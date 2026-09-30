"""PRISM-P model: energy-conditioned per-beamlet proton dose queries.

The proton analogue of :class:`prismx.model.PrismX`. It REUSES the photon
transport encoder (:class:`prismx.encoder.TransportEncoder`) and the continuous
decoders + physics-residual composition (:mod:`prismx.decoder`) UNCHANGED; only
the channel / feature counts and the analytic baseline change (PROTON_SPEC.md
section 3):

- the encoder consumes the 13 proton channels of
  :data:`prismx.proton_features.PROTON_ENCODER_CHANNELS` on the PARALLEL proton
  grid (no inverse-square, energy conditioning instead of MLC aperture);
- the decoders consume the 15 proton analytic features of
  :data:`prismx.proton_features.PROTON_DECODER_FEATURES`;
- :func:`prismx.decoder.compose_dose` rides on the proton baseline ``A`` from
  :func:`prismx.proton_analytic.proton_baseline`; there is NO inverse-square, so
  the additive head's decaying prefactor is the Bragg ``IDD(WED; E)`` alone
  (``isq = 1``) instead of the photon ``isq * p_wed``.

``encode_beamlet`` builds the parallel grid + latent transport memory; ``query``
routes continuous world points through the main (in-box) or far-field decoder;
``predict_dose_grid`` is region-restricted exactly like ``PrismX``. The decoder
heads are zero-initialized (in :mod:`prismx.decoder`), so the UNTRAINED PrismP
reproduces the proton analytic baseline on in-beam points.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from prismx.config import DecoderConfig, EncoderConfig, PrismXConfig
from prismx.cone_grid import CTVolume
from prismx.decoder import FarFieldDecoder, MainDecoder, compose_dose
from prismx.encoder import TransportEncoder
from prismx.geometry import BeamGeometry
from prismx.model import _sample_latent
from prismx.proton_analytic import BraggIDD
from prismx.proton_features import (
    PROTON_DECODER_FEATURES,
    PROTON_ENCODER_CHANNELS,
    _BeamletProxy,
    decoder_features_proton,
    encoder_channels_proton,
)
from prismx.proton_grid import ProtonGrid, ProtonGridSpec, build_proton_grid

# Number of proton encoder channels / decoder features (kept here so the configs
# below stay in sync with prismx.proton_features without importing the tuples at
# class-definition time).
N_PROTON_CHANNELS: int = len(PROTON_ENCODER_CHANNELS)   # 13
N_PROTON_FEATURES: int = len(PROTON_DECODER_FEATURES)   # 15

# OOD guard: bound the proton additive TERM at compose_dose. softplus(g_a) is
# unbounded; on MR-net density (out-of-distribution for this CT-commissioned
# engine) the residual additive explodes to ~1.6e5/voxel (proton/MRI integral 30x,
# wrecking sMAE/MAE while invisible to the local >10% gamma). In-distribution
# proton_ct's additive maxes at ~2.3, so this cap sits far above legit and far
# below the blow-up -> restores proton_ct AND keeps proton/MRI metrics meaningful.
# Clamping the TERM (not g_a) is essential: legit + OOD both reach softplus ~6e6;
# only the idd*softplus PRODUCT separates them. Photon does NOT pass add_clamp.
PROTON_ADD_CLAMP: float = 30.0

# The learned encoder runs only on the lateral bounding box of the beamlet's
# dose-carrying core plus this margin per side (mirrors PrismX.ENCODER_BOX_MARGIN
# for the spot footprint); the latent is zero outside.
ENCODER_BOX_MARGIN_MM: float = 30.0

# Default lateral radius (mm) from the central axis for the ``core`` region of
# :meth:`PrismP.predict_dose_grid`: the in-spot + lateral-tail shell that carries
# essentially all the beamlet dose (a few sigma of the ~6-9 mm proton spread).
CORE_REGION_RADIUS_MM: float = 40.0

# Density threshold (RSP surrogate) for the ``body`` region.
BODY_REGION_DENSITY: float = 0.05

# Regions accepted by :meth:`PrismP.predict_dose_grid`.
_PREDICT_REGIONS: tuple = ("full", "body", "core")


def proton_encoder_config(base: EncoderConfig = EncoderConfig()) -> EncoderConfig:
    """Photon :class:`EncoderConfig` with ``in_channels`` set to the proton count.

    Everything else (TCN width, dilations, latent channels) is reused unchanged,
    so the encoder is byte-for-byte the photon transport memory with a different
    input projection.
    """
    return replace(base, in_channels=N_PROTON_CHANNELS)


def proton_decoder_config(base: DecoderConfig = DecoderConfig()) -> DecoderConfig:
    """Photon :class:`DecoderConfig` with ``n_analytic`` set to the proton count."""
    return replace(base, n_analytic=N_PROTON_FEATURES)


@dataclass
class BeamletState:
    """Per-beamlet state produced by :meth:`PrismP.encode_beamlet`."""

    grid: ProtonGrid
    spot_uv: tuple
    energy_mev: float
    h: torch.Tensor  # [latent_channels, D, U, V]


class PrismP(nn.Module):
    """Energy-conditioned proton physics-residual dose engine (CPU reference).

    Reuses the photon encoder / decoders verbatim; only the I/O widths and the
    analytic baseline are proton-specific. Construct with a commissioned
    :class:`prismx.proton_analytic.BraggIDD` (range + Bragg shape); without one
    it falls back to :meth:`BraggIDD.default` (the Bortfeld seed).
    """

    def __init__(
        self,
        bragg: BraggIDD | None = None,
        cfg: PrismXConfig = PrismXConfig(),
        grid_spec: ProtonGridSpec = ProtonGridSpec(),
        sigma0_mm: float | None = None,
        sigma_growth: tuple | None = None,
        halo: tuple | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.grid_spec = grid_spec
        self.bragg = bragg if bragg is not None else BraggIDD.default()
        self.sigma0_mm = sigma0_mm
        # Commissioned lateral growth (a, b) of sigma^2 = sigma0^2 + a * wed^b;
        # None keeps the water seeds. MUST match the values the checkpoint was
        # trained with (the residual heads are paired to this baseline) -- the
        # trainer records them in the checkpoint as "sigma_params_used".
        self.sigma_growth = (
            (float(sigma_growth[0]), float(sigma_growth[1]))
            if sigma_growth is not None else None
        )
        # Double-Gaussian nuclear-halo term (halo_w, halo_ratio) of the lateral
        # kernel; None = single Gaussian (recom-v2/v3 pairing). Same pairing
        # contract: recorded as "halo_params_used" in every checkpoint.
        self.halo = (
            (float(halo[0]), float(halo[1])) if halo is not None else None
        )
        self.encoder = TransportEncoder(proton_encoder_config(cfg.encoder))
        self.dcfg = proton_decoder_config(cfg.decoder)
        self.main_decoder = MainDecoder(self.dcfg)
        self.far_decoder = FarFieldDecoder(self.dcfg)

    def n_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        """Device of the model parameters (first parameter's device).

        The pipeline device is defined by ``ct.density.device``; callers must
        keep the model on the same device (``model.to(ct.density.device)``).
        """
        return next(self.parameters()).device

    def _encoder_box(self, grid: ProtonGrid) -> tuple:
        """Lateral index slices (u, v) of the beamlet core (spot + margin).

        The spot sits at the lattice center (``u_rel = v_rel = 0``); the box is
        the central ``+-(CORE_REGION_RADIUS_MM + margin)`` window, clamped to the
        grid. Mirrors :meth:`PrismX._encoder_box` for the parallel spot footprint.
        """
        spec = grid.spec
        half = CORE_REGION_RADIUS_MM + ENCODER_BOX_MARGIN_MM

        def to_slice(n: int, step: float) -> slice:
            center = (n - 1) / 2
            i_lo = int(max(0, round(center - half / step)))
            i_hi = int(min(n, round(center + half / step) + 1))
            return slice(i_lo, max(i_hi, i_lo + 1))

        return to_slice(spec.nu, spec.du_mm), to_slice(spec.nv, spec.dv_mm)

    def encode_beamlet(
        self,
        ct: CTVolume,
        beam: BeamGeometry,
        spot_uv: tuple,
        energy_mev: float,
    ) -> BeamletState:
        """Build the parallel grid, encoder channels and latent transport memory.

        The encoder runs with gradients enabled (training differentiates through
        the latent) on the beamlet-core lateral sub-box; ``h`` is the full-grid
        latent with zeros outside that box. Everything runs on
        ``ct.density.device``; the model parameters must be on that device too.
        """
        grid = build_proton_grid(ct, beam, spot_uv, self.grid_spec)
        x = encoder_channels_proton(
            grid, beam, spot_uv, energy_mev, self.bragg, self.sigma0_mm,
            self.sigma_growth,
        )  # [13, D, U, V]
        su, sv = self._encoder_box(grid)
        h_box = self.encoder(x[:, :, su, sv].unsqueeze(0)).squeeze(0)
        h = torch.zeros(
            self.encoder.cfg.latent_channels,
            grid.n_depth,
            grid.spec.nu,
            grid.spec.nv,
            dtype=h_box.dtype,
            device=h_box.device,
        )
        h[:, :, su, sv] = h_box
        return BeamletState(
            grid=grid,
            spot_uv=(float(spot_uv[0]), float(spot_uv[1])),
            energy_mev=float(energy_mev),
            h=h,
        )

    def query(
        self,
        state: BeamletState,
        ct: CTVolume,
        beam: BeamGeometry,
        world_pts: torch.Tensor,
    ) -> torch.Tensor:
        """Predict dose at continuous world points ``[N, 3]`` -> ``[N]``.

        Composes the physics-residual dose on the proton baseline ``A``: NO
        inverse-square (``isq = 1``), with the Bragg ``IDD`` as the additive
        head's decaying depth prefactor.
        """
        feats, aux = decoder_features_proton(
            state.grid, ct, beam, state.spot_uv, state.energy_mev,
            world_pts, self.bragg, self.sigma0_mm, self.sigma_growth,
            self.halo,
        )
        if getattr(self, "baseline_only", False):
            # Pure analytic baseline A (no learned residual): the diagnostic
            # ``baseline_only`` path for the pure-A integral check (∫A/∫MC). No
            # surviving script sets this flag (set it manually when diagnosing).
            return aux["A"]
        in_box = aux["in_box"]
        far = ~in_box
        n = world_pts.shape[0]
        g_m = torch.zeros(n, dtype=torch.float32, device=state.h.device)
        g_a = torch.zeros(n, dtype=torch.float32, device=state.h.device)
        if bool(in_box.any()):
            latent = _sample_latent(state.grid, state.h, aux["uvd"][in_box])
            gm_in, ga_in = self.main_decoder(latent, feats[in_box])
            g_m[in_box] = gm_in.to(g_m.dtype)
            g_a[in_box] = ga_in.to(g_a.dtype)
        if bool(far.any()):
            gm_far, ga_far = self.far_decoder(feats[far])
            g_m[far] = gm_far.to(g_m.dtype)
            g_a[far] = ga_far.to(g_a.dtype)
        # No inverse-square for the parallel proton beam: isq = 1; the additive
        # head rides on the Bragg IDD depth factor (the proton analogue of the
        # photon p_wed * isq prefactor).
        isq = torch.ones_like(g_m)
        return compose_dose(aux["A"], aux["idd"], isq, g_m, g_a, self.dcfg, add_clamp=PROTON_ADD_CLAMP)

    def _grid_world_points(
        self,
        grid_origin_mm: tuple,
        grid_spacing_mm: tuple,
        grid_shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """World centers ``[Z*Y*X, 3]`` of a regular ``[Z, Y, X]`` voxel grid.

        Row-major over ``(z, y, x)`` so a flat result reshapes to ``[Z, Y, X]``
        (identical to :meth:`prismx.model.PrismX._grid_world_points`).
        """
        nz, ny, nx = (int(n) for n in grid_shape)
        sx, sy, sz = (float(s) for s in grid_spacing_mm)
        ox, oy, oz = (float(o) for o in grid_origin_mm)
        x = ox + sx * torch.arange(nx, dtype=torch.float32, device=device)
        y = oy + sy * torch.arange(ny, dtype=torch.float32, device=device)
        z = oz + sz * torch.arange(nz, dtype=torch.float32, device=device)
        gz, gy, gx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=1)

    def _region_mask(
        self,
        region: str,
        state: BeamletState,
        ct: CTVolume,
        beam: BeamGeometry,
        world_pts: torch.Tensor,
        radius_mm: float,
    ) -> torch.Tensor | None:
        """Boolean keep-mask ``[N]`` selecting query voxels for ``region``.

        ``full`` returns ``None`` (keep every voxel). ``body`` keeps voxels whose
        CT density exceeds :data:`BODY_REGION_DENSITY`. ``core`` keeps voxels
        inside the proton grid AND within ``radius_mm`` of the central axis
        (``r_perp <= radius_mm``) -- the in-spot + lateral-tail shell carrying
        the beamlet dose. Runs on ``world_pts.device`` with no host syncs.
        """
        if region == "full":
            return None
        if region == "body":
            return ct.sample_world(world_pts) > BODY_REGION_DENSITY
        if region == "core":
            geom = _BeamletProxy(beam, state.spot_uv)
            uvd = geom.world_to_beamlet(world_pts)
            r_perp = torch.sqrt(uvd[:, 0] ** 2 + uvd[:, 1] ** 2)
            return state.grid.contains(uvd) & (r_perp <= radius_mm)
        raise ValueError(f"region must be one of {_PREDICT_REGIONS}, got {region!r}")

    def predict_dose_grid(
        self,
        ct: CTVolume,
        beam: BeamGeometry,
        spot_uv: tuple,
        energy_mev: float,
        grid_origin_mm: tuple,
        grid_spacing_mm: tuple,
        grid_shape: tuple,
        region: str = "core",
        radius_mm: float = CORE_REGION_RADIUS_MM,
        batch_points: int = 2_000_000,
        fill: float = 0.0,
    ) -> torch.Tensor:
        """Predict dose on a regular voxel grid, restricted to ``region``.

        Encodes the beamlet ONCE, predicts only the voxels selected by ``region``
        and scatters them into a ``[Z, Y, X]`` volume initialized to ``fill``.
        The result is voxel-identical to a full-grid :meth:`query` on the
        selected voxels (the in-spot + tail for ``core``).

        Args:
            spot_uv / energy_mev: the beamlet's lateral spot and energy.
            grid_origin_mm: world position of voxel ``[0, 0, 0]``.
            grid_spacing_mm: ``(sx, sy, sz)`` voxel spacing in mm.
            grid_shape: ``(Z, Y, X)`` voxel counts.
            region: ``"full"`` (all voxels), ``"body"`` (CT density >
                :data:`BODY_REGION_DENSITY`) or ``"core"`` (inside the proton
                grid AND within ``radius_mm`` of the central axis).
            radius_mm: lateral radius for ``region="core"``.
            batch_points: chunk size for the query (bounds memory).
            fill: value of out-of-region voxels.

        Returns:
            ``[Z, Y, X]`` float32 dose tensor on ``ct.density.device``.
        """
        if region not in _PREDICT_REGIONS:
            raise ValueError(
                f"region must be one of {_PREDICT_REGIONS}, got {region!r}"
            )
        device = ct.density.device
        nz, ny, nx = (int(n) for n in grid_shape)
        n_vox = nz * ny * nx
        pts = self._grid_world_points(grid_origin_mm, grid_spacing_mm, grid_shape, device)

        state = self.encode_beamlet(ct, beam, spot_uv, energy_mev)
        mask = self._region_mask(region, state, ct, beam, pts, radius_mm)
        sel = pts if mask is None else pts[mask]
        preds = torch.empty(sel.shape[0], dtype=torch.float32, device=device)
        for start in range(0, sel.shape[0], batch_points):
            chunk = sel[start:start + batch_points]
            preds[start:start + batch_points] = self.query(state, ct, beam, chunk)

        flat = torch.full((n_vox,), float(fill), dtype=torch.float32, device=device)
        if mask is None:
            flat = preds.to(torch.float32)
        else:
            flat[mask] = preds.to(torch.float32)
        return flat.reshape(nz, ny, nx)

    def forward(
        self,
        ct: CTVolume,
        beam: BeamGeometry,
        spot_uv: tuple,
        energy_mev: float,
        world_pts: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the beamlet and query ``world_pts`` in one call."""
        state = self.encode_beamlet(ct, beam, spot_uv, energy_mev)
        return self.query(state, ct, beam, world_pts)
