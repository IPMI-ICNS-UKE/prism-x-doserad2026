"""PRISM-X model: per-CP transport-memory encoding + continuous dose queries."""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from prismx.analytic import PDDTable
from prismx.aperture import ApertureRaster, MLCAperture, build_raster
from prismx.cone_grid import ConeGrid, CTVolume, build_cone_grid
from prismx.config import PrismXConfig
from prismx.decoder import FarFieldDecoder, MainDecoder, compose_dose
from prismx.encoder import TransportEncoder
from prismx.features import decoder_features, encoder_channels
from prismx.geometry import BeamGeometry, beam_to_cone

ENCODER_BOX_MARGIN_MM: float = 60.0
CONE_REGION_MARGIN_MM: float = 40.0
BODY_REGION_DENSITY: float = 0.05
_PREDICT_REGIONS: tuple = ("full", "body", "cone")


@dataclass
class CPState:
    """Per-control-point state produced by :meth:`PrismX.encode_cp`."""

    cone: ConeGrid
    raster: ApertureRaster
    aperture: MLCAperture
    h: torch.Tensor


@dataclass(frozen=True)
class PreparedGridQuery:
    """Image-level query coordinates reusable across control points."""

    origin_mm: tuple[float, float, float]
    spacing_mm: tuple[float, float, float]
    shape: tuple[int, int, int]
    selected_world_pts: torch.Tensor
    flat_mask: torch.Tensor | None


def _coarse_bath_margin() -> float | None:
    """Deploy lever (speed round two): in-plane coarse-bath dual resolution."""
    if os.environ.get("PRISMX_COARSE_BATH", "0") != "1":
        return None
    return float(os.environ.get("PRISMX_COARSE_BATH_MARGIN_MM", "40"))


def _encoder_shape_bucket() -> bool:
    """Deploy lever (speed round two): encoder-box shape bucketing."""
    return os.environ.get("PRISMX_ENCODER_BUCKET", "0") == "1"


def _sample_latent(cone: ConeGrid, h: torch.Tensor, uvd: torch.Tensor) -> torch.Tensor:
    """Trilinear sample of the latent ``h`` [C, D, U, V] at ``uvd`` [N, 3]."""
    spec = cone.spec
    n_c, n_d = (h.shape[0], h.shape[1])
    q = uvd.to(torch.float32)
    fu = q[:, 0] / spec.du_mm + (spec.nu - 1) / 2
    fv = q[:, 1] / spec.dv_mm + (spec.nv - 1) / 2
    fd = (q[:, 2] - cone.d_start) / spec.dd_mm
    inside = (
        (fu >= 0)
        & (fu <= spec.nu - 1)
        & (fv >= 0)
        & (fv <= spec.nv - 1)
        & (fd >= 0)
        & (fd <= n_d - 1)
    )
    fu = fu.clamp(0.0, spec.nu - 1.0)
    fv = fv.clamp(0.0, spec.nv - 1.0)
    fd = fd.clamp(0.0, float(n_d - 1))
    iu = fu.floor().long().clamp(0, max(spec.nu - 2, 0))
    iv = fv.floor().long().clamp(0, max(spec.nv - 2, 0))
    id0 = fd.floor().long().clamp(0, max(n_d - 2, 0))
    ju = (iu + 1).clamp(max=spec.nu - 1)
    jv = (iv + 1).clamp(max=spec.nv - 1)
    jd = (id0 + 1).clamp(max=n_d - 1)
    wu = fu - iu.to(fu.dtype)
    wv = fv - iv.to(fv.dtype)
    wd = fd - id0.to(fd.dtype)
    flat = h.reshape(n_c, -1)

    def lin(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (a * spec.nu + b) * spec.nv + c

    n_q = id0.shape[0]
    idx = torch.stack(
        (
            lin(id0, iu, iv),
            lin(id0, iu, jv),
            lin(id0, ju, iv),
            lin(id0, ju, jv),
            lin(jd, iu, iv),
            lin(jd, iu, jv),
            lin(jd, ju, iv),
            lin(jd, ju, jv),
        )
    )
    corners = flat[:, idx.reshape(-1)].reshape(n_c, 8, n_q)
    v000, v001, v010, v011, v100, v101, v110, v111 = corners.unbind(1)
    c00 = v000 * (1 - wv) + v001 * wv
    c01 = v010 * (1 - wv) + v011 * wv
    c10 = v100 * (1 - wv) + v101 * wv
    c11 = v110 * (1 - wv) + v111 * wv
    c0 = c00 * (1 - wu) + c01 * wu
    c1 = c10 * (1 - wu) + c11 * wu
    out = c0 * (1 - wd) + c1 * wd
    return (out * inside.to(out.dtype)).transpose(0, 1)


def _sample_latent_grid(
    cone: ConeGrid, h: torch.Tensor, uvd: torch.Tensor
) -> torch.Tensor:
    """Sample ``h`` with CUDA's fused trilinear ``grid_sample`` kernel."""
    spec = cone.spec
    n_d = h.shape[1]
    q = uvd.to(torch.float32)
    fu = q[:, 0] / spec.du_mm + (spec.nu - 1) / 2
    fv = q[:, 1] / spec.dv_mm + (spec.nv - 1) / 2
    fd = (q[:, 2] - cone.d_start) / spec.dd_mm
    gv = 2.0 * fv / max(spec.nv - 1, 1) - 1.0
    gu = 2.0 * fu / max(spec.nu - 1, 1) - 1.0
    gd = 2.0 * fd / max(n_d - 1, 1) - 1.0
    grid = torch.stack((gv, gu, gd), dim=-1).reshape(1, 1, 1, -1, 3)
    return (
        F.grid_sample(
            h.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        .reshape(h.shape[0], -1)
        .transpose(0, 1)
    )


def _sample_latent_inference(
    cone: ConeGrid, h: torch.Tensor, uvd: torch.Tensor
) -> torch.Tensor:
    if os.environ.get("PRISMX_LATENT_GRID_SAMPLE", "0") == "1":
        return _sample_latent_grid(cone, h, uvd)
    return _sample_latent(cone, h, uvd)


def _aperture_bounds(aperture: MLCAperture) -> tuple[float, float, float, float]:
    """Tight open-leaf rectangle on the isocenter plane, clipped by jaws."""
    open_mask = aperture.right_mm > aperture.left_mm
    if not bool(open_mask.any()):
        return (
            float(aperture.jaw_u_mm[0]),
            float(aperture.jaw_u_mm[1]),
            float(aperture.jaw_v_mm[0]),
            float(aperture.jaw_v_mm[1]),
        )
    open_indices = torch.nonzero(open_mask, as_tuple=False).squeeze(1)
    leaf_width = float(aperture.mlc.leaf_width_mm)
    u_lo = max(float(aperture.left_mm[open_mask].min()), float(aperture.jaw_u_mm[0]))
    u_hi = min(float(aperture.right_mm[open_mask].max()), float(aperture.jaw_u_mm[1]))
    v_lo = max(
        float(aperture.v_lo_mm + int(open_indices.min()) * leaf_width),
        float(aperture.jaw_v_mm[0]),
    )
    v_hi = min(
        float(aperture.v_lo_mm + (int(open_indices.max()) + 1) * leaf_width),
        float(aperture.jaw_v_mm[1]),
    )
    return (u_lo, u_hi, v_lo, v_hi)


class PrismX(nn.Module):
    """Physics-residual implicit scatter memory dose engine (CPU reference)."""

    def __init__(self, cfg: PrismXConfig = PrismXConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransportEncoder(cfg.encoder)
        self.main_decoder = MainDecoder(cfg.decoder)
        self.far_decoder = FarFieldDecoder(cfg.decoder)
        self.pdd = PDDTable.default()

    def n_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum((p.numel() for p in self.parameters()))

    @property
    def device(self) -> torch.device:
        """Device of the model parameters (first parameter's device)."""
        return next(self.parameters()).device

    def _encoder_box(self, cone: ConeGrid, ap: MLCAperture) -> tuple:
        """Lateral index slices (u, v) of the open-aperture box + margin."""
        spec = cone.spec
        open_mask = ap.right_mm > ap.left_mm
        if not bool(open_mask.any()):
            return (slice(0, spec.nu), slice(0, spec.nv))
        u_lo = max(float(ap.left_mm[open_mask].min()), ap.jaw_u_mm[0])
        u_hi = min(float(ap.right_mm[open_mask].max()), ap.jaw_u_mm[1])
        open_idx = torch.nonzero(open_mask, as_tuple=False).squeeze(1)
        w = ap.mlc.leaf_width_mm
        v_lo = max(ap.v_lo_mm + float(open_idx.min()) * w, ap.jaw_v_mm[0])
        v_hi = min(ap.v_lo_mm + float(open_idx.max() + 1) * w, ap.jaw_v_mm[1])

        def to_slice(lo: float, hi: float, n: int, step: float) -> slice:
            center = (n - 1) / 2
            i_lo = int(max(0, math.floor(lo / step + center)))
            i_hi = int(min(n, math.ceil(hi / step + center) + 1))
            return slice(i_lo, max(i_hi, i_lo + 1))

        m = ENCODER_BOX_MARGIN_MM
        return (
            to_slice(u_lo - m, u_hi + m, spec.nu, spec.du_mm),
            to_slice(v_lo - m, v_hi + m, spec.nv, spec.dv_mm),
        )

    def encode_cp(
        self,
        ct: CTVolume,
        beam: BeamGeometry,
        ap: MLCAperture,
        cone_cache: dict | None = None,
    ) -> CPState:
        """Build cone grid, raster stack and latent transport memory for one CP."""
        device = ct.density.device
        if cone_cache is not None:
            key = (
                float(beam.gantry_deg),
                float(beam.collimator_deg),
                float(beam.couch_deg),
                tuple(map(float, beam.isocenter_mm)),
                float(beam.sad_mm),
            )
            cone = cone_cache.get(key)
            if cone is None:
                cone = build_cone_grid(ct, beam, self.cfg.cone)
                cone_cache[key] = cone
        else:
            cone = build_cone_grid(ct, beam, self.cfg.cone)
        raster = build_raster(ap.to(device), self.cfg.raster)
        x = encoder_channels(cone, raster, ap)
        su, sv = self._encoder_box(cone, ap)
        xb = x[:, :, su, sv].unsqueeze(0)
        if _encoder_shape_bucket():
            d0, u0, v0 = xb.shape[-3:]
            pad = (-v0 % 16, -u0 % 16, -d0 % 16)
            if any(pad):
                xb = F.pad(xb, (0, pad[0], 0, pad[1], 0, pad[2]))
            h_box = self.encoder(xb, pad_valid=(d0, u0, v0)).squeeze(0)[
                :, :d0, :u0, :v0
            ]
        else:
            h_box = self.encoder(xb).squeeze(0)
        h = torch.zeros(
            self.cfg.encoder.latent_channels,
            cone.n_depth,
            cone.spec.nu,
            cone.spec.nv,
            dtype=h_box.dtype,
            device=h_box.device,
        )
        h[:, :, su, sv] = h_box
        return CPState(cone=cone, raster=raster, aperture=ap, h=h)

    def query(
        self,
        state: CPState,
        ct: CTVolume,
        beam: BeamGeometry,
        world_pts: torch.Tensor,
        fp32: bool = False,
    ) -> torch.Tensor:
        """Predict dose at continuous world points ``[N, 3]`` -> ``[N]``."""
        feats, aux = decoder_features(
            state.cone,
            state.raster,
            state.aperture,
            ct,
            beam,
            world_pts,
            self.pdd,
            fp32=fp32,
        )
        in_box = aux["in_box"]
        far = ~in_box
        n = world_pts.shape[0]
        g_m = torch.zeros(n, dtype=torch.float32, device=state.h.device)
        g_a = torch.zeros(n, dtype=torch.float32, device=state.h.device)
        if bool(in_box.any()):
            latent = _sample_latent_inference(state.cone, state.h, aux["uvd"][in_box])
            gm_in, ga_in = self.main_decoder(latent, feats[in_box])
            g_m[in_box] = gm_in.to(g_m.dtype)
            g_a[in_box] = ga_in.to(g_a.dtype)
        if bool(far.any()):
            gm_far, ga_far = self.far_decoder(feats[far])
            g_m[far] = gm_far.to(g_m.dtype)
            g_a[far] = ga_far.to(g_a.dtype)
        return compose_dose(
            aux["A"], aux["p_wed"], aux["isq"], g_m, g_a, self.cfg.decoder
        )

    def _predict_prepared_coarse_bath(
        self,
        state: CPState,
        ct: CTVolume,
        beam: BeamGeometry,
        prepared: PreparedGridQuery,
        shape: tuple,
        batch_points: int,
        margin_mm: float,
    ) -> torch.Tensor:
        """Dual-resolution prediction over a prepared grid (see"""
        nz, ny, nx = shape
        sel = prepared.selected_world_pts
        device = sel.device
        cache = getattr(prepared, "_coarse_bath_cache", None)
        if cache is None:
            if prepared.flat_mask is not None:
                flat_idx = prepared.flat_mask.nonzero(as_tuple=False).squeeze(1)
            else:
                flat_idx = torch.arange(nz * ny * nx, device=device)
            rem = flat_idx % (ny * nx)
            lattice = (rem // nx % 2 == 0) & (rem % nx % 2 == 0)
            cache = (flat_idx, lattice)
            object.__setattr__(prepared, "_coarse_bath_cache", cache)
        flat_idx, lattice = cache
        if os.environ.get("PRISMX_COARSE_BATH_AABB", "0") == "1":
            near = self.aperture_margin_query_mask(
                prepared, beam, state.aperture, margin_mm
            )
        else:
            uvd = beam.world_to_beam(sel, fp32=True)
            sad = float(beam.sad_mm)
            scale_iso = sad / (uvd[:, 2] + sad).clamp_min(0.001)
            sd = state.raster.sample(
                "sdf", uvd[:, 0] * scale_iso, uvd[:, 1] * scale_iso
            )
            near = sd > -float(margin_mm)
        eval_mask = near | lattice
        sel_eval = sel[eval_mask]
        preds_eval = torch.empty(sel_eval.shape[0], dtype=torch.float32, device=device)
        for start in range(0, sel_eval.shape[0], batch_points):
            chunk = sel_eval[start : start + batch_points]
            preds_eval[start : start + batch_points] = self.query(
                state, ct, beam, chunk, fp32=True
            )
        preds = torch.zeros(sel.shape[0], dtype=torch.float32, device=device)
        preds[eval_mask] = preds_eval
        need = ~eval_mask
        if bool(need.any()):
            with torch.autocast(device.type, enabled=False):
                vol = torch.zeros(nz * ny * nx, dtype=torch.float32, device=device)
                ind = torch.zeros_like(vol)
                vol[flat_idx[lattice]] = preds[lattice]
                ind[flat_idx[lattice]] = 1.0
                coarse_v = vol.reshape(nz, ny, nx)[:, ::2, ::2]
                coarse_i = ind.reshape(nz, ny, nx)[:, ::2, ::2]
                up_v = F.interpolate(
                    coarse_v[None, None],
                    size=(nz, ny, nx),
                    mode="trilinear",
                    align_corners=False,
                )[0, 0]
                up_i = F.interpolate(
                    coarse_i[None, None],
                    size=(nz, ny, nx),
                    mode="trilinear",
                    align_corners=False,
                )[0, 0]
                interp = torch.where(
                    up_i > 0.001, up_v / up_i.clamp_min(0.001), torch.zeros_like(up_v)
                )
                preds[need] = interp.reshape(-1)[flat_idx[need]]
        return preds

    def _grid_world_points(
        self,
        grid_origin_mm: tuple,
        grid_spacing_mm: tuple,
        grid_shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """World centers ``[Z*Y*X, 3]`` of a regular ``[Z, Y, X]`` voxel grid."""
        nz, ny, nx = (int(n) for n in grid_shape)
        sx, sy, sz = (float(s) for s in grid_spacing_mm)
        ox, oy, oz = (float(o) for o in grid_origin_mm)
        x = ox + sx * torch.arange(nx, dtype=torch.float32, device=device)
        y = oy + sy * torch.arange(ny, dtype=torch.float32, device=device)
        z = oz + sz * torch.arange(nz, dtype=torch.float32, device=device)
        gz, gy, gx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=1)

    def prepare_grid_query(
        self,
        ct: CTVolume,
        grid_origin_mm: tuple,
        grid_spacing_mm: tuple,
        grid_shape: tuple,
        voxel_mask: torch.Tensor | None = None,
    ) -> PreparedGridQuery:
        """Prepare image-level coordinates for repeated ``region="full"`` calls."""
        device = ct.density.device
        shape = tuple((int(n) for n in grid_shape))
        if len(shape) != 3:
            raise ValueError(f"grid_shape must have three axes, got {shape}")
        origin = tuple((float(v) for v in grid_origin_mm))
        spacing = tuple((float(v) for v in grid_spacing_mm))
        pts = self._grid_world_points(origin, spacing, shape, device)
        mask = None
        if voxel_mask is not None:
            custom = torch.as_tensor(voxel_mask, device=device)
            if tuple(custom.shape) != shape:
                raise ValueError(
                    f"voxel_mask shape {tuple(custom.shape)} does not match grid_shape {shape}"
                )
            mask = custom.to(torch.bool).reshape(-1)
            pts = pts[mask]
        return PreparedGridQuery(
            origin_mm=origin,
            spacing_mm=spacing,
            shape=shape,
            selected_world_pts=pts,
            flat_mask=mask,
        )

    def aperture_margin_query_mask(
        self,
        prepared_grid: PreparedGridQuery,
        beam: BeamGeometry,
        aperture: MLCAperture,
        margin_mm: float,
    ) -> torch.Tensor:
        """Select cached query points inside an aperture-aligned rectangle."""
        margin = float(margin_mm)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("margin_mm must be finite and non-negative")
        points = prepared_grid.selected_world_pts
        uvd = beam_to_cone(
            beam.world_to_beam(points, fp32=True), beam.sad_mm, fp32=True
        )
        u_lo, u_hi, v_lo, v_hi = _aperture_bounds(aperture)
        u, v = (uvd[:, 0], uvd[:, 1])
        return (
            (u >= u_lo - margin)
            & (u <= u_hi + margin)
            & (v >= v_lo - margin)
            & (v <= v_hi + margin)
        )

    def _region_mask(
        self,
        region: str,
        state: CPState,
        ct: CTVolume,
        beam: BeamGeometry,
        world_pts: torch.Tensor,
        margin_mm: float,
    ) -> torch.Tensor | None:
        """Boolean keep-mask ``[N]`` selecting query voxels for ``region``."""
        if region == "full":
            return None
        if region == "body":
            return ct.sample_world(world_pts) > BODY_REGION_DENSITY
        if region == "cone":
            uvd = beam_to_cone(
                beam.world_to_beam(world_pts, fp32=True), beam.sad_mm, fp32=True
            )
            in_cone = state.cone.contains(uvd)
            sdf = state.raster.sample("sdf", uvd[:, 0], uvd[:, 1])
            return in_cone & (sdf >= -margin_mm)
        raise ValueError(f"region must be one of {_PREDICT_REGIONS}, got {region!r}")

    def predict_dose_grid(
        self,
        ct: CTVolume,
        beam: BeamGeometry,
        ap: MLCAperture,
        grid_origin_mm: tuple,
        grid_spacing_mm: tuple,
        grid_shape: tuple,
        region: str = "cone",
        margin_mm: float = CONE_REGION_MARGIN_MM,
        batch_points: int = 2000000,
        fill: float = 0.0,
        amp: bool = False,
        voxel_mask: torch.Tensor | None = None,
        prepared_grid: PreparedGridQuery | None = None,
        prepared_selection: torch.Tensor | None = None,
        cone_cache: dict | None = None,
    ) -> torch.Tensor:
        """Predict dose on a regular voxel grid, restricted to ``region``."""
        if region not in _PREDICT_REGIONS:
            raise ValueError(
                f"region must be one of {_PREDICT_REGIONS}, got {region!r}"
            )
        device = ct.density.device
        use_amp = bool(amp) and device.type == "cuda"
        nz, ny, nx = (int(n) for n in grid_shape)
        n_vox = nz * ny * nx
        if prepared_selection is not None and prepared_grid is None:
            raise ValueError("prepared_selection requires prepared_grid")
        if prepared_grid is not None:
            if region != "full":
                raise ValueError("prepared_grid requires region='full'")
            if voxel_mask is not None:
                raise ValueError("prepared_grid and voxel_mask are mutually exclusive")
            expected_origin = tuple((float(v) for v in grid_origin_mm))
            expected_spacing = tuple((float(v) for v in grid_spacing_mm))
            expected_shape = (nz, ny, nx)
            if (
                prepared_grid.origin_mm != expected_origin
                or prepared_grid.spacing_mm != expected_spacing
                or prepared_grid.shape != expected_shape
            ):
                raise ValueError(
                    "prepared_grid geometry does not match prediction grid"
                )
            if prepared_grid.selected_world_pts.device != device:
                raise ValueError("prepared_grid and CT must be on the same device")
            pts = prepared_grid.selected_world_pts
            mask = prepared_grid.flat_mask
            if mask is not None and (
                mask.device != device
                or mask.dtype != torch.bool
                or mask.numel() != n_vox
            ):
                raise ValueError("prepared_grid scatter mask is incompatible")
            selection = None
            if prepared_selection is not None:
                selection = torch.as_tensor(prepared_selection, device=device)
                if (
                    selection.dtype != torch.bool
                    or selection.ndim != 1
                    or selection.numel() != pts.shape[0]
                ):
                    raise ValueError(
                        "prepared_selection must be a boolean vector over prepared_grid.selected_world_pts"
                    )
        else:
            pts = self._grid_world_points(
                grid_origin_mm, grid_spacing_mm, grid_shape, device
            )
            mask = None
            selection = None
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            state = self.encode_cp(ct, beam, ap, cone_cache=cone_cache)
            if prepared_grid is None:
                mask = self._region_mask(region, state, ct, beam, pts, margin_mm)
                if voxel_mask is not None:
                    custom = torch.as_tensor(voxel_mask, device=device)
                    if tuple(custom.shape) != (nz, ny, nx):
                        raise ValueError(
                            f"voxel_mask shape {tuple(custom.shape)} does not match grid_shape {(nz, ny, nx)}"
                        )
                    custom = custom.to(torch.bool).reshape(-1)
                    mask = custom if mask is None else mask & custom
                sel = pts if mask is None else pts[mask]
            else:
                sel = pts if selection is None else pts[selection]
            coarse_margin = (
                _coarse_bath_margin()
                if prepared_grid is not None and selection is None
                else None
            )
            if coarse_margin is not None:
                preds = self._predict_prepared_coarse_bath(
                    state,
                    ct,
                    beam,
                    prepared_grid,
                    (nz, ny, nx),
                    batch_points,
                    coarse_margin,
                )
            else:
                preds = torch.empty(sel.shape[0], dtype=torch.float32, device=device)
                for start in range(0, sel.shape[0], batch_points):
                    chunk = sel[start : start + batch_points]
                    preds[start : start + batch_points] = self.query(
                        state, ct, beam, chunk, fp32=True
                    )
        if prepared_grid is not None and selection is not None:
            prepared_values = torch.full(
                (pts.shape[0],), float(fill), dtype=torch.float32, device=device
            )
            prepared_values[selection] = preds.to(torch.float32)
            preds = prepared_values
        flat = torch.full((n_vox,), float(fill), dtype=torch.float32, device=device)
        if mask is None:
            flat = preds.to(torch.float32)
        else:
            flat[mask] = preds.to(torch.float32)
        return flat.reshape(nz, ny, nx)

    def forward(
        self, ct: CTVolume, beam: BeamGeometry, ap: MLCAperture, world_pts: torch.Tensor
    ) -> torch.Tensor:
        """Encode the control point and query ``world_pts`` in one call."""
        return self.query(self.encode_cp(ct, beam, ap), ct, beam, world_pts)
