"""Feature assembly for PRISM-P (the proton arm of PRISM-X).

Mirrors :mod:`prismx.features` but for PARALLEL pencil beamlets (PROTON_SPEC.md
section 3). The two responsibilities are the same:

- :func:`encoder_channels_proton` builds the 13 encoder input channels on the
  parallel proton grid ``[13, D, U, V]`` in ``PROTON_ENCODER_CHANNELS`` order.
- :func:`decoder_features_proton` evaluates the 15 continuous analytic features
  (in ``PROTON_DECODER_FEATURES`` order) plus the analytic-baseline auxiliaries
  at arbitrary world query points, including a far-field PARALLEL ray-march WED
  for points outside the proton grid.

The proton channel set differs from the photon one only in what physics it
encodes, NOT in spirit:

- There is NO inverse-square term and NO MLC/aperture conditioning (the photon
  ``log_isq``, ``psi_*``, ``sd_*``, ``eq_sq`` channels are gone).
- ENERGY conditioning replaces the aperture conditioning: the Bragg range
  ``R(E)`` (the physical conditioning variable, PROTON_SPEC section 5) and the
  residual range ``R(E) - WED`` enter as features, and the integrated Bragg
  depth dose ``IDD(WED; E)`` and lateral spread ``sigma(WED, E)`` are sampled
  from the commissioned :class:`prismx.proton_analytic.BraggIDD`.
- The 3-exp spectral attenuation surrogate (a photon scatter feature) is gone;
  WED, the residual range and the Bragg IDD are the depth-transport features.

Lateral coordinates ``(u_rel, v_rel)`` are spot-relative and constant along a
beamlet (parallel geometry); ``r_perp = sqrt(u_rel^2 + v_rel^2)`` is the
distance from the central axis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prismx.cone_grid import CTVolume
from prismx.config import SURFACE_DENSITY_THRESHOLD
from prismx.geometry import BeamGeometry
from prismx.proton_analytic import BraggIDD, lateral_sigma, proton_baseline
from prismx.proton_grid import DEPTH_BELOW_SURFACE_CLAMP_MM, ProtonGrid

# Ordered proton encoder input channels (PROTON_SPEC.md section 3). NO
# inverse-square, NO aperture: energy conditioning (R(E), residual range, IDD,
# lateral sigma) replaces the photon ISQ / MLC channels.
PROTON_ENCODER_CHANNELS: tuple = (
    "rho_e",          # RSP-surrogate density rho(u, v, d)
    "drho_dd",        # d rho / dd along the (parallel) depth axis
    "rho_mean_3x3",   # lateral 3x3 box mean of rho
    "rho_mean_9x9",   # lateral 9x9 box mean of rho
    "wed",            # water-equivalent depth (cm)
    "resid_range",    # R(E) - WED, residual range (cm); sign flips past the peak
    "idd_bragg",      # commissioned integrated Bragg depth dose IDD(WED; E)
    "sigma_lat",      # lateral MCS spread sigma(WED, E) (cm)
    "r_perp",         # distance from the central axis (cm)
    "u_rel",          # spot-relative u (normalized)
    "v_rel",          # spot-relative v (normalized)
    "energy_norm",    # beamlet energy E broadcast (normalized)
    "range_norm",     # Bragg range R(E) broadcast (normalized)
)

# Ordered proton analytic decoder features (PROTON_SPEC.md section 3): the
# continuous analogues of the encoder channels at arbitrary world points, plus
# the baseline log dose, the surface/depth bookkeeping and the in-cell offsets.
PROTON_DECODER_FEATURES: tuple = (
    "log_A",                # log of the analytic baseline A = G * IDD
    "wed",                  # water-equivalent depth (cm)
    "resid_range",          # R(E) - WED (cm)
    "idd_bragg",            # IDD(WED; E)
    "sigma_lat",            # lateral sigma(WED, E) (cm)
    "r_perp",               # distance from the central axis (cm)
    "rho_e",                # local RSP-surrogate density
    "drho_dd",              # d rho / dd along the beamlet axis
    "ssd",                  # surface depth d_surface (cm)
    "depth_below_surface",  # d - d_surface (cm)
    "energy_norm",          # beamlet energy (normalized)
    "range_norm",           # Bragg range R(E) (normalized)
    "frac_du",              # in-cell u offset to the nearest grid node
    "frac_dv",              # in-cell v offset
    "frac_dd",              # in-cell depth offset
)

assert len(PROTON_ENCODER_CHANNELS) == 13
assert len(PROTON_DECODER_FEATURES) == 15

# Normalization scales: lateral coords / 60 mm (the default proton grid lateral
# half-extent), WED / depth in cm, energy / 250 MeV and range / 300 mm so the
# 31.7-200.8 MeV / ~45-200 mm WED span maps to O(1). Matches the spirit of the
# photon INTERFACES.md normalizations (lateral / extent, depth in cm).
_UV_NORM_MM: float = 60.0
_MM_PER_CM: float = 10.0
_ENERGY_NORM_MEV: float = 250.0
_RANGE_NORM_MM: float = 300.0
_LOG_EPS: float = 1e-6

# Far-field WED: direct PARALLEL ray marching (source plane -> point), the
# analogue of the photon divergent ray march but along the constant beam axis.
RAY_MARCH_STEPS: int = 128
_RAY_MARCH_CHUNK: int = 4096


def _sigma_kwargs(sigma0_mm: float | None, sigma_growth: tuple | None) -> dict:
    """Kwargs for :func:`lateral_sigma` / :func:`proton_baseline` from the
    commissioned overrides: ``sigma0_mm`` (spot floor) and ``sigma_growth``
    ``(a, b)`` of ``sigma^2 = sigma0^2 + a * wed^b``. ``None`` keeps the seed."""
    kw: dict = {}
    if sigma0_mm is not None:
        kw["sigma0_mm"] = float(sigma0_mm)
    if sigma_growth is not None:
        kw["a"], kw["b"] = float(sigma_growth[0]), float(sigma_growth[1])
    return kw


def _lateral_mean(rho: torch.Tensor, kernel: int) -> torch.Tensor:
    """Lateral (u, v) box mean of a [D, U, V] field, 'same' padding.

    Edge cells average only the in-grid neighbours (``count_include_pad=False``),
    so homogeneous fields stay exactly constant. Identical to the photon
    :func:`prismx.features._lateral_mean`.
    """
    x = rho.unsqueeze(1)  # [D, 1, U, V]; depth acts as the batch dim
    out = F.avg_pool2d(
        x, kernel, stride=1, padding=kernel // 2, count_include_pad=False
    )
    return out.squeeze(1)


def encoder_channels_proton(
    grid: ProtonGrid,
    beam: BeamGeometry,
    spot_uv: tuple,
    energy_mev: float,
    bragg: BraggIDD,
    sigma0_mm: float | None = None,
    sigma_growth: tuple | None = None,
) -> torch.Tensor:
    """Assemble the 13 proton encoder input channels on the parallel grid.

    Returns ``[13, D, U, V]`` float32 in ``PROTON_ENCODER_CHANNELS`` order. The
    energy is a per-beamlet scalar: ``R(E)``, the residual range ``R(E) - WED``,
    the Bragg ``IDD(WED; E)``, the lateral ``sigma(WED, E)`` and the normalized
    energy / range are broadcast over the lattice. NO inverse-square term.

    ``grid`` must live on the pipeline device (``grid.rho.device``), which is
    also the output device; ``beam`` / ``spot_uv`` describe the same beamlet the
    grid was built from. ``sigma0_mm`` / ``sigma_growth`` ``(a, b)`` override the
    commissioned lateral-spread parameters (``None`` keeps the
    :mod:`prismx.proton_analytic` seeds).
    """
    spec = grid.spec
    rho = grid.rho
    n_d = grid.n_depth
    dev = rho.device
    u_ax, v_ax, _ = grid.axes()

    drho = torch.gradient(rho, spacing=(float(spec.dd_mm),), dim=0)[0]

    e = torch.tensor(float(energy_mev), device=dev)
    r_e = bragg.range_table.R_of_E(e)  # Bragg range R(E) in WED (mm), scalar
    wed = grid.wed  # [D, U, V]
    resid_range = (r_e - wed)  # mm; > 0 proximal to the peak
    idd = bragg(wed, e, is_energy=True)  # [D, U, V] relative IDD
    sig_kw = _sigma_kwargs(sigma0_mm, sigma_growth)
    sigma_lat = lateral_sigma(wed, e, **sig_kw)  # [D, U, V] mm

    uu = u_ax.view(1, spec.nu, 1).expand(n_d, spec.nu, spec.nv)
    vv = v_ax.view(1, 1, spec.nv).expand(n_d, spec.nu, spec.nv)
    r_perp = torch.sqrt(uu * uu + vv * vv)

    channels = {
        "rho_e": rho,
        "drho_dd": drho,
        "rho_mean_3x3": _lateral_mean(rho, 3),
        "rho_mean_9x9": _lateral_mean(rho, 9),
        "wed": wed / _MM_PER_CM,
        "resid_range": resid_range / _MM_PER_CM,
        "idd_bragg": idd,
        "sigma_lat": sigma_lat / _MM_PER_CM,
        "r_perp": r_perp / _MM_PER_CM,
        "u_rel": (uu / _UV_NORM_MM),
        "v_rel": (vv / _UV_NORM_MM),
        "energy_norm": torch.full(
            (n_d, spec.nu, spec.nv), float(energy_mev) / _ENERGY_NORM_MEV, device=dev
        ),
        "range_norm": torch.full(
            (n_d, spec.nu, spec.nv), float(r_e) / _RANGE_NORM_MM, device=dev
        ),
    }
    return torch.stack(
        [channels[name].to(torch.float32) for name in PROTON_ENCODER_CHANNELS], dim=0
    )


def _bilinear_uv(
    grid: ProtonGrid, field: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Bilinear sample of a per-column [U, V] field at spot-relative (u, v) mm.

    Edge-clamped; mirrors :func:`prismx.features._bilinear_uv` on the proton
    grid's lateral spacing.
    """
    spec = grid.spec
    fu = (u / spec.du_mm + (spec.nu - 1) / 2).clamp(0.0, spec.nu - 1.0)
    fv = (v / spec.dv_mm + (spec.nv - 1) / 2).clamp(0.0, spec.nv - 1.0)
    i0 = fu.floor().long().clamp(0, max(spec.nu - 2, 0))
    j0 = fv.floor().long().clamp(0, max(spec.nv - 2, 0))
    i1 = (i0 + 1).clamp(max=spec.nu - 1)
    j1 = (j0 + 1).clamp(max=spec.nv - 1)
    wu = fu - i0.to(fu.dtype)
    wv = fv - j0.to(fv.dtype)
    return (
        field[i0, j0] * (1 - wu) * (1 - wv)
        + field[i1, j0] * wu * (1 - wv)
        + field[i0, j1] * (1 - wu) * wv
        + field[i1, j1] * wu * wv
    )


def _frac_offsets(grid: ProtonGrid, uvd: torch.Tensor) -> torch.Tensor:
    """In-cell offsets (delta_u, delta_v, delta_d) [N, 3], in [-0.5, 0.5].

    Offset of each query from its NEAREST grid node, in grid-spacing units --
    the proton analogue of :meth:`prismx.cone_grid.ConeGrid.frac_offsets`
    (``ProtonGrid`` does not carry this method).
    """
    fu, fv, fd = grid._frac_indices(uvd)
    return torch.stack(
        [fu - fu.round(), fv - fv.round(), fd - fd.round()], dim=-1
    )


def _ray_march_parallel(
    ct: CTVolume,
    beam: BeamGeometry,
    spot_uv: tuple,
    uvd: torch.Tensor,
    pts_world: torch.Tensor,
    miss_ssd: float,
) -> tuple:
    """Far-field WED and surface depth by marching the PARALLEL beamlet axis.

    Unlike the photon divergent march (source -> point), the proton ray is
    parallel: WED at a point is the integral of density from the source plane
    (``d = 0``) down to the point's axis depth ``d``, along the column through
    its spot-relative ``(u, v)``. Each ray is sampled at ``RAY_MARCH_STEPS``
    midpoints between the source plane and the point; the surface depth is the
    distance to the first sample above ``SURFACE_DENSITY_THRESHOLD``
    (``miss_ssd`` when the column never crosses tissue).

    Returns ``(wed_mm [N], ssd_mm [N])`` on ``pts_world.device``. ``uvd`` are the
    beamlet coords (``u_rel, v_rel, d``) of ``pts_world`` (spot already removed).
    """
    dev = pts_world.device
    u0, v0 = float(spot_uv[0]), float(spot_uv[1])
    sad = float(beam.sad_mm)
    n_steps = RAY_MARCH_STEPS
    t_mid = (torch.arange(n_steps, dtype=torch.float32, device=dev) + 0.5) / n_steps
    step_idx = torch.arange(n_steps, device=dev)

    wed_parts, ssd_parts = [], []
    for u_c, v_c, d_c in zip(
        uvd[:, 0].split(_RAY_MARCH_CHUNK),
        uvd[:, 1].split(_RAY_MARCH_CHUNK),
        uvd[:, 2].split(_RAY_MARCH_CHUNK),
    ):
        m = u_c.shape[0]
        # March from the source plane (d=0) to each point's depth d along its
        # own parallel column (u0 + u_rel, v0 + v_rel) in the beam frame.
        depth = d_c.clamp(min=0.0)  # [M]
        d_samp = t_mid.view(1, -1) * depth.view(m, 1)  # [M, S] axis depth
        ub = (u0 + u_c).view(m, 1).expand(m, n_steps)
        vb = (v0 + v_c).view(m, 1).expand(m, n_steps)
        zb = d_samp - sad  # beam-frame z = d - SAD
        pts_beam = torch.stack(
            [ub.reshape(-1), vb.reshape(-1), zb.reshape(-1)], dim=-1
        )
        pts_w = beam.beam_to_world(pts_beam)
        rho = ct.sample_world(pts_w).reshape(m, n_steps)
        wed_parts.append(rho.sum(dim=1) * depth / n_steps)
        hit = rho > SURFACE_DENSITY_THRESHOLD
        first = torch.where(hit, step_idx, n_steps).amin(dim=1)
        d_surf = (first.to(torch.float32) + 0.5) / n_steps * depth
        ssd_parts.append(
            torch.where(first < n_steps, d_surf, torch.full_like(d_surf, miss_ssd))
        )
    return torch.cat(wed_parts), torch.cat(ssd_parts)


def decoder_features_proton(
    grid: ProtonGrid,
    ct: CTVolume,
    beam: BeamGeometry,
    spot_uv: tuple,
    energy_mev: float,
    world_pts: torch.Tensor,
    bragg: BraggIDD,
    sigma0_mm: float | None = None,
    sigma_growth: tuple | None = None,
    halo: tuple | None = None,
) -> tuple:
    """Continuous proton analytic features at world query points.

    Returns ``(features [N, 15], aux)`` with features in
    ``PROTON_DECODER_FEATURES`` order and
    ``aux = {"A", "idd", "wed", "r_perp", "uvd", "in_box"}``. In-box points take
    WED / SSD from the proton grid; far-field points march the CT along the
    PARALLEL beamlet axis (:func:`_ray_march_parallel`). NO inverse-square term.

    The baseline ``A`` is :func:`prismx.proton_analytic.proton_baseline`
    (``G(r_perp; sigma(WED, E)) * IDD(WED; E)``); ``idd`` is its on-axis depth
    factor (used by :func:`prismx.decoder.compose_dose` as the additive head's
    decaying prefactor, the proton analogue of the photon ``p_wed``).
    """
    spec = grid.spec
    pts = world_pts.to(torch.float32)
    geom = _BeamletProxy(beam, spot_uv)
    uvd = geom.world_to_beamlet(pts)  # (u_rel, v_rel, d)
    u, v, d = uvd[:, 0], uvd[:, 1], uvd[:, 2]
    r_perp = torch.sqrt(u * u + v * v)

    u_half = (spec.nu - 1) / 2 * spec.du_mm
    v_half = (spec.nv - 1) / 2 * spec.dv_mm
    in_box = grid.contains(uvd) & (u.abs() <= u_half) & (v.abs() <= v_half)
    far = ~in_box

    # SSD convention for columns that never cross tissue mirrors the grid:
    # surface placed one step past the deepest slice.
    miss_ssd = grid.d_start + grid.n_depth * spec.dd_mm
    wed_mm = torch.zeros_like(d)
    ssd_mm = torch.full_like(d, miss_ssd)
    if bool(in_box.any()):
        wed_mm[in_box] = grid.sample(grid.wed, uvd[in_box], outside=0.0)
        ssd_mm[in_box] = _bilinear_uv(grid, grid.ssd, u[in_box], v[in_box])
    if bool(far.any()):
        wed_mm[far], ssd_mm[far] = _ray_march_parallel(
            ct, beam, spot_uv, uvd[far], pts[far], miss_ssd
        )

    dbs_mm = (d - ssd_mm).clamp(
        -DEPTH_BELOW_SURFACE_CLAMP_MM, DEPTH_BELOW_SURFACE_CLAMP_MM
    )

    e = torch.tensor(float(energy_mev), device=pts.device)
    r_e = bragg.range_table.R_of_E(e)
    idd = bragg(wed_mm, e, is_energy=True)
    sig_kw = _sigma_kwargs(sigma0_mm, sigma_growth)
    # sigma_lat feature stays the CORE sigma (stable, well-defined input); the
    # halo enters the model ONLY through the baseline A / log_A.
    sigma_lat = lateral_sigma(wed_mm, e, **sig_kw)
    base_kw = dict(sig_kw)
    if halo is not None:
        base_kw["halo_w"], base_kw["halo_ratio"] = float(halo[0]), float(halo[1])
    a_val = proton_baseline(r_perp, wed_mm, e, bragg, **base_kw)

    rho_e = ct.sample_world(pts)
    # d rho / dd: central difference of one depth step along the PARALLEL axis
    # (constant direction, unlike the photon source ray).
    axis = geom.axis_world().to(device=pts.device, dtype=torch.float32)
    axis = axis / axis.norm().clamp_min(1e-6)
    half_step = 0.5 * spec.dd_mm * axis
    drho_dd = (
        ct.sample_world(pts + half_step) - ct.sample_world(pts - half_step)
    ) / spec.dd_mm

    frac = _frac_offsets(grid, uvd)  # [N, 3]
    feats = {
        "log_A": torch.log(a_val + _LOG_EPS),
        "wed": wed_mm / _MM_PER_CM,
        "resid_range": (r_e - wed_mm) / _MM_PER_CM,
        "idd_bragg": idd,
        "sigma_lat": sigma_lat / _MM_PER_CM,
        "r_perp": r_perp / _MM_PER_CM,
        "rho_e": rho_e,
        "drho_dd": drho_dd,
        "ssd": ssd_mm / _MM_PER_CM,
        "depth_below_surface": dbs_mm / _MM_PER_CM,
        "energy_norm": torch.full_like(d, float(energy_mev) / _ENERGY_NORM_MEV),
        "range_norm": torch.full_like(d, float(r_e) / _RANGE_NORM_MM),
        "frac_du": frac[:, 0],
        "frac_dv": frac[:, 1],
        "frac_dd": frac[:, 2],
    }
    features = torch.stack(
        [feats[name].to(torch.float32) for name in PROTON_DECODER_FEATURES], dim=-1
    )
    aux = {
        "A": a_val,
        "idd": idd,
        "wed": wed_mm,
        "r_perp": r_perp,
        "uvd": uvd,
        "in_box": in_box,
    }
    return features, aux


class _BeamletProxy:
    """Lightweight (beam, spot) -> beamlet-coordinate helper.

    Equivalent to :class:`prismx.proton_data.BeamletGeometry` but constructed
    inline so :func:`decoder_features_proton` does not depend on the dataset
    layer. ``world_to_beamlet`` and ``axis_world`` match that class exactly.
    """

    def __init__(self, beam: BeamGeometry, spot_uv: tuple) -> None:
        self.beam = beam
        self.spot_uv = (float(spot_uv[0]), float(spot_uv[1]))
        self.sad_mm = float(beam.sad_mm)

    def world_to_beamlet(self, pts: torch.Tensor) -> torch.Tensor:
        beam = self.beam.world_to_beam(pts)
        u_rel = beam[..., 0] - self.spot_uv[0]
        v_rel = beam[..., 1] - self.spot_uv[1]
        d = beam[..., 2] + self.sad_mm
        return torch.stack([u_rel, v_rel, d], dim=-1)

    def axis_world(self) -> torch.Tensor:
        origin = self.beam.beam_to_world(torch.zeros(1, 3))[0]
        tip = self.beam.beam_to_world(torch.tensor([[0.0, 0.0, 1.0]]))[0]
        return tip - origin
