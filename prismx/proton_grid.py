"""Parallel pencil-beam grid for protons (the analogue of the photon cone grid).

PROTON_SPEC.md section 2/4: a beamlet's density is resampled on a rotated
Cartesian lattice aligned with the beam axis. Unlike the divergent cone there is
no central projection - lateral coordinates ``(u, v)`` are spot-relative and
constant along depth, and ``d`` runs along the (parallel) beam axis. World
sample points are ``source + d * axis + (u0 + u) * u_hat + (v0 + v) * v_hat``,
which in the beam frame is simply ``(u0 + u, v0 + v, d - SAD)`` fed through
:meth:`prismx.geometry.BeamGeometry.beam_to_world`.

Water-equivalent depth is the inclusive prefix-scan ``cumsum(rho * dd)`` of the
(RSP-surrogate) density along ``d``; the depth window is entry-aligned exactly
like the cone grid (slice ``0`` sits ``entry_margin_mm`` upstream of the first
depth at which any ray exceeds ``SURFACE_DENSITY_THRESHOLD``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from prismx.config import SURFACE_DENSITY_THRESHOLD
from prismx.cone_grid import CTVolume, _trilinear
from prismx.geometry import BeamGeometry
from prismx.proton_data import SAD_MM

# Clamp for depth_below_surface in mm (matches the cone grid contract).
DEPTH_BELOW_SURFACE_CLAMP_MM: float = 500.0


@dataclass(frozen=True)
class ProtonGridSpec:
    """Parallel-beam grid sampling spec.

    Lateral extent spans ``+-((nu-1)/2) * du`` around the spot; the default
    ``+-60 mm`` covers the ~6-9 mm proton sigma plus its tail. ``dd`` is the
    depth step (range resolution); the depth window is entry-aligned and clipped
    to ``max_depth_steps`` slices.
    """

    nu: int = 61
    nv: int = 61
    du_mm: float = 2.0
    dv_mm: float = 2.0
    dd_mm: float = 2.0
    max_depth_steps: int = 320
    entry_margin_mm: float = 10.0


@dataclass
class ProtonGrid:
    """Entry-aligned parallel-beam grid; field tensors are ``[D, U, V]``.

    Depth index ``k`` maps to ``d_k = d_start + k * dd_mm`` (distance from the
    source plane along the beam axis); lateral index ``i`` maps to the
    spot-relative offset ``u_i = (i - (nu-1)/2) * du_mm`` (same for ``v``).

    All field tensors live on the device of the CT the grid was built from;
    ``sample`` / ``contains`` run on that device (callers pass ``uvd`` there).
    """

    spec: ProtonGridSpec
    beam: BeamGeometry
    spot_uv: tuple                     # (u0, v0) lateral spot, mm
    d_start: float
    n_depth: int
    rho: torch.Tensor                  # [D, U, V] RSP-surrogate density
    wed: torch.Tensor                  # [D, U, V] inclusive cumsum(rho * dd_mm)
    depth_below_surface: torch.Tensor  # [D, U, V] d_k - d_surface(u, v), clamped
    ssd: torch.Tensor                  # [U, V] d_surface per beamlet column

    def axes(self) -> tuple:
        """1-D coords ``(u [U], v [V], d [D])`` in mm (spot-relative u/v), on device."""
        s = self.spec
        dev = self.rho.device
        u = (torch.arange(s.nu, dtype=torch.float32, device=dev) - (s.nu - 1) / 2) * s.du_mm
        v = (torch.arange(s.nv, dtype=torch.float32, device=dev) - (s.nv - 1) / 2) * s.dv_mm
        d = self.d_start + torch.arange(self.n_depth, dtype=torch.float32, device=dev) * s.dd_mm
        return u, v, d

    def _frac_indices(self, uvd: torch.Tensor) -> tuple:
        """Continuous (u, v, d) -> fractional grid indices (fu, fv, fd)."""
        s = self.spec
        q = uvd.to(torch.float32)
        fu = q[..., 0] / s.du_mm + (s.nu - 1) / 2
        fv = q[..., 1] / s.dv_mm + (s.nv - 1) / 2
        fd = (q[..., 2] - self.d_start) / s.dd_mm
        return fu, fv, fd

    def sample(
        self, field: torch.Tensor, uvd: torch.Tensor, outside: float = 0.0
    ) -> torch.Tensor:
        """Trilinear sample of ``field`` [D, U, V] at coords ``uvd`` [N, 3] -> [N].

        ``uvd`` are spot-relative ``(u, v)`` and axis depth ``d`` (the same
        coords :meth:`prismx.proton_data.BeamletGeometry.world_to_beamlet`
        returns).
        """
        fu, fv, fd = self._frac_indices(uvd)
        return _trilinear(field.to(torch.float32), fd, fu, fv, outside)

    def contains(self, uvd: torch.Tensor) -> torch.Tensor:
        """Boolean mask [N]: True where ``uvd`` lies inside the grid support."""
        s = self.spec
        fu, fv, fd = self._frac_indices(uvd)
        return (
            (fu >= 0) & (fu <= s.nu - 1)
            & (fv >= 0) & (fv <= s.nv - 1)
            & (fd >= 0) & (fd <= self.n_depth - 1)
        )


def _sample_rho(
    ct: CTVolume,
    beam: BeamGeometry,
    spec: ProtonGridSpec,
    spot_uv: tuple,
    d_axis: torch.Tensor,
) -> torch.Tensor:
    """Density on the full [D, U, V] parallel lattice with one ``sample_world`` call.

    Beam-frame point ``(u0 + u, v0 + v, d - SAD)`` -> world. Runs entirely on
    ``ct.density.device``.
    """
    dev = ct.density.device
    n_d = int(d_axis.numel())
    u0, v0 = float(spot_uv[0]), float(spot_uv[1])
    sad = float(beam.sad_mm)
    u = (torch.arange(spec.nu, dtype=torch.float32, device=dev) - (spec.nu - 1) / 2) * spec.du_mm + u0
    v = (torch.arange(spec.nv, dtype=torch.float32, device=dev) - (spec.nv - 1) / 2) * spec.dv_mm + v0
    uu = u.view(1, spec.nu, 1).expand(n_d, spec.nu, spec.nv)
    vv = v.view(1, 1, spec.nv).expand(n_d, spec.nu, spec.nv)
    zz = (d_axis.to(device=dev, dtype=torch.float32) - sad).view(n_d, 1, 1).expand(n_d, spec.nu, spec.nv)
    pts_beam = torch.stack([uu, vv, zz], dim=-1).reshape(-1, 3)
    pts_world = beam.beam_to_world(pts_beam)
    return ct.sample_world(pts_world).reshape(n_d, spec.nu, spec.nv)


def build_proton_grid(
    ct: CTVolume,
    beam: BeamGeometry,
    spot_uv: tuple,
    spec: ProtonGridSpec,
) -> ProtonGrid:
    """Resample ``ct`` along the parallel beamlet and build the entry-aligned grid.

    Candidate depths span the CT support along the beam axis (nearest to
    farthest projection of the voxel-center bounding box onto the axis); the
    window starts ``entry_margin_mm`` upstream of the first candidate depth at
    which ANY column density exceeds ``SURFACE_DENSITY_THRESHOLD``, floored at
    one step ``dd_mm`` (depth from the source plane), and clipped to
    ``max_depth_steps`` slices.

    All grid tensors are built on ``ct.density.device``; the scalar window
    placement is one-time BUILD-time work and may sync to the host.
    """
    dev = ct.density.device
    dd = spec.dd_mm
    nz, ny, nx = ct.density.shape
    sx, sy, sz = (float(s) for s in ct.spacing_mm)
    ox, oy, oz = (float(o) for o in ct.origin_mm)
    box_lo = torch.tensor([ox, oy, oz], dtype=torch.float64)
    box_hi = torch.tensor(
        [ox + (nx - 1) * sx, oy + (ny - 1) * sy, oz + (nz - 1) * sz],
        dtype=torch.float64,
    )
    corners = torch.stack(
        [
            torch.where(
                torch.tensor([(m >> a) & 1 for a in range(3)], dtype=torch.bool),
                box_hi,
                box_lo,
            )
            for m in range(8)
        ]
    )

    # Depth window along the PARALLEL axis: project the bounding-box corners
    # onto the axis (d measured from the source plane), spanning the support.
    sad = float(beam.sad_mm)
    src_plane = beam.beam_to_world(
        torch.tensor([[float(spot_uv[0]), float(spot_uv[1]), -sad]])
    )[0].to(torch.float64)
    axis = (
        beam.beam_to_world(torch.tensor([[0.0, 0.0, 1.0]]))[0]
        - beam.beam_to_world(torch.zeros(1, 3))[0]
    ).to(torch.float64)
    d_proj = ((corners - src_plane) @ axis)
    d_lo = max(dd, float(d_proj.min()))
    d_hi = max(float(d_proj.max()), d_lo + dd)

    # Candidate pass: locate the entry depth on a dd-spaced lattice.
    n_cand = int(math.ceil((d_hi - d_lo) / dd)) + 1
    d_cand = d_lo + torch.arange(n_cand, dtype=torch.float32, device=dev) * dd
    rho_cand = _sample_rho(ct, beam, spec, spot_uv, d_cand)
    hit_depth = (rho_cand > SURFACE_DENSITY_THRESHOLD).reshape(n_cand, -1).any(dim=1)
    hits = hit_depth.nonzero()
    if hits.numel() > 0:
        d_entry = d_lo + int(hits[0]) * dd
        d_start = max(dd, d_entry - spec.entry_margin_mm)
    else:
        d_start = d_lo  # no tissue in the beamlet: keep the CT-spanning window

    n_depth = max(1, min(spec.max_depth_steps, int(math.ceil((d_hi - d_start) / dd)) + 1))
    d_axis = d_start + torch.arange(n_depth, dtype=torch.float32, device=dev) * dd

    rho = _sample_rho(ct, beam, spec, spot_uv, d_axis)
    wed = torch.cumsum(rho * dd, dim=0)

    # Per-column surface: first depth index above threshold; misses get index D,
    # which yields ssd = d_start + D * dd.
    surf = rho > SURFACE_DENSITY_THRESHOLD
    k_idx = torch.arange(n_depth, device=dev).view(-1, 1, 1).expand_as(surf)
    k_surf = torch.where(surf, k_idx, torch.full_like(k_idx, n_depth)).amin(dim=0)
    ssd = d_start + k_surf.to(torch.float32) * dd
    depth_below_surface = (d_axis.view(-1, 1, 1) - ssd.unsqueeze(0)).clamp(
        -DEPTH_BELOW_SURFACE_CLAMP_MM, DEPTH_BELOW_SURFACE_CLAMP_MM
    )

    return ProtonGrid(
        spec=spec,
        beam=beam,
        spot_uv=(float(spot_uv[0]), float(spot_uv[1])),
        d_start=float(d_start),
        n_depth=n_depth,
        rho=rho,
        wed=wed,
        depth_below_surface=depth_below_surface,
        ssd=ssd,
    )
