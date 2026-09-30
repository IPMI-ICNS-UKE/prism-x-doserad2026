"""CT volume sampling and the per-control-point beam's-eye-view cone grid."""

from __future__ import annotations
import math
import os as _os
from dataclasses import dataclass, replace
import torch
from prismx.config import AIR_DENSITY, SURFACE_DENSITY_THRESHOLD, ConeGridSpec
from prismx.geometry import BeamGeometry, cone_to_beam

DEPTH_BELOW_SURFACE_CLAMP_MM: float = 500.0


def _trilinear(
    volume: torch.Tensor,
    f0: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    outside: float,
) -> torch.Tensor:
    """Trilinear interpolation of ``volume`` [N0, N1, N2] at fractional indices."""
    n0, n1, n2 = volume.shape
    inside = (
        (f0 >= 0)
        & (f0 <= n0 - 1)
        & (f1 >= 0)
        & (f1 <= n1 - 1)
        & (f2 >= 0)
        & (f2 <= n2 - 1)
    )
    f0 = f0.clamp(0.0, float(n0 - 1))
    f1 = f1.clamp(0.0, float(n1 - 1))
    f2 = f2.clamp(0.0, float(n2 - 1))
    i0 = f0.floor().long().clamp(0, max(n0 - 2, 0))
    i1 = f1.floor().long().clamp(0, max(n1 - 2, 0))
    i2 = f2.floor().long().clamp(0, max(n2 - 2, 0))
    j0 = (i0 + 1).clamp(max=n0 - 1)
    j1 = (i1 + 1).clamp(max=n1 - 1)
    j2 = (i2 + 1).clamp(max=n2 - 1)
    w0 = f0 - i0.to(f0.dtype)
    w1 = f1 - i1.to(f1.dtype)
    w2 = f2 - i2.to(f2.dtype)
    flat = volume.reshape(-1)

    def lin(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (a * n1 + b) * n2 + c

    corners = flat[
        torch.stack(
            (
                lin(i0, i1, i2),
                lin(i0, i1, j2),
                lin(i0, j1, i2),
                lin(i0, j1, j2),
                lin(j0, i1, i2),
                lin(j0, i1, j2),
                lin(j0, j1, i2),
                lin(j0, j1, j2),
            )
        )
    ]
    v000, v001, v010, v011, v100, v101, v110, v111 = corners.unbind(0)
    c00 = v000 * (1 - w2) + v001 * w2
    c01 = v010 * (1 - w2) + v011 * w2
    c10 = v100 * (1 - w2) + v101 * w2
    c11 = v110 * (1 - w2) + v111 * w2
    c0 = c00 * (1 - w1) + c01 * w1
    c1 = c10 * (1 - w1) + c11 * w1
    out = c0 * (1 - w0) + c1 * w0
    return torch.where(inside, out, torch.full_like(out, float(outside)))


@dataclass
class CTVolume:
    """A world-aligned relative-electron-density volume."""

    density: torch.Tensor
    spacing_mm: tuple
    origin_mm: tuple

    def to(self, device) -> "CTVolume":
        """Copy of the volume with ``density`` on ``device``."""
        return CTVolume(
            density=self.density.to(torch.device(device)),
            spacing_mm=self.spacing_mm,
            origin_mm=self.origin_mm,
        )

    def sample_world(
        self, pts: torch.Tensor, outside_value: float = AIR_DENSITY
    ) -> torch.Tensor:
        """Trilinear density at world points [N, 3] (mm); outside -> ``outside_value``."""
        sx, sy, sz = (float(s) for s in self.spacing_mm)
        ox, oy, oz = (float(o) for o in self.origin_mm)
        p = pts.to(torch.float32)
        fx = (p[..., 0] - ox) / sx
        fy = (p[..., 1] - oy) / sy
        fz = (p[..., 2] - oz) / sz
        return _trilinear(self.density.to(torch.float32), fz, fy, fx, outside_value)


@dataclass
class ConeGrid:
    """Entry-aligned beam's-eye-view grid; field tensors are ``[D, U, V]``."""

    spec: ConeGridSpec
    beam: BeamGeometry
    d_start: float
    n_depth: int
    rho: torch.Tensor
    wed: torch.Tensor
    depth_below_surface: torch.Tensor
    ssd: torch.Tensor

    def axes(self) -> tuple:
        """1-D coordinate tensors ``(u [U], v [V], d [D])`` in mm, on ``rho.device``."""
        s = self.spec
        dev = self.rho.device
        u = (
            torch.arange(s.nu, dtype=torch.float32, device=dev) - (s.nu - 1) / 2
        ) * s.du_mm
        v = (
            torch.arange(s.nv, dtype=torch.float32, device=dev) - (s.nv - 1) / 2
        ) * s.dv_mm
        d = (
            self.d_start
            + torch.arange(self.n_depth, dtype=torch.float32, device=dev) * s.dd_mm
        )
        return (u, v, d)

    def _frac_indices(self, uvd: torch.Tensor) -> tuple:
        """Continuous (u, v, d) -> fractional grid indices (fu, fv, fd)."""
        s = self.spec
        q = uvd.to(torch.float32)
        fu = q[..., 0] / s.du_mm + (s.nu - 1) / 2
        fv = q[..., 1] / s.dv_mm + (s.nv - 1) / 2
        fd = (q[..., 2] - self.d_start) / s.dd_mm
        return (fu, fv, fd)

    def sample(
        self, field: torch.Tensor, uvd: torch.Tensor, outside: float = 0.0
    ) -> torch.Tensor:
        """Trilinear sample of ``field`` [D, U, V] at cone coords ``uvd`` [N, 3] -> [N]."""
        fu, fv, fd = self._frac_indices(uvd)
        return _trilinear(field.to(torch.float32), fd, fu, fv, outside)

    def frac_offsets(self, uvd: torch.Tensor) -> torch.Tensor:
        """In-cell offsets (delta_u, delta_v, delta_d) [N, 3], index units in [-0.5, 0.5]."""
        fu, fv, fd = self._frac_indices(uvd)
        return torch.stack([fu - fu.round(), fv - fv.round(), fd - fd.round()], dim=-1)

    def contains(self, uvd: torch.Tensor) -> torch.Tensor:
        """Boolean mask [N]: True where ``uvd`` lies inside the grid support."""
        s = self.spec
        fu, fv, fd = self._frac_indices(uvd)
        return (
            (fu >= 0)
            & (fu <= s.nu - 1)
            & (fv >= 0)
            & (fv <= s.nv - 1)
            & (fd >= 0)
            & (fd <= self.n_depth - 1)
        )


def _sample_rho(
    ct: CTVolume, beam: BeamGeometry, spec: ConeGridSpec, d_axis: torch.Tensor
) -> torch.Tensor:
    """Density on the full [D, U, V] cone lattice with one ``sample_world`` call."""
    dev = ct.density.device
    n_d = int(d_axis.numel())
    u = (
        torch.arange(spec.nu, dtype=torch.float32, device=dev) - (spec.nu - 1) / 2
    ) * spec.du_mm
    v = (
        torch.arange(spec.nv, dtype=torch.float32, device=dev) - (spec.nv - 1) / 2
    ) * spec.dv_mm
    uu = u.view(1, spec.nu, 1).expand(n_d, spec.nu, spec.nv)
    vv = v.view(1, 1, spec.nv).expand(n_d, spec.nu, spec.nv)
    dd = (
        d_axis.to(device=dev, dtype=torch.float32)
        .view(n_d, 1, 1)
        .expand(n_d, spec.nu, spec.nv)
    )
    uvd = torch.stack([uu, vv, dd], dim=-1).reshape(-1, 3)
    pts_world = beam.beam_to_world(cone_to_beam(uvd, beam.sad_mm))
    return ct.sample_world(pts_world).reshape(n_d, spec.nu, spec.nv)


def _place_window(ct: CTVolume, beam: BeamGeometry, spec: ConeGridSpec) -> tuple:
    """Auto-place the cone depth window ``(d_start, n_depth)`` from the entry surface."""
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
    src = beam.source_world().cpu().to(torch.float64)
    d_lo = max(dd, float((src - torch.max(box_lo, torch.min(src, box_hi))).norm()))
    corners = torch.stack(
        [
            torch.where(
                torch.tensor([m >> a & 1 for a in range(3)], dtype=torch.bool),
                box_hi,
                box_lo,
            )
            for m in range(8)
        ]
    )
    d_hi = float((corners - src).norm(dim=1).max())
    d_hi = max(d_hi, d_lo + dd)
    stride = max(1, int(getattr(spec, "window_scout_stride", 1)))
    scout_spec = spec
    extra_margin = 0.0
    if stride > 1:
        nu_s = max(2, (spec.nu - 1) // stride + 1)
        nv_s = max(2, (spec.nv - 1) // stride + 1)
        scout_spec = replace(
            spec,
            nu=nu_s,
            nv=nv_s,
            du_mm=(spec.nu - 1) * spec.du_mm / (nu_s - 1),
            dv_mm=(spec.nv - 1) * spec.dv_mm / (nv_s - 1),
        )
        extra_margin = 2.0 * dd
    n_cand = int(math.ceil((d_hi - d_lo) / dd)) + 1
    d_cand = d_lo + torch.arange(n_cand, dtype=torch.float32, device=dev) * dd
    rho_cand = _sample_rho(ct, beam, scout_spec, d_cand)
    hit_depth = (rho_cand > SURFACE_DENSITY_THRESHOLD).reshape(n_cand, -1).any(dim=1)
    hits = hit_depth.nonzero()
    if hits.numel() > 0:
        d_entry = d_lo + int(hits[0]) * dd
        d_start = max(dd, d_entry - spec.entry_margin_mm - extra_margin)
        _exit_margin = float(_os.environ.get("PRISMX_EXIT_MARGIN_MM", "0") or 0.0)
        if _exit_margin > 0.0:
            d_exit = d_lo + int(hits[-1]) * dd
            d_hi = max(d_start + dd, min(d_hi, d_exit + _exit_margin + extra_margin))
    else:
        d_start = d_lo
    n_depth = max(
        1, min(spec.max_depth_steps, int(math.ceil((d_hi - d_start) / dd)) + 1)
    )
    if extra_margin > 0.0 and n_depth >= spec.max_depth_steps:
        d_start = max(dd, d_start + extra_margin)
        n_depth = max(
            1, min(spec.max_depth_steps, int(math.ceil((d_hi - d_start) / dd)) + 1)
        )
    return (d_start, n_depth)


def build_cone_grid(
    ct: CTVolume,
    beam: BeamGeometry,
    spec: ConeGridSpec,
    *,
    d_start: float | None = None,
    n_depth: int | None = None,
) -> ConeGrid:
    """Ray-march the cone through ``ct`` and build the entry-aligned grid."""
    dev = ct.density.device
    dd = spec.dd_mm
    if d_start is None or n_depth is None:
        d_start, n_depth = _place_window(ct, beam, spec)
    else:
        d_start, n_depth = (max(dd, float(d_start)), max(1, int(n_depth)))
    d_axis = d_start + torch.arange(n_depth, dtype=torch.float32, device=dev) * dd
    rho = _sample_rho(ct, beam, spec, d_axis)
    wed = torch.cumsum(rho * dd, dim=0)
    surf = rho > SURFACE_DENSITY_THRESHOLD
    k_idx = torch.arange(n_depth, device=dev).view(-1, 1, 1).expand_as(surf)
    k_surf = torch.where(surf, k_idx, torch.full_like(k_idx, n_depth)).amin(dim=0)
    ssd = d_start + k_surf.to(torch.float32) * dd
    depth_below_surface = (d_axis.view(-1, 1, 1) - ssd.unsqueeze(0)).clamp(
        -DEPTH_BELOW_SURFACE_CLAMP_MM, DEPTH_BELOW_SURFACE_CLAMP_MM
    )
    return ConeGrid(
        spec=spec,
        beam=beam,
        d_start=float(d_start),
        n_depth=n_depth,
        rho=rho,
        wed=wed,
        depth_below_surface=depth_below_surface,
        ssd=ssd,
    )
