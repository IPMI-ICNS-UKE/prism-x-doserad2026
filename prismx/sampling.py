"""Stratified training-query sampling (SPEC.md section 9)."""

from __future__ import annotations
import numpy as np
import torch
from prismx.config import SURFACE_DENSITY_THRESHOLD, LossConfig, SamplingConfig
from prismx.geometry import beam_to_cone

STRATUM_NAMES: tuple = ("in_field", "penumbra", "buildup", "interface", "out_of_field")
IN_FIELD, PENUMBRA, BUILDUP, INTERFACE, OUT_OF_FIELD = range(5)


def _voxel_world_coords(
    shape: tuple, origin: tuple, spacing: tuple, device: torch.device
) -> torch.Tensor:
    """World coordinates (x, y, z) of all voxel centers of a [Z, Y, X] grid."""
    nz, ny, nx = shape
    x = origin[0] + spacing[0] * torch.arange(nx, dtype=torch.float32, device=device)
    y = origin[1] + spacing[1] * torch.arange(ny, dtype=torch.float32, device=device)
    z = origin[2] + spacing[2] * torch.arange(nz, dtype=torch.float32, device=device)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((gx, gy, gz), dim=-1).reshape(-1, 3)


def _trilinear_world(
    grid: torch.Tensor, origin: tuple, spacing: tuple, pts: torch.Tensor
) -> torch.Tensor:
    """Edge-clamped trilinear interpolation of a world-aligned [Z, Y, X] grid."""
    nz, ny, nx = grid.shape
    fx = ((pts[:, 0] - origin[0]) / spacing[0]).clamp(0.0, nx - 1)
    fy = ((pts[:, 1] - origin[1]) / spacing[1]).clamp(0.0, ny - 1)
    fz = ((pts[:, 2] - origin[2]) / spacing[2]).clamp(0.0, nz - 1)
    x0, y0, z0 = (fx.floor().long(), fy.floor().long(), fz.floor().long())
    x1 = (x0 + 1).clamp(max=nx - 1)
    y1 = (y0 + 1).clamp(max=ny - 1)
    z1 = (z0 + 1).clamp(max=nz - 1)
    tx, ty, tz = (fx - x0, fy - y0, fz - z0)
    flat = grid.reshape(-1)

    def at(zi: torch.Tensor, yi: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return flat[(zi * ny + yi) * nx + xi]

    c00 = at(z0, y0, x0) * (1 - tx) + at(z0, y0, x1) * tx
    c01 = at(z0, y1, x0) * (1 - tx) + at(z0, y1, x1) * tx
    c10 = at(z1, y0, x0) * (1 - tx) + at(z1, y0, x1) * tx
    c11 = at(z1, y1, x0) * (1 - tx) + at(z1, y1, x1) * tx
    return (c00 * (1 - ty) + c01 * ty) * (1 - tz) + (c10 * (1 - ty) + c11 * ty) * tz


def stratified_query_points(
    dose: torch.Tensor,
    grid_origin_mm: tuple,
    grid_spacing_mm: tuple,
    beam,
    raster,
    cone,
    cfg: SamplingConfig,
    loss_cfg: LossConfig,
    rng: np.random.Generator,
    n: int | None = None,
) -> dict:
    """Draw stratified training queries from one MC dose grid."""
    dose = dose.to(torch.float32)
    dev = dose.device
    n_total = int(n) if n is not None else cfg.n_queries_per_cp
    origin = tuple((float(o) for o in grid_origin_mm))
    spacing = tuple((float(s) for s in grid_spacing_mm))
    pts = _voxel_world_coords(tuple(dose.shape), origin, spacing, dev)
    flat = dose.reshape(-1)
    d_max = float(flat.max())
    uvd = beam_to_cone(beam.world_to_beam(pts), beam.sad_mm)
    rho_pt = cone.sample(cone.rho, uvd, outside=0.0)
    body = rho_pt > SURFACE_DENSITY_THRESHOLD
    sdf = raster.sample("sdf", uvd[:, 0], uvd[:, 1])
    dbs = cone.sample(cone.depth_below_surface, uvd, outside=-1000000.0)
    drho = torch.gradient(cone.rho, spacing=(float(cone.spec.dd_mm),), dim=0)[0]
    grad_pt = cone.sample(drho.abs(), uvd, outside=0.0)
    masks = {
        IN_FIELD: flat > cfg.in_field_isodose * d_max,
        PENUMBRA: sdf.abs() <= loss_cfg.penumbra_band_mm,
        BUILDUP: (dbs >= 0.0) & (dbs <= cfg.buildup_depth_mm),
        INTERFACE: grad_pt >= cfg.interface_grad_threshold_per_mm,
        OUT_OF_FIELD: body & (flat <= cfg.out_of_field_isodose * d_max),
    }
    fracs = (
        cfg.frac_in_field,
        cfg.frac_penumbra,
        cfg.frac_buildup,
        cfg.frac_interface,
        cfg.frac_out_of_field,
    )
    counts = [int(round(f * n_total)) for f in fracs]
    counts[IN_FIELD] += n_total - sum(counts)
    body_idx = torch.nonzero(body, as_tuple=False).squeeze(1).cpu().numpy()
    if body_idx.size == 0:
        body_idx = np.arange(flat.numel())
    spacing_t = torch.tensor(spacing, dtype=torch.float32, device=dev)
    out_pts, out_tgt, out_w, out_code, out_jit = ([], [], [], [], [])
    for code, count in enumerate(counts):
        if count <= 0:
            continue
        pool = torch.nonzero(masks[code], as_tuple=False).squeeze(1).cpu().numpy()
        if pool.size == 0:
            pool = body_idx
        idx_np = rng.choice(pool, size=count, replace=pool.size < count)
        idx = torch.from_numpy(np.ascontiguousarray(idx_np, dtype=np.int64)).to(dev)
        p = pts[idx].clone()
        tgt = flat[idx].clone()
        jit = torch.zeros(count, dtype=torch.bool, device=dev)
        n_jit = int(round(cfg.frac_jitter * count))
        if n_jit > 0:
            jit[:n_jit] = True
            offs = rng.uniform(-0.5, 0.5, size=(n_jit, 3)).astype(np.float32)
            p[:n_jit] += torch.from_numpy(offs).to(dev) * spacing_t
            tgt[:n_jit] = _trilinear_world(dose, origin, spacing, p[:n_jit])
        w = torch.ones(count, dtype=torch.float32, device=dev)
        w[jit] = loss_cfg.jitter_weight
        if code == PENUMBRA:
            w *= loss_cfg.penumbra_weight
        out_pts.append(p)
        out_tgt.append(tgt)
        out_w.append(w)
        out_code.append(torch.full((count,), code, dtype=torch.int8, device=dev))
        out_jit.append(jit)
    return {
        "world_pts": torch.cat(out_pts),
        "target": torch.cat(out_tgt),
        "m_weight": torch.cat(out_w),
        "stratum": torch.cat(out_code),
        "jittered": torch.cat(out_jit),
    }
