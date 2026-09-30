"""GPU-parallel signed Euclidean distance transforms via Jump Flooding."""

from __future__ import annotations
import torch

_UNSEEDED: float = 1e18


def _jump_offsets(h: int, w: int) -> list:
    """Powers-of-two jump steps for a ``[h, w]`` grid: ``..., 4, 2, 1``."""
    k = 1
    while k < max(h, w):
        k *= 2
    steps = []
    while k >= 1:
        steps.append(k)
        k //= 2
    steps.append(1)
    return steps


def _flood(
    seed_y: torch.Tensor,
    seed_x: torch.Tensor,
    best_d2: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
    sy_mm: float,
    sx_mm: float,
    offsets: list,
    steps: list,
) -> tuple:
    """Run the jump-flooding passes in place over the seed-coordinate fields."""
    for k in steps:
        for dy, dx in offsets:
            sy = seed_y.roll(shifts=(dy * k, dx * k), dims=(0, 1))
            sx = seed_x.roll(shifts=(dy * k, dx * k), dims=(0, 1))
            if dy != 0:
                if dy * k > 0:
                    sy[: dy * k, :] = -1
                else:
                    sy[dy * k :, :] = -1
            if dx != 0:
                if dx * k > 0:
                    sy[:, : dx * k] = -1
                else:
                    sy[:, dx * k :] = -1
            valid = sy >= 0
            dyy = (yy - sy).to(best_d2.dtype) * sy_mm
            dxx = (xx - sx).to(best_d2.dtype) * sx_mm
            cand_d2 = dyy * dyy + dxx * dxx
            cand_d2 = torch.where(valid, cand_d2, best_d2.new_full((), _UNSEEDED))
            take = cand_d2 < best_d2
            best_d2 = torch.where(take, cand_d2, best_d2)
            seed_y = torch.where(take, sy, seed_y)
            seed_x = torch.where(take, sx, seed_x)
    return (seed_y, seed_x, best_d2)


def jfa_edt(mask: torch.Tensor, spacing: tuple = (1.0, 1.0)) -> torch.Tensor:
    """Approximate Euclidean distance transform of a boolean mask via JFA."""
    device = mask.device
    mask = mask.to(torch.bool)
    h, w = mask.shape
    sy, sx = (float(spacing[0]), float(spacing[1]))
    if not bool(mask.any()):
        return torch.zeros((h, w), dtype=torch.float32, device=device)
    if bool(mask.all()):
        return torch.full((h, w), float(_UNSEEDED), dtype=torch.float32, device=device)
    yy = torch.arange(h, device=device).view(h, 1).expand(h, w)
    xx = torch.arange(w, device=device).view(1, w).expand(h, w)
    seed = ~mask
    seed_y = torch.where(seed, yy, torch.full_like(yy, -1))
    seed_x = torch.where(seed, xx, torch.full_like(xx, -1))
    best_d2 = torch.where(
        seed,
        torch.zeros((h, w), dtype=torch.float32, device=device),
        torch.full((h, w), _UNSEEDED, dtype=torch.float32, device=device),
    )
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    seed_y, seed_x, _ = _flood(
        seed_y, seed_x, best_d2, yy, xx, sy, sx, offsets, _jump_offsets(h, w)
    )
    reached = seed_y >= 0
    dy = (yy - seed_y).to(torch.float32) * sy
    dx = (xx - seed_x).to(torch.float32) * sx
    dist = torch.sqrt(dy * dy + dx * dx)
    return torch.where(reached, dist, torch.full_like(dist, float(_UNSEEDED)))


def edt_1d_axis(mask: torch.Tensor, spacing_mm: float, axis: int = 0) -> torch.Tensor:
    """Exact 1-D Euclidean distance to the nearest ``False`` along one axis."""
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    work = mask if axis == 0 else mask.transpose(0, 1)
    out = _edt_1d_along_rows(work, spacing_mm)
    return out if axis == 0 else out.transpose(0, 1)


def _edt_1d_along_rows(mask: torch.Tensor, spacing_mm: float) -> torch.Tensor:
    """1-D Euclidean distance to the nearest ``False`` along dim 0, per column."""
    h = mask.shape[0]
    device = mask.device
    big = h + 1
    rows = (
        torch.arange(h, device=device).view(h, *[1] * (mask.dim() - 1)).expand_as(mask)
    )
    is_bg = ~mask
    pos_fwd = torch.where(is_bg, rows, torch.full_like(rows, -big))
    nearest_above = torch.cummax(pos_fwd, dim=0).values
    dist_above = torch.where(
        nearest_above >= 0, rows - nearest_above, torch.full_like(rows, big)
    )
    pos_bwd = torch.where(is_bg, rows, torch.full_like(rows, big + h))
    flip = torch.flip(pos_bwd, dims=[0])
    nearest_below = torch.flip(torch.cummin(flip, dim=0).values, dims=[0])
    dist_below = torch.where(
        nearest_below <= h, nearest_below - rows, torch.full_like(rows, big)
    )
    d_idx = torch.minimum(dist_above, dist_below).to(torch.float32)
    d_idx = torch.where(is_bg, torch.zeros_like(d_idx), d_idx)
    return d_idx * float(spacing_mm)
