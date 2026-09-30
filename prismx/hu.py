"""HU -> density conversion matching the ground-truth simulations."""

from __future__ import annotations
import numpy as np
import torch
from prismx.cone_grid import CTVolume

HU_DENSITY_TABLE: tuple = (
    (-1024.0, 0.0012),
    (-999.0, 0.00121),
    (-200.0, 0.8043754),
    (-199.0, 0.8183035),
    (-10.0, 1.006579),
    (-9.0, 0.9966749),
    (120.0, 1.126553),
    (121.0, 1.095097),
    (3000.0, 3.027294),
    (4000.0, 3.698428),
)


def hu_to_density(hu: torch.Tensor, table: tuple = HU_DENSITY_TABLE) -> torch.Tensor:
    """Convert HU values to density (same shape, float32, edge-clamped)."""
    arr = np.asarray(table, dtype=np.float64)
    out = np.interp(hu.numpy().astype(np.float64), arr[:, 0], arr[:, 1])
    return torch.from_numpy(out.astype(np.float32))


def ct_volume_from_hu(
    hu: torch.Tensor,
    spacing_mm: tuple,
    origin_mm: tuple,
    table: tuple = HU_DENSITY_TABLE,
) -> CTVolume:
    """Build a :class:`CTVolume` (density) from an HU volume [Z, Y, X]."""
    return CTVolume(
        density=hu_to_density(hu, table),
        spacing_mm=tuple((float(s) for s in spacing_mm)),
        origin_mm=tuple((float(o) for o in origin_mm)),
    )
