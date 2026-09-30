"""Procedural training phantoms (SPEC.md section 9)."""

from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from prismx.config import AIR_DENSITY
from prismx.cone_grid import CTVolume

DENSITY_MIN: float = 0.05
DENSITY_MAX: float = 2.0
_PERLIN_OCTAVES: int = 3
_PERLIN_AMPLITUDE: float = 0.35


def _grid(size_mm: tuple, spacing_mm) -> tuple:
    """Return (x, y, z) voxel-center coordinate arrays, spacing and origin."""
    if isinstance(spacing_mm, (int, float)):
        spacing = (float(spacing_mm),) * 3
    else:
        spacing = tuple((float(s) for s in spacing_mm))
    counts = tuple((max(1, int(round(size_mm[i] / spacing[i]))) for i in range(3)))
    coords = tuple(
        (
            (np.arange(counts[i], dtype=np.float64) - (counts[i] - 1) / 2) * spacing[i]
            for i in range(3)
        )
    )
    origin = tuple((float(c[0]) for c in coords))
    return (coords, spacing, origin)


def _volume(density_zyx: np.ndarray, spacing: tuple, origin: tuple) -> CTVolume:
    return CTVolume(
        density=torch.from_numpy(density_zyx.astype(np.float32)),
        spacing_mm=spacing,
        origin_mm=origin,
    )


def _broadcast_coords(coords: tuple) -> tuple:
    """Reshape (x, y, z) axes for [Z, Y, X] broadcasting."""
    x, y, z = coords
    return (x[None, None, :], y[None, :, None], z[:, None, None])


def water_box(size_mm: tuple = (300.0, 300.0, 300.0), spacing_mm=2.0) -> CTVolume:
    """Homogeneous water box (density 1.0 everywhere) centered on the origin."""
    coords, spacing, origin = _grid(size_mm, spacing_mm)
    shape = (coords[2].size, coords[1].size, coords[0].size)
    return _volume(np.ones(shape, dtype=np.float32), spacing, origin)


def slab_phantom(
    slabs: list, size_mm: tuple = (300.0, 300.0, 300.0), spacing_mm=2.0
) -> CTVolume:
    """Water box with density slabs stacked along world z."""
    coords, spacing, origin = _grid(size_mm, spacing_mm)
    zc = coords[2]
    arr = np.ones((zc.size, coords[1].size, coords[0].size), dtype=np.float32)
    for z_lo, z_hi, rho in slabs:
        sel = (zc >= z_lo) & (zc < z_hi)
        arr[sel, :, :] = np.clip(rho, DENSITY_MIN, DENSITY_MAX)
    return _volume(arr, spacing, origin)


def cylinder_phantom(
    radius_mm: float = 120.0,
    axis: str = "y",
    core_density: float | None = None,
    core_radius_mm: float = 40.0,
    density: float = 1.0,
    size_mm: tuple = (300.0, 300.0, 300.0),
    spacing_mm=2.0,
) -> CTVolume:
    """Cylinder of tissue (air outside), optionally with a coaxial core."""
    coords, spacing, origin = _grid(size_mm, spacing_mm)
    x, y, z = _broadcast_coords(coords)
    if axis == "x":
        r2 = y**2 + z**2
    elif axis == "y":
        r2 = x**2 + z**2
    elif axis == "z":
        r2 = x**2 + y**2
    else:
        raise ValueError(f"axis must be x, y or z, got {axis!r}")
    shape = (coords[2].size, coords[1].size, coords[0].size)
    r2 = np.broadcast_to(r2, shape)
    arr = np.full(shape, AIR_DENSITY, dtype=np.float32)
    arr[r2 <= radius_mm**2] = np.clip(density, DENSITY_MIN, DENSITY_MAX)
    if core_density is not None:
        arr[r2 <= core_radius_mm**2] = np.clip(core_density, DENSITY_MIN, DENSITY_MAX)
    return _volume(arr, spacing, origin)


def phantom_from_id(phantom_id: str) -> CTVolume:
    """Reconstruct a procedural phantom from its string id."""
    parts = phantom_id.split(":")
    kind = parts[0]
    if kind == "water":
        size, spacing = (float(parts[1]), float(parts[2]))
        return water_box(size_mm=(size,) * 3, spacing_mm=spacing)
    if kind == "slab_lung":
        size, spacing = (float(parts[1]), float(parts[2]))
        slab = (-size / 8, size / 4, 0.26)
        return slab_phantom([slab], size_mm=(size,) * 3, spacing_mm=spacing)
    if kind == "perlin":
        seed, size, spacing = (int(parts[1]), float(parts[2]), float(parts[3]))
        return perlin_phantom(
            np.random.default_rng(seed), size_mm=(size,) * 3, spacing_mm=spacing
        )
    raise ValueError(f"unknown phantom id {phantom_id!r}")


def perlin_phantom(
    rng: np.random.Generator,
    size_mm: tuple = (300.0, 300.0, 300.0),
    spacing_mm=2.0,
    corr_mm: float = 25.0,
    n_inserts: int = 5,
) -> CTVolume:
    """Smooth heterogeneous ellipsoid body with random spherical inserts."""
    coords, spacing, origin = _grid(size_mm, spacing_mm)
    shape = (coords[2].size, coords[1].size, coords[0].size)
    noise = np.zeros(shape, dtype=np.float64)
    amp_sum = 0.0
    for octave in range(_PERLIN_OCTAVES):
        corr = corr_mm / 2**octave
        sigma_vox = (corr / spacing[2], corr / spacing[1], corr / spacing[0])
        field = gaussian_filter(rng.standard_normal(shape), sigma=sigma_vox)
        field /= max(float(field.std()), 1e-06)
        amp = 0.5**octave
        noise += amp * field
        amp_sum += amp
    density = 1.0 + _PERLIN_AMPLITUDE * noise / amp_sum
    x, y, z = _broadcast_coords(coords)
    semi = tuple((max(s / 2, 1e-06) for s in size_mm))
    body = (x / semi[0]) ** 2 + (y / semi[1]) ** 2 + (z / semi[2]) ** 2 <= 1.0
    body = np.broadcast_to(body, shape)
    for _ in range(n_inserts):
        center = rng.uniform(-0.6, 0.6, size=3) * np.asarray(semi)
        radius = rng.uniform(0.05, 0.15) * min(size_mm)
        rho = rng.uniform(DENSITY_MIN, DENSITY_MAX)
        sphere = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (
            z - center[2]
        ) ** 2 <= radius**2
        density[np.broadcast_to(sphere, shape) & body] = rho
    arr = np.where(body, np.clip(density, DENSITY_MIN, DENSITY_MAX), AIR_DENSITY)
    return _volume(arr.astype(np.float32), spacing, origin)
