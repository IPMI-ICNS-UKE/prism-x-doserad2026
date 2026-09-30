"""Beam geometry: world / beam / cone coordinate transforms."""

from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from prismx.config import SAD_MM

_F64 = torch.float64


def _rot_y(deg: float) -> torch.Tensor:
    """Right-handed rotation about the y axis, [3, 3] float64."""
    c, s = (math.cos(math.radians(deg)), math.sin(math.radians(deg)))
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=_F64)


def _rot_z(deg: float) -> torch.Tensor:
    """Right-handed rotation about the z axis, [3, 3] float64."""
    c, s = (math.cos(math.radians(deg)), math.sin(math.radians(deg)))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=_F64)


_BASE = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=_F64)


@dataclass
class BeamGeometry:
    """Rigid geometry of one control point (angles in degrees, lengths in mm)."""

    gantry_deg: float
    collimator_deg: float = 0.0
    couch_deg: float = 0.0
    isocenter_mm: tuple = (0.0, 0.0, 0.0)
    sad_mm: float = SAD_MM

    def _rotation64(self) -> torch.Tensor:
        return (
            _rot_z(self.collimator_deg)
            @ _BASE
            @ _rot_z(-self.gantry_deg)
            @ _rot_y(self.couch_deg)
        )

    def _iso64(self) -> torch.Tensor:
        return torch.tensor(self.isocenter_mm, dtype=_F64)

    def rotation(self) -> torch.Tensor:
        """World -> beam rotation matrix, [3, 3] float32."""
        return self._rotation64().to(torch.float32)

    def world_to_beam(self, pts: torch.Tensor, fp32: bool = False) -> torch.Tensor:
        """Map world points [N, 3] (mm) into the beam frame, [N, 3] float32."""
        compute = torch.float32 if fp32 else _F64
        rot = self._rotation64().to(device=pts.device, dtype=compute)
        iso = self._iso64().to(device=pts.device, dtype=compute)
        return ((pts.to(compute) - iso) @ rot.T).to(torch.float32)

    def beam_to_world(self, pts: torch.Tensor, fp32: bool = False) -> torch.Tensor:
        """Inverse of :meth:`world_to_beam`: beam-frame points [N, 3] -> world."""
        compute = torch.float32 if fp32 else _F64
        rot = self._rotation64().to(device=pts.device, dtype=compute)
        iso = self._iso64().to(device=pts.device, dtype=compute)
        return (pts.to(compute) @ rot + iso).to(torch.float32)

    def source_world(self) -> torch.Tensor:
        """World position [3] of the point source (beam frame ``(0, 0, -SAD)``)."""
        src_beam = torch.tensor([[0.0, 0.0, -self.sad_mm]], dtype=_F64)
        return self.beam_to_world(src_beam)[0]


def beam_to_cone(
    pts_beam: torch.Tensor, sad_mm: float = SAD_MM, fp32: bool = False
) -> torch.Tensor:
    """Map beam-frame points [N, 3] to cone coordinates (u, v, d), [N, 3]."""
    compute = torch.float32 if fp32 else _F64
    p = pts_beam.to(compute)
    x, y, z = (p[..., 0], p[..., 1], p[..., 2])
    zs = z + sad_mm
    u = x * sad_mm / zs
    v = y * sad_mm / zs
    d = torch.sqrt(x * x + y * y + zs * zs)
    return torch.stack([u, v, d], dim=-1).to(torch.float32)


def cone_to_beam(
    uvd: torch.Tensor, sad_mm: float = SAD_MM, fp32: bool = False
) -> torch.Tensor:
    """Inverse of :func:`beam_to_cone`: (u, v, d) [N, 3] -> beam-frame (x, y, z)."""
    compute = torch.float32 if fp32 else _F64
    q = uvd.to(compute)
    u, v, d = (q[..., 0], q[..., 1], q[..., 2])
    scale = d / torch.sqrt(u * u + v * v + sad_mm * sad_mm)
    x = u * scale
    y = v * scale
    z = sad_mm * scale - sad_mm
    return torch.stack([x, y, z], dim=-1).to(torch.float32)
