"""Native reader for the proton training-data layout (PRISM-P).

Case folder layout (e.g. ``data/proton/training/1ABB006``)::

    <case>/<case>.json                  plan: beams -> rays -> beamlets
    <case>/image/ct.mha                 HU volume (same grid as the dose files)
    <case>/dose/Dose_B{b}_R{r}_L{l}.mha per (beam, ray, layer=energy) dose

Conventions (PROTON_SPEC.md section 1, verified against ``1ABB006``):

- The plan JSON has ``iso_center`` and ``beams`` (36, gantry 0-350 deg in 10 deg
  steps). Each beam: ``beam_idx``, ``gantry_angle``, ``rays``. Each ray:
  ``ray_idx``, ``ray_source`` [x,y,z], ``ray_target`` [x,y,z], ``beamlets``
  [{``beamlet_idx``, ``energy`` (MeV)}]. One beamlet = one ray at one energy;
  the dose-file layer index ``l`` is the beamlet index within the ray.
- **Geometry: PARALLEL pencil beams.** ``axis = normalize(target - source)`` is
  constant across the rays of a beam; ``|target - source| = SAD = 1000 mm``. The
  gantry convention is the photon one (:class:`prismx.geometry.BeamGeometry`,
  source at ``iso + SAD * (sin g, -cos g, 0)``): at gantry 0 the beam axis is
  ``+y_world`` and rays differ only in the lateral spot ``(u0, v0)`` of the
  isocenter-plane crossing. There is NO divergence and NO inverse-square term.
  The spot ``(u0, v0)`` is the lateral (``u``, ``v``) of ``ray_target`` in the
  beam frame (the target sits on the isocenter plane, ``z_beam ~ 0``); a beamlet
  travels parallel to the beam axis through that spot.

``HU -> density`` reuses the photon G4DCM table (``prismx.hu``); density is the
RSP surrogate. The dose files carry no uncertainty map, so ``sigma`` is the same
Poisson-consistent placeholder as the photon reader (:mod:`prismx.plan_data`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from prismx.cone_grid import CTVolume
from prismx.geometry import BeamGeometry
from prismx.hu import ct_volume_from_hu

SIGMA_REL_DEFAULT: float = 0.03
SIGMA_FLOOR_REL_DEFAULT: float = 0.002

# |ray_target - ray_source| in the proton plan (source-to-isocenter distance).
SAD_MM: float = 1000.0


def _read_mha(path: Path) -> tuple:
    """Read a MetaImage file -> (array [Z, Y, X] float32, spacing, origin)."""
    try:
        import SimpleITK as sitk
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "SimpleITK is required to read .mha files: pip install SimpleITK"
        ) from err
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, tuple(img.GetSpacing()), tuple(img.GetOrigin())


@dataclass
class BeamletGeometry:
    """A PARALLEL pencil beamlet: a beam frame plus a lateral spot ``(u0, v0)``.

    The beamlet travels parallel to the beam axis (beam-frame ``+z``) through the
    spot ``(u0, v0)`` on the isocenter plane. World points are mapped to beamlet
    coordinates ``(u_rel, v_rel, d)`` by rotating into the beam frame and then
    offsetting laterally by the spot; ``d = z_beam + SAD`` is the depth measured
    from the source plane along the axis. Unlike the divergent cone there is no
    central projection: ``u_rel`` / ``v_rel`` are constant along a beamlet.
    """

    beam: BeamGeometry
    spot_uv: tuple  # (u0, v0) in mm, beam-frame lateral coords of the ray target
    sad_mm: float = SAD_MM

    def world_to_beamlet(self, pts: torch.Tensor) -> torch.Tensor:
        """Map world points [N, 3] (mm) to beamlet coords ``(u_rel, v_rel, d)``.

        Runs on ``pts.device`` (the beam transform inherits it). ``r_perp`` is
        ``sqrt(u_rel**2 + v_rel**2)`` from the returned lateral pair.
        """
        beam = self.beam.world_to_beam(pts)
        u0, v0 = float(self.spot_uv[0]), float(self.spot_uv[1])
        u_rel = beam[..., 0] - u0
        v_rel = beam[..., 1] - v0
        d = beam[..., 2] + self.sad_mm
        return torch.stack([u_rel, v_rel, d], dim=-1)

    def axis_world(self) -> torch.Tensor:
        """Unit beam axis in world coords [3] (beam-frame ``+z``), on CPU."""
        origin = self.beam.beam_to_world(torch.zeros(1, 3))[0]
        tip = self.beam.beam_to_world(torch.tensor([[0.0, 0.0, 1.0]]))[0]
        return tip - origin


def beamlet_geometry_from_ray(
    gantry_deg: float,
    isocenter_mm: tuple,
    ray_source: tuple,
    ray_target: tuple,
    sad_mm: float = SAD_MM,
) -> BeamletGeometry:
    """Build a :class:`BeamletGeometry` from a plan ray.

    The spot ``(u0, v0)`` is the lateral ``(u, v)`` of ``ray_target`` in the beam
    frame; the beam axis is pinned by the gantry convention (validated to match
    ``normalize(ray_target - ray_source)`` for every beam of ``1ABB006``).
    """
    beam = BeamGeometry(
        gantry_deg=float(gantry_deg),
        isocenter_mm=tuple(float(v) for v in isocenter_mm),
        sad_mm=float(sad_mm),
    )
    target = torch.tensor([list(ray_target)], dtype=torch.float32)
    tgt_beam = beam.world_to_beam(target)[0]
    spot_uv = (float(tgt_beam[0]), float(tgt_beam[1]))
    return BeamletGeometry(beam=beam, spot_uv=spot_uv, sad_mm=float(sad_mm))


@dataclass
class ProtonBeamlet:
    """One proton pencil beamlet (ray at one energy), ready for training/eval."""

    dose: torch.Tensor  # [Z, Y, X], raw file units
    sigma: torch.Tensor  # [Z, Y, X], same units (placeholder estimate)
    grid_origin_mm: tuple
    grid_spacing_mm: tuple
    beam: BeamGeometry
    spot_uv: tuple  # (u0, v0) lateral spot in the beam frame, mm
    energy_mev: float
    ct: CTVolume
    case_id: str
    beam_idx: int
    ray_idx: int
    layer_idx: int

    def beamlet_geometry(self) -> BeamletGeometry:
        """Parallel-beamlet geometry helper for this beamlet."""
        return BeamletGeometry(
            beam=self.beam, spot_uv=self.spot_uv, sad_mm=float(self.beam.sad_mm)
        )

    def to(self, device) -> "ProtonBeamlet":
        """Copy with ``dose``/``sigma`` on ``device``.

        ``ct`` is shared per case and moved separately by the caller
        (``beamlet.ct.to(device)``); ``beam`` is a device-neutral plan
        description and stays as it is.
        """
        device = torch.device(device)
        return replace(
            self, dose=self.dose.to(device), sigma=self.sigma.to(device)
        )


class ProtonCase:
    """One case folder: plan JSON + CT + per-beamlet dose files (lazy loading)."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.case_id = self.root.name
        with open(self.root / f"{self.case_id}.json") as f:
            self.plan = json.load(f)
        self._ct: CTVolume | None = None

    @property
    def isocenter_mm(self) -> tuple:
        return tuple(float(v) for v in self.plan["iso_center"])

    @property
    def beams(self) -> list:
        return self.plan["beams"]

    def ct(self) -> CTVolume:
        if self._ct is None:
            hu, spacing, origin = _read_mha(self.root / "image" / "ct.mha")
            self._ct = ct_volume_from_hu(torch.from_numpy(hu), spacing, origin)
        return self._ct

    def dose_path(self, beam_idx: int, ray_idx: int, layer_idx: int) -> Path:
        return (
            self.root
            / "dose"
            / f"Dose_B{beam_idx}_R{ray_idx}_L{layer_idx}.mha"
        )

    def beam_geometry(self, beam_idx: int) -> BeamGeometry:
        beam = self.beams[beam_idx]
        return BeamGeometry(
            gantry_deg=float(beam["gantry_angle"]),
            isocenter_mm=self.isocenter_mm,
            sad_mm=SAD_MM,
        )

    def beamlet_geometry(self, beam_idx: int, ray_idx: int) -> BeamletGeometry:
        beam = self.beams[beam_idx]
        ray = beam["rays"][ray_idx]
        return beamlet_geometry_from_ray(
            gantry_deg=float(beam["gantry_angle"]),
            isocenter_mm=self.isocenter_mm,
            ray_source=tuple(ray["ray_source"]),
            ray_target=tuple(ray["ray_target"]),
            sad_mm=SAD_MM,
        )

    def energy_mev(self, beam_idx: int, ray_idx: int, layer_idx: int) -> float:
        ray = self.beams[beam_idx]["rays"][ray_idx]
        return float(ray["beamlets"][layer_idx]["energy"])


class ProtonBeamletDataset:
    """Flattened (case, beam, ray, layer) dataset over proton case folders.

    Only beamlets whose dose file exists are listed. CT volumes are cached per
    case (:class:`ProtonCase`); dose files are read lazily per item. Mirrors
    :class:`prismx.plan_data.PlanCPDataset`.
    """

    def __init__(
        self,
        roots,
        sigma_rel: float = SIGMA_REL_DEFAULT,
        sigma_floor_rel: float = SIGMA_FLOOR_REL_DEFAULT,
    ) -> None:
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.cases = [ProtonCase(r) for r in roots]
        self.sigma_rel = float(sigma_rel)
        self.sigma_floor_rel = float(sigma_floor_rel)
        self.items: list = []
        for ci, case in enumerate(self.cases):
            for bi, beam in enumerate(case.beams):
                for ri, ray in enumerate(beam["rays"]):
                    for li in range(len(ray["beamlets"])):
                        if case.dose_path(bi, ri, li).exists():
                            self.items.append((ci, bi, ri, li))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ProtonBeamlet:
        ci, bi, ri, li = self.items[index]
        case = self.cases[ci]
        dose_np, spacing, origin = _read_mha(case.dose_path(bi, ri, li))
        dose = torch.from_numpy(dose_np)
        d_max = float(dose.max())
        sigma = (
            self.sigma_rel * torch.sqrt(dose.clamp(min=0.0) * d_max)
            + self.sigma_floor_rel * d_max
        )
        geo = case.beamlet_geometry(bi, ri)
        return ProtonBeamlet(
            dose=dose,
            sigma=sigma,
            grid_origin_mm=origin,
            grid_spacing_mm=spacing,
            beam=geo.beam,
            spot_uv=geo.spot_uv,
            energy_mev=case.energy_mev(bi, ri, li),
            ct=case.ct(),
            case_id=case.case_id,
            beam_idx=bi,
            ray_idx=ri,
            layer_idx=li,
        )
