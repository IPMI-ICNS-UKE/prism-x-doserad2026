"""Native reader for the photon training-data layout."""

from __future__ import annotations
import json
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import torch
from prismx.aperture import MLCAperture
from prismx.cone_grid import CTVolume
from prismx.config import MLCSpec
from prismx.geometry import BeamGeometry
from prismx.hu import ct_volume_from_hu

SIGMA_REL_DEFAULT: float = 0.03
SIGMA_FLOOR_REL_DEFAULT: float = 0.002


def _read_mha(path: Path) -> tuple:
    """Read a MetaImage file -> (array [Z, Y, X] float32, spacing, origin)."""
    try:
        import SimpleITK as sitk
    except ImportError as err:
        raise ImportError(
            "SimpleITK is required to read .mha files: pip install SimpleITK"
        ) from err
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return (arr, tuple(img.GetSpacing()), tuple(img.GetOrigin()))


@dataclass
class PlanCP:
    """One control point of a plan case, ready for training/evaluation."""

    dose: torch.Tensor
    sigma: torch.Tensor
    grid_origin_mm: tuple
    grid_spacing_mm: tuple
    beam: BeamGeometry
    aperture: MLCAperture
    ct: CTVolume
    case_id: str
    beam_idx: int
    cp_idx: int
    mu_weight: float = 1.0

    def to(self, device) -> "PlanCP":
        """Copy with ``dose``/``sigma`` on ``device``."""
        device = torch.device(device)
        return replace(self, dose=self.dose.to(device), sigma=self.sigma.to(device))


class PlanCase:
    """One case folder: plan JSON + CT + per-CP dose files (lazy loading)."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.case_id = self.root.name
        with open(self.root / f"{self.case_id}.json") as f:
            self.plan = json.load(f)
        self._ct: CTVolume | None = None

    @property
    def beams(self) -> list:
        return self.plan["beams"]

    def ct(self) -> CTVolume:
        if self._ct is None:
            hu, spacing, origin = _read_mha(self.root / "image" / "ct.mha")
            self._ct = ct_volume_from_hu(torch.from_numpy(hu), spacing, origin)
        return self._ct

    def dose_path(self, beam_idx: int, cp_idx: int) -> Path:
        return self.root / "dose" / f"Dose_B{beam_idx}_CP{cp_idx:03d}.mha"

    def beam_geometry(self, beam_idx: int, cp_idx: int) -> BeamGeometry:
        beam = self.beams[beam_idx]
        cp = beam["control_points"][cp_idx]
        return BeamGeometry(
            gantry_deg=float(cp["gantry_angle"]),
            isocenter_mm=tuple((float(v) for v in beam["iso_center"])),
            sad_mm=float(beam.get("SAD", 1000.0)),
        )

    def aperture(self, beam_idx: int, cp_idx: int) -> MLCAperture:
        beam = self.beams[beam_idx]
        cp = beam["control_points"][cp_idx]
        n_pairs = int(beam["num_mlc_leaf_pairs"])
        left = torch.tensor(cp["mlc_left_int_mm"][::-1], dtype=torch.float32)
        right = torch.tensor(cp["mlc_right_int_mm"][::-1], dtype=torch.float32)
        return MLCAperture(
            left_mm=left,
            right_mm=right,
            jaw_u_mm=(-200.0, 200.0),
            jaw_v_mm=(-200.0, 200.0),
            mlc=MLCSpec(n_pairs=n_pairs),
        )


class PlanCPDataset:
    """Flattened (case, beam, control point) dataset over case folders."""

    def __init__(
        self,
        roots,
        sigma_rel: float = SIGMA_REL_DEFAULT,
        sigma_floor_rel: float = SIGMA_FLOOR_REL_DEFAULT,
    ) -> None:
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.cases = [PlanCase(r) for r in roots]
        self.sigma_rel = float(sigma_rel)
        self.sigma_floor_rel = float(sigma_floor_rel)
        self.items: list = []
        for ci, case in enumerate(self.cases):
            for bi, beam in enumerate(case.beams):
                for cp in range(len(beam["control_points"])):
                    if case.dose_path(bi, cp).exists():
                        self.items.append((ci, bi, cp))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> PlanCP:
        ci, bi, cp = self.items[index]
        case = self.cases[ci]
        dose_np, spacing, origin = _read_mha(case.dose_path(bi, cp))
        dose = torch.from_numpy(dose_np)
        d_max = float(dose.max())
        sigma = (
            self.sigma_rel * torch.sqrt(dose.clamp(min=0.0) * d_max)
            + self.sigma_floor_rel * d_max
        )
        return PlanCP(
            dose=dose,
            sigma=sigma,
            grid_origin_mm=origin,
            grid_spacing_mm=spacing,
            beam=case.beam_geometry(bi, cp),
            aperture=case.aperture(bi, cp),
            ct=case.ct(),
            case_id=case.case_id,
            beam_idx=bi,
            cp_idx=cp,
        )
