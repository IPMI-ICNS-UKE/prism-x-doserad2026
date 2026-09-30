"""MR loading + per-scan normalization + paired (MR, CT-density, dose) triplets."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import torch
from prismx.cone_grid import CTVolume
from prismx.hu import hu_to_density
from prismx.plan_data import PlanCase, PlanCP, PlanCPDataset, _read_mha

IQR_TO_STD: float = 1.349
BODY_FRAC_DEFAULT: float = 0.02
CLIP_LO_DEFAULT: float = -5.0
CLIP_HI_DEFAULT: float = 5.0


def mr_body_mask(
    mr: torch.Tensor, body_frac: float = BODY_FRAC_DEFAULT
) -> torch.Tensor:
    """Boolean body mask [Z, Y, X]: ``mr > body_frac * mr.max()``."""
    mx = float(mr.max())
    if mx <= 0.0:
        return torch.zeros_like(mr, dtype=torch.bool)
    return mr > body_frac * mx


def normalize_mr(
    mr: torch.Tensor,
    body_mask: torch.Tensor | None = None,
    body_frac: float = BODY_FRAC_DEFAULT,
    clip_lo: float = CLIP_LO_DEFAULT,
    clip_hi: float = CLIP_HI_DEFAULT,
) -> torch.Tensor:
    """Per-scan, body-masked robust standardization of an MR volume [Z, Y, X]."""
    mr = mr.to(torch.float32)
    if body_mask is None:
        body_mask = mr_body_mask(mr, body_frac)
    out = torch.full_like(mr, float(clip_lo))
    body = mr[body_mask]
    if body.numel() == 0:
        return out
    median = body.median()
    q1, q3 = torch.quantile(
        body, torch.tensor([0.25, 0.75], device=mr.device, dtype=mr.dtype)
    )
    iqr = q3 - q1
    scale = (iqr / IQR_TO_STD).clamp_min(torch.finfo(mr.dtype).eps)
    z = ((body - median) / scale).clamp(float(clip_lo), float(clip_hi))
    out[body_mask] = z
    return out


@dataclass
class MRCase:
    """One photon case: normalized MR + the anchor CT density on the SAME grid."""

    case_id: str
    mr_norm: torch.Tensor
    ct_density: torch.Tensor
    ct_hu: torch.Tensor
    grid_origin_mm: tuple
    grid_spacing_mm: tuple
    plan: PlanCase

    def ct_volume(self) -> CTVolume:
        """Anchor CT-density volume (``CTVolume``) on this case's grid."""
        return CTVolume(
            density=self.ct_density,
            spacing_mm=self.grid_spacing_mm,
            origin_mm=self.grid_origin_mm,
        )

    def n_cps(self) -> int:
        """Number of control points of this case that have a dose file."""
        return sum(
            (
                1
                for bi, beam in enumerate(self.plan.beams)
                for cp in range(len(beam["control_points"]))
                if self.plan.dose_path(bi, cp).exists()
            )
        )

    def iter_cps(self) -> Iterator[PlanCP]:
        """Iterate the case's per-CP dose samples (``PlanCP``) with dose files."""
        ds = PlanCPDataset(self.plan.root)
        for i in range(len(ds)):
            yield ds[i]

    def cp(self, beam_idx: int, cp_idx: int) -> PlanCP:
        """Fetch a single control point ``(beam_idx, cp_idx)`` as a ``PlanCP``."""
        ds = PlanCPDataset(self.plan.root)
        for i, (_, bi, cp) in enumerate(ds.items):
            if bi == beam_idx and cp == cp_idx:
                return ds[i]
        raise KeyError(
            f"control point (beam {beam_idx}, cp {cp_idx}) has no dose file in {self.plan.root}"
        )


class MRPatientDataset:
    """Per-case MR triplet reader over photon case folders."""

    def __init__(
        self,
        roots,
        body_frac: float = BODY_FRAC_DEFAULT,
        clip_lo: float = CLIP_LO_DEFAULT,
        clip_hi: float = CLIP_HI_DEFAULT,
    ) -> None:
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [Path(r) for r in roots]
        self.body_frac = float(body_frac)
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.roots)

    def case_id(self, index: int) -> str:
        return self.roots[index].name

    def __getitem__(self, index: int) -> MRCase:
        root = self.roots[index]
        if root in self._cache:
            return self._cache[root]
        mr_np, mr_spacing, mr_origin = _read_mha(root / "image" / "mr.mha")
        hu_np, ct_spacing, ct_origin = _read_mha(root / "image" / "ct.mha")
        if mr_np.shape != hu_np.shape:
            raise ValueError(
                f"{root.name}: MR grid {mr_np.shape} != CT grid {hu_np.shape}"
            )
        ct_hu = torch.from_numpy(hu_np)
        mr = torch.from_numpy(mr_np)
        case = MRCase(
            case_id=root.name,
            mr_norm=normalize_mr(
                mr, body_frac=self.body_frac, clip_lo=self.clip_lo, clip_hi=self.clip_hi
            ),
            ct_density=hu_to_density(ct_hu),
            ct_hu=ct_hu,
            grid_origin_mm=ct_origin,
            grid_spacing_mm=ct_spacing,
            plan=PlanCase(root),
        )
        self._cache[root] = case
        return case
