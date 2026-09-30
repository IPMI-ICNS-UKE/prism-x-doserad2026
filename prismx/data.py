"""Control-point training records and dataset (npz on disk, no pickling)."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


def discover_cases(paths) -> list:
    """Resolve plan-case directories from case dirs and/or roots of case dirs."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list = []
    seen: set = set()

    def add(case: Path) -> None:
        key = case.resolve()
        if key not in seen:
            seen.add(key)
            out.append(case)

    for entry in paths:
        p = Path(entry)
        if not p.is_dir():
            raise FileNotFoundError(f"--data entry is not a directory: {p}")
        if (p / f"{p.name}.json").is_file():
            add(p)
            continue
        children = sorted(
            (d for d in p.iterdir() if d.is_dir() and (d / f"{d.name}.json").is_file())
        )
        if not children:
            raise FileNotFoundError(
                f"{p} is neither a case dir (no {p.name}.json) nor a root containing case dirs one level down"
            )
        for child in children:
            add(child)
    return out


def split_items(items, val_beam: int) -> tuple:
    """Partition ``(case, beam_idx, cp)`` item tuples by held-out beam index."""
    val_beam = int(val_beam)
    train = [item for item in items if item[1] != val_beam]
    val = [item for item in items if item[1] == val_beam]
    if not train or not val:
        raise ValueError(
            f"empty split for val_beam={val_beam}: {len(train)} train / {len(val)} val items"
        )
    return (train, val)


def split_items_by_case(items, case_ids, val_case) -> tuple:
    """Partition item tuples by held-out case id(s) (patient holdout)."""
    if isinstance(val_case, str):
        wanted = [c.strip() for c in val_case.split(",") if c.strip()]
    else:
        wanted = [str(c) for c in val_case]
    unknown = [c for c in wanted if c not in case_ids]
    if unknown:
        raise ValueError(
            f"val_case(s) {unknown!r} not among cases {sorted(set(case_ids))}"
        )
    wanted_set = set(wanted)
    val_idx = {i for i, cid in enumerate(case_ids) if cid in wanted_set}
    train = [item for item in items if item[0] not in val_idx]
    val = [item for item in items if item[0] in val_idx]
    if not train or not val:
        raise ValueError(
            f"empty split for val_case={wanted!r}: {len(train)} train / {len(val)} val items"
        )
    return (train, val)


@dataclass
class CPRecord:
    """One control point: MC dose grid + machine state, serializable to npz."""

    dose: np.ndarray
    sigma: np.ndarray
    grid_origin_mm: tuple
    grid_spacing_mm: tuple
    gantry_deg: float
    collimator_deg: float
    couch_deg: float
    isocenter_mm: tuple
    left_mm: np.ndarray
    right_mm: np.ndarray
    jaw_u_mm: tuple
    jaw_v_mm: tuple
    phantom_id: str
    mu_weight: float = 1.0

    def save(self, path) -> None:
        """Write the record as a compressed npz (no object arrays / pickling)."""
        np.savez_compressed(
            Path(path),
            dose=np.asarray(self.dose, dtype=np.float32),
            sigma=np.asarray(self.sigma, dtype=np.float32),
            grid_origin_mm=np.asarray(self.grid_origin_mm, dtype=np.float64),
            grid_spacing_mm=np.asarray(self.grid_spacing_mm, dtype=np.float64),
            gantry_deg=np.asarray(self.gantry_deg, dtype=np.float64),
            collimator_deg=np.asarray(self.collimator_deg, dtype=np.float64),
            couch_deg=np.asarray(self.couch_deg, dtype=np.float64),
            isocenter_mm=np.asarray(self.isocenter_mm, dtype=np.float64),
            left_mm=np.asarray(self.left_mm, dtype=np.float32),
            right_mm=np.asarray(self.right_mm, dtype=np.float32),
            jaw_u_mm=np.asarray(self.jaw_u_mm, dtype=np.float64),
            jaw_v_mm=np.asarray(self.jaw_v_mm, dtype=np.float64),
            phantom_id=np.asarray(self.phantom_id),
            mu_weight=np.asarray(self.mu_weight, dtype=np.float64),
        )

    @staticmethod
    def load(path) -> "CPRecord":
        """Load a record written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as z:
            return CPRecord(
                dose=z["dose"],
                sigma=z["sigma"],
                grid_origin_mm=tuple((float(v) for v in z["grid_origin_mm"])),
                grid_spacing_mm=tuple((float(v) for v in z["grid_spacing_mm"])),
                gantry_deg=float(z["gantry_deg"]),
                collimator_deg=float(z["collimator_deg"]),
                couch_deg=float(z["couch_deg"]),
                isocenter_mm=tuple((float(v) for v in z["isocenter_mm"])),
                left_mm=z["left_mm"],
                right_mm=z["right_mm"],
                jaw_u_mm=tuple((float(v) for v in z["jaw_u_mm"])),
                jaw_v_mm=tuple((float(v) for v in z["jaw_v_mm"])),
                phantom_id=str(z["phantom_id"][()]),
                mu_weight=float(z["mu_weight"]),
            )

    def beam(self):
        """Reconstruct the :class:`BeamGeometry` of this control point."""
        from prismx.geometry import BeamGeometry

        return BeamGeometry(
            gantry_deg=self.gantry_deg,
            collimator_deg=self.collimator_deg,
            couch_deg=self.couch_deg,
            isocenter_mm=self.isocenter_mm,
        )

    def aperture_obj(self):
        """Reconstruct the :class:`MLCAperture` of this control point."""
        from prismx.aperture import MLCAperture
        from prismx.config import MLCSpec

        return MLCAperture(
            left_mm=torch.as_tensor(self.left_mm, dtype=torch.float32),
            right_mm=torch.as_tensor(self.right_mm, dtype=torch.float32),
            jaw_u_mm=tuple(self.jaw_u_mm),
            jaw_v_mm=tuple(self.jaw_v_mm),
            mlc=MLCSpec(n_pairs=len(self.left_mm)),
        )


class CPDataset:
    """Directory-backed dataset of :class:`CPRecord` npz files (sorted)."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.npz"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> CPRecord:
        return CPRecord.load(self.paths[index])
