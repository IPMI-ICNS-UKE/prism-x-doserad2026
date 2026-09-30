"""Analytic baseline components for PRISM-X (SPEC.md section 2)."""

from __future__ import annotations
import numpy as np
import torch
from prismx.config import SAD_MM, THREE_EXP_MU_PER_MM, THREE_EXP_WEIGHTS

_PDD_BUILDUP_MM: float = 12.0
_PDD_WED_MAX_MM: float = 600.0
_PDD_BIN_MM: float = 0.5


class PDDTable:
    """Tabulated water primary depth-dose vs WED; linear interp, edge clamp."""

    def __init__(self, wed_mm: torch.Tensor, values: torch.Tensor):
        wed = torch.as_tensor(wed_mm, dtype=torch.float32).reshape(-1)
        vals = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
        if wed.numel() != vals.numel() or wed.numel() < 2:
            raise ValueError("wed_mm and values must be 1-D with matching length >= 2")
        if not bool((wed[1:] > wed[:-1]).all()):
            raise ValueError("wed_mm must be strictly increasing")
        self.wed_mm = wed
        self.values = vals
        self._device_tables: dict = {wed.device: (wed, vals)}

    @staticmethod
    def default() -> "PDDTable":
        """Placeholder curve P(w) = (1 - exp(-w/12)) * sum_k w_k exp(-mu_k w),"""
        w = torch.arange(0.0, _PDD_WED_MAX_MM + 0.5 * _PDD_BIN_MM, _PDD_BIN_MM)
        weights = torch.tensor(THREE_EXP_WEIGHTS)
        mus = torch.tensor(THREE_EXP_MU_PER_MM)
        spectrum = (weights * torch.exp(-w.unsqueeze(-1) * mus)).sum(-1)
        p = (1.0 - torch.exp(-w / _PDD_BUILDUP_MM)) * spectrum
        return PDDTable(w, p / p.max())

    def save_npz(self, path) -> None:
        """Save the table to ``path`` as npz (arrays ``wed_mm`` and ``values``)."""
        np.savez(
            path, wed_mm=self.wed_mm.cpu().numpy(), values=self.values.cpu().numpy()
        )

    @staticmethod
    def from_npz(path) -> "PDDTable":
        """Load a table from an npz holding ``wed_mm`` and ``values``."""
        with np.load(path) as data:
            return PDDTable(
                torch.from_numpy(np.asarray(data["wed_mm"], dtype=np.float32)),
                torch.from_numpy(np.asarray(data["values"], dtype=np.float32)),
            )

    def _tables_on(self, device: torch.device) -> tuple:
        """Lookup tables on ``device`` (cached after the first transfer)."""
        tables = self._device_tables.get(device)
        if tables is None:
            tables = (self.wed_mm.to(device), self.values.to(device))
            self._device_tables[device] = tables
        return tables

    def __call__(self, wed: torch.Tensor) -> torch.Tensor:
        wed = torch.as_tensor(wed, dtype=torch.float32)
        wed_mm, values = self._tables_on(wed.device)
        x = wed.reshape(-1).clamp(wed_mm[0], wed_mm[-1])
        idx = torch.searchsorted(wed_mm, x, right=True).clamp(1, wed_mm.numel() - 1)
        x0 = wed_mm[idx - 1]
        x1 = wed_mm[idx]
        t = (x - x0) / (x1 - x0)
        y = values[idx - 1] * (1.0 - t) + values[idx] * t
        return y.reshape(wed.shape)


def three_exp_transmission(wed: torch.Tensor) -> torch.Tensor:
    """Spectral attenuation surrogate T_k = exp(-mu_k * WED); [...] -> [..., 3]."""
    wed = torch.as_tensor(wed, dtype=torch.float32)
    mus = torch.tensor(THREE_EXP_MU_PER_MM, dtype=wed.dtype, device=wed.device)
    return torch.exp(-wed.unsqueeze(-1) * mus)


def inverse_square(d: torch.Tensor, sad_mm: float = SAD_MM) -> torch.Tensor:
    """(SAD / d)^2; equals 1 at d = SAD."""
    d = torch.as_tensor(d, dtype=torch.float32)
    return (sad_mm / d) ** 2


def analytic_baseline(
    psi: torch.Tensor, p_wed: torch.Tensor, isq: torch.Tensor
) -> torch.Tensor:
    """A = psi * P(WED) * ISQ."""
    return psi * p_wed * isq
