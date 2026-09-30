"""Analytic baseline components for PRISM-P (PROTON_SPEC.md section 2).

The proton baseline is a parallel-beam pencil kernel — NO inverse-square term::

    A(x) = G(r_perp; sigma(WED, E)) * IDD_Bragg(WED; E)

where ``G`` is the lateral MCS Gaussian (Fermi-Eyges sigma growing with depth)
and ``IDD_Bragg`` is the integrated depth-dose Bragg curve in water-equivalent
depth (WED) for a beamlet of energy ``E``. The Bragg peak sits at the range
``R(E)``; the distal falloff (R80 -> R20 over a few mm) is the range-critical,
steep part of the curve.

Three commissionable physics models, mirroring ``prismx.analytic``'s
``PDDTable`` + 3-exp surrogate but for protons:

- :class:`RangeEnergyTable` — monotonic R80(E) range-from-energy curve and its
  inverse, fit from ``(energy, measured R80)`` pairs.
- :class:`BraggIDD` — a canonical Bortfeld-1997 analytic Bragg depth-dose,
  evaluated in WED for a given range (or energy via the range table).
- :func:`lateral_sigma` — Highland/Fermi-Eyges MCS spread sigma(WED, E).

``proton_baseline`` composes them into a differentiable torch kernel.
Commissioning fits (:func:`fit_range_energy`, :func:`fit_sigma`) operate on
pooled per-beamlet samples by least squares.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.special import gamma as _gamma_fn
from scipy.special import pbdv as _pbdv

# Bortfeld (1997, Med. Phys. 24:2024) analytic Bragg-curve constants for water.
# Range-energy power law R(E) = alpha * E^p with E in MeV, R in mm (alpha in
# mm/MeV^p). The classic water values are alpha ~ 2.2e-3 cm/MeV^p, p ~ 1.77;
# converted to mm: alpha_mm = 2.2e-2 mm/MeV^p. PROTON_SPEC reports R ~ E^1.77.
_BORTFELD_ALPHA_MM: float = 0.022
_BORTFELD_P: float = 1.77

# Bortfeld depth-dose shape constants (water):
#   p          range-energy exponent (slope of the falloff width vs range)
#   beta       slope of the linear fluence reduction (per mm), ~0.012 /cm
#   gamma_frac fraction of locally absorbed energy from nuclear interactions
#   epsilon    fraction of primaries leaving the field (here 0 — single spot)
_BORTFELD_BETA_PER_MM: float = 0.0012
_BORTFELD_GAMMA_FRAC: float = 0.6
# Energy / range straggling: sigma_mono(R) = a0 + a1 * R (mm). a0 is the spot
# energy spread floor; a1 the per-mm range-straggling growth (~0.012 * R^0.935
# in water, linearized here). Seeded to give a sharp mono-energetic peak with an
# ~R80->R20 distal width of ~3 mm (the range-critical falloff).
_STRAGGLE_A0_MM: float = 0.3
_STRAGGLE_A1: float = 0.008

# Lateral MCS spread: sigma^2(wed, E) = sigma0^2 + a * wed^b (mm). Seeded so the
# spot reaches ~6-9 mm at the Bragg peak for clinical ranges (PROTON_SPEC: 6-9
# mm). With sigma0 = 3 mm, a = 0.02 mm^(2-b), b = 1.5 a 150 mm-range spot grows
# to sqrt(9 + 0.02 * 150^1.5) ~ 6.8 mm (and ~8 mm at 200 mm WED).
_SIGMA0_MM: float = 3.0
_SIGMA_GROWTH_A: float = 0.02
_SIGMA_GROWTH_B: float = 1.5

# Spec (energy, R80-from-source) anchors. These are GEOMETRIC depths from the
# source through patient tissue; the WED-space range table is commissioned from
# data. We seed RangeEnergyTable.default() from the Bortfeld power law (water /
# WED) so the un-commissioned baseline is self-consistent in WED, and keep the
# spec pairs documented here for reference / sanity checks.
SPEC_ENERGY_R80_FROM_SOURCE_MM: tuple = (
    (45.6, 973.0),
    (72.3, 992.0),
    (114.5, 1042.0),
    (126.1, 1065.0),
    (173.4, 1136.0),
    (180.5, 1139.0),
)

_EPS: float = 1e-8


def _as_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


class RangeEnergyTable:
    """Monotonic R80(E) range-from-energy curve with an analytic inverse.

    Holds strictly-increasing ``(energy_mev, range_mm)`` anchors; ``R_of_E`` and
    ``E_of_R`` linearly interpolate (edge-clamped) on the device of their input.
    Ranges are water-equivalent depths (WED, mm) — the physical Bragg-peak depth
    along the beamlet.
    """

    def __init__(self, energy_mev: torch.Tensor, range_mm: torch.Tensor):
        e = _as_f32(energy_mev).reshape(-1)
        r = _as_f32(range_mm).reshape(-1)
        if e.numel() != r.numel() or e.numel() < 2:
            raise ValueError("energy_mev and range_mm must be 1-D with matching length >= 2")
        if not bool((e[1:] > e[:-1]).all()):
            raise ValueError("energy_mev must be strictly increasing")
        if not bool((r[1:] > r[:-1]).all()):
            raise ValueError("range_mm must be strictly increasing (R(E) monotonic)")
        self.energy_mev = e
        self.range_mm = r
        self._device_tables: dict = {e.device: (e, r)}

    @staticmethod
    def default() -> "RangeEnergyTable":
        """Bortfeld water power law R(E) = alpha * E^p over 30-210 MeV.

        Seeded in WED so the un-commissioned baseline is self-consistent; covers
        the PROTON_SPEC energy span (31.7-200.8 MeV). At E=126 MeV the range is
        ~0.022 * 126^1.77 ~ 115 mm WED, matching the ~11.5 cm water range of a
        126 MeV proton.
        """
        e = torch.linspace(30.0, 210.0, 64)
        r = _BORTFELD_ALPHA_MM * e.pow(_BORTFELD_P)
        return RangeEnergyTable(e, r)

    def _tables_on(self, device: torch.device) -> tuple:
        tables = self._device_tables.get(device)
        if tables is None:
            tables = (self.energy_mev.to(device), self.range_mm.to(device))
            self._device_tables[device] = tables
        return tables

    @staticmethod
    def _interp(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """1-D linear interpolation of ys(xs) at x, edge-clamped, shape-preserving."""
        q = _as_f32(x)
        flat = q.reshape(-1).clamp(xs[0], xs[-1])
        idx = torch.searchsorted(xs, flat, right=True).clamp(1, xs.numel() - 1)
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        t = (flat - x0) / (x1 - x0)
        return (y0 * (1.0 - t) + y1 * t).reshape(q.shape)

    def R_of_E(self, energy_mev: torch.Tensor) -> torch.Tensor:
        """Range (WED, mm) for beamlet energy (MeV); edge-clamped interpolation."""
        e = _as_f32(energy_mev)
        es, rs = self._tables_on(e.device)
        return self._interp(e, es, rs)

    def E_of_R(self, range_mm: torch.Tensor) -> torch.Tensor:
        """Inverse: energy (MeV) for a given range (WED, mm); edge-clamped."""
        r = _as_f32(range_mm)
        es, rs = self._tables_on(r.device)
        return self._interp(r, rs, es)

    def save_npz(self, path) -> None:
        np.savez(
            path,
            re_energy_mev=self.energy_mev.cpu().numpy(),
            re_range_mm=self.range_mm.cpu().numpy(),
        )

    @staticmethod
    def from_npz(path) -> "RangeEnergyTable":
        with np.load(path) as data:
            return RangeEnergyTable(
                torch.from_numpy(np.asarray(data["re_energy_mev"], dtype=np.float32)),
                torch.from_numpy(np.asarray(data["re_range_mm"], dtype=np.float32)),
            )


def _parabolic_cylinder_d(nu: float, x: np.ndarray) -> np.ndarray:
    """Parabolic cylinder function D_nu(x) via scipy.special.pbdv (vectorized)."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    out = np.array([_pbdv(nu, float(v))[0] for v in flat], dtype=np.float64)
    return out.reshape(x.shape)


def bortfeld_idd(
    wed_mm: torch.Tensor,
    range_mm: torch.Tensor,
    sigma_mm: torch.Tensor,
    epsilon: float = 0.0,
    gamma_frac: float = _BORTFELD_GAMMA_FRAC,
    p: float = _BORTFELD_P,
    beta_per_mm: float = _BORTFELD_BETA_PER_MM,
) -> torch.Tensor:
    """Bortfeld (1997) analytic Bragg integrated depth-dose D(z), unnormalized.

    Closed form (Bortfeld eq. 28) for the depth dose of a proton pencil beam in
    water, valid around the peak where the energy straggling ``sigma_mm`` (mm)
    Gaussian-broadens the otherwise cusped peak::

        zeta = (range_mm - wed_mm) / sigma_mm
        D ∝ sigma^(1/p) * exp(-zeta^2/4) * [ c1 * D_{-1/p}(-zeta)
                                           + c2 * D_{-1/p-1}(-zeta) ]

    with ``c1 = Gamma(1/p)/Gamma(1/p+...)`` style prefactors folded into the
    parabolic cylinder functions ``D_nu``. Computed in float64 on CPU via scipy
    (the parabolic cylinder functions have no torch primitive), then returned as
    a float32 torch tensor on the inputs' device. Differentiability w.r.t.
    ``range_mm``/``sigma_mm`` is provided by :class:`BraggIDD` through a smooth
    autograd-friendly surrogate; this function is the reference shape used for
    commissioning and table generation.
    """
    wed = _as_f32(wed_mm)
    dev = wed.device
    w = wed.reshape(-1).detach().cpu().numpy().astype(np.float64)
    R = float(_as_f32(range_mm).reshape(()).item())
    sig = float(_as_f32(sigma_mm).reshape(()).item())
    sig = max(sig, 1e-3)

    depth_left = R - w  # residual range; > 0 proximal to the peak
    zeta = depth_left / sig
    nu = 1.0 / p

    # Bortfeld eq. 28 (parabolic-cylinder form) is only well-conditioned in a
    # window |zeta| <= ZETA_MAX around the peak; beyond it exp(-zeta^2/4) and
    # D_nu(-zeta) over/underflow to inf*0 = nan. Outside the window use the
    # asymptotic branches: proximal (zeta >> 0) is the unbroadened power-law
    # plateau (eq. 27, dose ∝ depth_left^(1-1/p)); distal (zeta << 0, past the
    # range) the dose is ~0. ZETA_MAX = 6 keeps both factors finite (exp(-9) ~
    # 1e-4) while covering the few-mm straggling-broadened peak.
    _ZETA_MAX = 6.0

    def _eq28(z: np.ndarray) -> np.ndarray:
        """Bortfeld eq. 28 parabolic-cylinder depth dose at reduced depth z."""
        pref = (
            np.exp(-(z**2) / 4.0)
            * sig ** (1.0 / p)
            / (np.sqrt(2.0 * np.pi) * p ** (1.0 / p) * _gamma_fn(1.0 / p))
        )
        term = (
            _gamma_fn(nu) * _parabolic_cylinder_d(-nu, -z)
            + (1.0 / np.sqrt(2.0) / sig)
            * (gamma_frac * sig + epsilon * R / p)
            * _gamma_fn(nu + 1.0)
            * _parabolic_cylinder_d(-nu - 1.0, -z)
        )
        return pref * term

    near = np.abs(zeta) <= _ZETA_MAX
    dose = np.zeros_like(w)
    if near.any():
        dose[near] = _eq28(zeta[near])

    # Proximal plateau (well before the peak): Bortfeld eq. 27 unbroadened
    # depth dose D(z) ∝ depth_left^(1/p - 1) (exponent ~ -0.435, so the dose
    # falls gently from the peak back toward the entrance), matched continuously
    # to the eq.-28 window at zeta = ZETA_MAX (computed in-line, no recursion).
    plateau_exp = 1.0 / p - 1.0
    plateau = zeta > _ZETA_MAX
    if plateau.any():
        dl_match = _ZETA_MAX * sig
        d_match = float(_eq28(np.array([_ZETA_MAX]))[0])
        scale = d_match / (dl_match**plateau_exp)
        dose[plateau] = scale * depth_left[plateau] ** plateau_exp

    # Linear fluence reduction with depth + nuclear buildup; clamp tiny/neg tails.
    dose = dose * (1.0 + beta_per_mm * np.maximum(R - w, 0.0))
    dose = np.clip(dose, 0.0, None)
    return torch.from_numpy(dose.astype(np.float32)).reshape(wed.shape).to(dev)


class BraggIDD:
    """Canonical mono-energetic Bragg integrated depth-dose vs WED.

    Holds a peak-normalized shape table in *residual-range* coordinates
    ``s = R - WED`` (mm), shared across energies and scaled per beamlet by its
    range ``R`` (and a per-range straggling width). ``__call__(wed, E_or_R)``
    interpolates the shape at ``s = R - wed`` and returns a relative dose that
    peaks near ``wed = R`` with a steep distal falloff.

    The table is built once from :func:`bortfeld_idd` at a reference range; the
    range-coordinate representation makes it differentiable and cheap to
    evaluate in torch (no parabolic-cylinder calls at query time).
    """

    def __init__(
        self,
        s_mm: torch.Tensor,
        shape: torch.Tensor,
        range_table: RangeEnergyTable | None = None,
    ):
        s = _as_f32(s_mm).reshape(-1)
        sh = _as_f32(shape).reshape(-1)
        if s.numel() != sh.numel() or s.numel() < 2:
            raise ValueError("s_mm and shape must be 1-D with matching length >= 2")
        if not bool((s[1:] > s[:-1]).all()):
            raise ValueError("s_mm (residual range) must be strictly increasing")
        self.s_mm = s
        self.shape = sh
        self.range_table = range_table or RangeEnergyTable.default()
        self._device_tables: dict = {s.device: (s, sh)}

    @staticmethod
    def default(
        range_ref_mm: float = 150.0,
        s_min_mm: float = -40.0,
        s_max_mm: float = 260.0,
        s_step_mm: float = 0.5,
        range_table: RangeEnergyTable | None = None,
    ) -> "BraggIDD":
        """Bortfeld shape at ``range_ref_mm``, peak-normalized, in s = R - wed.

        ``s > 0`` is proximal to the peak (entrance plateau), ``s < 0`` is the
        distal falloff. The shape table extends a few cm past the peak (negative
        s) to resolve the R80->R20 falloff and well into the plateau.
        """
        s = torch.arange(s_min_mm, s_max_mm + 0.5 * s_step_mm, s_step_mm)
        wed = range_ref_mm - s
        sigma = _STRAGGLE_A0_MM + _STRAGGLE_A1 * range_ref_mm
        dose = bortfeld_idd(wed, torch.tensor(range_ref_mm), torch.tensor(sigma))
        dose = dose / dose.max().clamp_min(_EPS)
        return BraggIDD(s, dose, range_table)

    def _tables_on(self, device: torch.device) -> tuple:
        tables = self._device_tables.get(device)
        if tables is None:
            tables = (self.s_mm.to(device), self.shape.to(device))
            self._device_tables[device] = tables
        return tables

    def __call__(self, wed: torch.Tensor, E_or_R: torch.Tensor, is_energy: bool = True) -> torch.Tensor:
        """Relative Bragg dose at ``wed`` (mm) for energy (default) or range.

        ``E_or_R`` is broadcast against ``wed``. When ``is_energy`` it is mapped
        to a range via the range table; otherwise it is treated as the range
        directly. Output is edge-clamped, differentiable, shape = broadcast.
        """
        wed = _as_f32(wed)
        s_tab, shape = self._tables_on(wed.device)
        r = (
            self.range_table.R_of_E(E_or_R).to(wed.device)
            if is_energy
            else _as_f32(E_or_R).to(wed.device)
        )
        s = r - wed  # residual range (s>0 proximal, s<0 distal); broadcasts
        s_flat = s.reshape(-1)
        flat = s_flat.clamp(s_tab[0], s_tab[-1])
        idx = torch.searchsorted(s_tab, flat, right=True).clamp(1, s_tab.numel() - 1)
        s0, s1 = s_tab[idx - 1], s_tab[idx]
        y0, y1 = shape[idx - 1], shape[idx]
        t = (flat - s0) / (s1 - s0)
        y = y0 * (1.0 - t) + y1 * t
        # No dose DISTAL to the table (deep past the range). Edge-clamping here
        # would extrapolate shape[0] -- a nonzero floor in a smeared table -- as a
        # constant dose bath for all depths past the peak, which over-integrates;
        # physically the Bragg dose is ~0 past the distal falloff.
        y = torch.where(s_flat < s_tab[0], torch.zeros_like(y), y)
        return y.reshape(s.shape)

    def save_npz(self, path) -> None:
        np.savez(
            path,
            bragg_s_mm=self.s_mm.cpu().numpy(),
            bragg_shape=self.shape.cpu().numpy(),
            re_energy_mev=self.range_table.energy_mev.cpu().numpy(),
            re_range_mm=self.range_table.range_mm.cpu().numpy(),
        )

    @staticmethod
    def from_npz(path) -> "BraggIDD":
        with np.load(path) as data:
            rt = RangeEnergyTable(
                torch.from_numpy(np.asarray(data["re_energy_mev"], dtype=np.float32)),
                torch.from_numpy(np.asarray(data["re_range_mm"], dtype=np.float32)),
            )
            return BraggIDD(
                torch.from_numpy(np.asarray(data["bragg_s_mm"], dtype=np.float32)),
                torch.from_numpy(np.asarray(data["bragg_shape"], dtype=np.float32)),
                rt,
            )


def lateral_sigma(
    wed: torch.Tensor,
    energy_mev: torch.Tensor | None = None,
    sigma0_mm: float = _SIGMA0_MM,
    a: float = _SIGMA_GROWTH_A,
    b: float = _SIGMA_GROWTH_B,
) -> torch.Tensor:
    """MCS lateral spread sigma(WED) (mm), Highland/Fermi-Eyges growth.

    ``sigma^2 = sigma0^2 + a * wed^b`` — an initial spot ``sigma0`` plus
    multiple-Coulomb-scattering growth with water-equivalent depth. Monotonically
    increasing in ``wed``. ``energy_mev`` is accepted for interface symmetry with
    the rest of the module (lower energies scatter more, but the dominant
    dependence is via depth); when provided it is currently unused — the depth
    growth captures the leading behaviour. Returns float32 on the input device.
    """
    w = _as_f32(wed).clamp_min(0.0)
    var = sigma0_mm**2 + a * w.pow(b)
    return var.clamp_min(_EPS).sqrt()


def lateral_gaussian(
    r_perp: torch.Tensor, sigma: torch.Tensor, normalize: str = "peak"
) -> torch.Tensor:
    """Lateral Gaussian G(r_perp; sigma); ``r_perp`` (mm) from the central axis.

    ``"peak"`` (the only supported normalization): ``exp(-r_perp^2 / (2 sigma^2))``
    — unit on the axis. This is the form used by :func:`proton_baseline`, where
    the commissioned ``IDD_Bragg`` is already the CENTRAL-AXIS depth dose, so the
    on-axis baseline equals the IDD and the depth profile peaks at the Bragg
    range. (An area-normalized variant was A/B-tested against MC and rejected —
    see :func:`proton_baseline`.)
    """
    r = _as_f32(r_perp)
    s = _as_f32(sigma).to(r.device).clamp_min(_EPS)
    g = torch.exp(-(r**2) / (2.0 * s**2))
    if normalize != "peak":
        raise ValueError("normalize must be 'peak'")
    return g


def proton_baseline(
    r_perp: torch.Tensor,
    wed: torch.Tensor,
    energy_mev: torch.Tensor,
    bragg: BraggIDD,
    weights: torch.Tensor | None = None,
    sigma0_mm: float = _SIGMA0_MM,
    a: float = _SIGMA_GROWTH_A,
    b: float = _SIGMA_GROWTH_B,
    halo_w: float = 0.0,
    halo_ratio: float = 3.0,
) -> torch.Tensor:
    """Proton analytic baseline A = G(r_perp; sigma(wed, E)) * IDD_Bragg(wed; E).

    PARALLEL pencil beam: NO inverse-square term (parallel geometry, no
    divergence). The lateral factor is the peak-normalized Gaussian (on-axis
    value 1) and ``IDD_Bragg`` is the commissioned central-axis depth dose, so
    on the central axis ``A(0, wed, E) = IDD_Bragg(wed, E)`` and the depth
    profile peaks at the Bragg range ``R(E)``; off axis it falls as a Gaussian
    whose width ``sigma(wed, E)`` grows with depth (Fermi-Eyges MCS).

    All inputs broadcast against each other; ``energy_mev`` selects the Bragg
    range and modulates the lateral spread. ``weights`` (optional,
    broadcastable) scales the per-beamlet contribution linearly (the
    plan-assembly weight, like photon MU); defaults to 1. ``sigma0_mm`` /
    ``a`` / ``b`` are the commissioned :func:`lateral_sigma` parameters
    (``sigma^2 = sigma0^2 + a * wed^b``); the defaults are the water seeds.

    DOUBLE-GAUSSIAN lateral (recom-v4): with ``halo_w > 0`` the lateral kernel is
    ``(1 - halo_w) * G(r; sigma) + halo_w * G(r; halo_ratio * sigma)`` -- the
    commissioned NUCLEAR-HALO term the single Gaussian structurally lacks. Both
    components are peak-normalized, so on-axis A == IDD stays exactly true (the
    dose_scale commissioning convention). ``halo_w = 0`` (default) reproduces the
    single-Gaussian baseline bit-exactly; recom-v2 compensated the missing halo
    by accident via its too-wide seed sigma, which is why fixing sigma alone
    (recom-v3) LOST plan gamma while winning per-beam MAE.
    Returns a finite, non-negative, differentiable float32 tensor.
    """
    r = _as_f32(r_perp)
    w = _as_f32(wed).to(r.device)
    e = _as_f32(energy_mev).to(r.device)
    sigma = lateral_sigma(w, e, sigma0_mm=sigma0_mm, a=a, b=b)
    # Peak-normalized lateral kernel (on-axis = IDD). An area-normalized variant
    # (transverse integral = IDD) was A/B-tested against MC and rejected: it
    # over-fills the mid-field (+25%) under core-weighted calibration, fighting
    # the additive halo head g_a; peak-norm leaves the nuclear halo for g_a.
    lateral = lateral_gaussian(r, sigma, normalize="peak")
    if halo_w > 0.0:
        halo = lateral_gaussian(r, float(halo_ratio) * sigma, normalize="peak")
        lateral = (1.0 - float(halo_w)) * lateral + float(halo_w) * halo
    idd = bragg(w, e, is_energy=True)
    a = lateral * idd
    if weights is not None:
        a = a * _as_f32(weights).to(r.device)
    return a.clamp_min(0.0)


# --------------------------------------------------------------------------- #
# Commissioning fits (least squares on pooled per-beamlet samples).
# --------------------------------------------------------------------------- #


def fit_range_energy(
    energy_mev: torch.Tensor,
    range_mm: torch.Tensor,
    n_table: int = 64,
) -> RangeEnergyTable:
    """Fit a monotonic R(E) = alpha * E^p power law to (energy, R80) samples.

    Least-squares in log space (``log R = log alpha + p log E``, linear and
    well-conditioned), then samples a strictly-increasing table over the
    observed energy span. Returns a :class:`RangeEnergyTable`. Robust to noisy
    per-beamlet R80 estimates; the power law is the physical water relation.
    """
    e = _as_f32(energy_mev).reshape(-1).cpu().numpy().astype(np.float64)
    r = _as_f32(range_mm).reshape(-1).cpu().numpy().astype(np.float64)
    keep = (e > 0) & (r > 0)
    e, r = e[keep], r[keep]
    if e.size < 2:
        raise ValueError("fit_range_energy: need at least two positive (E, R) pairs")
    # Linear fit log R = log alpha + p log E.
    p_coef, log_alpha = np.polyfit(np.log(e), np.log(r), 1)
    alpha = float(np.exp(log_alpha))
    p = float(p_coef)
    e_tab = torch.linspace(float(e.min()), float(e.max()), n_table)
    r_tab = alpha * e_tab.pow(p)
    return RangeEnergyTable(e_tab, r_tab)


def fit_sigma(
    r_perp: torch.Tensor,
    wed: torch.Tensor,
    sigma_obs_mm: torch.Tensor,
) -> tuple:
    """Fit the lateral MCS growth ``sigma^2 = sigma0^2 + a * wed^b`` from data.

    ``sigma_obs_mm`` is a per-sample observed lateral spread (e.g. from fitting a
    Gaussian to the off-axis dose profile at depth ``wed``). Least-squares the
    three parameters (``sigma0``, ``a``, ``b``) with positive bounds. ``r_perp``
    is accepted for signature symmetry with the sampler (the fit operates on the
    pre-extracted per-depth sigma); returns ``(sigma0_mm, a, b)`` floats.
    """
    w = _as_f32(wed).reshape(-1).cpu().numpy().astype(np.float64)
    sig = _as_f32(sigma_obs_mm).reshape(-1).cpu().numpy().astype(np.float64)
    keep = (w >= 0) & (sig > 0)
    w, sig = w[keep], sig[keep]
    if w.size < 3:
        raise ValueError("fit_sigma: need at least three (wed, sigma) samples")

    def model(x, sigma0, a, b):
        return np.sqrt(np.clip(sigma0**2 + a * np.power(np.clip(x, 0, None), b), 1e-12, None))

    popt, _ = curve_fit(
        model,
        w,
        sig,
        p0=[_SIGMA0_MM, _SIGMA_GROWTH_A, _SIGMA_GROWTH_B],
        bounds=([1e-3, 0.0, 0.5], [50.0, 100.0, 3.0]),
        maxfev=100000,
    )
    return float(popt[0]), float(popt[1]), float(popt[2])
