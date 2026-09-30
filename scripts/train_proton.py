"""Per-beamlet training of PRISM-P on proton Geant4 dose (PROTON_SPEC.md).

The proton analogue of ``scripts/train_stage0.py``. Trains :class:`PrismP`
(energy-conditioned per-beamlet physics-residual dose engine) on the per-beamlet
Geant4 doses read by :class:`prismx.proton_data.ProtonBeamletDataset`, holding
out ALL beamlets of one or more cases for validation (patient holdout, the
proton default holds out two: ``1ABB118,1THB011``). Runs unchanged on CPU
(smoke) and CUDA (real host); the pipeline device is the CT volume's device.

Data handling (mirrors train_stage0):
    * the CTVolume of each case is moved to the device once and cached;
    * per-beamlet dose/sigma tensors are LRU-cached in HOST RAM up to
      ``--cache-gb``, divided by ``--dose-scale`` once at load time so the raw
      Geant4 units land in the residual heads' clamp range (the proton baseline
      ``A`` is peak-normalized; raw MC is ~3.7e-3 of it -- see the proton
      residual pilot's dataset-wide scale fit);
    * the parallel proton grid + encoder channels are rebuilt per step by
      :meth:`PrismP.encode_beamlet` on the device CT.

Query sampling reuses the proton residual-pilot STRATIFICATION
(:func:`stratified_beamlet_queries`): in-field dose-weighted core, the
distal-falloff band within +-5 mm WED of the beamlet's R80 (the RANGE-critical
stratum, the proton analogue of penumbra and the most important validation
number), the off-axis MCS tail, and the low-dose floor.

Commissioning: ``--bragg`` (default ``commissioning/proton_bragg.npz``) is loaded
via :meth:`prismx.proton_analytic.BraggIDD.from_npz` and injected into PrismP so
the analytic baseline rides on the COMMISSIONED Bragg/sigma tables (the same
tables the residual pilot's scale was fit against). The ``sigma_params[0]`` floor
overrides the lateral-spread seed exactly like the pilot.

Usage:
    PYTHONPATH=. python scripts/train_proton.py --data data/proton/training \
        --out runs/proton [--val-case 1ABB118,1THB011] [--steps 40000] ...

Single-case smoke (no second case to hold out): point ``--data`` at a directory
with ONE case and pass ``--val-beam`` (or set ``--val-case`` to that case id and
it auto-falls back to a beam holdout) -- all beamlets of the held-out beam index
become validation. See the bottom of this docstring's module constants.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from prismx.config import PrismXConfig
from prismx.data import discover_cases
from prismx.losses import prismx_loss
from prismx.proton_analytic import BraggIDD
from prismx.proton_data import ProtonBeamletDataset
from prismx.proton_grid import ProtonGridSpec, build_proton_grid
from prismx.proton_model import PrismP

LR_FINAL: float = 1e-5
EMA_DECAY: float = 0.999
TRAIN_LOSS_EMA_DECAY: float = 0.98
REL_ERR_FLOOR: float = 0.05  # fraction of per-beamlet max dose in the denominator
VAL_SEED_BASE: int = 20_000  # fixed val queries: rng seeded by val-beamlet index

# Dataset-wide raw-dose-units per unit proton baseline (median of dose / A over
# pooled on-axis near-peak voxels; scripts/proton_residual_pilot.py). Dividing
# dose/sigma by this brings the raw Geant4 targets onto the analytic-baseline
# scale so the multiplicative head (clamped to +-2.5 in log space) can reach
# them. Pass 1.0 to disable.
DOSE_UNITS_PER_BASELINE: float = 0.0036971147514955442

# Proton query strata (the proton residual-pilot population; the binding
# constraint is RANGE, not penumbra). Distal-falloff is the range-critical band.
STRATUM_NAMES: tuple = ("in_field", "distal_falloff", "off_axis", "low_dose", "far_suppress")
IN_FIELD, DISTAL_FALLOFF, OFF_AXIS, LOW_DOSE, FAR_SUPPRESS = range(5)
REPORT_STRATA: tuple = (IN_FIELD, DISTAL_FALLOFF, OFF_AXIS, LOW_DOSE, FAR_SUPPRESS)

# Stratum fractions of the per-step query budget (sum to 1.0). In-field carries
# the remainder so the total is exact.
FRAC_IN_FIELD: float = 0.30
FRAC_DISTAL_FALLOFF: float = 0.25
FRAC_OFF_AXIS: float = 0.18
FRAC_LOW_DOSE: float = 0.12
FRAC_FAR_SUPPRESS: float = 0.15

# Stratum geometry (PROTON_SPEC section 3 / the residual pilot).
GATE_DOSE_FRAC: float = 0.10    # in-field / off-axis gate: dose > 10% of bl max
LOW_DOSE_FRAC: float = 0.01     # outer skirt floor: 1% < dose <= 10% of bl max
ON_AXIS_R_PERP_MM: float = 3.0  # central-axis radius (Bragg depth dose)
DISTAL_BAND_MM: float = 5.0     # +-5 mm WED of R80 (range-critical stratum)
R80_FRAC: float = 0.8
# Far-suppression stratum: near-zero voxels (dose <= LOW_DOSE_FRAC * max) inside
# the >1%-footprint bbox grown by this margin. These are NEVER in the in-support
# keep/grid.contains pool, so without this stratum the far_decoder is wholly
# unsupervised and leaks the analytic baseline as a low-dose bath -- which sums,
# over ~1080 overlapping beamlets, into a ~26% plan-level halo (invisible to the
# peak-normalized per-beamlet gamma). Targets are the TRUE (~0) MC dose.
FAR_SUPPRESS_EXPAND_MM: float = 30.0

# Proton grid sampling for the WED projection (commissioning / pilot default).
GRID_DD_MM: float = 2.0

# IDD-consistency loss: WED bin width (mm) for the on-axis depth-dose curve match.
IDD_BIN_MM: float = 2.0


# --------------------------------------------------------------------------- #
# Case / beam splits over the (case, beam, ray, layer) proton item tuples.
# prismx.data.split_items_by_case assumes 3-tuples ((ci, bi, cp)); the proton
# dataset items are 4-tuples (ci, bi, ri, li), so we split on item[0] (case) /
# item[1] (beam) directly here. Both raise on an empty side, like the photon
# helpers.
# --------------------------------------------------------------------------- #


def split_beamlets_by_case(items, case_ids, val_case) -> tuple:
    """Partition (case, beam, ray, layer) items by held-out case id(s).

    Args:
        items: ``(case_idx, beam_idx, ray_idx, layer_idx)`` tuples.
        case_ids: case-id string per case index (``dataset.cases`` order).
        val_case: a case id (str), a comma-separated string of ids, or an
            iterable of ids whose beamlets form the validation side.

    Returns:
        ``(train_items, val_items)`` lists (order preserved).

    Raises:
        ValueError: an unknown case id, or an empty split side.
    """
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
    train = [it for it in items if it[0] not in val_idx]
    val = [it for it in items if it[0] in val_idx]
    if not train or not val:
        raise ValueError(
            f"empty split for val_case={wanted!r}: "
            f"{len(train)} train / {len(val)} val beamlets"
        )
    return train, val


def split_beamlets_by_beam(items, val_beam: int) -> tuple:
    """Partition (case, beam, ray, layer) items by held-out beam index.

    The single-case smoke fallback (mirrors ``train_stage0``'s ``--val-beam``):
    all beamlets whose ``beam_idx`` (``item[1]``) equals ``val_beam`` validate.

    Raises:
        ValueError: either side is empty.
    """
    val_beam = int(val_beam)
    train = [it for it in items if it[1] != val_beam]
    val = [it for it in items if it[1] == val_beam]
    if not train or not val:
        raise ValueError(
            f"empty split for val_beam={val_beam}: "
            f"{len(train)} train / {len(val)} val beamlets"
        )
    return train, val


# --------------------------------------------------------------------------- #
# Stratified per-beamlet query sampling (the proton residual-pilot strata,
# factored into a reusable sampler). Voxel centers of the beamlet dose grid are
# projected to (r_perp, WED) via the parallel grid; R80 places the range-
# critical distal band per beamlet.
# --------------------------------------------------------------------------- #


def _distal_r80_torch(r_perp: torch.Tensor, wed: torch.Tensor, dose: torch.Tensor) -> float:
    """Deepest WED where the on-axis depth dose crosses 80% of its peak (mm).

    The same 1 mm-binned on-axis detector as ``proton_residual_pilot._distal_r80``
    and ``proton_commission._distal_r80``; reimplemented on the already-gated
    torch tensors so the sampler needs no extra projection pass. Returns NaN when
    the on-axis profile is too sparse to locate the falloff.
    """
    on_axis = r_perp < ON_AXIS_R_PERP_MM
    if int(on_axis.sum()) < 10:
        return float("nan")
    w = wed[on_axis].detach().cpu().numpy()
    d = dose[on_axis].detach().cpu().numpy()
    bins = np.arange(w.min(), w.max() + 1.0, 1.0)
    if bins.size < 3:
        return float("nan")
    prof = np.array(
        [
            d[(w >= bins[i]) & (w < bins[i] + 1.0)].mean()
            if ((w >= bins[i]) & (w < bins[i] + 1.0)).any()
            else 0.0
            for i in range(len(bins) - 1)
        ]
    )
    if prof.size < 3 or prof.max() <= 0:
        return float("nan")
    peak = prof.max()
    k = int(prof.argmax())
    below = np.nonzero(prof[k:] < R80_FRAC * peak)[0]
    if below.size == 0:
        return float("nan")
    return float(bins[k + below[0]] + 0.5)


def stratified_beamlet_queries(
    beamlet,
    grid,
    rng: np.random.Generator,
    n: int,
    fracs: tuple = (
        FRAC_IN_FIELD, FRAC_DISTAL_FALLOFF, FRAC_OFF_AXIS, FRAC_LOW_DOSE,
        FRAC_FAR_SUPPRESS,
    ),
) -> dict:
    """Draw ``n`` stratified training queries from one beamlet's MC dose grid.

    Reuses the proton residual-pilot stratification (PROTON_SPEC section 3, the
    binding stratum is RANGE):

        ``in_field``        dose > 10% of beamlet max AND r_perp <= 3 mm (the
                            in-spot / central-axis dose-carrying core);
        ``distal_falloff``  gated AND within +-5 mm WED of the beamlet R80 (the
                            range-critical distal band -- the most important
                            stratum);
        ``off_axis``        gated AND r_perp > 3 mm (the lateral MCS tail);
        ``low_dose``        1% < dose <= 10% of beamlet max (the gate's outer
                            skirt).

    Every gated voxel center is projected to ``(r_perp, WED)`` through the
    parallel grid (``grid.contains`` / ``grid.sample(grid.wed, ...)``); R80 is
    the on-axis 1 mm-binned 80% crossing (:func:`_distal_r80_torch`). Empty
    strata fall back to the pooled in-field voxels. Counts are split by ``fracs``
    (in-field carries the remainder so the total is exactly ``n``); points are
    voxel centers (no sub-voxel jitter -- the proton dose grids are 2 mm and the
    distal band is only a few mm wide, so jitter would smear the range stratum).

    Args:
        beamlet: a :class:`prismx.proton_data.ProtonBeamlet` (its ``dose`` may be
            pre-scaled; ``grid_origin_mm`` / ``grid_spacing_mm`` /
            ``beamlet_geometry()`` describe the same grid).
        grid: the :class:`prismx.proton_grid.ProtonGrid` for this beamlet, built
            on the same device as ``beamlet.dose``.
        rng: numpy random generator (per-step query stream).
        n: number of queries to return.
        fracs: per-stratum fractions (in_field, distal_falloff, off_axis,
            low_dose); need not sum to 1 (in-field absorbs the remainder).

    Returns:
        dict with ``world_pts [n,3]``, ``target [n]``, ``stratum [n] int8``,
        ``m_weight [n]``, ``wed [n]`` / ``r_perp [n]`` (beamlet depth / off-axis
        coords; NaN / +inf for far-suppress points), ``r80`` (float, mm WED) --
        tensors on ``beamlet.dose.device``.
    """
    dose = beamlet.dose.to(torch.float32)
    dev = dose.device
    geom = beamlet.beamlet_geometry()
    sx, sy, sz = (float(s) for s in beamlet.grid_spacing_mm)
    ox, oy, oz = (float(o) for o in beamlet.grid_origin_mm)
    nz, ny, nx = dose.shape

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=dev),
        torch.arange(ny, dtype=torch.float32, device=dev),
        torch.arange(nx, dtype=torch.float32, device=dev),
        indexing="ij",
    )
    pts = torch.stack(
        [ox + xx.reshape(-1) * sx, oy + yy.reshape(-1) * sy, oz + zz.reshape(-1) * sz],
        dim=1,
    )
    dose_flat = dose.reshape(-1)
    d_max = float(dose_flat.max())

    # Far-suppression pool: near-zero voxels in the >1%-footprint bbox grown by
    # FAR_SUPPRESS_EXPAND_MM. Indices into the FULL flat grid (NOT the in-support
    # keep/contains pool); their TRUE (~0) MC dose is the target that teaches the
    # far_decoder to suppress the analytic baseline instead of leaking a bath.
    thr_lo = LOW_DOSE_FRAC * max(d_max, 1e-30)
    supp3d = dose > thr_lo
    sup_idx = torch.nonzero(supp3d, as_tuple=False)
    if sup_idx.numel() > 0:
        gz, gy, gx = (max(1, int(FAR_SUPPRESS_EXPAND_MM / s)) for s in (sz, sy, sx))
        zlo = max(0, int(sup_idx[:, 0].min()) - gz); zhi = min(nz, int(sup_idx[:, 0].max()) + gz + 1)
        ylo = max(0, int(sup_idx[:, 1].min()) - gy); yhi = min(ny, int(sup_idx[:, 1].max()) + gy + 1)
        xlo = max(0, int(sup_idx[:, 2].min()) - gx); xhi = min(nx, int(sup_idx[:, 2].max()) + gx + 1)
        box = torch.zeros_like(supp3d)
        box[zlo:zhi, ylo:yhi, xlo:xhi] = True
        far_pool = torch.nonzero((box & ~supp3d).reshape(-1), as_tuple=False).squeeze(1)
    else:  # degenerate all-zero beamlet
        far_pool = torch.arange(dose_flat.numel(), device=dev)

    # Pre-gate on the low-dose skirt to keep the projection cheap, then beamlet
    # coords + grid WED (exactly the residual-pilot order).
    keep = dose_flat > LOW_DOSE_FRAC * max(d_max, 1e-30)
    pts_k, dose_k = pts[keep], dose_flat[keep]
    if dose_k.numel() == 0:  # degenerate (all-zero) beamlet: fall back to all voxels
        pts_k, dose_k = pts, dose_flat
    uvd = geom.world_to_beamlet(pts_k)
    r_perp = torch.sqrt(uvd[:, 0] ** 2 + uvd[:, 1] ** 2)
    inside = grid.contains(uvd)
    if bool(inside.any()):
        pts_k, dose_k, r_perp, uvd = (
            pts_k[inside], dose_k[inside], r_perp[inside], uvd[inside],
        )
    wed = grid.sample(grid.wed, uvd, outside=0.0)

    r80 = _distal_r80_torch(r_perp, wed, dose_k)

    gate = dose_k > GATE_DOSE_FRAC * max(d_max, 1e-30)
    in_field = gate & (r_perp <= ON_AXIS_R_PERP_MM)
    off_axis = gate & (r_perp > ON_AXIS_R_PERP_MM)
    low_dose = (dose_k > LOW_DOSE_FRAC * max(d_max, 1e-30)) & ~gate
    if math.isfinite(r80):
        distal = gate & ((wed - r80).abs() <= DISTAL_BAND_MM)
    else:
        distal = torch.zeros_like(gate)

    masks = [None, None, None, None, None]
    masks[IN_FIELD] = in_field
    masks[DISTAL_FALLOFF] = distal
    masks[OFF_AXIS] = off_axis
    masks[LOW_DOSE] = low_dose
    # masks[FAR_SUPPRESS] is sampled from far_pool (full grid), handled below.

    # Counts: each stratum gets round(frac * n); in-field absorbs the remainder.
    counts = [int(round(f * n)) for f in fracs]
    counts[IN_FIELD] += n - sum(counts)
    if counts[IN_FIELD] < 0:  # fracs summed > 1: clamp, steal from the tail strata
        counts[IN_FIELD] = 0

    # Fallback pool when a stratum is empty: the pooled in-field voxels (or any
    # gated voxel, or -- degenerate -- every kept voxel).
    fallback = torch.nonzero(in_field, as_tuple=False).squeeze(1)
    if fallback.numel() == 0:
        fallback = torch.nonzero(gate, as_tuple=False).squeeze(1)
    if fallback.numel() == 0:
        fallback = torch.arange(dose_k.numel(), device=dev)
    fallback_np = fallback.cpu().numpy()

    out_pts, out_tgt, out_code, out_w, out_wed, out_rp = [], [], [], [], [], []
    for code, count in enumerate(counts):
        if count <= 0:
            continue
        if code == FAR_SUPPRESS:
            # Sample from the full-grid far pool (true ~0 targets); supervises
            # the far_decoder directly, unlike the contains-gated strata above.
            pool = far_pool.cpu().numpy()
            idx_np = rng.choice(pool, size=count, replace=pool.size < count)
            idx = torch.from_numpy(np.ascontiguousarray(idx_np, dtype=np.int64)).to(dev)
            out_pts.append(pts[idx])
            out_tgt.append(dose_flat[idx])
            out_code.append(torch.full((count,), code, dtype=torch.int8, device=dev))
            out_w.append(torch.full((count,), 1.0, dtype=torch.float32, device=dev))
            # Far-suppress points come from the FULL grid (no beamlet projection),
            # so they carry no (wed, r_perp); sentinel them out of the on-axis IDD
            # / R80 reductions (NaN wed, +inf r_perp).
            out_wed.append(torch.full((count,), float("nan"), dtype=torch.float32, device=dev))
            out_rp.append(torch.full((count,), float("inf"), dtype=torch.float32, device=dev))
            continue
        pool = torch.nonzero(masks[code], as_tuple=False).squeeze(1).cpu().numpy()
        if pool.size == 0:
            pool = fallback_np
        idx_np = rng.choice(pool, size=count, replace=pool.size < count)
        idx = torch.from_numpy(np.ascontiguousarray(idx_np, dtype=np.int64)).to(dev)
        out_pts.append(pts_k[idx])
        out_tgt.append(dose_k[idx])
        out_code.append(torch.full((count,), code, dtype=torch.int8, device=dev))
        # The distal-falloff (range-critical) stratum is up-weighted, like the
        # photon penumbra weight.
        wval = 2.0 if code == DISTAL_FALLOFF else 1.0
        out_w.append(torch.full((count,), wval, dtype=torch.float32, device=dev))
        out_wed.append(wed[idx])
        out_rp.append(r_perp[idx])

    return {
        "world_pts": torch.cat(out_pts),
        "target": torch.cat(out_tgt),
        "stratum": torch.cat(out_code),
        "m_weight": torch.cat(out_w),
        "wed": torch.cat(out_wed),
        "r_perp": torch.cat(out_rp),
        "r80": r80,
    }


# --------------------------------------------------------------------------- #
# Training scaffolding (EMA, LR schedule, host-RAM LRU cache, checkpoints) --
# the proton counterparts of train_stage0's helpers.
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="proton case dir or root of case dirs (repeatable; autodetected)",
    )
    parser.add_argument(
        "--val-case",
        type=str,
        default="1ABB118,1THB011",
        help="hold out ALL beamlets of these case id(s) for validation "
        "(comma-separated; patient holdout). If only one case is present it "
        "auto-falls back to a --val-beam holdout.",
    )
    parser.add_argument(
        "--val-beam",
        type=int,
        default=2,
        help="single-case fallback: hold out ALL beamlets of this beam_idx",
    )
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-beamlets", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--cache-gb",
        type=float,
        default=8.0,
        help="host-RAM budget for the LRU cache of per-beamlet dose/sigma "
        "tensors (kept modest by default; the shared host is OOM-prone)",
    )
    parser.add_argument(
        "--bragg",
        type=Path,
        default=Path("commissioning/proton_bragg.npz"),
        help="commissioned Bragg npz (R(E), shape, sigma_params); loaded via "
        "BraggIDD.from_npz and injected into PrismP so the analytic baseline "
        "uses the COMMISSIONED tables",
    )
    parser.add_argument(
        "--dose-scale",
        type=float,
        default=DOSE_UNITS_PER_BASELINE,
        help="raw dose-file units per unit proton baseline; dose/sigma are "
        "divided by this at load time (1.0 disables)",
    )
    parser.add_argument(
        "--lambda-idd",
        type=float,
        default=0.0,
        help="COMPLEMENTARY on-axis depth-dose (IDD) consistency loss -- NOT the "
        "lateral-halo fix (that is the re-commissioned sigma). Matches the ON-AXIS "
        "depth-dose CURVE (mean dose per WED bin) of pred to MC, putting gradient "
        "on the depth-dose shape/range the relative point loss is blind to. Guards "
        "the distal/range shape; 0 = off.",
    )
    parser.add_argument(
        "--sel-lambda",
        type=float,
        default=0.0,
        help="best.pt selection: add sel_lambda * mean|1 - sum(pred)/sum(tgt)| "
        "(high-dose integral bias) to val_loss. val_loss alone is scale/integral-"
        "cancelling, so it keeps selecting the over-integrating checkpoint; this "
        "makes selection SEE the magnitude/range offset. 0 = val_loss only.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup to --lr over --warmup-steps, cosine to LR_FINAL at --steps."""
    if step <= args.warmup_steps:
        return args.lr * step / max(1, args.warmup_steps)
    t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return LR_FINAL + 0.5 * (args.lr - LR_FINAL) * (1.0 + math.cos(math.pi * t))


class EMA:
    """Exponential moving average of the model state (floating entries)."""

    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:  # pragma: no cover - PrismP has no integer buffers today
                s.copy_(v)

    @contextmanager
    def swap(self, model: torch.nn.Module):
        """Temporarily load the EMA weights into ``model``."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(backup)


class HostBeamletCache:
    """LRU cache of host-RAM :class:`ProtonBeamlet` records bounded by tensor bytes.

    Dose and sigma are divided by ``dose_scale`` once at load time, bringing the
    raw Geant4 targets onto the proton analytic-baseline scale (module docstring).
    The CT is shared per case and cached separately by the caller, so only the
    per-beamlet dose/sigma tensors count against the byte budget here.
    """

    def __init__(self, dataset: ProtonBeamletDataset, limit_gb: float, dose_scale: float):
        self.dataset = dataset
        self.index_of = {item: i for i, item in enumerate(dataset.items)}
        self.limit_bytes = int(limit_gb * 2**30)
        self.dose_scale = float(dose_scale)
        self.entries: OrderedDict = OrderedDict()  # item -> (beamlet, d_max, nbytes)
        self.total_bytes = 0

    def get(self, item: tuple):
        """Return ``(ProtonBeamlet on host, d_max)`` for one (ci, bi, ri, li) item."""
        entry = self.entries.get(item)
        if entry is not None:
            self.entries.move_to_end(item)
            return entry[0], entry[1]
        bl = self.dataset[self.index_of[item]]
        bl.dose = bl.dose / self.dose_scale
        bl.sigma = bl.sigma / self.dose_scale
        d_max = float(bl.dose.max())
        nbytes = (bl.dose.numel() + bl.sigma.numel()) * bl.dose.element_size()
        self.entries[item] = (bl, d_max, nbytes)
        self.total_bytes += nbytes
        while self.total_bytes > self.limit_bytes and len(self.entries) > 1:
            _, (_, _, nb) = self.entries.popitem(last=False)
            self.total_bytes -= nb
        return bl, d_max


def proton_grid_spec(model: PrismP) -> ProtonGridSpec:
    """The grid spec used for query stratification (model's spec at 2 mm depth).

    The model's own ``grid_spec`` lateral extent/resolution is reused so the
    sampler's ``grid.contains`` matches the encode-time grid support; the depth
    step is pinned to the commissioning/pilot ``GRID_DD_MM`` for a consistent WED
    projection.
    """
    spec = model.grid_spec
    return ProtonGridSpec(
        nu=spec.nu,
        nv=spec.nv,
        du_mm=spec.du_mm,
        dv_mm=spec.dv_mm,
        dd_mm=GRID_DD_MM,
        max_depth_steps=spec.max_depth_steps,
        entry_margin_mm=spec.entry_margin_mm,
    )


def idd_consistency_loss(pred, target, wed, r_perp) -> tuple:
    """On-axis depth-dose (IDD) curve consistency loss + signed integral bias.

    Density/energy reach proton dose ONLY through the depth-dose curve, yet the
    relative point loss (``prismx_loss``) is blind to a COHERENT range/integral
    bias -- the engine deposits dose past the Bragg range and over-integrates
    (the proton twin of the MR WED-bias). This bins the ON-AXIS queries
    (``r_perp <= ON_AXIS_R_PERP_MM``) by WED and penalizes the per-bin MEAN dose
    mismatch ``|mean_pred - mean_tgt|`` -- the MEAN (not the sum) is independent
    of the stratified sampling density, so it is the depth-dose CURVE, not the
    sampling histogram. Normalized by the peak bin (so ``lambda_idd`` is O(1)).

    Returns ``(loss, signed_bias)`` with ``signed_bias`` = the on-axis integral
    ratio ``sum(mean_pred)/sum(mean_tgt) - 1`` (+ over-integrates), a detached
    float for logging. ``wed`` / ``r_perp`` are query coordinates (no grad);
    gradient flows only through ``pred``.
    """
    on_axis = torch.isfinite(wed) & (r_perp <= ON_AXIS_R_PERP_MM)
    if int(on_axis.sum()) < 2:
        return pred.new_zeros(()), 0.0
    w = wed[on_axis].detach()
    p = pred[on_axis]
    t = target[on_axis].detach()
    bin_idx = ((w - w.min()) / IDD_BIN_MM).long().clamp_min(0)
    nb = int(bin_idx.max().item()) + 1
    cnt = torch.zeros(nb, device=p.device).scatter_add(0, bin_idx, torch.ones_like(t)).clamp_min(1.0)
    mean_p = torch.zeros(nb, device=p.device).scatter_add(0, bin_idx, p) / cnt
    mean_t = torch.zeros(nb, device=p.device).scatter_add(0, bin_idx, t) / cnt
    ref = mean_t.max().clamp_min(1e-30)
    loss = ((mean_p - mean_t) / ref).square().mean()
    bias = float((mean_p.sum() / mean_t.sum().clamp_min(1e-30) - 1.0).detach())
    return loss, bias


def run_train_step(model, opt, ema, ct, beamlet, d_max, sample_spec, args, query_rng) -> tuple:
    """One optimizer step on a single beamlet (encode -> stratified query -> loss).

    ``ct`` is the per-case CT on the device; ``beamlet`` is the host-cached,
    dose-scaled :class:`ProtonBeamlet`. The query stream is ``query_rng``
    (consumed once here). Returns ``(loss_f, comps, n_distal, idd_bias)``.
    """
    dev = ct.density.device
    bl_dev = beamlet.to(dev)  # device copy of dose/sigma (CT/beam shared)
    grid = build_proton_grid(ct, beamlet.beam, beamlet.spot_uv, sample_spec)
    state = model.encode_beamlet(ct, beamlet.beam, beamlet.spot_uv, beamlet.energy_mev)

    batch = stratified_beamlet_queries(bl_dev, grid, query_rng, args.queries)
    pred = model.query(state, ct, beamlet.beam, batch["world_pts"])
    target = batch["target"]
    sigma = _approx_sigma(bl_dev, batch["world_pts"], target, d_max)
    loss, comps = prismx_loss(
        pred, target, sigma, d_max, batch["m_weight"], model.cfg.loss
    )
    idd_bias = 0.0
    if args.lambda_idd > 0.0:
        l_idd, idd_bias = idd_consistency_loss(
            pred, target, batch["wed"], batch["r_perp"]
        )
        loss = loss + args.lambda_idd * l_idd
        comps = {**comps, "idd": l_idd.detach()}
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    ema.update(model)
    n_distal = int((batch["stratum"] == DISTAL_FALLOFF).sum())
    return float(loss), comps, n_distal, idd_bias


def _approx_sigma(beamlet, world_pts, target, d_max: float) -> torch.Tensor:
    """Per-query MC sigma, trilinearly interpolated from the beamlet sigma map.

    The proton reader carries a Poisson-consistent per-voxel placeholder
    ``sigma`` (already divided by ``--dose-scale`` in the cache), so unlike
    train_stage0 -- which re-derives sigma from the target -- we interpolate the
    stored sigma map at the (possibly off-grid) query points. Runs on the dose
    grid's device.
    """
    sigma_grid = beamlet.sigma.to(world_pts.device, torch.float32)
    return _trilinear_world(
        sigma_grid, beamlet.grid_origin_mm, beamlet.grid_spacing_mm, world_pts
    )


def _trilinear_world(grid: torch.Tensor, origin, spacing, pts: torch.Tensor) -> torch.Tensor:
    """Edge-clamped trilinear interpolation of a world-aligned [Z, Y, X] grid.

    Identical to :func:`prismx.sampling._trilinear_world`; inlined so the script
    does not depend on the photon sampler (its signature is photon-specific).
    """
    nz, ny, nx = grid.shape
    ox, oy, oz = (float(o) for o in origin)
    sx, sy, sz = (float(s) for s in spacing)
    fx = ((pts[:, 0] - ox) / sx).clamp(0.0, nx - 1)
    fy = ((pts[:, 1] - oy) / sy).clamp(0.0, ny - 1)
    fz = ((pts[:, 2] - oz) / sz).clamp(0.0, nz - 1)
    x0, y0, z0 = fx.floor().long(), fy.floor().long(), fz.floor().long()
    x1 = (x0 + 1).clamp(max=nx - 1)
    y1 = (y0 + 1).clamp(max=ny - 1)
    z1 = (z0 + 1).clamp(max=nz - 1)
    tx, ty, tz = fx - x0, fy - y0, fz - z0
    flat = grid.reshape(-1)

    def at(zi, yi, xi):
        return flat[(zi * ny + yi) * nx + xi]

    c00 = at(z0, y0, x0) * (1 - tx) + at(z0, y0, x1) * tx
    c01 = at(z0, y1, x0) * (1 - tx) + at(z0, y1, x1) * tx
    c10 = at(z1, y0, x0) * (1 - tx) + at(z1, y0, x1) * tx
    c11 = at(z1, y1, x0) * (1 - tx) + at(z1, y1, x1) * tx
    return (c00 * (1 - ty) + c01 * ty) * (1 - tz) + (c10 * (1 - ty) + c11 * ty) * tz


def validate(model, ema, val_sel, cache, ct_cache, sample_spec, args, device) -> dict:
    """Loss + per-region |pred-target|/(target + 0.05 d_max) on fixed val beamlets.

    Runs with the EMA weights under ``no_grad``; every beamlet uses a FIXED query
    set (rng seeded by its val index) so the metric is comparable across
    validations. The distal_falloff region is the range-critical number.
    """
    losses = []
    abs_sum = {"overall": 0.0}
    cnt = {"overall": 0}
    for code in REPORT_STRATA:
        abs_sum[STRATUM_NAMES[code]] = 0.0
        cnt[STRATUM_NAMES[code]] = 0
    # high-dose integral bias (total / off-axis only) + signed R80 range error
    int_biases, off_int_biases, dr80s = [], [], []

    with torch.no_grad(), ema.swap(model):
        for vi, item in enumerate(val_sel):
            bl, d_max = cache.get(item)
            ct = ct_cache[item[0]]
            bl_dev = bl.to(device)
            grid = build_proton_grid(ct, bl.beam, bl.spot_uv, sample_spec)
            rng = np.random.default_rng(VAL_SEED_BASE + vi)
            batch = stratified_beamlet_queries(bl_dev, grid, rng, args.queries)
            pred = model.query(model_state(model, ct, bl), ct, bl.beam, batch["world_pts"])
            target = batch["target"]
            sigma = _approx_sigma(bl_dev, batch["world_pts"], target, d_max)
            loss, _ = prismx_loss(
                pred, target, sigma, d_max, batch["m_weight"], model.cfg.loss
            )
            losses.append(float(loss))
            rel = (pred - target).abs() / (target + REL_ERR_FLOOR * d_max)
            abs_sum["overall"] += float(rel.sum())
            cnt["overall"] += rel.numel()
            for code in REPORT_STRATA:
                mask = batch["stratum"] == code
                name = STRATUM_NAMES[code]
                abs_sum[name] += float(rel[mask].sum())
                cnt[name] += int(mask.sum())

            # Integral/range diagnostics (the over-integration the relative loss
            # is blind to): high-dose integral bias and signed R80 range error.
            hi = target > GATE_DOSE_FRAC * d_max
            st = float(target[hi].sum())
            if st > 0:
                int_biases.append(abs(1.0 - float(pred[hi].sum()) / st))
            # Off-axis-only integral bias: the IDD loss is on-axis only and the
            # lateral fix (1b) is deferred, so this channel is UNCONSTRAINED -- log
            # it to catch over-integration merely moving off-axis.
            off = hi & (batch["r_perp"] > ON_AXIS_R_PERP_MM)
            sto = float(target[off].sum())
            if sto > 0:
                off_int_biases.append(abs(1.0 - float(pred[off].sum()) / sto))
            r80_p = _distal_r80_torch(batch["r_perp"], batch["wed"], pred)
            r80_t = _distal_r80_torch(batch["r_perp"], batch["wed"], target)
            if math.isfinite(r80_p) and math.isfinite(r80_t):
                dr80s.append(r80_p - r80_t)

    rel_err = {
        k: (abs_sum[k] / cnt[k] if cnt[k] > 0 else float("nan")) for k in abs_sum
    }
    return {
        "loss": float(np.mean(losses)),
        "rel_err": rel_err,
        "int_bias": float(np.mean(int_biases)) if int_biases else float("nan"),
        "off_int_bias": float(np.mean(off_int_biases)) if off_int_biases else float("nan"),
        "dr80": float(np.mean(dr80s)) if dr80s else float("nan"),
    }


def model_state(model: PrismP, ct, beamlet):
    """Encode the beamlet's latent transport memory (validation helper)."""
    return model.encode_beamlet(ct, beamlet.beam, beamlet.spot_uv, beamlet.energy_mev)


def save_checkpoint(path: Path, model, ema, opt, step, args, best_val) -> None:
    def jsonable(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [jsonable(x) for x in v]
        return v

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "optimizer": opt.state_dict(),
            "step": step,
            "args": {k: jsonable(v) for k, v in vars(args).items()},
            "best_val": best_val,
        },
        path,
    )


def resolve_path(path: Path) -> Path:
    """Try the given path, then relative to the repo root (parent of scripts/)."""
    if path.is_file() or path.is_absolute():
        return path
    candidate = Path(__file__).resolve().parent.parent / path
    return candidate if candidate.is_file() else path


def build_model(args: argparse.Namespace, device: torch.device) -> PrismP:
    """Construct PrismP on the commissioned Bragg/sigma tables, encoder checkpointed.

    Loads ``--bragg`` via :meth:`BraggIDD.from_npz` (R(E) + Bragg shape) and
    pulls ``sigma_params[0]`` as the lateral-spread floor override, exactly like
    the residual pilot, so the analytic baseline rides on the COMMISSIONED
    tables. ``encoder.grad_checkpoint`` is forced True to bound peak training
    memory on the large proton beamlet sub-boxes.
    """
    bragg_path = resolve_path(args.bragg)
    sigma0 = None
    if bragg_path.is_file():
        bragg = BraggIDD.from_npz(bragg_path)
        with np.load(bragg_path) as data:
            sig = data.get("sigma_params")
            sigma0 = float(sig[0]) if sig is not None else None
        print(
            f"Bragg: commissioned table from {bragg_path}"
            + (f" (sigma0={sigma0:.2f} mm)" if sigma0 is not None else "")
        )
    else:
        bragg = BraggIDD.default()
        print(
            f"Bragg: seeded default ({args.bragg} not found) -- the analytic "
            "baseline will mix commissioning error into training"
        )
    model = PrismP(bragg=bragg, cfg=PrismXConfig(), sigma0_mm=sigma0).to(device)
    model.encoder.grad_checkpoint = True
    return model


def split_train_val(dataset: ProtonBeamletDataset, args: argparse.Namespace) -> tuple:
    """Hold out ``--val-case`` beamlets; fall back to ``--val-beam`` for one case.

    With >= 2 cases (or a held-out case different from the only case) this is a
    patient holdout. When a single case is present and it IS the requested
    holdout (the smoke), there is no other case to train on, so we fall back to
    holding out one BEAM index of that case (mirrors train_stage0's --val-beam).
    Returns ``(train_items, val_items, holdout_desc)``.
    """
    case_ids = [c.case_id for c in dataset.cases]
    wanted = [c.strip() for c in args.val_case.split(",") if c.strip()]
    present_wanted = [c for c in wanted if c in case_ids]
    # Single-case smoke: every requested holdout case is the only case present
    # (or no holdout case is present at all) -> beam-holdout fallback.
    single_case = len(case_ids) == 1
    if single_case and (not present_wanted or set(present_wanted) == set(case_ids)):
        train, val = split_beamlets_by_beam(dataset.items, args.val_beam)
        return train, val, f"beam_idx == {args.val_beam} (single-case fallback)"
    train, val = split_beamlets_by_case(dataset.items, case_ids, args.val_case)
    return train, val, f"case in {present_wanted or wanted}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # stream logs when piped
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sel_rng = np.random.default_rng(args.seed)
    query_rng = np.random.default_rng(args.seed + 1)

    cases = discover_cases(args.data)
    print(f"cases ({len(cases)}): {', '.join(c.name for c in cases)}")
    dataset = ProtonBeamletDataset(cases)

    try:
        train_items, val_items, holdout = split_train_val(dataset, args)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    print(
        f"beamlets: {len(train_items)} train, {len(val_items)} val ({holdout})"
    )

    model = build_model(args, device)
    print(f"PrismP parameters: {model.n_parameters():,}  device: {device}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    ema = EMA(model, EMA_DECAY)

    start_step = 0
    best_val = math.inf
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        if ckpt.get("best_val") is not None:
            best_val = float(ckpt["best_val"])
        print(f"resumed from {args.resume} at step {start_step}")

    # Prefer the commissioned dose-scale saved in the Bragg npz unless --dose-scale
    # was overridden. Re-commissioning (new sigma/Bragg) RE-DERIVES this scalar, so
    # the hard-coded default would be stale and silently re-introduce an integral
    # offset (the net trains to RELATIVE error and would absorb a wrong scale).
    if args.dose_scale == DOSE_UNITS_PER_BASELINE:
        bpath = resolve_path(args.bragg)
        if bpath.is_file():
            with np.load(bpath) as _d:
                _ds = _d.get("dose_scale")
            if _ds is not None and float(_ds) > 0:
                args.dose_scale = float(_ds)
                print(f"dose scale: using commissioned value from {bpath}")

    print(
        f"dose scale: dividing MC dose/sigma by {args.dose_scale:.4e} "
        "(raw units per unit baseline)"
    )

    # Per-case CT volumes on the device, once.
    ct_cache = {ci: case.ct().to(device) for ci, case in enumerate(dataset.cases)}
    cache = HostBeamletCache(dataset, args.cache_gb, args.dose_scale)
    sample_spec = proton_grid_spec(model)

    # Fixed, evenly spaced validation beamlets.
    n_val = min(args.val_beamlets, len(val_items))
    sel_pos = np.unique(
        np.linspace(0, len(val_items) - 1, num=n_val).round().astype(int)
    )
    val_sel = [val_items[i] for i in sel_pos]
    print(
        f"validation: {len(val_sel)} fixed beamlets "
        f"{[f'{it[0]}/B{it[1]}/R{it[2]}/L{it[3]}' for it in val_sel]}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    train_ema = None
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        lr = lr_at(step, args)
        for group in opt.param_groups:
            group["lr"] = lr
        item = train_items[int(sel_rng.integers(len(train_items)))]
        bl, d_max = cache.get(item)
        ct = ct_cache[item[0]]
        loss_f, comps, n_distal, idd_bias = run_train_step(
            model, opt, ema, ct, bl, d_max, sample_spec, args, query_rng
        )

        train_ema = (
            loss_f
            if train_ema is None
            else TRAIN_LOSS_EMA_DECAY * train_ema + (1.0 - TRAIN_LOSS_EMA_DECAY) * loss_f
        )

        if step % args.log_every == 0 or step == start_step + 1:
            dt = (time.time() - t0) / (step - start_step)
            idd_str = (
                f"  idd {float(comps['idd']):.5f}({idd_bias:+.3f})" if "idd" in comps else ""
            )
            print(
                f"step {step:6d}  lr {lr:.2e}  loss {loss_f:.5f}  "
                f"ema {train_ema:.5f}  (mse {float(comps['mse']):.5f}  "
                f"log {float(comps['log']):.5f}{idd_str})  distal {n_distal:4d}  "
                f"{dt:.2f} s/step",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(
                model, ema, val_sel, cache, ct_cache, sample_spec, args, device
            )
            rel = metrics["rel_err"]
            ibias, dr80 = metrics["int_bias"], metrics["dr80"]
            sel = metrics["loss"] + args.sel_lambda * (ibias if math.isfinite(ibias) else 0.0)
            print(f"  validation @ step {step} ({len(val_sel)} beamlets)")
            print(f"    {'val_loss':<24s}{metrics['loss']:.5f}")
            print(
                f"    {'sel (loss+lam*ibias)':<24s}{sel:.5f}"
                f"   int_bias {ibias:.4f} (off {metrics['off_int_bias']:.4f})"
                f"  dR80 {dr80:+.2f} mm"
            )
            for key in ("overall", *(STRATUM_NAMES[c] for c in REPORT_STRATA)):
                print(f"    {'rel_err ' + key:<24s}{rel[key]:.4f}")
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "lr": lr,
                            "train_loss_ema": train_ema,
                            "val_loss": metrics["loss"],
                            "val_sel": sel,
                            "val_int_bias": ibias,
                            "val_off_int_bias": metrics["off_int_bias"],
                            "val_dr80": dr80,
                            "val_rel_err": rel,
                        }
                    )
                    + "\n"
                )
            if sel < best_val:
                best_val = sel
                save_checkpoint(
                    args.out / "best.pt", model, ema, opt, step, args, best_val
                )
                print(f"    new best sel {best_val:.5f} -> best.pt")
            save_checkpoint(
                args.out / "last.pt", model, ema, opt, step, args, best_val
            )

    n_done = args.steps - start_step
    print(
        f"done: {n_done} steps in {time.time() - t0:.1f} s, "
        f"best sel {best_val:.5f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
