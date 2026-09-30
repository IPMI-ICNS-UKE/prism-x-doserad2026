"""Train the Photon analytical-prior correction network."""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from prismx.analytic import PDDTable
from prismx.config import synthrad_config
from prismx.data import discover_cases, split_items, split_items_by_case
from prismx.losses import prismx_loss
from prismx.model import PrismX
from prismx.plan_data import SIGMA_FLOOR_REL_DEFAULT, SIGMA_REL_DEFAULT, PlanCPDataset
from prismx.inference import _filled_body_voxel_mask
from prismx.losses import sample_grid_at_world, stratified_plan_sum_loss
from prismx.sampling import (
    IN_FIELD,
    OUT_OF_FIELD,
    PENUMBRA,
    STRATUM_NAMES,
    stratified_query_points,
)
from prismx.train_prep import encode_from_channels, prepare_cp, reconstruct_on_device

LR_FINAL: float = 1e-05
EMA_DECAY: float = 0.999
DOSE_UNITS_PER_BASELINE: float = 5.437449455848826e-05
TRAIN_LOSS_EMA_DECAY: float = 0.98
REL_ERR_FLOOR: float = 0.05
VAL_SEED_BASE: int = 10000
FULL_IDD_VAL_SEED_BASE: int = 20000
REPORT_STRATA: tuple = (IN_FIELD, PENUMBRA, OUT_OF_FIELD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--width-scale", type=float, default=1.0, help="width scale")
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="case dir or root of case dirs (repeatable; autodetected)",
    )
    parser.add_argument(
        "--val-beam",
        type=int,
        default=2,
        help="hold out ALL control points of this beam_idx for validation",
    )
    parser.add_argument(
        "--val-case",
        type=str,
        default=None,
        help="hold out ALL control points of this case id instead (patient holdout; overrides --val-beam)",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-cps", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--cache-gb",
        type=float,
        default=8.0,
        help="host-RAM budget for the LRU cache of PlanCP dose/sigma tensors",
    )
    parser.add_argument(
        "--dose-scale",
        type=float,
        default=DOSE_UNITS_PER_BASELINE,
        help="raw dose-file units per unit analytic baseline; dose/sigma are divided by this at load time (1.0 disables)",
    )
    parser.add_argument(
        "--pdd",
        type=Path,
        default=Path("commissioning/synthrad_photon_pdd.npz"),
        help="commissioned PDD npz; used instead of the placeholder if it exists",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume model, EMA, optimizer, and global step",
    )
    checkpoint_group.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="initialize model from a checkpoint's EMA weights, but reset optimizer, LR schedule, step, and best validation score (fine-tuning)",
    )
    parser.add_argument(
        "--challenge-weight",
        type=float,
        default=0.0,
        help="weight of sampled DoseRAD masked MAE (target >=10%% CP max) added to the original PRISM-X loss; 0 preserves legacy training",
    )
    parser.add_argument("--lambda-idd", type=float, default=0.0, help="lambda idd")
    parser.add_argument("--lambda-plan", type=float, default=0.0, help="lambda plan")
    parser.add_argument(
        "--plan-cps",
        type=int,
        default=3,
        help="sibling control points added to the step's own CP to form the partial plan. Each costs a full encode_cp + query.",
    )
    parser.add_argument(
        "--plan-bands",
        type=int,
        default=4,
        help="equal-count dose strata of the summed target, mirroring the stratification of metric 2.1.",
    )
    parser.add_argument(
        "--lambda-full-idd", type=float, default=0.0, help="lambda full idd"
    )
    parser.add_argument(
        "--full-idd-cutoff-raw",
        type=float,
        default=3e-07,
        help="raw-dose cutoff used by the differentiable full-slice IDD loss",
    )
    parser.add_argument(
        "--full-idd-val-cutoffs",
        default="0,1e-7,3e-7,1e-6",
        help="comma-separated raw-dose cutoffs reported during validation",
    )
    parser.add_argument(
        "--full-idd-output-scale",
        type=float,
        default=None,
        help="raw output units per model-dose unit before applying the cutoff; defaults to --dose-scale (the Photon/CT deployment path)",
    )
    parser.add_argument(
        "--full-idd-tau-frac",
        type=float,
        default=0.25,
        help="soft-gate temperature as a fraction of the normalized cutoff; the forward remains the exact hard cutoff",
    )
    parser.add_argument(
        "--full-idd-slices",
        type=int,
        default=24,
        help="stratified array-depth slices sampled per control point",
    )
    parser.add_argument(
        "--full-idd-samples-per-slice",
        type=int,
        default=256,
        help="uniform full-plane samples per selected depth slice",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("loss", "challenge_mae", "idd", "full_idd"),
        default="loss",
        help="validation metric used to select best.pt",
    )
    parser.add_argument(
        "--beam-mae-guard-baseline",
        type=float,
        default=None,
        help="optional sampled metric-1.1 baseline for guarded full-IDD selection",
    )
    parser.add_argument(
        "--beam-mae-guard-tolerance",
        type=float,
        default=0.0,
        help="allowed metric-1.1 increase before the selector guard activates",
    )
    parser.add_argument(
        "--beam-mae-guard-weight",
        type=float,
        default=0.0,
        help="quadratic Beam-MAE guard weight added to full-IDD selection",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="bank a numbered checkpoint every N steps (0 disables)",
    )
    parser.add_argument(
        "--calibrate-full-idd-only",
        action="store_true",
        help="measure step-0 base/full-IDD losses and parameter-gradient norms on fixed balanced validation CPs, write calibration.json, then exit",
    )
    parser.add_argument(
        "--calibration-cps",
        type=int,
        default=4,
        help="number of patient-balanced held-out CPs used by --calibrate-full-idd-only",
    )
    parser.add_argument(
        "--in-field-isodose",
        type=float,
        default=None,
        help="override the in-field sampling threshold (0.1 aligns it with metric 1.1)",
    )
    parser.add_argument(
        "--sampling-fractions",
        default=None,
        help="override in-field,penumbra,buildup,interface,out-of-field fractions as five comma-separated values summing to 1",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="num workers")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="batches each worker prefetches ahead (only with --num-workers>0)",
    )
    return parser.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup to --lr over --warmup-steps, cosine to LR_FINAL at --steps."""
    if step <= args.warmup_steps:
        return args.lr * step / max(1, args.warmup_steps)
    t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return LR_FINAL + 0.5 * (args.lr - LR_FINAL) * (1.0 + math.cos(math.pi * t))


def approx_sigma(target: torch.Tensor, d_max: float) -> torch.Tensor:
    """Poisson-consistent MC-noise placeholder evaluated at the query targets."""
    return (
        SIGMA_REL_DEFAULT * torch.sqrt(target.clamp(min=0.0) * d_max)
        + SIGMA_FLOOR_REL_DEFAULT * d_max
    )


class EMA:
    """Exponential moving average of the model state (floating entries)."""

    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
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


class HostCPCache:
    """LRU cache of host-RAM :class:`PlanCP` records bounded by tensor bytes."""

    def __init__(self, dataset: PlanCPDataset, limit_gb: float, dose_scale: float):
        self.dataset = dataset
        self.index_of = {item: i for i, item in enumerate(dataset.items)}
        self.limit_bytes = int(limit_gb * 2**30)
        self.dose_scale = float(dose_scale)
        self.entries: OrderedDict = OrderedDict()
        self.total_bytes = 0

    def get(self, item: tuple):
        """Return ``(PlanCP on host, d_max)`` for one ``(ci, bi, cp)`` item."""
        entry = self.entries.get(item)
        if entry is not None:
            self.entries.move_to_end(item)
            return (entry[0], entry[1])
        cp = self.dataset[self.index_of[item]]
        cp.dose = cp.dose / self.dose_scale
        cp.sigma = cp.sigma / self.dose_scale
        d_max = float(cp.dose.max())
        nbytes = (cp.dose.numel() + cp.sigma.numel()) * cp.dose.element_size()
        self.entries[item] = (cp, d_max, nbytes)
        self.total_bytes += nbytes
        while self.total_bytes > self.limit_bytes and len(self.entries) > 1:
            _, (_, _, nb) = self.entries.popitem(last=False)
            self.total_bytes -= nb
        return (cp, d_max)


def make_query_batch(state, cp, dose_dev, cfg, rng, n: int) -> dict:
    """Stratified queries for one CP with the dose grid already on the device."""
    return stratified_query_points(
        dose_dev,
        cp.grid_origin_mm,
        cp.grid_spacing_mm,
        cp.beam,
        state.raster,
        state.cone,
        cfg.sampling,
        cfg.loss,
        rng,
        n=n,
    )


def challenge_masked_mae(
    pred: torch.Tensor, target: torch.Tensor, d_max: float
) -> torch.Tensor:
    """Differentiable sampled analogue of official metric 1.1."""
    scale = float(d_max)
    if scale <= 0.0:
        return pred.new_zeros(())
    mask = target >= 0.1 * scale
    error_sum = ((pred - target).abs() * mask).sum()
    count = mask.sum().clamp_min(1)
    return error_sum / count / scale


IDD_BIN_MM: float = 2.0


def idd_curve_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    world_pts: torch.Tensor,
    d_max: float,
    bin_mm: float = IDD_BIN_MM,
) -> torch.Tensor:
    """In-field depth-profile consistency; HELPS metric 1.2 via the low-dose bath."""
    scale = float(d_max)
    if scale <= 0.0:
        return pred.new_zeros(())
    mask = target >= 0.1 * scale
    if int(mask.sum()) < 2:
        return pred.new_zeros(())
    depth = world_pts[mask, 2].detach()
    p = pred[mask]
    t = target[mask].detach()
    bin_idx = ((depth - depth.min()) / float(bin_mm)).long().clamp_min(0)
    n_bins = int(bin_idx.max().item()) + 1
    zeros = torch.zeros(n_bins, device=p.device, dtype=p.dtype)
    counts = zeros.scatter_add(0, bin_idx, torch.ones_like(p))
    cnt = counts.clamp_min(1.0)
    mean_p = zeros.scatter_add(0, bin_idx, p) / cnt
    mean_t = zeros.scatter_add(0, bin_idx, t.to(p.dtype)) / cnt
    ref = mean_t.max().clamp_min(1e-30)
    occupied = counts > 0
    return ((mean_p - mean_t) / ref).square()[occupied].mean()


def parse_cutoff_grid(value: str) -> tuple[float, ...]:
    """Parse a non-empty comma-separated grid of finite, non-negative cutoffs."""
    try:
        cutoffs = tuple((float(v.strip()) for v in value.split(",") if v.strip()))
    except ValueError as err:
        raise ValueError("cutoff grid must contain numbers") from err
    if not cutoffs or any((not math.isfinite(v) or v < 0.0 for v in cutoffs)):
        raise ValueError("cutoff grid must contain finite non-negative numbers")
    return cutoffs


def stratified_depth_samples(
    n_depth: int, n_slices: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one uniform slice from each contiguous depth stratum."""
    n_depth, n_slices = (int(n_depth), int(n_slices))
    if n_depth <= 0 or n_slices <= 0:
        raise ValueError("n_depth and n_slices must be positive")
    n_strata = min(n_depth, n_slices)
    edges = np.floor(np.linspace(0, n_depth, n_strata + 1)).astype(np.int64)
    edges[-1] = n_depth
    slices, widths = ([], [])
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(int(hi), int(lo) + 1)
        slices.append(int(rng.integers(int(lo), hi)))
        widths.append(hi - int(lo))
    return (
        np.asarray(slices, dtype=np.int64),
        np.asarray(widths, dtype=np.float64) / float(n_depth),
    )


def make_full_slice_batch(
    dose: torch.Tensor,
    body_mask: torch.Tensor,
    grid_origin_mm: tuple,
    grid_spacing_mm: tuple,
    rng: np.random.Generator,
    n_slices: int,
    samples_per_slice: int,
) -> dict:
    """Uniform complete-plane samples for the official fixed-axis IDD."""
    if dose.ndim != 3:
        raise ValueError(f"dose must be [Z,Y,X], got {tuple(dose.shape)}")
    if tuple(body_mask.shape) != tuple(dose.shape):
        raise ValueError(
            f"body mask {tuple(body_mask.shape)} does not match dose {tuple(dose.shape)}"
        )
    if int(samples_per_slice) <= 0:
        raise ValueError("samples_per_slice must be positive")
    dose = dose.to(torch.float32)
    device = dose.device
    nz, ny, nx = (int(v) for v in dose.shape)
    z_np, weight_np = stratified_depth_samples(nz, n_slices, rng)
    n_selected = int(z_np.size)
    sample_shape = (n_selected, int(samples_per_slice))
    y_np = rng.integers(0, ny, size=sample_shape, dtype=np.int64)
    x_np = rng.integers(0, nx, size=sample_shape, dtype=np.int64)
    z_grid_np = np.broadcast_to(z_np[:, None], sample_shape).copy()
    z = torch.from_numpy(z_grid_np).to(device)
    y = torch.from_numpy(np.ascontiguousarray(y_np)).to(device)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    selected_z = torch.from_numpy(z_np).to(device)
    sx, sy, sz = (float(v) for v in grid_spacing_mm)
    ox, oy, oz = (float(v) for v in grid_origin_mm)
    world_pts = torch.stack(
        (
            ox + sx * x.reshape(-1).to(torch.float32),
            oy + sy * y.reshape(-1).to(torch.float32),
            oz + sz * z.reshape(-1).to(torch.float32),
        ),
        dim=1,
    )
    target_samples = dose[z, y, x]
    target_slice_mean_all = dose.mean(dim=(1, 2))
    return {
        "world_pts": world_pts,
        "target_samples": target_samples,
        "body_samples": body_mask.to(torch.bool)[z, y, x],
        "target_slice_mean": target_slice_mean_all[selected_z],
        "target_peak": target_slice_mean_all.max(),
        "slice_weights": torch.from_numpy(weight_np.astype(np.float32)).to(device),
        "selected_z": selected_z,
        "sample_indices": (z, y, x),
    }


def straight_through_cutoff(
    pred: torch.Tensor, cutoff_raw: float, output_scale: float, tau_frac: float
) -> torch.Tensor:
    """Exact hard-cut forward with a smooth backward gate."""
    cutoff_raw, output_scale, tau_frac = (
        float(cutoff_raw),
        float(output_scale),
        float(tau_frac),
    )
    if not math.isfinite(cutoff_raw) or cutoff_raw < 0.0:
        raise ValueError("cutoff_raw must be finite and non-negative")
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("output_scale must be finite and positive")
    if not math.isfinite(tau_frac) or tau_frac <= 0.0:
        raise ValueError("tau_frac must be finite and positive")
    if cutoff_raw == 0.0:
        return pred
    cutoff = pred.new_tensor(cutoff_raw / output_scale)
    tau = cutoff * tau_frac
    soft = pred * torch.sigmoid((pred - cutoff) / tau)
    hard = torch.where(pred > cutoff, pred, torch.zeros_like(pred))
    return soft + (hard - soft).detach()


def estimate_cutoff_slice_means(
    pred_samples: torch.Tensor,
    batch: dict,
    cutoff_raw: float,
    output_scale: float,
    tau_frac: float,
) -> torch.Tensor:
    """Estimate selected prediction slice means with an exact-target control variate."""
    shape = batch["target_samples"].shape
    pred = pred_samples.reshape(shape)
    pred = torch.where(batch["body_samples"], pred, torch.zeros_like(pred))
    cut = straight_through_cutoff(pred, cutoff_raw, output_scale, tau_frac)
    delta = (cut - batch["target_samples"]).mean(dim=1)
    return batch["target_slice_mean"] + delta


def full_slice_idd_loss_from_samples(
    pred_samples: torch.Tensor,
    batch: dict,
    cutoff_raw: float,
    output_scale: float,
    tau_frac: float,
) -> torch.Tensor:
    """RMS official-IDD estimator using an exact-target control variate."""
    target_peak = batch["target_peak"]
    if float(target_peak) <= 0.0:
        return pred_samples.sum() * 0.0
    pred_slice_mean = estimate_cutoff_slice_means(
        pred_samples, batch, cutoff_raw, output_scale, tau_frac
    )
    relative_sq = (
        (pred_slice_mean - batch["target_slice_mean"]) / target_peak
    ).square()
    mean_square = (relative_sq * batch["slice_weights"]).sum()
    eps = mean_square.new_tensor(1e-12)
    return torch.sqrt(mean_square + eps) - torch.sqrt(eps)


def query_full_slice_idd(
    model,
    state,
    ct,
    cp_meta,
    dose_dev: torch.Tensor,
    body_mask: torch.Tensor,
    args,
    rng: np.random.Generator,
    cutoffs: tuple[float, ...] | None = None,
) -> dict[float, torch.Tensor]:
    """Query the INR once and evaluate one or more raw cutoff hypotheses."""
    batch = make_full_slice_batch(
        dose_dev,
        body_mask,
        cp_meta.grid_origin_mm,
        cp_meta.grid_spacing_mm,
        rng,
        int(getattr(args, "full_idd_slices", 24)),
        int(getattr(args, "full_idd_samples_per_slice", 256)),
    )
    pred = model.query(state, ct, cp_meta.beam, batch["world_pts"])
    output_scale = getattr(args, "full_idd_output_scale", None)
    if output_scale is None:
        output_scale = float(getattr(args, "dose_scale", DOSE_UNITS_PER_BASELINE))
    tau_frac = float(getattr(args, "full_idd_tau_frac", 0.25))
    if cutoffs is None:
        cutoffs = (float(getattr(args, "full_idd_cutoff_raw", 3e-07)),)
    return {
        float(cutoff): full_slice_idd_loss_from_samples(
            pred, batch, float(cutoff), float(output_scale), tau_frac
        )
        for cutoff in cutoffs
    }


def _combined_loss(pred, batch, sigma, d_max, cfg, args):
    base, comps = prismx_loss(
        pred, batch["target"], sigma, d_max, batch["m_weight"], cfg.loss
    )
    challenge = challenge_masked_mae(pred, batch["target"], d_max)
    weight = float(getattr(args, "challenge_weight", 0.0))
    total = base + weight * challenge
    comps = dict(comps)
    comps["base"] = base.detach()
    comps["challenge"] = challenge.detach()
    idd = idd_curve_loss(pred, batch["target"], batch["world_pts"], d_max)
    comps["idd"] = idd.detach()
    lambda_idd = float(getattr(args, "lambda_idd", 0.0))
    if lambda_idd > 0.0:
        total = total + lambda_idd * idd
    return (total, comps)


def make_step_rngs(seed: int) -> tuple:
    """Two independent, reproducible RNG streams: (CP selection, query sampling)."""
    sel_ss, query_ss = np.random.SeedSequence(int(seed)).spawn(2)
    return (np.random.default_rng(sel_ss), np.random.default_rng(query_ss))


def make_step_items(train_items, sel_rng: np.random.Generator, n_steps: int) -> list:
    """Deterministic per-step ``(case, beam, cp)`` sequence drawn from ``sel_rng``."""
    return [
        train_items[int(sel_rng.integers(len(train_items)))] for _ in range(n_steps)
    ]


def balanced_validation_selection(
    val_items: list, index_of: dict, n_val: int
) -> list[tuple[int, tuple]]:
    """Evenly allocate fixed validation CPs across held-out patients."""
    if not val_items or int(n_val) <= 0:
        return []
    groups: OrderedDict[int, list] = OrderedDict()
    for item in val_items:
        groups.setdefault(int(item[0]), []).append(item)
    target = min(int(n_val), len(val_items))
    case_ids = list(groups)
    quotas = {ci: min(len(groups[ci]), target // len(case_ids)) for ci in case_ids}
    remaining = target - sum(quotas.values())
    while remaining > 0:
        progressed = False
        for ci in case_ids:
            if quotas[ci] < len(groups[ci]):
                quotas[ci] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    selected = []
    for ci in case_ids:
        items = groups[ci]
        quota = quotas[ci]
        if quota <= 0:
            continue
        positions = np.unique(
            np.linspace(0, len(items) - 1, num=quota).round().astype(int)
        )
        selected.extend(((index_of[items[pos]], items[pos]) for pos in positions))
    return selected


class PrismXStepDataset(torch.utils.data.Dataset):
    """CPU-prep Dataset over a FIXED per-step ``(case, beam, cp)`` sequence."""

    def __init__(self, dataset, step_items, cfg, dose_scale: float):
        self.dataset = dataset
        self.index_of = {item: i for i, item in enumerate(dataset.items)}
        self.step_items = list(step_items)
        self.cfg = cfg
        self.dose_scale = float(dose_scale)

    def __len__(self) -> int:
        return len(self.step_items)

    def __getitem__(self, i: int) -> dict:
        item = self.step_items[i]
        cp = self.dataset[self.index_of[item]]
        prep = prepare_cp(cp, self.cfg, self.dose_scale)
        prep["step_item"] = item
        return prep


def _identity_collate(batch: list):
    """Return the single prepped item unchanged (batch size is always 1)."""
    return batch[0]


class _CPMeta:
    """Light stand-in for a :class:`PlanCP` carrying only what the query needs."""

    __slots__ = ("grid_origin_mm", "grid_spacing_mm", "beam")

    def __init__(self, grid_origin_mm, grid_spacing_mm, beam):
        self.grid_origin_mm = grid_origin_mm
        self.grid_spacing_mm = grid_spacing_mm
        self.beam = beam


def plan_sum_term(
    model, ct, cp_meta, siblings, world_pts, pred0, target0, n_bands: int
):
    """Partial plan sum: the step's control point plus ``siblings`` from the case."""
    preds, targets = ([pred0], [target0])
    for sib, sib_dose in siblings:
        state_s = model.encode_cp(ct, sib.beam, sib.aperture)
        preds.append(model.query(state_s, ct, sib.beam, world_pts))
        targets.append(
            sample_grid_at_world(
                sib_dose, sib.grid_origin_mm, sib.grid_spacing_mm, world_pts
            )
        )
    return stratified_plan_sum_loss(
        torch.stack(preds), torch.stack(targets), n_bands=n_bands
    )


def run_train_step(
    model,
    opt,
    ema,
    state,
    ct,
    cp_meta,
    dose_dev,
    d_max,
    cfg,
    args,
    query_rng,
    body_mask=None,
    siblings=None,
) -> tuple:
    """One optimizer step shared by the synchronous and prefetched paths."""
    batch = make_query_batch(state, cp_meta, dose_dev, cfg, query_rng, args.queries)
    pred = model.query(state, ct, cp_meta.beam, batch["world_pts"])
    sigma = approx_sigma(batch["target"], d_max)
    loss, comps = _combined_loss(pred, batch, sigma, d_max, cfg, args)
    full_weight = float(getattr(args, "lambda_full_idd", 0.0))
    if full_weight > 0.0:
        if body_mask is None:
            raise ValueError("body_mask is required when --lambda-full-idd > 0")
        cutoff = float(getattr(args, "full_idd_cutoff_raw", 3e-07))
        full = query_full_slice_idd(
            model,
            state,
            ct,
            cp_meta,
            dose_dev,
            body_mask,
            args,
            query_rng,
            cutoffs=(cutoff,),
        )[cutoff]
        loss = loss + full_weight * full
        comps["full_idd"] = full.detach()
    else:
        comps["full_idd"] = pred.new_zeros(())
    plan_weight = float(getattr(args, "lambda_plan", 0.0))
    if plan_weight > 0.0 and siblings:
        plan = plan_sum_term(
            model,
            ct,
            cp_meta,
            siblings,
            batch["world_pts"],
            pred,
            batch["target"],
            int(getattr(args, "plan_bands", 4)),
        )
        loss = loss + plan_weight * plan
        comps["plan"] = plan.detach()
    else:
        comps["plan"] = pred.new_zeros(())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    ema.update(model)
    return (float(loss), comps)


def validate(
    model, ema, val_sel, cache, ct_cache, cfg, args, device, body_cache=None
) -> dict:
    """Loss + |pred-target|/(target + 0.05 d_max) on the fixed val items."""
    losses, base_losses, challenge_maes, idds = ([], [], [], [])
    full_enabled = (
        float(getattr(args, "lambda_full_idd", 0.0)) > 0.0
        or getattr(args, "selection_metric", "loss") == "full_idd"
    )
    main_cutoff = float(getattr(args, "full_idd_cutoff_raw", 3e-07))
    try:
        val_cutoffs = parse_cutoff_grid(
            getattr(args, "full_idd_val_cutoffs", "0,1e-7,3e-7,1e-6")
        )
    except ValueError as err:
        raise ValueError(f"invalid --full-idd-val-cutoffs: {err}") from err
    if main_cutoff not in val_cutoffs:
        val_cutoffs = (*val_cutoffs, main_cutoff)
    full_idds = {cutoff: [] for cutoff in val_cutoffs}
    abs_sum = {"overall": 0.0}
    cnt = {"overall": 0}
    for code in REPORT_STRATA:
        abs_sum[STRATUM_NAMES[code]] = 0.0
        cnt[STRATUM_NAMES[code]] = 0
    with torch.no_grad(), ema.swap(model):
        for ds_index, item in val_sel:
            cp, d_max = cache.get(item)
            ct = ct_cache[item[0]]
            dose_dev = cp.dose.to(device)
            state = model.encode_cp(ct, cp.beam, cp.aperture)
            rng = np.random.default_rng(VAL_SEED_BASE + ds_index)
            batch = make_query_batch(state, cp, dose_dev, cfg, rng, args.queries)
            pred = model.query(state, ct, cp.beam, batch["world_pts"])
            target = batch["target"]
            sigma = approx_sigma(target, d_max)
            loss, comps = _combined_loss(pred, batch, sigma, d_max, cfg, args)
            losses.append(float(loss))
            base_losses.append(float(comps["base"]))
            challenge_maes.append(float(comps["challenge"]))
            idds.append(float(comps["idd"]))
            if full_enabled:
                if body_cache is None:
                    raise ValueError("body_cache is required for full-IDD validation")
                full_rng = np.random.default_rng(FULL_IDD_VAL_SEED_BASE + ds_index)
                cutoff_losses = query_full_slice_idd(
                    model,
                    state,
                    ct,
                    cp,
                    dose_dev,
                    body_cache[item[0]],
                    args,
                    full_rng,
                    cutoffs=val_cutoffs,
                )
                for cutoff, value in cutoff_losses.items():
                    full_idds[cutoff].append(float(value))
            rel = (pred - target).abs() / (target + REL_ERR_FLOOR * d_max)
            abs_sum["overall"] += float(rel.sum())
            cnt["overall"] += rel.numel()
            for code in REPORT_STRATA:
                mask = batch["stratum"] == code
                name = STRATUM_NAMES[code]
                abs_sum[name] += float(rel[mask].sum())
                cnt[name] += int(mask.sum())
    rel_err = {k: abs_sum[k] / cnt[k] if cnt[k] > 0 else float("nan") for k in abs_sum}
    result = {
        "loss": float(np.mean(losses)),
        "base_loss": float(np.mean(base_losses)),
        "challenge_mae": float(np.mean(challenge_maes)),
        "idd": float(np.mean(idds)),
        "rel_err": rel_err,
    }
    if full_enabled:
        result["full_idd"] = float(np.mean(full_idds[main_cutoff]))
        result["full_idd_by_cutoff"] = {
            cutoff: float(np.mean(values)) for cutoff, values in full_idds.items()
        }
    return result


def checkpoint_selection_score(metrics: dict, args) -> float:
    """Configured validation selector with an optional one-sided Beam-MAE guard."""
    metric = getattr(args, "selection_metric", "loss")
    score = float(metrics[metric])
    if metric != "full_idd":
        return score
    baseline = getattr(args, "beam_mae_guard_baseline", None)
    weight = float(getattr(args, "beam_mae_guard_weight", 0.0))
    tolerance = float(getattr(args, "beam_mae_guard_tolerance", 0.0))
    if baseline is None or weight <= 0.0:
        return score
    excess = max(0.0, float(metrics["challenge_mae"]) - float(baseline) - tolerance)
    return score + weight * excess * excess


def _parameter_gradient_l2(model: torch.nn.Module) -> float:
    """Global L2 norm of currently populated parameter gradients."""
    total = None
    for parameter in model.parameters():
        if parameter.grad is not None:
            value = parameter.grad.detach().float().square().sum()
            total = value if total is None else total + value
    return float(torch.sqrt(total)) if total is not None else 0.0


def calibrate_full_idd_weight(
    model, selected, cache, ct_cache, body_cache, cfg, args, device
) -> dict:
    """Measure raw term and gradient scales before choosing full-IDD lambda."""
    rows = []
    model.train()
    for ds_index, item in selected:
        cp, d_max = cache.get(item)
        ct = ct_cache[item[0]]
        dose_dev = cp.dose.to(device)
        model.zero_grad(set_to_none=True)
        state = model.encode_cp(ct, cp.beam, cp.aperture)
        base_rng = np.random.default_rng(30000 + ds_index)
        batch = make_query_batch(state, cp, dose_dev, cfg, base_rng, args.queries)
        pred = model.query(state, ct, cp.beam, batch["world_pts"])
        sigma = approx_sigma(batch["target"], d_max)
        base, _ = _combined_loss(pred, batch, sigma, d_max, cfg, args)
        base.backward()
        base_grad = _parameter_gradient_l2(model)
        base_value = float(base.detach())
        del state, batch, pred, sigma, base
        model.zero_grad(set_to_none=True)
        state = model.encode_cp(ct, cp.beam, cp.aperture)
        full_rng = np.random.default_rng(40000 + ds_index)
        cutoff = float(args.full_idd_cutoff_raw)
        full = query_full_slice_idd(
            model,
            state,
            ct,
            cp,
            dose_dev,
            body_cache[item[0]],
            args,
            full_rng,
            cutoffs=(cutoff,),
        )[cutoff]
        full.backward()
        full_grad = _parameter_gradient_l2(model)
        full_value = float(full.detach())
        ratio = base_grad / full_grad if full_grad > 0.0 else float("inf")
        row = {
            "dataset_index": int(ds_index),
            "item": [int(v) for v in item],
            "case_id": dataset_case_id(cache.dataset, item[0]),
            "base_loss": base_value,
            "full_idd_rms": full_value,
            "base_grad_l2": base_grad,
            "full_idd_grad_l2": full_grad,
            "lambda_equal_gradient": ratio,
        }
        rows.append(row)
        print(
            "  calibration {case_id}/B{beam}/CP{cp:03d}: base={base:.6g} full_idd={full:.6g} |g_base|={gb:.6g} |g_full|={gf:.6g} lambda_equal={ratio:.4g}".format(
                case_id=row["case_id"],
                beam=item[1],
                cp=item[2],
                base=base_value,
                full=full_value,
                gb=base_grad,
                gf=full_grad,
                ratio=ratio,
            ),
            flush=True,
        )
        del state, full, dose_dev
    model.zero_grad(set_to_none=True)
    finite_ratios = [
        row["lambda_equal_gradient"]
        for row in rows
        if math.isfinite(row["lambda_equal_gradient"])
    ]
    equal = float(np.median(finite_ratios)) if finite_ratios else float("inf")
    aggregate = {
        "n_cps": len(rows),
        "median_base_loss": float(np.median([row["base_loss"] for row in rows])),
        "median_full_idd_rms": float(np.median([row["full_idd_rms"] for row in rows])),
        "median_base_grad_l2": float(np.median([row["base_grad_l2"] for row in rows])),
        "median_full_idd_grad_l2": float(
            np.median([row["full_idd_grad_l2"] for row in rows])
        ),
        "median_lambda_equal_gradient": equal,
        "suggested_lambda_for_gradient_fraction": {
            str(fraction): equal * fraction for fraction in (0.1, 0.25, 0.5)
        },
    }
    return {"aggregate": aggregate, "per_cp": rows}


def dataset_case_id(dataset: PlanCPDataset, case_index: int) -> str:
    """Return a stable case identifier for diagnostics."""
    return str(dataset.cases[int(case_index)].case_id)


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


def resolve_pdd_path(pdd: Path) -> Path:
    """Try the given path, then relative to the repo root (parent of scripts/)."""
    if pdd.is_file() or pdd.is_absolute():
        return pdd
    candidate = Path(__file__).resolve().parent.parent / pdd
    return candidate if candidate.is_file() else pdd


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sel_rng, query_rng = make_step_rngs(args.seed)
    cases = discover_cases(args.data)
    print(f"cases ({len(cases)}): {', '.join((c.name for c in cases))}")
    dataset = PlanCPDataset(cases)
    try:
        if args.val_case is not None:
            case_ids = [c.case_id for c in dataset.cases]
            train_items, val_items = split_items_by_case(
                dataset.items, case_ids, args.val_case
            )
            holdout = f"case == {args.val_case}"
        else:
            train_items, val_items = split_items(dataset.items, args.val_beam)
            holdout = f"beam_idx == {args.val_beam}"
    except ValueError as err:
        raise SystemExit(str(err)) from err
    print(f"control points: {len(train_items)} train, {len(val_items)} val ({holdout})")
    cfg = synthrad_config()
    width_scale = float(getattr(args, "width_scale", 1.0) or 1.0)
    if width_scale != 1.0:
        from prismx.config import scaled_prismx_config

        enc = cfg.encoder
        try:
            cfg = scaled_prismx_config(cfg, width_scale)
        except ValueError as err:
            raise SystemExit(str(err)) from err
        print(
            f"width scale {width_scale:g}: encoder {enc.width}->{cfg.encoder.width}, latent {enc.latent_channels}->{cfg.encoder.latent_channels}, decoder {cfg.decoder.width}"
        )
    sampling_updates = {}
    if args.in_field_isodose is not None:
        if not 0.0 < args.in_field_isodose < 1.0:
            raise SystemExit("--in-field-isodose must be between 0 and 1")
        sampling_updates["in_field_isodose"] = float(args.in_field_isodose)
    if args.sampling_fractions is not None:
        try:
            fractions = [float(x.strip()) for x in args.sampling_fractions.split(",")]
        except ValueError as err:
            raise SystemExit("--sampling-fractions must contain five numbers") from err
        if len(fractions) != 5 or any((x < 0.0 for x in fractions)):
            raise SystemExit(
                "--sampling-fractions must contain five non-negative numbers"
            )
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-08):
            raise SystemExit("--sampling-fractions must sum to 1")
        for key, value in zip(
            (
                "frac_in_field",
                "frac_penumbra",
                "frac_buildup",
                "frac_interface",
                "frac_out_of_field",
            ),
            fractions,
        ):
            sampling_updates[key] = value
    if sampling_updates:
        cfg = replace(cfg, sampling=replace(cfg.sampling, **sampling_updates))
    if args.challenge_weight < 0.0 or args.lambda_idd < 0.0:
        raise SystemExit("--challenge-weight and --lambda-idd must be non-negative")
    if args.lambda_full_idd < 0.0:
        raise SystemExit("--lambda-full-idd must be non-negative")
    if not math.isfinite(args.full_idd_cutoff_raw) or args.full_idd_cutoff_raw < 0.0:
        raise SystemExit("--full-idd-cutoff-raw must be finite and non-negative")
    try:
        parse_cutoff_grid(args.full_idd_val_cutoffs)
    except ValueError as err:
        raise SystemExit(f"invalid --full-idd-val-cutoffs: {err}") from err
    if args.full_idd_output_scale is None:
        args.full_idd_output_scale = float(args.dose_scale)
    if (
        not math.isfinite(args.full_idd_output_scale)
        or args.full_idd_output_scale <= 0.0
    ):
        raise SystemExit("--full-idd-output-scale must be finite and positive")
    if not math.isfinite(args.full_idd_tau_frac) or args.full_idd_tau_frac <= 0.0:
        raise SystemExit("--full-idd-tau-frac must be finite and positive")
    if args.full_idd_slices <= 0 or args.full_idd_samples_per_slice <= 0:
        raise SystemExit("full-IDD slice and sample counts must be positive")
    if args.beam_mae_guard_tolerance < 0.0 or args.beam_mae_guard_weight < 0.0:
        raise SystemExit("Beam-MAE guard tolerance and weight must be non-negative")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be non-negative")
    if args.calibration_cps <= 0:
        raise SystemExit("--calibration-cps must be positive")
    print(
        "metric alignment: challenge_weight={} lambda_idd={} lambda_full_idd={} selection={} in_field>={} fractions={}".format(
            args.challenge_weight,
            args.lambda_idd,
            args.lambda_full_idd,
            args.selection_metric,
            cfg.sampling.in_field_isodose,
            (
                cfg.sampling.frac_in_field,
                cfg.sampling.frac_penumbra,
                cfg.sampling.frac_buildup,
                cfg.sampling.frac_interface,
                cfg.sampling.frac_out_of_field,
            ),
        )
    )
    if (
        args.lambda_full_idd > 0.0
        or args.selection_metric == "full_idd"
        or args.calibrate_full_idd_only
    ):
        print(
            "full-slice IDD: cutoff_raw={:.3g} val_cutoffs={} output_scale={:.8g} tau_frac={} slices={} samples/slice={}".format(
                args.full_idd_cutoff_raw,
                args.full_idd_val_cutoffs,
                args.full_idd_output_scale,
                args.full_idd_tau_frac,
                args.full_idd_slices,
                args.full_idd_samples_per_slice,
            )
        )
    model = PrismX(cfg).to(device)
    print(f"PrismX parameters: {model.n_parameters():,}  device: {device}")
    pdd_path = resolve_pdd_path(args.pdd)
    if pdd_path.is_file():
        model.pdd = PDDTable.from_npz(pdd_path)
        print(f"PDD: commissioned table from {pdd_path}")
    else:
        print(f"PDD: placeholder default ({args.pdd} not found)")
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
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        if ckpt.get("best_val") is not None:
            best_val = float(ckpt["best_val"])
        print(f"resumed from {args.resume} at step {start_step}")
    elif args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        weights = ckpt.get("ema", ckpt.get("model", ckpt))
        model.load_state_dict(weights)
        ema = EMA(model, EMA_DECAY)
        print(
            f"initialized from EMA weights in {args.init_from}; optimizer and step reset for fine-tuning"
        )
    print(
        f"dose scale: dividing MC dose/sigma by {args.dose_scale:.4e} (raw units per unit baseline)"
    )
    index_of = {item: i for i, item in enumerate(dataset.items)}
    ct_cache = {ci: case.ct().to(device) for ci, case in enumerate(dataset.cases)}
    full_idd_enabled = (
        args.lambda_full_idd > 0.0
        or args.selection_metric == "full_idd"
        or args.calibrate_full_idd_only
    )
    body_cache = (
        {ci: _filled_body_voxel_mask(ct.density) for ci, ct in ct_cache.items()}
        if full_idd_enabled
        else None
    )
    cache = HostCPCache(dataset, args.cache_gb, args.dose_scale)
    if args.calibrate_full_idd_only:
        calibration_sel = balanced_validation_selection(
            val_items, index_of, min(args.calibration_cps, len(val_items))
        )
        print(
            f"step-0 full-IDD calibration: {len(calibration_sel)} balanced CPs",
            flush=True,
        )
        calibration = calibrate_full_idd_weight(
            model, calibration_sel, cache, ct_cache, body_cache, cfg, args, device
        )
        args.out.mkdir(parents=True, exist_ok=True)
        calibration_path = args.out / "calibration.json"
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
        print(json.dumps(calibration["aggregate"], indent=2), flush=True)
        print(f"wrote {calibration_path}", flush=True)
        return
    n_val = min(args.val_cps, len(val_items))
    val_sel = balanced_validation_selection(val_items, index_of, n_val)
    print(
        f"validation: {len(val_sel)} fixed CPs {[f'{it[0]}/B{it[1]}/CP{it[2]:03d}' for _, it in val_sel]}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"
    if float(getattr(args, "lambda_plan", 0.0)) > 0.0 and int(args.num_workers) > 0:
        raise SystemExit(
            "--lambda-plan needs --num-workers 0: the plan term requires the synchronous path. The prefetched path hands run_train_step a _CPMeta, which carries no aperture, and a sibling cannot be encoded without one. Failing loudly beats training without the term you asked for."
        )
    n_steps = args.steps - start_step
    step_items = make_step_items(train_items, sel_rng, n_steps)
    by_case: dict = {}
    for it in train_items:
        by_case.setdefault(int(it[0]), []).append(it)
    plan_rng = np.random.default_rng(args.seed + 7000000)
    state_t = {"train_ema": None, "best_val": best_val, "t0": time.time()}

    def after_step(step, lr, loss_f, comps) -> None:
        """Shared per-step bookkeeping: EMA, logging, validation, checkpoints."""
        train_ema = state_t["train_ema"]
        train_ema = (
            loss_f
            if train_ema is None
            else TRAIN_LOSS_EMA_DECAY * train_ema
            + (1.0 - TRAIN_LOSS_EMA_DECAY) * loss_f
        )
        state_t["train_ema"] = train_ema
        if step % args.log_every == 0 or step == start_step + 1:
            dt = (time.time() - state_t["t0"]) / (step - start_step)
            plan_s = (
                f"plan {float(comps['plan']):.5f}  "
                if float(getattr(args, "lambda_plan", 0.0)) > 0.0
                else ""
            )
            print(
                f"step {step:6d}  lr {lr:.2e}  loss {loss_f:.5f}  {plan_s}ema {train_ema:.5f}  (mse {float(comps['mse']):.5f}  log {float(comps['log']):.5f}  challenge {float(comps['challenge']):.5f}  idd {float(comps['idd']):.6f}  full_idd {float(comps['full_idd']):.5f})  {dt:.2f} s/step",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(
                model,
                ema,
                val_sel,
                cache,
                ct_cache,
                cfg,
                args,
                device,
                body_cache=body_cache,
            )
            rel = metrics["rel_err"]
            print(f"  validation @ step {step} ({len(val_sel)} CPs)")
            print(f"    {'val_loss':<22s}{metrics['loss']:.5f}")
            print(f"    {'val_base_loss':<22s}{metrics['base_loss']:.5f}")
            print(f"    {'val_challenge_mae':<22s}{metrics['challenge_mae']:.5f}")
            print(
                f"    {'val_idd':<22s}{metrics['idd']:.6f}  (rms {math.sqrt(max(metrics['idd'], 0.0)):.5f})"
            )
            if "full_idd" in metrics:
                print(f"    {'val_full_idd':<22s}{metrics['full_idd']:.6f}")
                for cutoff, value in metrics["full_idd_by_cutoff"].items():
                    print(f"    full_idd@{cutoff:g}".ljust(26) + f"{value:.5f}")
            for key in ("overall", *(STRATUM_NAMES[c] for c in REPORT_STRATA)):
                print(f"    {'rel_err ' + key:<22s}{rel[key]:.4f}")
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "lr": lr,
                            "train_loss_ema": train_ema,
                            "val_loss": metrics["loss"],
                            "val_base_loss": metrics["base_loss"],
                            "val_challenge_mae": metrics["challenge_mae"],
                            "val_idd": metrics["idd"],
                            "val_rel_err": rel,
                            "val_full_idd": metrics.get("full_idd"),
                            "val_full_idd_by_cutoff": {
                                str(cutoff): value
                                for cutoff, value in metrics.get(
                                    "full_idd_by_cutoff", {}
                                ).items()
                            },
                        }
                    )
                    + "\n"
                )
            selection = checkpoint_selection_score(metrics, args)
            if selection < state_t["best_val"]:
                state_t["best_val"] = selection
                save_checkpoint(
                    args.out / "best.pt",
                    model,
                    ema,
                    opt,
                    step,
                    args,
                    state_t["best_val"],
                )
                print(
                    f"    new best {args.selection_metric} {state_t['best_val']:.5f} -> best.pt"
                )
            if args.checkpoint_every > 0 and (
                step % args.checkpoint_every == 0 or step == args.steps
            ):
                bank = args.out / "checkpoints"
                bank.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    bank / f"step_{step:06d}.pt",
                    model,
                    ema,
                    opt,
                    step,
                    args,
                    state_t["best_val"],
                )
            save_checkpoint(
                args.out / "last.pt", model, ema, opt, step, args, state_t["best_val"]
            )

    if args.num_workers > 0:
        print(
            f"prefetch: {args.num_workers} worker(s), prefetch_factor={args.prefetch_factor} (NAS read + raster/cone/channels overlap GPU compute)"
        )
        step_ds = PrismXStepDataset(dataset, step_items, cfg, args.dose_scale)
        import multiprocessing as _mp

        mp_ctx = (
            _mp.get_context("fork") if "fork" in _mp.get_all_start_methods() else None
        )
        loader = torch.utils.data.DataLoader(
            step_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=_identity_collate,
            persistent_workers=False,
            pin_memory=False,
            multiprocessing_context=mp_ctx,
        )
        for offset, prep in enumerate(loader):
            step = start_step + 1 + offset
            lr = lr_at(step, args)
            for group in opt.param_groups:
                group["lr"] = lr
            recon = reconstruct_on_device(prep, cfg, device)
            ct = ct_cache[prep["step_item"][0]]
            state = encode_from_channels(model, recon)
            cp_meta = _CPMeta(recon.grid_origin_mm, recon.grid_spacing_mm, recon.beam)
            loss_f, comps = run_train_step(
                model,
                opt,
                ema,
                state,
                ct,
                cp_meta,
                recon.dose,
                recon.d_max,
                cfg,
                args,
                query_rng,
                body_mask=body_cache[prep["step_item"][0]] if body_cache else None,
            )
            after_step(step, lr, loss_f, comps)
    else:
        for offset, item in enumerate(step_items):
            step = start_step + 1 + offset
            lr = lr_at(step, args)
            for group in opt.param_groups:
                group["lr"] = lr
            cp, d_max = cache.get(item)
            ct = ct_cache[item[0]]
            dose_dev = cp.dose.to(device)
            state = model.encode_cp(ct, cp.beam, cp.aperture)
            siblings = []
            if float(args.lambda_plan) > 0.0:
                pool = by_case.get(int(item[0]), [])
                pool = [p for p in pool if p != item]
                if pool:
                    k = min(int(args.plan_cps), len(pool))
                    pick = plan_rng.choice(len(pool), size=k, replace=False)
                    for j in pick:
                        sib, _ = cache.get(pool[int(j)])
                        siblings.append((sib, sib.dose.to(device)))
            loss_f, comps = run_train_step(
                model,
                opt,
                ema,
                state,
                ct,
                cp,
                dose_dev,
                d_max,
                cfg,
                args,
                query_rng,
                body_mask=body_cache[item[0]] if body_cache else None,
                siblings=siblings,
            )
            after_step(step, lr, loss_f, comps)
    n_done = args.steps - start_step
    print(
        f"done: {n_done} steps in {time.time() - state_t['t0']:.1f} s, best {args.selection_metric} selector {state_t['best_val']:.5f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
