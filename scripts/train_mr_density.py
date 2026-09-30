"""Train the MR density network: density supervision, optional frozen-Photon dose stage."""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from prismx.analytic import PDDTable
from prismx.aperture import MLCAperture
from prismx.config import synthrad_config, test_config
from prismx.data import discover_cases
from prismx.geometry import BeamGeometry
from prismx.losses import prismx_loss
from prismx.model import PrismX
from prismx.mr_data import MRPatientDataset, mr_body_mask
from prismx.mr_density import (
    MRDensityNet,
    ResEncDensityNet,
    air_loss,
    anchor_loss,
    resolution_augment,
    cap_loss,
    density_ctvolume,
)
from prismx.phantoms import water_box
from prismx.plan_data import (
    SIGMA_FLOOR_REL_DEFAULT,
    SIGMA_REL_DEFAULT,
    PlanCP,
    PlanCPDataset,
)
from prismx.sampling import stratified_query_points

LR_FINAL: float = 1e-05
EMA_DECAY: float = 0.999
DOSE_UNITS_PER_BASELINE: float = 5.437449455848826e-05
TRAIN_LOSS_EMA_DECAY: float = 0.98
REL_ERR_FLOOR: float = 0.05
VAL_SEED_BASE: int = 10000
NET_BASE_CHANNELS: int = 24
NET_LEVELS: int = 3
SMOKE_BASE_CHANNELS: int = 8
SMOKE_LEVELS: int = 2
SMOKE_SIZE_MM: float = 80.0
SMOKE_SPACING_MM: float = 8.0
SMOKE_FIELD_MM: float = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="case dir or root of case dirs (repeatable; autodetected). Not required with --smoke.",
    )
    parser.add_argument(
        "--engine-checkpoint",
        type=Path,
        default=None,
        help="stage-0 photon checkpoint (best.pt); its EMA weights are loaded into the FROZEN engine. Required unless --smoke.",
    )
    parser.add_argument(
        "--val-case",
        type=str,
        default=None,
        help="case id held out for validation (anchor MAE + a few CP dose masked-MAE); defaults to the last discovered case.",
    )
    parser.add_argument("--anchor-steps", type=int, default=2000)
    parser.add_argument("--dose-steps", type=int, default=8000)
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--res-aug", type=float, default=0.0, help="res aug")
    parser.add_argument(
        "--res-aug-max-voxels",
        type=int,
        default=20000000,
        help="random-crop budget after upscaling, to bound step memory",
    )
    parser.add_argument("--lambda-grad", type=float, default=0.0, help="lambda grad")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument(
        "--lambda-anchor",
        type=float,
        default=0.1,
        help="weight of the anchor regularizer added to the phase-2 dose loss",
    )
    parser.add_argument("--bone-weight", type=float, default=0.0, help="bone weight")
    parser.add_argument("--w-air", type=float, default=0.0, help="w air")
    parser.add_argument("--w-cap", type=float, default=0.0, help="w cap")
    parser.add_argument(
        "--cap-tau",
        type=float,
        default=1.1,
        help="soft-tissue density ceiling for --w-cap (matches the deploy soft-cap).",
    )
    parser.add_argument(
        "--arch",
        default="resenc",
        choices=["unet", "resenc"],
        help="density net backbone: 'unet' (MRDensityNet, 3.15M) or 'resenc' (ResEncDensityNet, stronger residual-encoder, nnU-Net-inspired)",
    )
    parser.add_argument(
        "--net-base-channels",
        type=int,
        default=0,
        help="density net base channels (0 = arch default)",
    )
    parser.add_argument(
        "--net-levels",
        type=int,
        default=0,
        help="density net down/up levels (0 = arch default)",
    )
    parser.add_argument(
        "--net-grad-checkpoint",
        action="store_true",
        help="gradient-checkpoint the resenc residual blocks (bounds dose-phase memory of a wide net; unnecessary for anchor-only training)",
    )
    parser.add_argument("--init-from", type=Path, default=None, help="init from")
    parser.add_argument(
        "--dose-scale",
        type=float,
        default=DOSE_UNITS_PER_BASELINE,
        help="raw dose-file units per unit analytic baseline; the prediction is multiplied by this to reach MC units (1.0 disables)",
    )
    parser.add_argument("--rel-floor", type=float, default=None, help="rel floor")
    parser.add_argument(
        "--pdd",
        type=Path,
        default=Path("commissioning/synthrad_photon_pdd.npz"),
        help="commissioned PDD npz swapped into the frozen engine if it exists",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-cps", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--keep-every", type=int, default=1000, help="keep every")
    parser.add_argument("--smoke", action="store_true", help="smoke")
    return parser.parse_args()


def lr_at(step: int, peak_lr: float, warmup: int, total: int) -> float:
    """Linear warmup to ``peak_lr`` then cosine to :data:`LR_FINAL` at ``total``."""
    if step <= warmup:
        return peak_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return LR_FINAL + 0.5 * (peak_lr - LR_FINAL) * (1.0 + math.cos(math.pi * t))


def approx_sigma(target: torch.Tensor, d_max: float) -> torch.Tensor:
    """Poisson-consistent MC-noise placeholder at the query targets."""
    return (
        SIGMA_REL_DEFAULT * torch.sqrt(target.clamp(min=0.0) * d_max)
        + SIGMA_FLOOR_REL_DEFAULT * d_max
    )


class EMA:
    """Exponential moving average of a module's state (floating entries)."""

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


def resolve_repo_path(path: Path) -> Path:
    """Try ``path`` as given, then relative to the repo root (parent of scripts/)."""
    if path.is_file() or path.is_absolute():
        return path
    candidate = Path(__file__).resolve().parent.parent / path
    return candidate if candidate.is_file() else path


def load_frozen_engine(checkpoint: Path, pdd: Path, device: torch.device) -> PrismX:
    """Build a ``synthrad_config`` PrismX, load the EMA weights, freeze, eval."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = synthrad_config()
    width_scale = float((ckpt.get("args") or {}).get("width_scale", 1.0) or 1.0)
    if width_scale != 1.0:
        from prismx.config import scaled_prismx_config

        cfg = scaled_prismx_config(cfg, width_scale)
        print(
            f"Width scale from checkpoint: x{width_scale:g} (encoder {cfg.encoder.width}, latent {cfg.encoder.latent_channels}, decoder {cfg.decoder.width})"
        )
    engine = PrismX(cfg).to(device)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt
    engine.load_state_dict(state)
    pdd_path = resolve_repo_path(pdd)
    if pdd_path.is_file():
        engine.pdd = PDDTable.from_npz(pdd_path)
        print(f"PDD: commissioned table from {pdd_path}")
    else:
        print(f"PDD: placeholder default ({pdd} not found)")
    engine.eval()
    engine.requires_grad_(False)
    return engine


def build_net(
    smoke: bool,
    device: torch.device,
    arch: str = "resenc",
    base_channels: int = 0,
    levels: int = 0,
    grad_checkpoint: bool = False,
):
    """The trainable density net."""
    if smoke:
        b, lv = (SMOKE_BASE_CHANNELS, SMOKE_LEVELS)
    else:
        b, lv = (base_channels or NET_BASE_CHANNELS, levels or NET_LEVELS)
    if arch == "resenc":
        return ResEncDensityNet(
            base_channels=b, levels=lv, grad_checkpoint=grad_checkpoint
        ).to(device)
    return MRDensityNet(b, lv).to(device)


def forward_net(net, mr_norm: torch.Tensor) -> tuple:
    """Run the net on a ``[Z, Y, X]`` MR -> ``(density [Z,Y,X], seg_logits|None)``."""
    out = net(mr_norm.unsqueeze(0).unsqueeze(0))
    return (out[0, 0], None)


@torch.no_grad()
def density_from_mr(net, mr_norm: torch.Tensor) -> torch.Tensor:
    """Run the net on a ``[Z, Y, X]`` MR and return a ``[Z, Y, X]`` density."""
    rho, _ = forward_net(net, mr_norm)
    return rho


def rebuild_net_from_ckpt(ckpt: dict, device: torch.device, weights: str = "ema"):
    """Rebuild + load the density net stored in a training checkpoint."""
    sd = ckpt.get(weights) or ckpt.get("net") or ckpt.get("model") or ckpt
    meta = ckpt.get("arch") or {}
    name = meta.get("name")
    has_seg = any(
        (k.startswith("seg_head.") or k == "seg_head.weight" for k in sd)
    ) or any((k.startswith("seg_ups.") for k in sd))
    if name == "anatomy_aware" or has_seg:
        raise SystemExit(
            "anatomy-aware checkpoints are no longer loadable (the experiment family was removed 2026-07-03 after the population verdict); check out an older commit to evaluate this checkpoint"
        )
    if name == "resenc" or any((k.startswith("stem.") for k in sd)):
        base = int(sd["stem.weight"].shape[0])
        levels = sum(
            (1 for k in sd if k.startswith("downs.") and k.endswith(".weight"))
        )
        net = ResEncDensityNet(base_channels=base, levels=levels).to(device)
        print(
            f"density net: resenc base={base} levels={levels} ({net.n_parameters():,} p)"
        )
    else:
        net = build_net(smoke=False, device=device)
        print(f"density net: unet ({net.n_parameters():,} p)")
    net.load_state_dict(sd)
    net.eval()
    return net


def anchor_step(
    net,
    mr_norm: torch.Tensor,
    ct_density: torch.Tensor,
    body_mask: torch.Tensor,
    bone_weight: float = 0.0,
    grad_weight: float = 0.0,
    w_air: float = 0.0,
    w_cap: float = 0.0,
    cap_tau: float = 1.1,
) -> torch.Tensor:
    """One phase-1 forward: body-masked anchor loss of ``D_theta(mr)`` vs CT,"""
    rho, _ = forward_net(net, mr_norm)
    loss = anchor_loss(
        rho, ct_density, body_mask, bone_weight=bone_weight, grad_weight=grad_weight
    )
    if w_air > 0.0:
        loss = loss + w_air * air_loss(rho, body_mask)
    if w_cap > 0.0:
        loss = loss + w_cap * cap_loss(rho, ct_density, body_mask, tau=cap_tau)
    return loss


def dose_step(
    net: MRDensityNet,
    engine: PrismX,
    mr_norm: torch.Tensor,
    ct_density: torch.Tensor,
    body_mask: torch.Tensor,
    cp: PlanCP,
    dose_scale: float,
    lambda_anchor: float,
    n_queries: int,
    rng: np.random.Generator,
    device: torch.device,
    bone_weight: float = 0.0,
    grad_weight: float = 0.0,
    w_air: float = 0.0,
    w_cap: float = 0.0,
    cap_tau: float = 1.1,
) -> tuple:
    """One phase-2 forward through the FROZEN engine on the MR-derived density."""
    rho, _ = forward_net(net, mr_norm)
    ct_pred = density_ctvolume(rho, cp.grid_spacing_mm, cp.grid_origin_mm)
    state = engine.encode_cp(ct_pred, cp.beam, cp.aperture)
    dose_dev = cp.dose.to(device)
    batch = stratified_query_points(
        dose_dev,
        cp.grid_origin_mm,
        cp.grid_spacing_mm,
        cp.beam,
        state.raster,
        state.cone,
        engine.cfg.sampling,
        engine.cfg.loss,
        rng,
        n=n_queries,
    )
    pred = engine.query(state, ct_pred, cp.beam, batch["world_pts"])
    d_ref = float(batch["target"].max())
    sigma = approx_sigma(batch["target"], d_ref)
    dose_loss, _ = prismx_loss(
        pred, batch["target"], sigma, d_ref, batch["m_weight"], engine.cfg.loss
    )
    a_loss = anchor_loss(
        rho, ct_density, body_mask, bone_weight=bone_weight, grad_weight=grad_weight
    )
    total = dose_loss + lambda_anchor * a_loss
    if w_air > 0.0:
        total = total + w_air * air_loss(rho, body_mask)
    if w_cap > 0.0:
        total = total + w_cap * cap_loss(rho, ct_density, body_mask, tau=cap_tau)
    return (total, float(dose_loss.detach()), float(a_loss.detach()))


def warm_start(net, init_from: Path, device: torch.device) -> None:
    """Warm-start ``net`` from ``init_from`` (EMA/model weights), PARTIAL-loading."""
    ck = torch.load(init_from, map_location=device, weights_only=False)
    sd = ck.get("ema") or ck.get("net") or ck.get("model") or ck
    tgt = net.state_dict()
    to_load, skipped_shape = ({}, [])
    for k, v in sd.items():
        if k in tgt and tgt[k].shape == v.shape:
            to_load[k] = v
        elif k in tgt:
            skipped_shape.append(k)
    missing = [k for k in tgt if k not in to_load]
    net.load_state_dict(to_load, strict=False)

    def _groups(keys):
        return sorted({k.split(".", 1)[0] for k in keys})

    print(
        f"warm-start: net loaded from {init_from} (step {ck.get('step', '?')}); {len(to_load)}/{len(tgt)} tensors copied"
    )
    print(f"  loaded groups: {_groups(to_load.keys())}")
    if missing:
        print(
            f"  FRESH (init, not in checkpoint): {_groups(missing)} ({len(missing)} tensors)"
        )
    if skipped_shape:
        print(
            f"  SKIPPED (shape mismatch): {_groups(skipped_shape)} ({len(skipped_shape)} tensors)"
        )
    source_only = [k for k in sd if k not in tgt]
    if source_only:
        print(
            f"  IGNORED (in checkpoint, not in net): {_groups(source_only)} ({len(source_only)} tensors)"
        )


def net_arch_meta(net) -> dict:
    """A small, self-describing arch tag stored in the checkpoint."""
    if isinstance(net, ResEncDensityNet):
        return {"name": "resenc", "levels": net.levels}
    return {"name": "unet"}


def save_checkpoint(path: Path, net, ema, opt, step, args, best_val) -> None:
    """Persist the density-net state, its EMA, the optimizer and run metadata."""

    def jsonable(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [jsonable(x) for x in v]
        return v

    torch.save(
        {
            "net": net.state_dict(),
            "ema": ema.shadow,
            "optimizer": opt.state_dict(),
            "step": step,
            "args": {k: jsonable(v) for k, v in vars(args).items()},
            "best_val": best_val,
            "arch": net_arch_meta(net),
        },
        path,
    )


def save_weights_only(path: Path, net, ema, step) -> None:
    """Persist ONLY the density net + its EMA + the arch tag (no optimizer)."""
    torch.save(
        {
            "net": net.state_dict(),
            "ema": ema.shadow,
            "step": step,
            "arch": net_arch_meta(net),
        },
        path,
    )


def validate_anchor(net, ema, mr_norm, ct_density, body_mask) -> tuple:
    """Body-masked density MAE of the EMA net on the held-out case (g/cm^3)."""
    with torch.no_grad(), ema.swap(net):
        rho = density_from_mr(net, mr_norm)
        mask = body_mask.to(rho.dtype)
        denom = float(mask.sum())
        if denom == 0:
            return (float("nan"), float("nan"), float("nan"))
        mae = float(((rho - ct_density).abs() * mask).sum() / denom)
        bone = body_mask.to(torch.bool) & (ct_density > 1.1)
        bone_src = "ct>1.1"
        if int(bone.sum()) > 0:
            diff = (rho - ct_density)[bone]
            bone_mae = float(diff.abs().mean())
            bone_bias = float(diff.mean())
        else:
            bone_mae = bone_bias = float("nan")
        print(
            f"    bone-region ({bone_src}) MAE {bone_mae:.5f}  signed bias {bone_bias:+.5f} g/cm^3",
            flush=True,
        )
        return (mae, bone_mae, bone_bias)


def validate_dose(
    net, ema, engine, mr_norm, val_cps, dose_scale, n_queries, device
) -> tuple:
    """HIGH-DOSE dose agreement (MR-density vs MC) on ONE held-out case's CPs."""
    if not val_cps:
        return (float("nan"), float("nan"), float("nan"))
    rels, integ = ([], [])
    p_sum, t_sum = (0.0, 0.0)
    with torch.no_grad(), ema.swap(net):
        rho = density_from_mr(net, mr_norm)
        for k, cp in enumerate(val_cps):
            ct_pred = density_ctvolume(rho, cp.grid_spacing_mm, cp.grid_origin_mm)
            state = engine.encode_cp(ct_pred, cp.beam, cp.aperture)
            dose_dev = cp.dose.to(device)
            rng = np.random.default_rng(VAL_SEED_BASE + k)
            batch = stratified_query_points(
                dose_dev,
                cp.grid_origin_mm,
                cp.grid_spacing_mm,
                cp.beam,
                state.raster,
                state.cone,
                engine.cfg.sampling,
                engine.cfg.loss,
                rng,
                n=n_queries,
            )
            pred = engine.query(state, ct_pred, cp.beam, batch["world_pts"])
            d_max = float(batch["target"].max())
            hi = batch["target"] > 0.1 * d_max
            if int(hi.sum()) > 0 and d_max > 0:
                rels.append(
                    float((pred[hi] - batch["target"][hi]).abs().mean() / d_max)
                )
                p_sum += float(pred[hi].sum())
                tsum = float(batch["target"][hi].sum())
                t_sum += tsum
                if tsum > 0:
                    integ.append(abs(float(pred[hi].sum()) - tsum) / tsum)
    rel_mae = float(np.mean(rels)) if rels else float("nan")
    integral_err = float(np.mean(integ)) if integ else float("nan")
    plan_ratio = p_sum / t_sum if t_sum > 0 else float("nan")
    return (rel_mae, integral_err, plan_ratio)


BONE_MAE_SEL_WEIGHT: float = 0.25


def selection_metric(plan_ratios, bone_mae: float) -> float:
    """SIGNED PLAN-LEVEL best.pt selector (lower is better)."""
    if isinstance(plan_ratios, (int, float)):
        plan_ratios = [plan_ratios]
    devs = [abs(float(pr) - 1.0) for pr in plan_ratios if math.isfinite(pr)]
    parts = []
    if devs:
        parts.append(max(devs))
    if math.isfinite(bone_mae):
        parts.append(BONE_MAE_SEL_WEIGHT * bone_mae)
    return sum(parts) if parts else math.inf


def pick_random_cp(ds: PlanCPDataset, rng: np.random.Generator) -> PlanCP:
    """A random control point (``PlanCP``) of ``ds`` (raw dose-file units)."""
    return ds[int(rng.integers(len(ds)))]


def make_smoke_case(device: torch.device):
    """A tiny synthetic case for --smoke (no disk I/O): water-box MR/CT + one CP."""
    ct = water_box(size_mm=(SMOKE_SIZE_MM,) * 3, spacing_mm=SMOKE_SPACING_MM).to(device)
    z, y, x = ct.density.shape
    mr_norm = torch.randn(z, y, x, device=device)
    body_mask = ct.density > 0.05
    beam = BeamGeometry(gantry_deg=0.0)
    ap = MLCAperture.rectangle(SMOKE_FIELD_MM, SMOKE_FIELD_MM)
    dose = (ct.density * 0.0001).to(device)
    sigma = torch.full_like(dose, 1e-06)
    cp = PlanCP(
        dose=dose,
        sigma=sigma,
        grid_origin_mm=ct.origin_mm,
        grid_spacing_mm=ct.spacing_mm,
        beam=beam,
        aperture=ap,
        ct=ct,
        case_id="smoke",
        beam_idx=0,
        cp_idx=0,
    )
    return (mr_norm, ct.density, body_mask, [cp])


class ValCase:
    """A single held-out validation case (HOST-resident tensors + fixed CPs)."""

    __slots__ = ("case_id", "mr_norm", "ct_density", "body_mask", "val_cps")

    def __init__(self, case_id, mr_norm, ct_density, body_mask, val_cps):
        self.case_id = case_id
        self.mr_norm = mr_norm
        self.ct_density = ct_density
        self.body_mask = body_mask
        self.val_cps = val_cps


def build_val_case(case_id, case_root, dose_scale, n_val_cps, device) -> ValCase:
    """Build a HOST-resident :class:`ValCase` (MR/CT/mask/klass + fixed dose CPs)."""
    mr_case = MRPatientDataset([case_root])[0]
    mr_norm = mr_case.mr_norm
    ct_density = mr_case.ct_density
    body_mask = mr_body_mask(mr_case.mr_norm) | (mr_case.ct_density > 0.05)
    cp_ds = PlanCPDataset(mr_case.plan.root)
    n_val = min(n_val_cps, len(cp_ds))
    sel_idx = np.unique(np.linspace(0, len(cp_ds) - 1, num=n_val).round().astype(int))
    val_cps = []
    for i in sel_idx:
        cp = cp_ds[int(i)]
        cp.dose = cp.dose / dose_scale
        cp.sigma = cp.sigma / dose_scale
        cp.ct = cp.ct.to(device)
        val_cps.append(cp.to(device))
    return ValCase(case_id, mr_norm, ct_density, body_mask, val_cps)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"
    if args.smoke:
        engine = PrismX(test_config()).to(device).eval()
        engine.requires_grad_(False)
        net = build_net(
            smoke=True,
            device=device,
            arch=args.arch,
            grad_checkpoint=args.net_grad_checkpoint,
        )
        mr_norm, ct_density, body_mask, val_cps = make_smoke_case(device)
        train_cases = [(mr_norm, ct_density, body_mask, None)]
        val_primary = "smoke"
        _mr2, _ct2, _bm2, _cps2 = make_smoke_case(device)
        extra_val_cases = [ValCase("smoke2", _mr2.cpu(), _ct2.cpu(), _bm2.cpu(), _cps2)]
        print(
            f"SMOKE: test_config engine ({engine.n_parameters():,} params, frozen) + {type(net).__name__} ({net.n_parameters():,} params) on a {tuple(ct_density.shape)} water-box stand-in, device {device}"
        )
    else:
        if not args.data:
            raise SystemExit("--data is required unless --smoke")
        if args.engine_checkpoint is None:
            raise SystemExit("--engine-checkpoint is required unless --smoke")
        engine = load_frozen_engine(args.engine_checkpoint, args.pdd, device)
        if args.rel_floor is not None:
            from dataclasses import replace as _dc_replace

            engine.cfg = _dc_replace(
                engine.cfg,
                loss=_dc_replace(engine.cfg.loss, rel_floor=float(args.rel_floor)),
            )
            print(
                f"dose loss: rel_floor override -> {args.rel_floor} (engine default was 0.005)"
            )
        engine.encoder.grad_checkpoint = True
        net = build_net(
            smoke=False,
            device=device,
            arch=args.arch,
            base_channels=args.net_base_channels,
            levels=args.net_levels,
            grad_checkpoint=args.net_grad_checkpoint,
        )
        cases = discover_cases(args.data)
        case_ids = [c.name for c in cases]
        val_case = args.val_case or case_ids[-1]
        val_list = [c.strip() for c in val_case.split(",") if c.strip()]
        unknown = [c for c in val_list if c not in case_ids]
        if unknown:
            raise SystemExit(f"--val-case {unknown!r} not among {case_ids}")
        val_set = set(val_list)
        val_primary = val_list[0]
        train_roots = [c for c in cases if c.name not in val_set]
        if not train_roots:
            raise SystemExit("no training cases left after holding out --val-case")
        print(
            f"engine: {engine.n_parameters():,} params (frozen)  {type(net).__name__}: {net.n_parameters():,} params (arch={args.arch})  device {device}"
        )
        print(
            f"cases: {len(train_roots)} train ({', '.join((c.name for c in train_roots))}), holdout {val_list} (readout on {val_primary})"
        )
        train_ds = MRPatientDataset(train_roots)
        train_cp_ds = [PlanCPDataset(r) for r in train_roots]
        val_mr = MRPatientDataset([next((c for c in cases if c.name == val_primary))])[
            0
        ]
        mr_norm = val_mr.mr_norm.to(device)
        ct_density = val_mr.ct_density.to(device)
        body_mask = mr_body_mask(val_mr.mr_norm).to(device) | (
            val_mr.ct_density > 0.05
        ).to(device)
        val_cp_ds = PlanCPDataset(val_mr.plan.root)
        n_val = min(args.val_cps, len(val_cp_ds))
        sel = np.unique(
            np.linspace(0, len(val_cp_ds) - 1, num=n_val).round().astype(int)
        )
        val_cps = []
        for i in sel:
            cp = val_cp_ds[int(i)]
            cp.dose = cp.dose / args.dose_scale
            cp.sigma = cp.sigma / args.dose_scale
            cp.ct = cp.ct.to(device)
            val_cps.append(cp.to(device))
        train_cases = []
        train_case_meta = []
        for ci, case in enumerate(train_roots):
            mr_case = train_ds[ci]
            train_cases.append(
                (
                    mr_case.mr_norm,
                    mr_case.ct_density,
                    mr_body_mask(mr_case.mr_norm) | (mr_case.ct_density > 0.05),
                    train_cp_ds[ci],
                )
            )
            train_case_meta.append(
                (case.name, mr_case.grid_spacing_mm, mr_case.grid_origin_mm)
            )
        extra_val_cases = [
            build_val_case(
                cid,
                next((c for c in cases if c.name == cid)),
                args.dose_scale,
                args.val_cps,
                device,
            )
            for cid in val_list[1:]
        ]
        if extra_val_cases:
            print(
                f"extra val cases (selector max-over-cases): {[vc.case_id for vc in extra_val_cases]} (+ primary {val_primary})"
            )
    if args.init_from is not None:
        warm_start(net, args.init_from, device)
    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    ema = EMA(net, EMA_DECAY)
    if hasattr(net, "set_grad_checkpoint"):
        net.set_grad_checkpoint(args.net_grad_checkpoint)
    if hasattr(net, "set_grad_checkpoint"):
        net.set_grad_checkpoint(args.net_grad_checkpoint)
    print(f"phase 1 (anchor): {args.anchor_steps} steps")
    train_ema = None
    t0 = time.time()
    for step in range(1, args.anchor_steps + 1):
        lr = lr_at(step, args.lr, args.warmup_steps, args.anchor_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        ci = int(rng.integers(len(train_cases)))
        mr_c, ct_c, mask_c, _ = train_cases[ci]
        mr_c, ct_c, mask_c = (mr_c.to(device), ct_c.to(device), mask_c.to(device))
        if args.res_aug > 0.0 and rng.random() < args.res_aug:
            mr_c, ct_c, mask_c = resolution_augment(
                mr_c, ct_c, mask_c, rng, max_voxels=args.res_aug_max_voxels
            )
        loss = anchor_step(
            net,
            mr_c,
            ct_c,
            mask_c,
            bone_weight=args.bone_weight,
            grad_weight=args.lambda_grad,
            w_air=args.w_air,
            w_cap=args.w_cap,
            cap_tau=args.cap_tau,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ema.update(net)
        lf = float(loss)
        train_ema = (
            lf
            if train_ema is None
            else TRAIN_LOSS_EMA_DECAY * train_ema + (1.0 - TRAIN_LOSS_EMA_DECAY) * lf
        )
        if step % args.log_every == 0 or step == 1:
            print(
                f"  anchor {step:6d}  lr {lr:.2e}  anchor_loss {lf:.5f}  ema {train_ema:.5f}  {(time.time() - t0) / step:.2f} s/step",
                flush=True,
            )
    if args.anchor_steps > 0:
        mae, bone_mae, bone_bias = validate_anchor(
            net, ema, mr_norm, ct_density, body_mask
        )
        print(
            f"  phase-1 val anchor density MAE: {mae:.5f} g/cm^3  (bone MAE {bone_mae:.5f}  bias {bone_bias:+.5f})"
        )
        save_checkpoint(
            args.out / "anchor.pt", net, ema, opt, args.anchor_steps, args, mae
        )
        print(f"  saved {args.out / 'anchor.pt'}")
    if hasattr(net, "set_grad_checkpoint"):
        net.set_grad_checkpoint(args.net_grad_checkpoint)
    print(f"phase 2 (dose): {args.dose_steps} steps")
    best_val = math.inf
    train_ema = None
    t0 = time.time()
    for step in range(1, args.dose_steps + 1):
        lr = lr_at(step, args.lr, args.warmup_steps, args.dose_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        ci = int(rng.integers(len(train_cases)))
        mr_c, ct_c, mask_c, cp_ds = train_cases[ci]
        mr_c, ct_c, mask_c = (mr_c.to(device), ct_c.to(device), mask_c.to(device))
        result, tag = (None, "dose")
        if result is None:
            if cp_ds is None:
                cp = val_cps[0]
            else:
                cp = pick_random_cp(cp_ds, rng)
                cp.dose = cp.dose / args.dose_scale
                cp.sigma = cp.sigma / args.dose_scale
                cp.ct = cp.ct.to(device)
                cp = cp.to(device)
            result = dose_step(
                net,
                engine,
                mr_c,
                ct_c,
                mask_c,
                cp,
                args.dose_scale,
                args.lambda_anchor,
                args.queries,
                rng,
                device,
                bone_weight=args.bone_weight,
                grad_weight=args.lambda_grad,
                w_air=args.w_air,
                w_cap=args.w_cap,
                cap_tau=args.cap_tau,
            )
        total, dose_l, anch_l = result
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ema.update(net)
        tf = float(total)
        train_ema = (
            tf
            if train_ema is None
            else TRAIN_LOSS_EMA_DECAY * train_ema + (1.0 - TRAIN_LOSS_EMA_DECAY) * tf
        )
        if step % args.log_every == 0 or step == 1:
            print(
                f"  {tag:>5s} {step:6d}  lr {lr:.2e}  total {tf:.5f}  ({tag} {dose_l:.5f}  anchor {anch_l:.5f})  ema {train_ema:.5f}  {(time.time() - t0) / step:.2f} s/step",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.dose_steps:
            d_mae, integ_err, plan_ratio = validate_dose(
                net,
                ema,
                engine,
                mr_norm,
                val_cps,
                args.dose_scale,
                args.queries,
                device,
            )
            a_mae, bone_mae, bone_bias = validate_anchor(
                net, ema, mr_norm, ct_density, body_mask
            )
            per_case = {
                val_primary: {
                    "plan_ratio": plan_ratio,
                    "bone_mae": bone_mae,
                    "anchor_mae": a_mae,
                    "bone_bias": bone_bias,
                }
            }
            plan_ratios = [plan_ratio]
            for vc in extra_val_cases:
                vc_mr = vc.mr_norm.to(device)
                vc_ct = vc.ct_density.to(device)
                vc_bm = vc.body_mask.to(device)
                _, _, vc_ratio = validate_dose(
                    net,
                    ema,
                    engine,
                    vc_mr,
                    vc.val_cps,
                    args.dose_scale,
                    args.queries,
                    device,
                )
                vc_a_mae, vc_bone_mae, vc_bone_bias = validate_anchor(
                    net, ema, vc_mr, vc_ct, vc_bm
                )
                per_case[vc.case_id] = {
                    "plan_ratio": vc_ratio,
                    "bone_mae": vc_bone_mae,
                    "anchor_mae": vc_a_mae,
                    "bone_bias": vc_bone_bias,
                }
                plan_ratios.append(vc_ratio)
            nan_ratio_cases = [
                c for c, d in per_case.items() if not math.isfinite(d["plan_ratio"])
            ]
            if nan_ratio_cases:
                print(
                    f"    WARNING: NaN plan_ratio for {nan_ratio_cases} -> dropped from the worst-case selector",
                    flush=True,
                )
            finite_ratios = [
                d["plan_ratio"]
                for d in per_case.values()
                if math.isfinite(d["plan_ratio"])
            ]
            if len(finite_ratios) >= 2:
                r_lo, r_hi = (min(finite_ratios), max(finite_ratios))
                plan_ratio_spread = (r_hi - r_lo) / (r_hi + r_lo)
                implied_scale = 2.0 / (r_lo + r_hi)
            else:
                plan_ratio_spread = float("nan")
                implied_scale = float("nan")
            finite_bone = [
                d["bone_mae"] for d in per_case.values() if math.isfinite(d["bone_mae"])
            ]
            bone_mae_max = max(finite_bone) if finite_bone else float("nan")
            sel = selection_metric(plan_ratios, bone_mae)
            ratio_str = "  ".join(
                (f"{cid} pr {d['plan_ratio']:.4f}" for cid, d in per_case.items())
            )
            print(
                f"  val @ dose step {step}: dose rel-MAE {d_mae:.5f}  integral err {integ_err:.5f}  anchor MAE {a_mae:.5f}  bone MAE {bone_mae:.5f}  bone bias {bone_bias:+.5f}  [{ratio_str}]  spread {plan_ratio_spread:.4f}  -> sel {sel:.5f}"
            )
            rec = {
                "phase": "dose",
                "step": step,
                "lr": lr,
                "train_loss_ema": train_ema,
                "val_dose_rel_mae": d_mae,
                "val_dose_integral_err": integ_err,
                "val_anchor_mae": a_mae,
                "val_bone_mae": bone_mae,
                "val_bone_bias": bone_bias,
                "val_plan_ratio": plan_ratio,
                "val_plan_ratio_per_case": {
                    c: d["plan_ratio"] for c, d in per_case.items()
                },
                "val_bone_mae_per_case": {
                    c: d["bone_mae"] for c, d in per_case.items()
                },
                "val_anchor_mae_per_case": {
                    c: d["anchor_mae"] for c, d in per_case.items()
                },
                "val_bone_bias_per_case": {
                    c: d["bone_bias"] for c, d in per_case.items()
                },
                "val_plan_ratio_spread": plan_ratio_spread,
                "val_implied_scale": implied_scale,
                "val_bone_mae_max": bone_mae_max,
                "val_selection": sel,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if sel < best_val:
                best_val = sel
                save_checkpoint(
                    args.out / "best.pt", net, ema, opt, step, args, best_val
                )
                sel_json = {
                    "proxy_best_step": step,
                    "proxy_best_selection": sel,
                    "plan_ratio_per_case": {
                        c: d["plan_ratio"] for c, d in per_case.items()
                    },
                    "bone_mae_per_case": {
                        c: d["bone_mae"] for c, d in per_case.items()
                    },
                    "anchor_mae_per_case": {
                        c: d["anchor_mae"] for c, d in per_case.items()
                    },
                    "bone_bias_per_case": {
                        c: d["bone_bias"] for c, d in per_case.items()
                    },
                    "plan_ratio_spread": plan_ratio_spread,
                    "implied_scale": implied_scale,
                    "bone_mae_max": bone_mae_max,
                    "selector": f"max_over_valcases(|plan_ratio-1|) + {BONE_MAE_SEL_WEIGHT}*bone_mae(primary)",
                    "val_cases": [val_primary] + [vc.case_id for vc in extra_val_cases],
                }
                with open(args.out / "selection.json", "w") as f:
                    json.dump(sel_json, f, indent=2)
                print(
                    f"    new best selection {best_val:.5f} (worst |plan_ratio-1| + 0.25*bone_mae) -> best.pt + selection.json"
                )
            save_checkpoint(args.out / "last.pt", net, ema, opt, step, args, best_val)
            if args.keep_every > 0 and step % args.keep_every == 0:
                ck_path = args.out / f"ckpt_{step}.pt"
                save_weights_only(ck_path, net, ema, step)
                print(f"    step-tagged {ck_path.name} (weights-only)")
    print(
        f"done: anchor {args.anchor_steps} + dose {args.dose_steps} steps, best selection {(best_val if math.isfinite(best_val) else float('nan')):.5f} (worst |plan_ratio-1| + 0.25*bone_mae) -> {args.out}"
    )


if __name__ == "__main__":
    main()
