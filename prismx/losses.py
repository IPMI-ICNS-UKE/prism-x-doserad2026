"""PRISM-X training loss (SPEC.md section 9)."""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from prismx.config import LossConfig

GRAD_REL_FLOOR: float = 0.02
PLAN_MAE_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("high", 0.8, None),
    ("mid", 0.3, 0.8),
    ("low", 0.1, 0.3),
)


def prismx_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma_mc: torch.Tensor,
    d_ref: float,
    m_weight: torch.Tensor,
    cfg: LossConfig,
    pred_grad: torch.Tensor | None = None,
    target_grad: torch.Tensor | None = None,
    grad_mask: torch.Tensor | None = None,
    sigma_grad: torch.Tensor | None = None,
) -> tuple:
    """Compute the PRISM-X training loss."""
    floor = cfg.rel_floor * d_ref
    resid = pred - target
    l_mse = (
        m_weight * resid.square() / ((target + floor).square() + sigma_mc.square())
    ).mean()
    eps = cfg.log_eps * d_ref
    l_log = (
        (torch.log(pred.clamp_min(0.0) + eps) - torch.log(target + eps)).abs().mean()
    )
    if pred_grad is not None and target_grad is not None:
        pg, tg, sg = (pred_grad, target_grad, sigma_grad)
        if grad_mask is not None:
            pg = pg[grad_mask]
            tg = tg[grad_mask]
            if sg is not None and sg.dim() > 0 and (sg.shape[0] == grad_mask.shape[0]):
                sg = sg[grad_mask]
        if tg.numel() == 0:
            l_grad = pred.new_zeros(())
        else:
            denom = (tg.abs() + GRAD_REL_FLOOR * d_ref).square()
            if sg is not None:
                while sg.dim() < tg.dim():
                    sg = sg.unsqueeze(-1)
                denom = denom + sg.square()
            l_grad = ((pg - tg).square() / denom).mean()
    else:
        l_grad = pred.new_zeros(())
    total = l_mse + cfg.lambda_log * l_log + cfg.lambda_grad * l_grad
    return (
        total,
        {"mse": l_mse.detach(), "log": l_log.detach(), "grad": l_grad.detach()},
    )


def sample_grid_at_world(dose, origin_mm, spacing_mm, world_pts):
    """Trilinear sample of a beamlet's dose grid at arbitrary world points."""
    nz, ny, nx = dose.shape
    o = torch.as_tensor(origin_mm, dtype=world_pts.dtype, device=world_pts.device)
    s = torch.as_tensor(spacing_mm, dtype=world_pts.dtype, device=world_pts.device)
    idx = (world_pts - o) / s
    gx = 2.0 * idx[:, 0] / max(nx - 1, 1) - 1.0
    gy = 2.0 * idx[:, 1] / max(ny - 1, 1) - 1.0
    gz = 2.0 * idx[:, 2] / max(nz - 1, 1) - 1.0
    grid = torch.stack((gx, gy, gz), dim=-1).view(1, 1, 1, -1, 3)
    out = F.grid_sample(
        dose.view(1, 1, nz, ny, nx),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return out.view(-1)


def straight_through_cutoff(
    pred: torch.Tensor, cutoff_raw: float, output_scale: float, tau_frac: float = 0.25
) -> torch.Tensor:
    """Apply the exact deployment cutoff in the forward pass."""
    cutoff_raw = float(cutoff_raw)
    output_scale = float(output_scale)
    tau_frac = float(tau_frac)
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


def official_plan_mae_loss(
    pred_plan: torch.Tensor,
    target_plan: torch.Tensor,
    prescription: float | torch.Tensor | None = None,
    *,
    return_bands: bool = False,
):
    """Differentiable form of public metric 2.1."""
    if pred_plan.shape != target_plan.shape:
        raise ValueError(
            f"pred plan {tuple(pred_plan.shape)} and target plan {tuple(target_plan.shape)} shapes differ"
        )
    if prescription is None:
        rx = target_plan.detach().amax()
    else:
        rx = torch.as_tensor(
            prescription, dtype=target_plan.dtype, device=target_plan.device
        )
    if rx.numel() != 1:
        raise ValueError("prescription must be scalar")
    if not bool(torch.isfinite(rx)) or float(rx) <= 0.0:
        raise ValueError("prescription must be finite and positive")
    reference_fraction = target_plan.detach() / rx
    terms: dict[str, torch.Tensor] = {}
    for name, lower, upper in PLAN_MAE_BANDS:
        mask = reference_fraction >= lower
        if upper is not None:
            mask = mask & (reference_fraction < upper)
        if bool(mask.any()):
            terms[name] = (pred_plan[mask] - target_plan[mask]).abs().mean() / rx
    combined = (
        torch.stack(tuple(terms.values())).mean() if terms else pred_plan.sum() * 0.0
    )
    if return_bands:
        return (combined, terms)
    return combined


def metric_aligned_plan_sum_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_plan: torch.Tensor | None = None,
    prescription: float | torch.Tensor | None = None,
    total_contributors: int | None = None,
    return_bands: bool = False,
):
    """Plan-MAE loss for exact or sampled collections of contributors."""
    if preds.shape != targets.shape or preds.ndim != 2:
        raise ValueError("preds and targets must have the same [K, M] shape")
    if preds.shape[0] == 0:
        raise ValueError("at least one plan contributor is required")
    residual_sum = (preds - targets).sum(dim=0)
    if target_plan is None:
        reference = targets.sum(dim=0)
        estimate = reference + residual_sum
    else:
        if target_plan.shape != preds.shape[1:]:
            raise ValueError(
                f"target_plan shape {tuple(target_plan.shape)} does not match query shape {tuple(preds.shape[1:])}"
            )
        reference = target_plan
        multiplier = 1.0
        if total_contributors is not None:
            total_contributors = int(total_contributors)
            if total_contributors < preds.shape[0]:
                raise ValueError("total_contributors must be >= sampled K")
            multiplier = total_contributors / preds.shape[0]
        estimate = reference + multiplier * residual_sum
    return official_plan_mae_loss(
        estimate, reference, prescription, return_bands=return_bands
    )


def _control_variate_plan_estimate(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_plan: torch.Tensor | None,
    total_contributors: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct (estimated plan, reference plan) from sampled contributors."""
    if preds.shape != targets.shape or preds.ndim != 2:
        raise ValueError("preds and targets must have the same [K, M] shape")
    if preds.shape[0] == 0:
        raise ValueError("at least one plan contributor is required")
    residual_sum = (preds - targets).sum(dim=0)
    if target_plan is None:
        reference = targets.sum(dim=0)
        return (reference + residual_sum, reference)
    if target_plan.shape != preds.shape[1:]:
        raise ValueError(
            f"target_plan shape {tuple(target_plan.shape)} does not match query shape {tuple(preds.shape[1:])}"
        )
    multiplier = 1.0
    if total_contributors is not None:
        total_contributors = int(total_contributors)
        if total_contributors < preds.shape[0]:
            raise ValueError("total_contributors must be >= sampled K")
        multiplier = total_contributors / preds.shape[0]
    return (target_plan + multiplier * residual_sum, target_plan)


def gamma_band_plan_loss(
    pred_plan: torch.Tensor,
    target_plan: torch.Tensor,
    prescription: float | torch.Tensor | None = None,
    *,
    tol_rel: float = 0.01,
    cutoff_frac: float = 0.1,
    power: float = 1.0,
    return_stats: bool = False,
):
    """Tolerance-banded plan objective shaped like the dose axis of local gamma."""
    if pred_plan.shape != target_plan.shape:
        raise ValueError(
            f"pred plan {tuple(pred_plan.shape)} and target plan {tuple(target_plan.shape)} shapes differ"
        )
    if not 0.0 <= float(tol_rel) < 1.0:
        raise ValueError("tol_rel must be in [0, 1)")
    if not 0.0 < float(cutoff_frac) < 1.0:
        raise ValueError("cutoff_frac must be in (0, 1)")
    if float(power) <= 0.0:
        raise ValueError("power must be positive")
    if prescription is None:
        rx = target_plan.detach().amax()
    else:
        rx = torch.as_tensor(
            prescription, dtype=target_plan.dtype, device=target_plan.device
        )
    if rx.numel() != 1:
        raise ValueError("prescription must be scalar")
    if not bool(torch.isfinite(rx)) or float(rx) <= 0.0:
        raise ValueError("prescription must be finite and positive")
    reference = target_plan.detach()
    mask = reference >= float(cutoff_frac) * rx
    if not bool(mask.any()):
        zero = pred_plan.sum() * 0.0
        if return_stats:
            return (zero, {"fail_frac": zero.detach(), "n_eval": 0})
        return zero
    rel = (pred_plan[mask] - target_plan[mask]).abs() / reference[mask]
    excess = (rel - float(tol_rel)).clamp_min(0.0)
    if float(power) != 1.0:
        excess = excess.pow(float(power))
    loss = excess.mean()
    if return_stats:
        stats = {
            "fail_frac": (rel.detach() > float(tol_rel)).float().mean(),
            "n_eval": int(mask.sum()),
        }
        return (loss, stats)
    return loss


def gamma_band_plan_sum_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_plan: torch.Tensor | None = None,
    prescription: float | torch.Tensor | None = None,
    total_contributors: int | None = None,
    tol_rel: float = 0.01,
    cutoff_frac: float = 0.1,
    power: float = 1.0,
    return_stats: bool = False,
):
    """Gamma-band objective on the (sampled or control-variate) summed plan."""
    estimate, reference = _control_variate_plan_estimate(
        preds, targets, target_plan, total_contributors
    )
    return gamma_band_plan_loss(
        estimate,
        reference,
        prescription,
        tol_rel=tol_rel,
        cutoff_frac=cutoff_frac,
        power=power,
        return_stats=return_stats,
    )


def plan_sum_loss(preds, targets, eps: float = 1e-08):
    """Relative error of the SUMMED partial plan, not of the individual beamlets."""
    ps, ts = (preds.sum(dim=0), targets.sum(dim=0))
    scale = ts.abs().mean().clamp_min(eps)
    return (ps - ts).abs().mean() / scale


def stratified_plan_sum_loss(preds, targets, n_bands: int = 4, eps: float = 1e-08):
    """Legacy equal-count partial-plan proxy retained for reproducibility."""
    ts = targets.sum(dim=0)
    order = torch.argsort(ts)
    chunks = torch.tensor_split(order, max(1, int(n_bands)))
    terms = []
    for idx in chunks:
        if idx.numel() == 0:
            continue
        terms.append(plan_sum_loss(preds[:, idx], targets[:, idx], eps=eps))
    if not terms:
        return preds.sum() * 0.0
    return torch.stack(terms).mean()
