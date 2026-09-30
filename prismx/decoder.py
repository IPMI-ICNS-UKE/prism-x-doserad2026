"""Decoders and physics-residual dose composition (SPEC.md section 5)."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from prismx.config import DecoderConfig

_GA_INDEX = 1


def _build_mlp(in_features: int, hidden: int, n_hidden: int) -> nn.Sequential:
    """SiLU MLP in_features -> hidden x n_hidden -> 2."""
    layers: list[nn.Module] = []
    width = in_features
    for _ in range(n_hidden):
        layers.append(nn.Linear(width, hidden))
        layers.append(nn.SiLU())
        width = hidden
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


def _zero_init_final(mlp: nn.Sequential, additive_bias_init: float) -> None:
    final = mlp[-1]
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    with torch.no_grad():
        final.bias[_GA_INDEX] += additive_bias_init


class MainDecoder(nn.Module):
    """In-box decoder: [latent ++ analytic] -> (g_m, g_a)."""

    def __init__(self, cfg: DecoderConfig = DecoderConfig()):
        super().__init__()
        self.cfg = cfg
        self.mlp = _build_mlp(
            cfg.latent_channels + cfg.n_analytic, cfg.width, cfg.n_hidden
        )
        _zero_init_final(self.mlp, cfg.additive_bias_init)

    def forward(self, latent: torch.Tensor, analytic: torch.Tensor) -> tuple:
        """latent [N, latent_channels], analytic [N, n_analytic] -> (g_m [N], g_a [N])."""
        out = self.mlp(torch.cat([latent, analytic], dim=-1))
        return (out[:, 0], out[:, _GA_INDEX])


class FarFieldDecoder(nn.Module):
    """Out-of-box decoder: analytic features only -> (g_m, g_a)."""

    def __init__(self, cfg: DecoderConfig = DecoderConfig()):
        super().__init__()
        self.cfg = cfg
        self.mlp = _build_mlp(cfg.n_analytic, cfg.farfield_width, 2)
        _zero_init_final(self.mlp, cfg.additive_bias_init)

    def forward(self, analytic: torch.Tensor) -> tuple:
        """analytic [N, n_analytic] -> (g_m [N], g_a [N])."""
        out = self.mlp(analytic)
        return (out[:, 0], out[:, _GA_INDEX])


def compose_dose(
    A: torch.Tensor,
    p_wed: torch.Tensor,
    isq: torch.Tensor,
    g_m: torch.Tensor,
    g_a: torch.Tensor,
    cfg: DecoderConfig,
    add_clamp: float | None = None,
    gm_lower_clamp: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Compose the analytic baseline with multiplicative and additive residuals."""
    if gm_lower_clamp is None:
        g_m = torch.clamp(g_m, -cfg.gm_clamp, cfg.gm_clamp)
    else:
        lower = torch.as_tensor(
            gm_lower_clamp, dtype=g_m.dtype, device=g_m.device
        ).clamp_min(0.0)
        g_m = torch.minimum(g_m, g_m.new_tensor(cfg.gm_clamp))
        g_m = torch.maximum(g_m, -lower)
    add = cfg.additive_scale * isq * p_wed * F.softplus(g_a)
    if add_clamp is not None:
        add = add.clamp(max=add_clamp)
    return A * torch.exp(g_m) + add
