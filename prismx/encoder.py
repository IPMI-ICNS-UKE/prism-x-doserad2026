"""Transport encoder: depthwise-separable gated causal TCN over the cone grid."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn
from prismx.config import EncoderConfig


class RMSNorm3d(nn.Module):
    """RMS normalization over the channel dimension of a [B, C, D, U, V] tensor."""

    def __init__(self, channels: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight.view(1, -1, 1, 1, 1)


class CausalGatedLayer(nn.Module):
    """One encoder layer: causal depthwise conv -> pointwise -> GLU -> residual -> RMSNorm."""

    def __init__(
        self,
        width: int,
        glu_hidden: int,
        depth_kernel: int,
        lateral_kernel: int,
        depth_dilation: int,
        lateral_dilation: int,
    ):
        super().__init__()
        if glu_hidden != 2 * width:
            raise ValueError("glu_hidden must equal 2*width so GLU returns to width")
        self.depth_pad = (depth_kernel - 1) * depth_dilation
        lat_pad = (lateral_kernel - 1) // 2 * lateral_dilation
        self.depthwise = nn.Conv3d(
            width,
            width,
            kernel_size=(depth_kernel, lateral_kernel, lateral_kernel),
            dilation=(depth_dilation, lateral_dilation, lateral_dilation),
            padding=(0, lat_pad, lat_pad),
            groups=width,
        )
        self.pointwise = nn.Conv3d(width, glu_hidden, kernel_size=1)
        self.norm = RMSNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (0, 0, 0, 0, self.depth_pad, 0))
        h = self.depthwise(h)
        h = self.pointwise(h)
        h = F.glu(h, dim=1)
        return self.norm(x + h)


class TransportEncoder(nn.Module):
    """Gated causal TCN producing the latent transport memory h(u, v, d)."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Conv3d(cfg.in_channels, cfg.width, kernel_size=1)
        self.forward_layers = nn.ModuleList(
            (
                CausalGatedLayer(
                    cfg.width,
                    cfg.glu_hidden,
                    cfg.depth_kernel,
                    cfg.lateral_kernel,
                    depth_dilation=dd,
                    lateral_dilation=ld,
                )
                for dd, ld in zip(cfg.depth_dilations, cfg.lateral_dilations)
            )
        )
        self.anticausal_layers = nn.ModuleList(
            (
                CausalGatedLayer(
                    cfg.width,
                    cfg.glu_hidden,
                    cfg.depth_kernel,
                    cfg.lateral_kernel,
                    depth_dilation=dd,
                    lateral_dilation=1,
                )
                for dd in cfg.anticausal_depth_dilations
            )
        )
        self.head = nn.Conv3d(cfg.width, cfg.latent_channels, kernel_size=1)
        self.grad_checkpoint = True

    def causal_test_offsets(self) -> int:
        """Number of downstream depth steps visible at any output position."""
        return sum(
            (
                (self.cfg.depth_kernel - 1) * d
                for d in self.cfg.anticausal_depth_dilations
            )
        )

    def forward(self, x: torch.Tensor, pad_valid: tuple | None = None) -> torch.Tensor:
        """[B, in_channels, D, U, V] -> [B, latent_channels, D, U, V]."""
        h = self.input_proj(x)
        mask = None
        if pad_valid is not None:
            d0, u0, v0 = (int(v) for v in pad_valid)
            if (d0, u0, v0) != tuple(h.shape[-3:]):
                mask = torch.zeros(1, 1, *h.shape[-3:], dtype=h.dtype, device=h.device)
                mask[:, :, :d0, :u0, :v0] = 1.0
                h = h * mask
        use_ckpt = self.grad_checkpoint and torch.is_grad_enabled() and h.requires_grad
        for layer in self.forward_layers:
            if use_ckpt:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
            if mask is not None:
                h = h * mask
        for layer in self.anticausal_layers:
            if use_ckpt:
                h = torch.flip(
                    checkpoint(layer, torch.flip(h, dims=[2]), use_reentrant=False),
                    dims=[2],
                )
            else:
                h = torch.flip(layer(torch.flip(h, dims=[2])), dims=[2])
            if mask is not None:
                h = h * mask
        return self.head(h)
