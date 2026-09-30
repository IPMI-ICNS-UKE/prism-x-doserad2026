"""Learned MR -> density module + a differentiable density-override adapter."""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from prismx.cone_grid import CTVolume

DENSITY_SCALE: float = 3.0


def _norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm for ``channels`` feature maps (batch-size independent)."""
    g = min(groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class _ConvBlock(nn.Module):
    """Two 3x3x3 convs (GroupNorm + SiLU each), 'same' padding."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            _norm(out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            _norm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MRDensityNet(nn.Module):
    """3D U-Net mapping a normalized MR volume to a relative-density volume."""

    def __init__(self, base_channels: int = 24, levels: int = 3):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = levels
        chans = [base_channels * 2**i for i in range(levels + 1)]
        self.in_block = _ConvBlock(1, chans[0])
        self.down = nn.ModuleList(
            (_ConvBlock(chans[i], chans[i + 1]) for i in range(levels))
        )
        self.up_convs = nn.ModuleList(
            (
                nn.ConvTranspose3d(chans[i + 1], chans[i], kernel_size=2, stride=2)
                for i in reversed(range(levels))
            )
        )
        self.up_blocks = nn.ModuleList(
            (_ConvBlock(2 * chans[i], chans[i]) for i in reversed(range(levels)))
        )
        self.head = nn.Conv3d(chans[0], 1, kernel_size=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def n_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum((p.numel() for p in self.parameters()))

    def forward(self, mr: torch.Tensor) -> torch.Tensor:
        """``[B, 1, Z, Y, X]`` MR -> ``[B, 1, Z, Y, X]`` density ``>= 0``."""
        z, y, x = mr.shape[-3:]
        mult = 2**self.levels
        pads = [(0, (mult - s % mult) % mult) for s in (x, y, z)]
        flat_pad = [
            pads[0][0],
            pads[0][1],
            pads[1][0],
            pads[1][1],
            pads[2][0],
            pads[2][1],
        ]
        h = F.pad(mr, flat_pad) if any(flat_pad) else mr
        skips: list[torch.Tensor] = []
        h = self.in_block(h)
        for block in self.down:
            skips.append(h)
            h = block(self.pool(h))
        for up_conv, up_block, skip in zip(
            self.up_convs, self.up_blocks, reversed(skips)
        ):
            h = up_conv(h)
            h = up_block(torch.cat([h, skip], dim=1))
        rho = DENSITY_SCALE * F.softplus(self.head(h))
        return rho[..., :z, :y, :x]


class _ResBlock(nn.Module):
    """Pre-activation-ish residual block: (Conv-GN-SiLU) x2 + identity skip."""

    def __init__(self, in_ch: int, out_ch: int, grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = _norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _norm(out_ch)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            residual = self.skip(x)
            h = self.conv1(x)
            del x
            h = self.act(self.norm1(h))
            h = self.norm2(self.conv2(h))
            return self.act(h + residual)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))

    @staticmethod
    def _conv_from_parts(
        a: torch.Tensor, b: torch.Tensor, conv: nn.Conv3d
    ) -> torch.Tensor:
        """Apply a groups=1 convolution to ``cat([a,b],1)`` without the cat buffer."""
        if conv.groups != 1 or a.shape[1] + b.shape[1] != conv.in_channels:
            raise ValueError(
                "split decoder convolution requires groups=1 matching channels"
            )
        split = int(a.shape[1])
        kwargs = {
            "stride": conv.stride,
            "padding": conv.padding,
            "dilation": conv.dilation,
            "groups": 1,
        }
        out = F.conv3d(a, conv.weight[:, :split], conv.bias, **kwargs)
        other = F.conv3d(b, conv.weight[:, split:], None, **kwargs)
        out.add_(other)
        return out

    def inference_from_parts(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Decoder inference without materializing ``cat([a,b], dim=1)``."""
        if torch.is_grad_enabled():
            raise RuntimeError("inference_from_parts is inference-only")
        if not isinstance(self.skip, nn.Conv3d):
            raise TypeError("decoder split path requires a convolutional residual")
        h = self._conv_from_parts(a, b, self.conv1)
        residual = self._conv_from_parts(a, b, self.skip)
        del a, b
        h = self.act(self.norm1(h))
        h = self.norm2(self.conv2(h))
        return self.act(h + residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and x.requires_grad:
            return checkpoint(self._body, x, use_reentrant=False)
        return self._body(x)


class _ResStage(nn.Module):
    """``n_blocks`` residual blocks; the first maps ``in_ch -> out_ch``."""

    def __init__(
        self, in_ch: int, out_ch: int, n_blocks: int, grad_checkpoint: bool = False
    ):
        super().__init__()
        blocks = [_ResBlock(in_ch, out_ch, grad_checkpoint)]
        blocks += [
            _ResBlock(out_ch, out_ch, grad_checkpoint)
            for _ in range(max(0, n_blocks - 1))
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def _resenc_widths(base_channels: int, levels: int, cap: int = 320) -> list:
    """Channel width per resolution level: ``base * 2**i`` capped at ``cap``."""
    return [min(base_channels * 2**i, cap) for i in range(levels + 1)]


class ResEncDensityNet(nn.Module):
    """Residual-ENCODER 3D U-Net for MR -> density (nnU-Net ResEnc-inspired)."""

    def __init__(
        self,
        base_channels: int = 32,
        levels: int = 4,
        blocks: tuple | None = None,
        cap: int = 320,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = levels
        w = _resenc_widths(base_channels, levels, cap)
        if blocks is None:
            blocks = tuple((min(1 + i, 4) for i in range(levels + 1)))
        if len(blocks) != levels + 1:
            raise ValueError(f"blocks must have length levels+1={levels + 1}")
        self.stem = nn.Conv3d(1, w[0], kernel_size=3, padding=1)
        self.enc0 = _ResStage(w[0], w[0], blocks[0], grad_checkpoint)
        self.downs = nn.ModuleList(
            (
                nn.Conv3d(w[i - 1], w[i], kernel_size=3, stride=2, padding=1)
                for i in range(1, levels + 1)
            )
        )
        self.enc = nn.ModuleList(
            (
                _ResStage(w[i], w[i], blocks[i], grad_checkpoint)
                for i in range(1, levels + 1)
            )
        )
        self.ups = nn.ModuleList(
            (
                nn.ConvTranspose3d(w[i], w[i - 1], kernel_size=2, stride=2)
                for i in reversed(range(1, levels + 1))
            )
        )
        self.dec = nn.ModuleList(
            (
                _ResBlock(2 * w[i - 1], w[i - 1], grad_checkpoint)
                for i in reversed(range(1, levels + 1))
            )
        )
        self.head = nn.Conv3d(w[0], 1, kernel_size=1)
        self.memory_efficient_inference = True

    def n_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters()))

    def set_grad_checkpoint(self, flag: bool) -> None:
        """Toggle gradient checkpointing on every residual block. Turn it OFF for"""
        for m in self.modules():
            if isinstance(m, _ResBlock):
                m.grad_checkpoint = flag

    def forward(self, mr: torch.Tensor) -> torch.Tensor:
        z, y, x = mr.shape[-3:]
        mult = 2**self.levels
        pads = [(0, (mult - s % mult) % mult) for s in (x, y, z)]
        flat_pad = [
            pads[0][0],
            pads[0][1],
            pads[1][0],
            pads[1][1],
            pads[2][0],
            pads[2][1],
        ]
        h = F.pad(mr, flat_pad) if any(flat_pad) else mr
        h = self.enc0(self.stem(h))
        skips = [h]
        for down, stage in zip(self.downs, self.enc):
            h = stage(down(h))
            skips.append(h)
        skips.pop()
        inference_cuda = not torch.is_grad_enabled() and h.is_cuda
        if inference_cuda:
            torch.cuda.empty_cache()
        for up, dec in zip(self.ups, self.dec):
            skip = skips.pop()
            h = up(h)
            if inference_cuda and self.memory_efficient_inference:
                h = dec.inference_from_parts(h, skip)
            else:
                joined = torch.cat([h, skip], dim=1)
                del h, skip
                h = dec(joined)
            if inference_cuda:
                torch.cuda.empty_cache()
        rho = DENSITY_SCALE * F.softplus(self.head(h))
        return rho[..., :z, :y, :x]


def density_ctvolume(
    density: torch.Tensor, spacing_mm: tuple, origin_mm: tuple
) -> CTVolume:
    """Wrap a (possibly grad-carrying) density tensor ``[Z, Y, X]`` as a CTVolume."""
    return CTVolume(
        density=density,
        spacing_mm=tuple((float(s) for s in spacing_mm)),
        origin_mm=tuple((float(o) for o in origin_mm)),
    )


TRAIN_SPACING_MM = (2.0, 2.0, 2.0)


def resample_to_spacing(arr, spacing, origin, target_spacing, order="linear"):
    """Resample a ``[Z,Y,X]`` volume to ``target_spacing`` (x, y, z order)."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.asarray(arr))
    img.SetSpacing(tuple((float(v) for v in spacing)))
    img.SetOrigin(tuple((float(v) for v in origin)))
    size = [
        max(1, int(round(n * s / t)))
        for n, s, t in zip(img.GetSize(), img.GetSpacing(), target_spacing)
    ]
    r = sitk.ResampleImageFilter()
    r.SetInterpolator(
        sitk.sitkLinear if order == "linear" else sitk.sitkNearestNeighbor
    )
    r.SetOutputSpacing(tuple((float(v) for v in target_spacing)))
    r.SetSize(size)
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
    out = r.Execute(img)
    return (
        sitk.GetArrayFromImage(out),
        tuple(out.GetSpacing()),
        tuple(out.GetOrigin()),
    )


def resample_like(arr, spacing, origin, ref_shape, ref_spacing, ref_origin):
    """Resample a ``[Z,Y,X]`` volume back onto a reference grid."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.asarray(arr))
    img.SetSpacing(tuple((float(v) for v in spacing)))
    img.SetOrigin(tuple((float(v) for v in origin)))
    ref = sitk.Image(
        [int(ref_shape[2]), int(ref_shape[1]), int(ref_shape[0])], sitk.sitkFloat32
    )
    ref.SetSpacing(tuple((float(v) for v in ref_spacing)))
    ref.SetOrigin(tuple((float(v) for v in ref_origin)))
    r = sitk.ResampleImageFilter()
    r.SetInterpolator(sitk.sitkLinear)
    r.SetReferenceImage(ref)
    return sitk.GetArrayFromImage(r.Execute(img))


def needs_resample(spacing, target_spacing=TRAIN_SPACING_MM, tol=0.05):
    """True when the case voxel size differs enough from the net's to matter."""
    return any(
        (abs(float(s) - float(t)) > tol for s, t in zip(spacing, target_spacing))
    )


def resolution_augment(
    mr, ct, mask, rng, z_range=(0.6, 1.0), xy_range=(1.0, 2.0), max_voxels=None
):
    """Resample a ``[Z,Y,X]`` training triplet to a random voxel size."""
    fz = float(rng.uniform(*z_range))
    fxy = float(rng.uniform(*xy_range))
    if abs(fz - 1.0) < 0.001 and abs(fxy - 1.0) < 0.001:
        return (mr, ct, mask)
    size = tuple((max(8, int(round(n * f))) for n, f in zip(mr.shape, (fz, fxy, fxy))))

    def _rs(x, mode):
        y = x.to(torch.float32)[None, None]
        y = F.interpolate(
            y,
            size=size,
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
        )
        return y[0, 0]

    mr_r, ct_r = (_rs(mr, "trilinear"), _rs(ct, "trilinear"))
    mask_r = _rs(mask.to(torch.float32), "nearest") > 0.5
    if max_voxels is not None and mr_r.numel() > max_voxels:
        keep = (max_voxels / mr_r.numel()) ** (1.0 / 3.0)
        crop = tuple((max(8, min(n, int(n * keep))) for n in mr_r.shape))
        off = tuple((int(rng.integers(0, n - c + 1)) for n, c in zip(mr_r.shape, crop)))
        sl = tuple((slice(o, o + c) for o, c in zip(off, crop)))
        mr_r, ct_r, mask_r = (mr_r[sl], ct_r[sl], mask_r[sl])
    return (mr_r, ct_r, mask_r)


def anchor_loss(
    density_pred: torch.Tensor,
    density_target: torch.Tensor,
    body_mask: torch.Tensor,
    beta: float = 0.0,
    bone_weight: float = 0.0,
    bone_thresh: float = 1.1,
    grad_weight: float = 0.0,
) -> torch.Tensor:
    """Body-masked density anchor loss (smooth-L1, ``beta=0`` -> L1)."""
    mask = body_mask.to(density_pred.dtype)
    if bone_weight > 0.0:
        bone = (density_target.to(density_pred.dtype) > bone_thresh).to(
            density_pred.dtype
        )
        mask = mask * (1.0 + bone_weight * bone)
    target = density_target.to(density_pred.dtype)
    per_voxel = F.smooth_l1_loss(density_pred, target, beta=beta, reduction="none")
    denom = mask.sum()
    if denom == 0:
        return density_pred.new_zeros(())
    out = (per_voxel * mask).sum() / denom
    if grad_weight > 0.0 and body_mask.shape == density_pred.shape:
        keep = body_mask.to(torch.bool)
        for d in (-3, -2, -1):
            n = density_pred.shape[d] - 1
            if n < 1:
                continue
            m = keep.narrow(d, 0, n) | keep.narrow(d, 1, n)
            gp = torch.diff(density_pred, dim=d)[m]
            if gp.numel() == 0:
                continue
            out = out + grad_weight * (gp - torch.diff(target, dim=d)[m]).abs().mean()
    return out


def air_loss(density_pred: torch.Tensor, body_mask: torch.Tensor) -> torch.Tensor:
    """L1-toward-0 of density OUTSIDE the body. The anchor loss is body-masked,"""
    outside = 1.0 - body_mask.to(density_pred.dtype)
    denom = outside.sum()
    if denom == 0:
        return density_pred.new_zeros(())
    return (density_pred * outside).sum() / denom


def cap_loss(
    density_pred: torch.Tensor,
    density_target: torch.Tensor,
    body_mask: torch.Tensor,
    tau: float = 1.1,
    bone_thresh: float = 1.1,
    bone_dilate: int = 1,
) -> torch.Tensor:
    """Soft-tissue density ceiling. Penalize predicted density above ``tau`` where"""
    tgt = density_target.to(density_pred.dtype)
    body = body_mask.to(density_pred.dtype)
    bone = (tgt > bone_thresh).to(density_pred.dtype)
    if bone_dilate > 0:
        k = 2 * bone_dilate + 1
        bone = F.max_pool3d(bone.reshape(1, 1, *bone.shape), k, 1, bone_dilate).reshape(
            bone.shape
        )
    nonbone = (1.0 - bone) * body
    denom = nonbone.sum()
    if denom == 0:
        return density_pred.new_zeros(())
    over = F.relu(density_pred - tau)
    return (over * over * nonbone).sum() / denom
