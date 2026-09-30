"""MLC aperture model and 2D fluence rasters for PRISM-X."""

from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from prismx.config import ApertureRasterSpec, MLCSpec
from prismx.jfa import edt_1d_axis, jfa_edt

INTERLEAF_STRIPE_WIDTH_MM: float = 0.4
_SQRT2 = math.sqrt(2.0)
_GAUSS_TRUNCATE: float = 4.0


def _gaussian_kernel1d(sigma_px: float, device, dtype) -> torch.Tensor:
    """1-D Gaussian weights matching ``scipy.ndimage.gaussian_filter1d``."""
    radius = int(_GAUSS_TRUNCATE * float(sigma_px) + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    w = torch.exp(-0.5 * (x / float(sigma_px)) ** 2)
    return w / w.sum()


def _gaussian_blur2d(img: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur of a 2-D field on its own device."""
    if sigma_px <= 0:
        return img.clone()
    k = _gaussian_kernel1d(sigma_px, img.device, img.dtype)
    radius = (k.numel() - 1) // 2
    x = img.unsqueeze(0).unsqueeze(0)
    kh = k.view(1, 1, -1, 1)
    kw = k.view(1, 1, 1, -1)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, kh)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kw)
    out = x.squeeze(0).squeeze(0)
    return out.clamp(img.min(), img.max())


@dataclass
class MLCAperture:
    """One MLC/jaw aperture. Pair ``j`` spans v in"""

    left_mm: torch.Tensor
    right_mm: torch.Tensor
    jaw_u_mm: tuple = (-200.0, 200.0)
    jaw_v_mm: tuple = (-200.0, 200.0)
    mlc: MLCSpec = field(default_factory=MLCSpec)

    def __post_init__(self) -> None:
        self.left_mm = torch.as_tensor(self.left_mm, dtype=torch.float32).reshape(-1)
        self.right_mm = torch.as_tensor(self.right_mm, dtype=torch.float32).reshape(-1)
        if self.left_mm.numel() != self.mlc.n_pairs:
            raise ValueError("left_mm/right_mm must have n_pairs entries")

    @property
    def v_lo_mm(self) -> float:
        return -0.5 * self.mlc.n_pairs * self.mlc.leaf_width_mm

    def to(self, device) -> "MLCAperture":
        """Copy of the aperture with the leaf banks on ``device`` (spec shared)."""
        device = torch.device(device)
        return MLCAperture(
            left_mm=self.left_mm.to(device),
            right_mm=self.right_mm.to(device),
            jaw_u_mm=self.jaw_u_mm,
            jaw_v_mm=self.jaw_v_mm,
            mlc=self.mlc,
        )

    @staticmethod
    def rectangle(
        field_u_mm: float, field_v_mm: float, mlc: MLCSpec = MLCSpec()
    ) -> "MLCAperture":
        """Rectangular field: pairs whose v-strip intersects the open"""
        n, w = (mlc.n_pairs, mlc.leaf_width_mm)
        v_lo = -0.5 * n * w
        strip_lo = v_lo + w * torch.arange(n, dtype=torch.float32)
        strip_hi = strip_lo + w
        is_open = (strip_lo < 0.5 * field_v_mm) & (strip_hi > -0.5 * field_v_mm)
        left = torch.where(is_open, torch.tensor(-0.5 * field_u_mm), torch.tensor(0.0))
        right = torch.where(is_open, torch.tensor(0.5 * field_u_mm), torch.tensor(0.0))
        return MLCAperture(
            left,
            right,
            jaw_u_mm=(-0.5 * field_u_mm, 0.5 * field_u_mm),
            jaw_v_mm=(-0.5 * field_v_mm, 0.5 * field_v_mm),
            mlc=mlc,
        )

    @staticmethod
    def random_walk(
        rng: np.random.Generator, mlc: MLCSpec = MLCSpec(), max_open_mm: float = 100.0
    ) -> "MLCAperture":
        """Procedural aperture: AR(1)-smooth per-bank leaf walks over a random"""
        n, w = (mlc.n_pairs, mlc.leaf_width_mm)
        half = 0.5 * n * w
        tip_bound = half - 2.0
        phi = 0.85

        def ar1(steps: int, scale: float) -> np.ndarray:
            x = np.zeros(steps)
            x[0] = rng.normal(0.0, scale)
            innov = scale * math.sqrt(1.0 - phi * phi)
            for i in range(1, steps):
                x[i] = phi * x[i - 1] + rng.normal(0.0, innov)
            return x

        n_active = int(rng.integers(max(4, n // 8), max(6, n // 2) + 1))
        n_active = min(n_active, n - 1)
        j0 = int(rng.integers(0, n - n_active + 1))
        c = float(rng.uniform(-0.25, 0.25)) * half
        gap0 = float(rng.uniform(0.2, 1.0)) * max_open_mm
        centers = c + ar1(n_active, 0.12 * max_open_mm)
        half_gap = np.clip(
            0.5 * gap0 + ar1(n_active, 0.2 * gap0), 1.0, 0.5 * max_open_mm
        )
        l_act = centers - half_gap
        r_act = centers + half_gap
        closed = rng.random(n_active) < 0.15
        closed[n_active // 2] = False
        l_act = np.where(closed, centers, l_act)
        r_act = np.where(closed, centers, r_act)
        left = np.full(n, c)
        right = np.full(n, c)
        left[j0 : j0 + n_active] = l_act
        right[j0 : j0 + n_active] = r_act
        left = np.clip(left, -tip_bound, tip_bound)
        right = np.clip(right, left, tip_bound)
        margin = 5.0
        jaw_u = (
            max(-half, float(left.min()) - margin),
            min(half, float(right.max()) + margin),
        )
        open_idx = np.nonzero(right > left)[0]
        v_lo = -half
        jaw_v = (
            max(-half, v_lo + float(open_idx.min()) * w - margin),
            min(half, v_lo + float(open_idx.max() + 1) * w + margin),
        )
        return MLCAperture(
            torch.tensor(left, dtype=torch.float32),
            torch.tensor(right, dtype=torch.float32),
            jaw_u_mm=jaw_u,
            jaw_v_mm=jaw_v,
            mlc=mlc,
        )

    def open_area_mm2(self) -> float:
        """Open aperture area inside the jaws, mm^2."""
        n, w = (self.mlc.n_pairs, self.mlc.leaf_width_mm)
        l = self.left_mm.clamp(min=self.jaw_u_mm[0])
        r = self.right_mm.clamp(max=self.jaw_u_mm[1])
        du = (r - l).clamp(min=0.0)
        strip_lo = self.v_lo_mm + w * torch.arange(
            n, dtype=torch.float32, device=self.left_mm.device
        )
        strip_hi = strip_lo + w
        dv = (
            strip_hi.clamp(max=self.jaw_v_mm[1]) - strip_lo.clamp(min=self.jaw_v_mm[0])
        ).clamp(min=0.0)
        return float((du * dv).sum())

    def equivalent_square_mm(self) -> float:
        """sqrt(open area inside jaws)."""
        return math.sqrt(self.open_area_mm2())


class ApertureRaster:
    """Precomputed 2D aperture stack on the iso-plane raster plus the"""

    def __init__(
        self,
        spec: ApertureRasterSpec,
        aperture: MLCAperture,
        psi_sharp: torch.Tensor,
        psi_blur: torch.Tensor,
        sdf: torch.Tensor,
        blurs: dict,
        sd_u: torch.Tensor,
        sd_v: torch.Tensor,
    ):
        self.spec = spec
        self.aperture = aperture
        self.psi_sharp = psi_sharp
        self.psi_blur = psi_blur
        self.sdf = sdf
        self.blurs = blurs
        self.sd_u = sd_u
        self.sd_v = sd_v

    @property
    def device(self) -> torch.device:
        """Device the raster tensors live on."""
        return self.psi_sharp.device

    def to(self, device) -> "ApertureRaster":
        """Copy of the raster with all tensors on ``device`` (spec/aperture shared)."""
        device = torch.device(device)
        return ApertureRaster(
            spec=self.spec,
            aperture=self.aperture,
            psi_sharp=self.psi_sharp.to(device),
            psi_blur=self.psi_blur.to(device),
            sdf=self.sdf.to(device),
            blurs={sig: val.to(device) for sig, val in self.blurs.items()},
            sd_u=self.sd_u.to(device),
            sd_v=self.sd_v.to(device),
        )

    def _field(self, name: str) -> torch.Tensor:
        if name in ("psi_sharp", "psi_blur", "sdf", "sd_u", "sd_v"):
            return getattr(self, name)
        if name.startswith("blur_"):
            sigma = float(name[len("blur_") :])
            for key, val in self.blurs.items():
                if abs(key - sigma) < 1e-06:
                    return val
        raise KeyError(f"unknown raster field {name!r}")

    def sample(self, name: str, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Bilinear sample of a raster field; outside points edge-clamp."""
        f = self._field(name)
        if name == "sdf" and f.shape == (1, 1):
            raise RuntimeError(
                "aperture SDF sampling is unavailable because PRISMX_SKIP_APERTURE_SDF=1"
            )
        u = torch.as_tensor(u, dtype=torch.float32, device=f.device)
        v = torch.as_tensor(v, dtype=torch.float32, device=f.device)
        s, h = (self.spec.spacing_mm, self.spec.half_extent_mm)
        nu, nv = f.shape
        gu = ((u.reshape(-1) + h) / s).clamp(0.0, nu - 1.0)
        gv = ((v.reshape(-1) + h) / s).clamp(0.0, nv - 1.0)
        i0 = gu.floor().long().clamp(0, nu - 2)
        j0 = gv.floor().long().clamp(0, nv - 2)
        fu = gu - i0.float()
        fv = gv - j0.float()
        out = (
            f[i0, j0] * (1 - fu) * (1 - fv)
            + f[i0 + 1, j0] * fu * (1 - fv)
            + f[i0, j0 + 1] * (1 - fu) * fv
            + f[i0 + 1, j0 + 1] * fu * fv
        )
        return out.reshape(u.shape)

    def sample_many(
        self, names: tuple[str, ...], u: torch.Tensor, v: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Bilinearly sample several fields with one coordinate projection."""
        if not names:
            return {}
        fields = [self._field(name) for name in names]
        first = fields[0]
        u = torch.as_tensor(u, dtype=torch.float32, device=first.device)
        v = torch.as_tensor(v, dtype=torch.float32, device=first.device)
        s, h = (self.spec.spacing_mm, self.spec.half_extent_mm)
        nu, nv = first.shape
        gu = ((u.reshape(-1) + h) / s).clamp(0.0, nu - 1.0)
        gv = ((v.reshape(-1) + h) / s).clamp(0.0, nv - 1.0)
        i0 = gu.floor().long().clamp(0, nu - 2)
        j0 = gv.floor().long().clamp(0, nv - 2)
        fu = gu - i0.float()
        fv = gv - j0.float()
        result = {}
        for name, field in zip(names, fields):
            out = (
                field[i0, j0] * (1 - fu) * (1 - fv)
                + field[i0 + 1, j0] * fu * (1 - fv)
                + field[i0, j0 + 1] * (1 - fu) * fv
                + field[i0 + 1, j0 + 1] * fu * fv
            )
            result[name] = out.reshape(u.shape)
        return result

    def fluence_exact(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Closed-form erf fluence at exact continuous (u, v)."""
        ap = self.aperture
        spec = self.spec
        dev = self.device
        u = torch.as_tensor(u, dtype=torch.float32, device=dev)
        v = torch.as_tensor(v, dtype=torch.float32, device=dev)
        shape = u.shape
        uu = u.reshape(-1)
        vv = v.reshape(-1)
        left_bank = ap.left_mm.to(dev)
        right_bank = ap.right_mm.to(dev)
        n, w = (ap.mlc.n_pairs, ap.mlc.leaf_width_mm)
        v_lo = ap.v_lo_mm
        lt = spec.leaf_transmission
        jt = spec.jaw_transmission
        sf = spec.sigma_focal_mm
        sig_u = math.sqrt(sf * sf + spec.tip_softening_mm**2 / 12.0)
        rel = (vv - v_lo) / w
        j = rel.floor().clamp(0, n - 1).long()
        frac = rel - j.float()
        upper = frac >= 0.5
        j2 = (j + torch.where(upper, 1, -1)).clamp(0, n - 1)
        boundary = v_lo + torch.where(upper, (j + 1).float(), j.float()) * w
        dist = (vv - boundary).abs()
        wgt = 0.5 * (1.0 + torch.erf(dist / (_SQRT2 * sf)))

        def pair_term(idx: torch.Tensor) -> torch.Tensor:
            left = left_bank[idx]
            right = right_bank[idx]
            core = 0.5 * (
                torch.erf((uu - left) / (_SQRT2 * sig_u))
                - torch.erf((uu - right) / (_SQRT2 * sig_u))
            )
            return lt + (1.0 - lt) * core

        psi_mlc = wgt * pair_term(j) + (1.0 - wgt) * pair_term(j2)
        in_u = 0.5 * (
            torch.erf((uu - ap.jaw_u_mm[0]) / (_SQRT2 * sf))
            - torch.erf((uu - ap.jaw_u_mm[1]) / (_SQRT2 * sf))
        )
        in_v = 0.5 * (
            torch.erf((vv - ap.jaw_v_mm[0]) / (_SQRT2 * sf))
            - torch.erf((vv - ap.jaw_v_mm[1]) / (_SQRT2 * sf))
        )
        f_jaw = jt + (1.0 - jt) * in_u * in_v
        psi = (psi_mlc * f_jaw).clamp(jt * lt, 1.0)
        return psi.reshape(shape)


def _signed_edt_clamped(
    open_mask: torch.Tensor, spacing: tuple, clamp: float
) -> torch.Tensor:
    """Signed distance (positive inside ``open_mask``) via parallel JFA, clamped."""
    if bool(open_mask.all()):
        return torch.full_like(open_mask, clamp, dtype=torch.float32)
    if not bool(open_mask.any()):
        return torch.full_like(open_mask, -clamp, dtype=torch.float32)
    inside = jfa_edt(open_mask, spacing)
    outside = jfa_edt(~open_mask, spacing)
    sdf = torch.where(open_mask, inside, -outside)
    return sdf.clamp(-clamp, clamp).to(torch.float32)


def build_raster(ap: MLCAperture, spec: ApertureRasterSpec) -> ApertureRaster:
    """Rasterize an aperture into the 2D transmission/distance stack."""
    device = ap.left_mm.device
    s, hf = (spec.spacing_mm, spec.half_extent_mm)
    n_cells = int(round(2.0 * hf / s)) + 1
    coords = torch.arange(n_cells, device=device, dtype=torch.float32) * s - hf
    u = coords.view(n_cells, 1)
    v = coords.view(1, n_cells)
    n, w = (ap.mlc.n_pairs, ap.mlc.leaf_width_mm)
    v_lo = -0.5 * n * w
    pidx = torch.floor((coords - v_lo) / w).long().clamp(0, n - 1)
    left = ap.left_mm.detach().to(device=device, dtype=torch.float32)
    right = ap.right_mm.detach().to(device=device, dtype=torch.float32)
    L = left[pidx].view(1, n_cells)
    R = right[pidx].view(1, n_cells)
    pair_open = R > L
    t = spec.tip_softening_mm
    ramp = ((u - L) / t + 0.5).clamp(0.0, 1.0) * ((R - u) / t + 0.5).clamp(0.0, 1.0)
    open_frac = torch.where(pair_open, ramp, torch.zeros_like(ramp))
    lt = spec.leaf_transmission
    psi = lt + (1.0 - lt) * open_frac
    jb = torch.round((coords - v_lo) / w).long()
    b_v = v_lo + jb.to(torch.float32) * w
    half_stripe = 0.5 * INTERLEAF_STRIPE_WIDTH_MM
    overlap = (
        torch.minimum(coords + 0.5 * s, b_v + half_stripe)
        - torch.maximum(coords - 0.5 * s, b_v - half_stripe)
    ).clamp(min=0.0)
    stripe_frac = torch.where(
        (jb >= 1) & (jb <= n - 1), overlap / s, torch.zeros_like(overlap)
    )
    psi = psi + spec.interleaf_transmission * stripe_frac.view(1, n_cells) * (
        1.0 - open_frac
    )
    in_jaw = (
        (u >= ap.jaw_u_mm[0])
        & (u <= ap.jaw_u_mm[1])
        & (v >= ap.jaw_v_mm[0])
        & (v <= ap.jaw_v_mm[1])
    )
    psi = torch.where(in_jaw, psi, psi * spec.jaw_transmission)
    psi = psi.clamp(spec.jaw_transmission * spec.leaf_transmission, 1.0).to(
        torch.float32
    )
    sigma_px = spec.sigma_focal_mm / s
    psi_blur = _gaussian_blur2d(psi, sigma_px)
    blurs = {float(sig): _gaussian_blur2d(psi, sig / s) for sig in spec.blur_sigmas_mm}
    clamp = spec.sd_clamp_mm
    skip_sdf = os.environ.get("PRISMX_SKIP_APERTURE_SDF", "0") == "1"
    if skip_sdf:
        sdf = torch.zeros((1, 1), dtype=torch.float32, device=device)
    else:
        open_mask = psi >= 0.5
        stride = max(1, int(getattr(spec, "edt_stride", 1)))
        if stride > 1 and (n_cells - 1) % stride == 0:
            coarse = open_mask[::stride, ::stride]
            sdf_c = _signed_edt_clamped(coarse, (s * stride, s * stride), clamp)
            sdf = F.interpolate(
                sdf_c[None, None],
                size=open_mask.shape,
                mode="bilinear",
                align_corners=True,
            )[0, 0].clamp(-clamp, clamp)
        else:
            sdf = _signed_edt_clamped(open_mask, (s, s), clamp)
    sd_u = torch.minimum(u - L, R - u).clamp(-clamp, clamp)
    sd_u = sd_u.expand(n_cells, n_cells).contiguous().to(torch.float32)
    open_un = (left.view(1, n) <= coords.view(n_cells, 1)) & (
        coords.view(n_cells, 1) <= right.view(1, n)
    )
    open_un = open_un & (right > left).view(1, n)
    open_v = open_un[:, pidx]
    if bool(open_v.all()):
        sd_v = torch.full_like(psi, clamp)
    elif not bool(open_v.any()):
        sd_v = torch.full_like(psi, -clamp)
    else:
        inside = edt_1d_axis(open_v, s, axis=1)
        outside = edt_1d_axis(~open_v, s, axis=1)
        sd_v = (
            torch.where(open_v, inside, -outside).clamp(-clamp, clamp).to(torch.float32)
        )
    return ApertureRaster(
        spec=spec,
        aperture=ap,
        psi_sharp=psi.contiguous(),
        psi_blur=psi_blur.contiguous(),
        sdf=sdf.contiguous(),
        blurs=blurs,
        sd_u=sd_u.contiguous(),
        sd_v=sd_v.contiguous(),
    )
