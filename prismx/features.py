"""Feature assembly for PRISM-X."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from prismx.analytic import (
    PDDTable,
    analytic_baseline,
    inverse_square,
    three_exp_transmission,
)
from prismx.aperture import ApertureRaster, MLCAperture
from prismx.cone_grid import DEPTH_BELOW_SURFACE_CLAMP_MM, ConeGrid, CTVolume
from prismx.config import DECODER_FEATURES, ENCODER_CHANNELS, SURFACE_DENSITY_THRESHOLD
from prismx.geometry import BeamGeometry, beam_to_cone

_UV_NORM_MM: float = 200.0
_EQ_SQ_NORM_MM: float = 250.0
_MM_PER_CM: float = 10.0
_LOG_EPS: float = 1e-06
RAY_MARCH_STEPS: int = 128
_RAY_MARCH_CHUNK: int = 4096


def _lateral_mean(rho: torch.Tensor, kernel: int) -> torch.Tensor:
    """Lateral (u, v) box mean of a [D, U, V] field, 'same' padding."""
    x = rho.unsqueeze(1)
    out = F.avg_pool2d(
        x, kernel, stride=1, padding=kernel // 2, count_include_pad=False
    )
    return out.squeeze(1)


def encoder_channels(
    cone: ConeGrid, raster: ApertureRaster, ap: MLCAperture
) -> torch.Tensor:
    """Assemble the 19 encoder input channels on the cone grid."""
    spec = cone.spec
    sad = cone.beam.sad_mm
    rho = cone.rho
    n_d = cone.n_depth
    u_ax, v_ax, d_ax = cone.axes()
    drho = torch.gradient(rho, spacing=(float(spec.dd_mm),), dim=0)[0]
    d_col = d_ax.view(n_d, 1, 1)
    log_isq = torch.log(inverse_square(d_col, sad)).expand(n_d, spec.nu, spec.nv)
    uu = u_ax.view(spec.nu, 1).expand(spec.nu, spec.nv)
    vv = v_ax.view(1, spec.nv).expand(spec.nu, spec.nv)

    def per_ray(name: str) -> torch.Tensor:
        return raster.sample(name, uu, vv).unsqueeze(0).expand(n_d, spec.nu, spec.nv)

    t123 = three_exp_transmission(cone.wed)
    d_iso = torch.sqrt(uu * uu + vv * vv + sad * sad)
    channels = {
        "rho_e": rho,
        "drho_dd": drho,
        "rho_mean_3x3": _lateral_mean(rho, 3),
        "rho_mean_9x9": _lateral_mean(rho, 9),
        "wed": cone.wed / _MM_PER_CM,
        "T1": t123[..., 0],
        "T2": t123[..., 1],
        "T3": t123[..., 2],
        "log_isq": log_isq,
        "psi_blur": per_ray("psi_blur"),
        "sd_u": per_ray("sd_u"),
        "sd_v": per_ray("sd_v"),
        "eq_sq": torch.full(
            (n_d, spec.nu, spec.nv),
            ap.equivalent_square_mm() / _EQ_SQ_NORM_MM,
            device=rho.device,
        ),
        "u_norm": (u_ax / _UV_NORM_MM)
        .view(1, spec.nu, 1)
        .expand(n_d, spec.nu, spec.nv),
        "v_norm": (v_ax / _UV_NORM_MM)
        .view(1, 1, spec.nv)
        .expand(n_d, spec.nu, spec.nv),
        "divergence": d_col / d_iso.unsqueeze(0),
    }
    for sig in raster.spec.blur_sigmas_mm:
        channels[f"psi_blur_{sig:g}mm"] = per_ray(f"blur_{sig:g}")
    return torch.stack(
        [channels[name].to(torch.float32) for name in ENCODER_CHANNELS], dim=0
    )


def _bilinear_uv(
    cone: ConeGrid, field: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Bilinear sample of a per-ray [U, V] field at (u, v) mm, edge clamp."""
    spec = cone.spec
    fu = (u / spec.du_mm + (spec.nu - 1) / 2).clamp(0.0, spec.nu - 1.0)
    fv = (v / spec.dv_mm + (spec.nv - 1) / 2).clamp(0.0, spec.nv - 1.0)
    i0 = fu.floor().long().clamp(0, max(spec.nu - 2, 0))
    j0 = fv.floor().long().clamp(0, max(spec.nv - 2, 0))
    i1 = (i0 + 1).clamp(max=spec.nu - 1)
    j1 = (j0 + 1).clamp(max=spec.nv - 1)
    wu = fu - i0.to(fu.dtype)
    wv = fv - j0.to(fv.dtype)
    return (
        field[i0, j0] * (1 - wu) * (1 - wv)
        + field[i1, j0] * wu * (1 - wv)
        + field[i0, j1] * (1 - wu) * wv
        + field[i1, j1] * wu * wv
    )


def _ray_march(
    ct: CTVolume, beam: BeamGeometry, pts_world: torch.Tensor, miss_d_surface: float
) -> tuple:
    """Far-field WED and surface depth by marching source -> point through ct."""
    dev = pts_world.device
    src = beam.source_world().to(device=dev, dtype=torch.float32)
    n_steps = RAY_MARCH_STEPS
    t_mid = (torch.arange(n_steps, dtype=torch.float32, device=dev) + 0.5) / n_steps
    step_idx = torch.arange(n_steps, device=dev)
    wed_parts, surf_parts = ([], [])
    for chunk in pts_world.split(_RAY_MARCH_CHUNK):
        delta = chunk - src
        length = delta.norm(dim=1)
        samples = src + t_mid.view(1, -1, 1) * delta.unsqueeze(1)
        rho = ct.sample_world(samples.reshape(-1, 3)).reshape(chunk.shape[0], n_steps)
        wed_parts.append(rho.sum(dim=1) * length / n_steps)
        hit = rho > SURFACE_DENSITY_THRESHOLD
        first = torch.where(hit, step_idx, n_steps).amin(dim=1)
        d_surf = (first.to(torch.float32) + 0.5) / n_steps * length
        surf_parts.append(
            torch.where(
                first < n_steps, d_surf, torch.full_like(d_surf, miss_d_surface)
            )
        )
    return (torch.cat(wed_parts), torch.cat(surf_parts))


def decoder_features(
    cone: ConeGrid,
    raster: ApertureRaster,
    ap: MLCAperture,
    ct: CTVolume,
    beam: BeamGeometry,
    world_pts: torch.Tensor,
    pdd: PDDTable,
    fp32: bool = False,
) -> tuple:
    """Continuous analytic features at world query points."""
    spec = cone.spec
    pts = world_pts.to(torch.float32)
    uvd = beam_to_cone(beam.world_to_beam(pts, fp32=fp32), beam.sad_mm, fp32=fp32)
    u, v, d = (uvd[:, 0], uvd[:, 1], uvd[:, 2])
    u_half = (spec.nu - 1) / 2 * spec.du_mm
    v_half = (spec.nv - 1) / 2 * spec.dv_mm
    in_box = cone.contains(uvd) & (u.abs() <= u_half) & (v.abs() <= v_half)
    far = ~in_box
    psi_exact = raster.fluence_exact(u, v)
    isq = inverse_square(d, beam.sad_mm)
    miss_ssd = cone.d_start + cone.n_depth * spec.dd_mm
    wed_mm = torch.zeros_like(d)
    ssd_mm = torch.full_like(d, miss_ssd)
    if bool(in_box.any()):
        wed_mm[in_box] = cone.sample(cone.wed, uvd[in_box], outside=0.0)
        ssd_mm[in_box] = _bilinear_uv(cone, cone.ssd, u[in_box], v[in_box])
    if bool(far.any()):
        wed_mm[far], ssd_mm[far] = _ray_march(ct, beam, pts[far], miss_ssd)
    dbs_mm = (d - ssd_mm).clamp(
        -DEPTH_BELOW_SURFACE_CLAMP_MM, DEPTH_BELOW_SURFACE_CLAMP_MM
    )
    p_wed = pdd(wed_mm)
    a_val = analytic_baseline(psi_exact, p_wed, isq)
    t123 = three_exp_transmission(wed_mm)
    frac = cone.frac_offsets(uvd)
    rho_e = ct.sample_world(pts)
    src = beam.source_world().to(device=pts.device, dtype=torch.float32)
    ray = pts - src
    ray = ray / ray.norm(dim=1, keepdim=True).clamp_min(1e-06)
    half_step = 0.5 * spec.dd_mm * ray
    drho_dd = (
        ct.sample_world(pts + half_step) - ct.sample_world(pts - half_step)
    ) / spec.dd_mm
    raster_names = (
        "sd_u",
        "sd_v",
        *(f"blur_{sig:g}" for sig in raster.spec.blur_sigmas_mm),
    )
    raster_values = raster.sample_many(raster_names, u, v)
    feats = {
        "psi_exact": psi_exact,
        "log_A": torch.log(a_val + _LOG_EPS),
        "wed": wed_mm / _MM_PER_CM,
        "p_wed": p_wed,
        "T1": t123[:, 0],
        "T2": t123[:, 1],
        "T3": t123[:, 2],
        "log_isq": torch.log(isq + _LOG_EPS),
        "off_axis_r": torch.sqrt(u * u + v * v) / _UV_NORM_MM,
        "sd_u": raster_values["sd_u"],
        "sd_v": raster_values["sd_v"],
        "eq_sq": torch.full_like(d, ap.equivalent_square_mm() / _EQ_SQ_NORM_MM),
        "rho_e": rho_e,
        "drho_dd": drho_dd,
        "ssd": ssd_mm / _MM_PER_CM,
        "depth_below_surface": dbs_mm / _MM_PER_CM,
        "frac_du": frac[:, 0],
        "frac_dv": frac[:, 1],
        "frac_dd": frac[:, 2],
    }
    for sig in raster.spec.blur_sigmas_mm:
        feats[f"psi_blur_{sig:g}mm"] = raster_values[f"blur_{sig:g}"]
    features = torch.stack(
        [feats[name].to(torch.float32) for name in DECODER_FEATURES], dim=-1
    )
    aux = {
        "A": a_val,
        "p_wed": p_wed,
        "isq": isq,
        "psi_exact": psi_exact,
        "uvd": uvd,
        "in_box": in_box,
    }
    return (features, aux)
