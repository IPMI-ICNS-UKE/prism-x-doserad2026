"""Central configuration for PRISM-X."""

from __future__ import annotations
from dataclasses import dataclass, field, replace

SAD_MM: float = 1000.0
ENCODER_CHANNELS: tuple = (
    "rho_e",
    "drho_dd",
    "rho_mean_3x3",
    "rho_mean_9x9",
    "wed",
    "T1",
    "T2",
    "T3",
    "log_isq",
    "psi_blur",
    "psi_blur_5mm",
    "psi_blur_15mm",
    "psi_blur_45mm",
    "sd_u",
    "sd_v",
    "eq_sq",
    "u_norm",
    "v_norm",
    "divergence",
)
DECODER_FEATURES: tuple = (
    "psi_exact",
    "log_A",
    "wed",
    "p_wed",
    "T1",
    "T2",
    "T3",
    "log_isq",
    "off_axis_r",
    "sd_u",
    "sd_v",
    "eq_sq",
    "psi_blur_5mm",
    "psi_blur_15mm",
    "psi_blur_45mm",
    "rho_e",
    "drho_dd",
    "ssd",
    "depth_below_surface",
    "frac_du",
    "frac_dv",
    "frac_dd",
)
assert len(ENCODER_CHANNELS) == 19
assert len(DECODER_FEATURES) == 22
THREE_EXP_WEIGHTS: tuple = (0.35, 0.45, 0.2)
THREE_EXP_MU_PER_MM: tuple = (0.0062, 0.0048, 0.0038)
AIR_DENSITY: float = 0.001
SURFACE_DENSITY_THRESHOLD: float = 0.05


@dataclass(frozen=True)
class ConeGridSpec:
    nu: int = 160
    nv: int = 160
    du_mm: float = 2.5
    dv_mm: float = 2.5
    dd_mm: float = 2.0
    max_depth_steps: int = 224
    entry_margin_mm: float = 10.0
    window_scout_stride: int = 1


@dataclass(frozen=True)
class MLCSpec:
    n_pairs: int = 80
    leaf_width_mm: float = 5.0


@dataclass(frozen=True)
class ApertureRasterSpec:
    spacing_mm: float = 0.5
    half_extent_mm: float = 200.0
    edt_stride: int = 1
    sigma_focal_mm: float = 1.0
    leaf_transmission: float = 0.02
    interleaf_transmission: float = 0.005
    jaw_transmission: float = 0.002
    tip_softening_mm: float = 3.0
    blur_sigmas_mm: tuple = (5.0, 15.0, 45.0)
    sd_clamp_mm: float = 50.0


@dataclass(frozen=True)
class EncoderConfig:
    in_channels: int = 19
    arch: str = "tcn"
    cond_channels: tuple = ()
    width: int = 48
    glu_hidden: int = 96
    latent_channels: int = 32
    depth_kernel: int = 2
    lateral_kernel: int = 3
    depth_dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 128, 1, 1)
    lateral_dilations: tuple = (1, 1, 2, 2, 4, 4, 8, 8, 1, 1)
    anticausal_depth_dilations: tuple = (1, 4)


@dataclass(frozen=True)
class DecoderConfig:
    latent_channels: int = 32
    n_analytic: int = 22
    width: int = 96
    n_hidden: int = 3
    gm_clamp: float = 2.5
    additive_scale: float = 0.05
    additive_bias_init: float = -4.0
    farfield_width: int = 32


@dataclass(frozen=True)
class LossConfig:
    rel_floor: float = 0.005
    lambda_log: float = 0.3
    lambda_grad: float = 0.1
    log_eps: float = 0.001
    penumbra_band_mm: float = 10.0
    penumbra_weight: float = 2.0
    jitter_weight: float = 0.7


@dataclass(frozen=True)
class SamplingConfig:
    n_queries_per_cp: int = 8192
    frac_in_field: float = 0.35
    frac_penumbra: float = 0.25
    frac_buildup: float = 0.1
    frac_interface: float = 0.1
    frac_out_of_field: float = 0.2
    frac_jitter: float = 0.2
    buildup_depth_mm: float = 20.0
    interface_grad_threshold_per_mm: float = 0.15
    in_field_isodose: float = 0.5
    out_of_field_isodose: float = 0.1


@dataclass(frozen=True)
class PrismXConfig:
    cone: ConeGridSpec = field(default_factory=ConeGridSpec)
    mlc: MLCSpec = field(default_factory=MLCSpec)
    raster: ApertureRasterSpec = field(default_factory=ApertureRasterSpec)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    sad_mm: float = SAD_MM


def test_config() -> PrismXConfig:
    """Small configuration for fast CPU tests."""
    return PrismXConfig(
        cone=ConeGridSpec(
            nu=32, nv=32, du_mm=5.0, dv_mm=5.0, dd_mm=4.0, max_depth_steps=48
        ),
        raster=ApertureRasterSpec(spacing_mm=2.0, half_extent_mm=80.0),
        mlc=MLCSpec(n_pairs=32, leaf_width_mm=5.0),
    )


def scaled_prismx_config(cfg: PrismXConfig, width_scale: float) -> PrismXConfig:
    """``PrismXConfig`` with encoder/decoder widths scaled by ``width_scale``."""
    import math as _math

    scale = float(width_scale)
    if scale == 1.0:
        return cfg
    if not _math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"width_scale must be finite and positive, got {width_scale!r}"
        )

    def _w(value: int) -> int:
        return max(8, int(round(value * scale / 8.0)) * 8)

    enc, dec = (cfg.encoder, cfg.decoder)
    enc_width = _w(enc.width)
    return replace(
        cfg,
        encoder=replace(
            enc,
            width=enc_width,
            glu_hidden=2 * enc_width,
            latent_channels=_w(enc.latent_channels),
        ),
        decoder=replace(
            dec,
            latent_channels=_w(dec.latent_channels),
            width=_w(dec.width),
            farfield_width=_w(dec.farfield_width),
        ),
    )


def synthrad_config(raster_spacing_mm: float = 0.5) -> PrismXConfig:
    """Configuration matched to the photon training simulations."""
    return PrismXConfig(
        raster=ApertureRasterSpec(
            spacing_mm=raster_spacing_mm,
            sigma_focal_mm=3.0,
            leaf_transmission=0.0,
            interleaf_transmission=0.0,
            jaw_transmission=0.0,
            tip_softening_mm=1.0,
        )
    )
