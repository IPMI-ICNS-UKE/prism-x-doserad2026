"""Picklable per-CP CPU prep for the prefetched Stage-0 training pipeline."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from prismx.aperture import ApertureRaster, MLCAperture, build_raster
from prismx.cone_grid import ConeGrid, build_cone_grid
from prismx.config import MLCSpec, PrismXConfig
from prismx.features import encoder_channels
from prismx.geometry import BeamGeometry


def _beam_descriptor(beam: BeamGeometry) -> dict:
    """Pure-python descriptor of a :class:`BeamGeometry` (picklable)."""
    return {
        "gantry_deg": float(beam.gantry_deg),
        "collimator_deg": float(beam.collimator_deg),
        "couch_deg": float(beam.couch_deg),
        "isocenter_mm": tuple((float(v) for v in beam.isocenter_mm)),
        "sad_mm": float(beam.sad_mm),
    }


def _beam_from_descriptor(d: dict) -> BeamGeometry:
    return BeamGeometry(
        gantry_deg=d["gantry_deg"],
        collimator_deg=d["collimator_deg"],
        couch_deg=d["couch_deg"],
        isocenter_mm=tuple(d["isocenter_mm"]),
        sad_mm=d["sad_mm"],
    )


def _aperture_descriptor(ap: MLCAperture) -> dict:
    """Picklable descriptor of an :class:`MLCAperture` (CPU bank tensors)."""
    return {
        "left_mm": ap.left_mm.detach().cpu().clone(),
        "right_mm": ap.right_mm.detach().cpu().clone(),
        "jaw_u_mm": tuple((float(v) for v in ap.jaw_u_mm)),
        "jaw_v_mm": tuple((float(v) for v in ap.jaw_v_mm)),
        "n_pairs": int(ap.mlc.n_pairs),
        "leaf_width_mm": float(ap.mlc.leaf_width_mm),
    }


def _aperture_from_descriptor(d: dict) -> MLCAperture:
    return MLCAperture(
        left_mm=d["left_mm"].clone(),
        right_mm=d["right_mm"].clone(),
        jaw_u_mm=tuple(d["jaw_u_mm"]),
        jaw_v_mm=tuple(d["jaw_v_mm"]),
        mlc=MLCSpec(n_pairs=d["n_pairs"], leaf_width_mm=d["leaf_width_mm"]),
    )


def _cone_fields(cone: ConeGrid) -> dict:
    """CPU copies of the :class:`ConeGrid` arrays + the scalar window placement."""
    return {
        "d_start": float(cone.d_start),
        "n_depth": int(cone.n_depth),
        "rho": cone.rho.detach().cpu().clone(),
        "wed": cone.wed.detach().cpu().clone(),
        "depth_below_surface": cone.depth_below_surface.detach().cpu().clone(),
        "ssd": cone.ssd.detach().cpu().clone(),
    }


def _cone_from_fields(
    fields: dict, spec, beam: BeamGeometry, device: torch.device
) -> ConeGrid:
    return ConeGrid(
        spec=spec,
        beam=beam,
        d_start=fields["d_start"],
        n_depth=fields["n_depth"],
        rho=fields["rho"].to(device),
        wed=fields["wed"].to(device),
        depth_below_surface=fields["depth_below_surface"].to(device),
        ssd=fields["ssd"].to(device),
    )


def _raster_fields(raster: ApertureRaster) -> dict:
    """CPU copies of every :class:`ApertureRaster` tensor field."""
    return {
        "psi_sharp": raster.psi_sharp.detach().cpu().clone(),
        "psi_blur": raster.psi_blur.detach().cpu().clone(),
        "sdf": raster.sdf.detach().cpu().clone(),
        "blurs": {
            float(sig): val.detach().cpu().clone() for sig, val in raster.blurs.items()
        },
        "sd_u": raster.sd_u.detach().cpu().clone(),
        "sd_v": raster.sd_v.detach().cpu().clone(),
    }


def _raster_from_fields(
    fields: dict, spec, aperture: MLCAperture, device: torch.device
) -> ApertureRaster:
    return ApertureRaster(
        spec=spec,
        aperture=aperture,
        psi_sharp=fields["psi_sharp"].to(device),
        psi_blur=fields["psi_blur"].to(device),
        sdf=fields["sdf"].to(device),
        blurs={sig: val.to(device) for sig, val in fields["blurs"].items()},
        sd_u=fields["sd_u"].to(device),
        sd_v=fields["sd_v"].to(device),
    )


@dataclass
class ReconstructedCP:
    """Device-side reconstruction of one prepped control point."""

    cone: ConeGrid
    raster: ApertureRaster
    aperture: MLCAperture
    beam: BeamGeometry
    channels: torch.Tensor
    dose: torch.Tensor
    d_max: float
    grid_origin_mm: tuple
    grid_spacing_mm: tuple
    item: tuple


def prepare_cp(cp, cfg: PrismXConfig, dose_scale: float) -> dict:
    """CPU-only prep for one control point; returns a picklable dict."""
    ct = cp.ct
    beam = cp.beam
    ap = cp.aperture
    cone = build_cone_grid(ct, beam, cfg.cone)
    raster = build_raster(ap, cfg.raster)
    channels = encoder_channels(cone, raster, ap)
    scale = float(dose_scale)
    dose = (cp.dose / scale).to(torch.float32).contiguous()
    d_max = float(dose.max())
    return {
        "item": (str(cp.case_id), int(cp.beam_idx), int(cp.cp_idx)),
        "channels": channels.detach().cpu().clone(),
        "cone_fields": _cone_fields(cone),
        "raster_fields": _raster_fields(raster),
        "beam": _beam_descriptor(beam),
        "aperture": _aperture_descriptor(ap),
        "dose": dose.detach().cpu().clone(),
        "d_max": d_max,
        "grid_origin_mm": tuple((float(o) for o in cp.grid_origin_mm)),
        "grid_spacing_mm": tuple((float(s) for s in cp.grid_spacing_mm)),
    }


def reconstruct_on_device(prep: dict, cfg: PrismXConfig, device) -> ReconstructedCP:
    """Rebuild a prepped CP on ``device`` for the encoder forward + query."""
    device = torch.device(device)
    beam = _beam_from_descriptor(prep["beam"])
    aperture = _aperture_from_descriptor(prep["aperture"])
    cone = _cone_from_fields(prep["cone_fields"], cfg.cone, beam, device)
    raster = _raster_from_fields(prep["raster_fields"], cfg.raster, aperture, device)
    channels = prep["channels"].to(device)
    dose = prep["dose"].to(device)
    return ReconstructedCP(
        cone=cone,
        raster=raster,
        aperture=aperture,
        beam=beam,
        channels=channels,
        dose=dose,
        d_max=float(prep["d_max"]),
        grid_origin_mm=tuple(prep["grid_origin_mm"]),
        grid_spacing_mm=tuple(prep["grid_spacing_mm"]),
        item=prep["item"],
    )


def encode_from_channels(model, recon: ReconstructedCP):
    """Run the encoder forward on prepped channels -> a ``CPState``."""
    from prismx.model import CPState

    cone = recon.cone
    ap = recon.aperture
    x = recon.channels
    su, sv = model._encoder_box(cone, ap)
    h_box = model.encoder(x[:, :, su, sv].unsqueeze(0)).squeeze(0)
    h = torch.zeros(
        model.cfg.encoder.latent_channels,
        cone.n_depth,
        cone.spec.nu,
        cone.spec.nv,
        dtype=h_box.dtype,
        device=h_box.device,
    )
    h[:, :, su, sv] = h_box
    return CPState(cone=cone, raster=recon.raster, aperture=ap, h=h)
