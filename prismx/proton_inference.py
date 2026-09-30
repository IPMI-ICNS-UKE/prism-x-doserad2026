"""CT/MR inference for the analytical Proton baseline and learned correction."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from prismx import inference as io
from prismx.loaders import load_density_net, load_proton_engine
from prismx.proton_data import beamlet_geometry_from_ray

ROOT = Path(__file__).resolve().parents[1]


def default_cfg(image_type):
    return {
        "engine": str(ROOT / "weights/proton.pt"),
        "bragg": str(ROOT / "commissioning/proton_bragg.npz"),
        "net": str(ROOT / "weights/mr-density.pt"),
    }


class Models:
    def __init__(self, image_type, device, cfg):
        if image_type not in ("ct", "mri"):
            raise ValueError(f"Unknown image type: {image_type}")
        io._verify_shipped_weights(cfg, image_type, "proton")
        self.modality, self.image_type = "proton", image_type
        self.device = torch.device(device)
        self.engine = load_proton_engine(Path(cfg["engine"]), Path(cfg["bragg"]), self.device)
        with np.load(cfg["bragg"]) as table:
            self.dose_scale = float(table["dose_scale"])
        self.net = None
        if image_type == "mri":
            self.net = load_density_net(Path(cfg["net"]), self.device)
            if self.device.type == "cuda":
                self.net.half()


def iter_beamlets(beams):
    for beam in beams:
        for ray in beam["rays"]:
            for beamlet in ray["beamlets"]:
                yield beamlet["output_info"], beam, ray, beamlet


@torch.no_grad()
def beamlet_dose(models, ct, body_mask, entry, beam, ray, beamlet, origin, spacing, shape):
    isocenter = entry.get("iso_center", beam.get("iso_center"))
    if isocenter is None:
        raise ValueError("Proton metadata requires an image-level iso_center")
    geometry = beamlet_geometry_from_ray(
        gantry_deg=float(beam["gantry_angle"]),
        isocenter_mm=tuple(isocenter),
        ray_source=tuple(ray["ray_source"]),
        ray_target=tuple(ray["ray_target"]),
    )
    dose = models.engine.predict_dose_grid(
        ct, geometry.beam, geometry.spot_uv, float(beamlet["energy"]),
        origin, spacing, shape, region="core", radius_mm=40.0, batch_points=131072,
    )
    dose.mul_(models.dose_scale)
    dose.masked_fill_(~body_mask, 0.0)
    return dose


@torch.no_grad()
def run(input_dir, output_dir, models):
    """Emit dose maps in the ten-slot DoseRAD format without reference doses."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    meta = json.loads((input_dir / "stacked-proton-beam-level-metadata.json").read_text())
    plan = io._routing_plan(meta, "proton")
    used_slots = {slot for _, groups in plan for slot in groups}
    active_writer = None
    count, started = 0, time.perf_counter()
    try:
        for entry, groups in plan:
            image_index = int(entry["image_file_idx"])
            folder = input_dir / "images" / f"radiation-dose-calculation-source-{models.image_type}-image-{image_index + 1}"
            arr, spacing, origin, reference = io._read_image(io._find_mha(folder))
            ct = io._ctvolume_for_image(models, arr, spacing, origin)
            body_mask = io._filled_body_voxel_mask(ct.density)
            for slot, items in sorted(groups.items()):
                target = output_dir / "images" / f"stacked-radiation-dose-map-{slot + 1}" / "output.mha"
                active_writer = io._StreamingSlotWriter(target, reference, arr.shape, len(items))
                for output_info, beam, ray, beamlet in items:
                    dose = beamlet_dose(models, ct, body_mask, entry, beam, ray, beamlet, origin, spacing, arr.shape)
                    io._apply_minimum_cutoff_(dose, float(output_info["minimum_cutoff"]))
                    active_writer.write_tensor(int(output_info["idx_in_output"]), dose)
                    count += 1
                active_writer.finish()
                active_writer = None
            del ct, body_mask, arr
        io._write_placeholders(output_dir, used_slots)
    finally:
        if active_writer is not None:
            active_writer.abort()
    print(f"[proton] wrote {count} dose maps in {time.perf_counter() - started:.3f}s", flush=True)
