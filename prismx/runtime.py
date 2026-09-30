"""Task selection for the Photon and analytical Proton engines."""
from prismx import inference as photon


def _default_cfg(modality, image_type):
    if modality == "photon":
        return photon._default_cfg(modality, image_type)
    if modality == "proton":
        from prismx.proton_inference import default_cfg
        return default_cfg(image_type)
    raise ValueError(f"Unknown modality: {modality}")


def Models(modality, image_type, device, cfg):
    if modality == "photon":
        return photon.Models(modality, image_type, device, cfg)
    if modality == "proton":
        from prismx.proton_inference import Models as ProtonModels
        return ProtonModels(image_type, device, cfg)
    raise ValueError(f"Unknown modality: {modality}")


def run(input_dir, output_dir, models):
    if models.modality == "photon":
        return photon.run(input_dir, output_dir, models)
    if models.modality == "proton":
        from prismx.proton_inference import run as run_proton
        return run_proton(input_dir, output_dir, models)
    raise ValueError(f"Unknown modality: {models.modality}")
