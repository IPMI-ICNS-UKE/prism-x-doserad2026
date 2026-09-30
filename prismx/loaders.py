"""Load the tensor-only release checkpoints."""

from pathlib import Path
import torch

from prismx.analytic import PDDTable
from prismx.config import synthrad_config
from prismx.model import PrismX
from prismx.mr_density import ResEncDensityNet


def load_frozen_engine(checkpoint: Path, pdd: Path, device) -> PrismX:
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    if float(checkpoint.get("args", {}).get("width_scale", 1.0)) != 1.0:
        raise ValueError("Expected the full-width Challenge engine")
    if not Path(pdd).is_file():
        raise FileNotFoundError(pdd)
    engine = PrismX(synthrad_config()).to(device)
    engine.load_state_dict(checkpoint["ema"], strict=True)
    engine.pdd = PDDTable.from_npz(pdd)
    return engine.eval().requires_grad_(False)


def load_density_net(checkpoint: Path, device):
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    state = checkpoint["ema"]
    base = int(state["stem.weight"].shape[0])
    levels = sum(k.startswith("downs.") and k.endswith(".weight") for k in state)
    if (base, levels) != (32, 4):
        raise ValueError("Expected the submitted ResEnc density network (32, 4)")
    net = ResEncDensityNet(base_channels=base, levels=levels).to(device)
    net.load_state_dict(state, strict=True)
    return net.eval().requires_grad_(False)


def load_proton_engine(checkpoint: Path, bragg: Path, device):
    """Load the analytical Proton engine with its paired lateral-spread model."""
    from prismx.config import PrismXConfig
    from prismx.proton_analytic import BraggIDD
    from prismx.proton_model import PrismP

    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    sigma = checkpoint["sigma_params_used"]
    engine = PrismP(
        bragg=BraggIDD.from_npz(bragg),
        cfg=PrismXConfig(),
        sigma0_mm=float(sigma[0]),
        sigma_growth=(float(sigma[1]), float(sigma[2])),
        halo=None,
    ).to(device)
    engine.load_state_dict(checkpoint["ema"], strict=True)
    return engine.eval().requires_grad_(False)
