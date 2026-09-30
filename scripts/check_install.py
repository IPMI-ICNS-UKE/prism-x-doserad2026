"""CPU shape/finite-value checks on synthetic inputs; not an accuracy benchmark."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from prismx.loaders import load_frozen_engine, load_density_net, load_proton_engine

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    engine = load_frozen_engine(
        ROOT / "weights/photon.pt",
        ROOT / "commissioning/synthrad_photon_pdd.npz",
        "cpu",
    )
    h = engine.encoder(torch.zeros(1, 19, 12, 5, 5))
    assert h.shape == (1, 32, 12, 5, 5)
    assert torch.isfinite(h).all()
    gm, ga = engine.main_decoder(torch.zeros(8, 32), torch.zeros(8, 22))
    assert gm.shape == ga.shape == (8,)
    assert torch.isfinite(gm).all() and torch.isfinite(ga).all()
    proton = load_proton_engine(ROOT / "weights/proton.pt", ROOT / "commissioning/proton_bragg.npz", "cpu")
    assert proton.n_parameters() == 95268
    ph = proton.encoder(torch.zeros(1, 13, 12, 5, 5))
    assert ph.shape == (1, 32, 12, 5, 5) and torch.isfinite(ph).all()
    pm, pa = proton.main_decoder(torch.zeros(8, 32), torch.zeros(8, 15))
    assert torch.isfinite(pm).all() and torch.isfinite(pa).all()
    net = load_density_net(ROOT / "weights/mr-density.pt", "cpu")
    density = net(torch.zeros(1, 1, 16, 16, 16))
    assert density.shape == (1, 1, 16, 16, 16)
    assert torch.isfinite(density).all() and (density >= 0).all()
    print(
        "PASS: checkpoint loading, Photon/Proton TCN/MLP, MR density forward (synthetic CPU inputs)"
    )


if __name__ == "__main__":
    main()
