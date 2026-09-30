"""Proton training, routing and output checks on synthetic data."""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import SimpleITK as sitk
import torch

from prismx.geometry import BeamGeometry
from prismx.inference import _routing_plan
from prismx.phantoms import water_box
from prismx.proton_data import ProtonBeamlet
from prismx.proton_grid import ProtonGridSpec
from prismx.proton_model import PrismP
from prismx.proton_inference import run
from scripts import train_proton as trainer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def small_engine():
    return PrismP(grid_spec=ProtonGridSpec(nu=9, nv=9, max_depth_steps=32))


class ProtonReleaseTests(unittest.TestCase):
    def test_two_training_updates(self):
        torch.manual_seed(7)
        model = small_engine()
        self.assertEqual(model.n_parameters(), 95268)
        ct = water_box((24., 24., 24.), spacing_mm=2.)
        beam = BeamGeometry(gantry_deg=0., isocenter_mm=(0., 0., 0.))
        with torch.no_grad():
            target = 1.2 * model.predict_dose_grid(
                ct, beam, (0., 0.), 70., ct.origin_mm, ct.spacing_mm, ct.density.shape,
            )
        self.assertGreater(float(target.max()), 0.)
        bl = ProtonBeamlet(target, target.abs() * .03 + .001, ct.origin_mm,
                           ct.spacing_mm, beam, (0., 0.), 70., ct, "synthetic", 0, 0, 0)
        opt = torch.optim.AdamW(model.parameters(), lr=.002, betas=(.9, .95), weight_decay=.01)
        ema = trainer.EMA(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        args = argparse.Namespace(queries=64, lambda_idd=.5)
        rng = np.random.default_rng(1)
        for _ in range(2):
            loss, *_ = trainer.run_train_step(model, opt, ema, ct, bl, float(target.max()),
                                              model.grid_spec, args, rng)
            self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(torch.isfinite(v).all() for v in ema.shadow.values()))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(file, model, ema, opt, 2, args, loss)
            saved = torch.load(file, weights_only=True)
            self.assertEqual(saved["step"], 2)
            self.assertTrue(saved["optimizer"]["state"])

    def test_recipe_and_sigma_pairing(self):
        recipe = json.loads((ROOT / "configs/training/proton.json").read_text())
        split = json.loads((ROOT / "configs/splits.json").read_text())
        self.assertEqual(recipe["args"]["val_case"].split(","), split["proton"]["validation"])
        self.assertEqual(recipe["args"]["steps"], 50000)
        model = trainer.build_model(argparse.Namespace(bragg=ROOT / "commissioning/proton_bragg.npz"), torch.device("cpu"))
        self.assertAlmostEqual(model.sigma0_mm, 4.950340270996094)
        self.assertIsNone(model.sigma_growth)
        self.assertIsNone(model.halo)

    def test_image_level_geometry_and_output(self):
        meta = [{"image_file_idx": 0, "anatomical_region": "abdominal", "iso_center": [0., 0., 0.],
                 "beams": [{"gantry_angle": 0., "rays": [
                     {"ray_source": [0., -1000., 0.], "ray_target": [0., 0., 0.], "beamlets": [
                         {"energy": 70., "output_info": {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 3e-7}}
                     ]}]}]}]
        self.assertEqual(len(_routing_plan(meta, "proton")[0][1][0]), 1)
        model = small_engine().eval().requires_grad_(False)
        models = SimpleNamespace(modality="proton", image_type="ct", device=torch.device("cpu"),
                                 engine=model, dose_scale=.001, net=None)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input", Path(directory) / "output"
            folder = source / "images/radiation-dose-calculation-source-ct-image-1"
            folder.mkdir(parents=True)
            image = sitk.GetImageFromArray(np.zeros((12, 12, 12), dtype=np.float32))
            image.SetSpacing((2., 2., 2.))
            image.SetOrigin((-11., -11., -11.))
            sitk.WriteImage(image, str(folder / "source.mha"))
            (source / "stacked-proton-beam-level-metadata.json").write_text(json.dumps(meta))
            run(source, output, models)
            result = sitk.ReadImage(str(output / "images/stacked-radiation-dose-map-1/output.mha"))
            dose = sitk.GetArrayFromImage(result)
            self.assertEqual(dose.shape, (1, 12, 12, 12))
            self.assertEqual(result.GetOrigin()[:3], image.GetOrigin())
            self.assertEqual(result.GetSpacing()[:3], image.GetSpacing())
            self.assertTrue(np.isfinite(dose).all())
            self.assertTrue(np.any(dose > 3e-7))
            self.assertTrue(np.all((dose == 0.) | (dose > 3e-7)))
            self.assertEqual(len(list(output.glob("images/*/output.mha"))), 10)


if __name__ == "__main__":
    unittest.main()
