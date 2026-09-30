"""Small source-only checks; no downloads, patient data or pretrained weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from prismx.config import DECODER_FEATURES, ENCODER_CHANNELS, synthrad_config
from prismx.data import split_items_by_case
from prismx.decoder import MainDecoder, compose_dose
from prismx.encoder import TransportEncoder
from prismx.inference import _apply_minimum_cutoff_, _iter_items
from prismx.mr_density import ResEncDensityNet

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ReleaseTests(unittest.TestCase):
    def test_photon_forward_and_backward(self):
        torch.manual_seed(0)
        cfg = synthrad_config()
        self.assertEqual((len(ENCODER_CHANNELS), len(DECODER_FEATURES)), (19, 22))
        encoder = TransportEncoder(cfg.encoder)
        decoder = MainDecoder(cfg.decoder)
        h = encoder(torch.randn(1, 19, 8, 5, 5))
        self.assertEqual(tuple(h.shape), (1, 32, 8, 5, 5))
        latent = h[0, :, :, 2, 2].T
        gm, ga = decoder(latent, torch.randn(8, 22))
        dose = compose_dose(torch.ones(8), torch.ones(8), torch.ones(8), gm, ga, cfg.decoder)
        self.assertTrue(torch.isfinite(dose).all())
        self.assertTrue((dose >= 0).all())
        (dose - 2).square().mean().backward()
        gradients = [p.grad for p in decoder.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))

    def test_mr_density_forward_and_backward(self):
        net = ResEncDensityNet(base_channels=8, levels=2)
        density = net(torch.randn(1, 1, 9, 11, 13))
        self.assertEqual(tuple(density.shape), (1, 1, 9, 11, 13))
        self.assertTrue(torch.isfinite(density).all())
        self.assertTrue((density >= 0).all())
        density.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_cutoff_boundary(self):
        values = torch.tensor([0., 2e-7, 3e-7, 4e-7])
        _apply_minimum_cutoff_(values, 3e-7)
        self.assertTrue(torch.equal(values[:3], torch.zeros(3)))
        self.assertGreater(float(values[3]), 3e-7)

    def test_unknown_modality(self):
        with self.assertRaises(ValueError):
            _iter_items('unknown', [])

    def test_patient_split_and_recipe(self):
        train, val = split_items_by_case([(0, 0, 0), (1, 0, 0)], ['train', 'val'], 'val')
        self.assertEqual(train, [(0, 0, 0)])
        self.assertEqual(val, [(1, 0, 0)])
        recipe = json.loads((ROOT / 'configs/training/photon.json').read_text())
        split = json.loads((ROOT / 'configs/splits.json').read_text())
        self.assertEqual(recipe['args']['val_case'].split(','), split['photon']['validation'])
        self.assertFalse(recipe['requires_initialization'])

    def test_cli_and_dry_run(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        for entry in ('run', 'train', 'scripts.train_stage0', 'scripts.train_proton', 'scripts.train_mr_density'):
            result = subprocess.run([sys.executable, '-m', entry, '--help'], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'not-created'
            result = subprocess.run([sys.executable, 'train.py', 'photon', '--data', directory,
                                     '--out', str(out)], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--steps 60000', result.stdout)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
