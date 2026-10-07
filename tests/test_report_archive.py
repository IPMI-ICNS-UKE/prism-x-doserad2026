"""Consistency of the report archive and matched-output scoring protocol."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/report"
spec = importlib.util.spec_from_file_location("matched_score", HERE / "score_matched_output.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


class ReportArchiveTest(unittest.TestCase):
    def test_range_archive_matches_results(self):
        archive = json.loads((HERE / "r80-beamlets.json").read_text())
        results = json.loads((HERE / "release-results.json").read_text())["results"]
        for run in archive["runs"]:
            rows = [dict(zip(archive["columns"], row)) for row in run["rows"]]
            expected = np.linspace(0, run["number_of_segments"] - 1, 150, dtype=int)
            self.assertEqual([row["segment"] for row in rows], expected.tolist())
            self.assertEqual(len({(r["beam"], r["ray"], r["layer"]) for r in rows}), 150)
            summary = next(r for r in results if (r["task"], r["case"]) ==
                           (run["task"], run["case"]))
            self.assertEqual(run["build_sha256"], summary["build_sha256"])
            self.assertEqual(run["plan_sha256"], summary["plan_sha256"])
            measured = scorer.range_metrics([{"range": row} for row in rows])
            for key, value in measured.items():
                self.assertAlmostEqual(value, summary[key], places=12)

    def test_protocol_and_commissioning_hashes(self):
        protocol = json.loads((HERE / "matched-output-evaluation.json").read_text())
        result = json.loads((HERE / "release-results.json").read_text())
        self.assertEqual(protocol, result["protocol"])
        original = HERE / protocol["original_build_protocol"]
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                         protocol["original_build_protocol_sha256"])
        for record in json.loads((HERE / "commissioning.json").read_text())["shape_fitting"].values():
            self.assertEqual(hashlib.sha256((ROOT / record["file"]).read_bytes()).hexdigest(),
                             record["sha256"])

    def test_cutoff_before_summation(self):
        cutoff = np.float32(3e-7)
        dose = np.array([0, cutoff / 2, cutoff, np.nextafter(cutoff, np.float32(1))], np.float32)
        expected = np.array([0, 0, 0, dose[-1]], np.float32)
        np.testing.assert_array_equal(scorer.threshold_segment(dose, cutoff), expected)
        self.assertEqual(float((2 * scorer.threshold_segment(dose, cutoff))[2]), 0.)
        self.assertGreater(float(scorer.threshold_segment(2 * dose, cutoff)[2]), 0.)

    def test_gamma_settings_and_infinite_failures(self):
        settings = json.loads((HERE / "matched-output-evaluation.json").read_text())["gamma"]
        dose = np.array([[[0., .2, .5, 1.]]])
        gamma = np.array([[[np.nan, np.inf, .5, 1.5]]])
        with patch.object(scorer.pymedphys, "gamma", return_value=gamma) as mocked:
            result = scorer.metrics(dose, dose, (2, 2, 2), settings)
        self.assertEqual(result["gamma_voxels"], 3)
        self.assertAlmostEqual(result["gamma_fraction"], 1 / 3)
        self.assertTrue(mocked.call_args.kwargs["local_gamma"])
        self.assertEqual(mocked.call_args.kwargs["interp_fraction"], 10)
        self.assertEqual(mocked.call_args.kwargs["max_gamma"], 2)
        self.assertEqual(result["mae_bands"], {"high": 0., "mid": 0., "low": 0.})


if __name__ == "__main__":
    unittest.main()
