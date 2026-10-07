"""Small metric checks; no patient data, weights or CUDA needed."""
import unittest
import numpy as np
from prismx.report_evaluation import frame_metrics, r80


class ReportMetricsTest(unittest.TestCase):
    def test_exact_frame(self):
        a = np.array([[[0, .1, .5, 1]]], np.float32)
        self.assertEqual(frame_metrics(a, a), 0.0)

    def test_known_relative_error(self):
        a = np.array([[[0, .01, .5, 1]]], np.float32)
        self.assertAlmostEqual(frame_metrics(a * 1.1, a), .075, places=6)

    def test_invalid_output(self):
        with self.assertRaises(ValueError):
            frame_metrics(np.array([[[-1.]]]), np.ones((1, 1, 1)))

    def test_range_and_undefined_falloff(self):
        w = np.repeat(np.arange(10, dtype=np.float32), 2)
        d = np.repeat(np.array([.1, .2, .3, .4, .5, 1, .9, .7, .4, .1]), 2)
        self.assertEqual(r80(w, d), 7.5)
        self.assertIsNone(r80(w, np.ones_like(w)))


if __name__ == "__main__":
    unittest.main()
