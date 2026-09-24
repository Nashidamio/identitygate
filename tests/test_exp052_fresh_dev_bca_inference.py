import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exp052_fresh_dev_bca_inference as m


class TestExp052(unittest.TestCase):
    def test_constant_bca_is_degenerate(self):
        observed = 0.125
        bootstrap = np.full(1000, observed, dtype=np.float64)
        jackknife = np.full(10, observed, dtype=np.float64)
        out = m.bca_interval(observed, bootstrap, jackknife, 0.95)
        self.assertEqual(out["ci_lower"], observed)
        self.assertEqual(out["ci_upper"], observed)
        self.assertAlmostEqual(out["bias_correction_z0"], 0.0)
        self.assertEqual(out["acceleration"], 0.0)

    def test_pooled_delta_uses_event_weighting(self):
        counts = np.asarray([1, 9], dtype=np.int64)
        a = np.asarray([1, 0], dtype=np.int64)
        b = np.asarray([0, 1], dtype=np.int64)
        self.assertAlmostEqual(m.pooled_delta(counts, a, b), 0.0)

    def test_cluster_bootstrap_duplicates_whole_video(self):
        counts = np.asarray([2, 3], dtype=np.int64)
        a = np.asarray([2, 0], dtype=np.int64)
        b = np.asarray([0, 3], dtype=np.int64)
        draws = np.asarray([[0, 0], [1, 1], [0, 1]], dtype=np.int64)
        out = m.bootstrap_delta(draws, counts, a, b)
        self.assertAlmostEqual(out[0], 1.0)
        self.assertAlmostEqual(out[1], -1.0)
        self.assertAlmostEqual(out[2], -0.2)

    def test_primary_interpretation_rules(self):
        self.assertEqual(
            m.interpret_primary(0.01, 0.05, 0.08),
            "SUPPORTED_POSITIVE_BELOW_PRACTICAL_THRESHOLD",
        )
        self.assertEqual(
            m.interpret_primary(-0.02, 0.06, 0.08),
            "NO_SUPPORTED_POSITIVE_AND_PRACTICAL_THRESHOLD_RULED_OUT",
        )
        self.assertEqual(
            m.interpret_primary(-0.02, 0.10, 0.08),
            "INCONCLUSIVE_BETWEEN_ZERO_AND_PRACTICAL_THRESHOLD",
        )


if __name__ == "__main__":
    unittest.main()
