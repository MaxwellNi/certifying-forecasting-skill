"""Exact-equivalence correction must not erase genuine sub-tolerance effects."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "correct_numeric_summaries.py"
spec = importlib.util.spec_from_file_location("numeric_summaries", path)
correction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correction)


class NumericalSummaryChecks(unittest.TestCase):
    def test_zero_weight_forecast_is_exactly_neutral(self):
        baseline = np.array([0.1, 0.5, 0.9, 0.2])
        candidate = np.array([0.9, 0.1, 0.3, 0.8])
        target = np.array([0.2, 0.8, 0.3, np.nan])
        augmentation = baseline + 0.0 * (candidate - baseline)
        self.assertTrue(np.array_equal(augmentation, baseline))
        self.assertEqual(correction.scalar_loss(augmentation, target), correction.scalar_loss(baseline, target))
        self.assertEqual(correction.paired_gain(augmentation, baseline, target), 0.0)
        optimum, index = correction.hindsight_forecast(baseline, augmentation[:, None], [False], target)
        self.assertEqual(index, -1)
        self.assertEqual(correction.paired_gain(optimum, augmentation, target), 0.0)

    def test_real_gain_smaller_than_mean_loss_rounding_is_preserved(self):
        baseline = np.full(128, 0.5)
        candidate = baseline.copy()
        candidate[0] = np.nextafter(0.5, 0.0)
        target = np.zeros(128)
        self.assertFalse(np.array_equal(candidate, baseline))
        self.assertEqual(correction.scalar_loss(candidate, target), correction.scalar_loss(baseline, target))
        # This real representable forecast difference has gain about 4e-19;
        # neither rounded scalar-mean subtraction nor an epsilon rule can keep it.
        gain = correction.paired_gain(candidate, baseline, target)
        self.assertGreater(gain, 0.0)
        self.assertLess(gain, 1e-16)
        self.assertEqual(correction.paired_gain(baseline, candidate, target), -gain)
        optimum, index = correction.hindsight_forecast(baseline, candidate[:, None], [False], target)
        self.assertEqual(index, 0)
        self.assertEqual(correction.paired_gain(optimum, candidate, target), 0.0)
        self.assertGreater(correction.paired_gain(optimum, baseline, target), 0.0)


if __name__ == "__main__":
    unittest.main()
