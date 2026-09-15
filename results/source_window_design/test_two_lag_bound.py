"""Checks of validation moments, correction direction, ranges and ordering."""
from fractions import Fraction as F
from itertools import product
from math import isfinite, log, sqrt
import unittest
import warnings

try:
    import numpy as np
except ImportError:
    np = None

from two_lag_bound import paired_lambda_samples, lower_bound


class BoundTests(unittest.TestCase):
    def test_raw_enumeration_validation_mean(self):
        for support, expected in [((0, 1), F(3, 64)), ((0, 1, 2), F(14, 243))]:
            raw = [[row[:2], row[2:4], row[4:]] for row in product(support, repeat=6)]
            vals = paired_lambda_samples(raw)
            self.assertEqual(sum(map(F, vals))/len(vals), expected)

    def test_signed_extremes(self):
        # Both endpoints occur for a legitimate raw-array comparison kernel.
        vals = paired_lambda_samples([[[1, 1], [0, 4], [3, -2]],
                                      [[1, 1], [0, 0], [3, 3]]])
        self.assertEqual(vals, [-0.5, 0.5])

    def test_large_integer_order(self):
        b = 2**60
        raw = [[[b+1, b+1], [b, b], [b+2, b+2]]]
        self.assertEqual(paired_lambda_samples(raw), [0.5])

    @unittest.skipIf(np is None, 'NumPy is optional for the fixed-width integer regression test.')
    def test_numpy_int64_sum_does_not_overflow(self):
        maximum = np.iinfo(np.int64).max
        raw = np.array([[[maximum, maximum], [0, 0], [maximum, maximum]],
                        [[maximum-1, maximum-1], [0, 0], [maximum, maximum]]], dtype=np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            self.assertEqual(paired_lambda_samples(raw), [0.125, 0.5])

    def test_range_and_bound_overflow_are_rejected(self):
        with self.assertRaises(ValueError):
            lower_bound([0.0], omega=0, coefficient_l1_a=1e308, coefficient_l1_b=1e308)
        with self.assertRaises(ValueError):
            lower_bound([0.0], omega=0, coefficient_l1_a=1e308,
                        coefficient_l1_b=2, error_probability=1e-300)
        with self.assertRaises(ValueError):
            lower_bound([0.0], omega=0, coefficient_l1_a=10**1000, coefficient_l1_b=2)
        with self.assertRaises(ValueError):
            lower_bound([10**1000], omega=0, coefficient_l1_a=2, coefficient_l1_b=2)
        with self.assertRaises(ValueError):
            lower_bound([1e-15], omega=0, coefficient_l1_a=0, coefficient_l1_b=2)

    def test_small_error_budgets_and_large_mean_are_finite(self):
        tiny = float.fromhex('0x0.0000000000001p-1022')
        result = lower_bound([0.0], omega=0, coefficient_l1_a=2,
                             coefficient_l1_b=2, error_probability=tiny)
        self.assertTrue(isfinite(result['lower_bound']))
        result = lower_bound([0.0], omega=0.5, coefficient_l1_a=2,
                             coefficient_l1_b=2, validation_raw=[[[0, 0]]*3],
                             validation_error_probability=tiny)
        self.assertTrue(isfinite(result['lower_bound']))
        result = lower_bound([5e307]*4, omega=0, coefficient_l1_a=1e308, coefficient_l1_b=2)
        self.assertEqual(result['score_mean'], 5e307)
        self.assertTrue(isfinite(result['lower_bound']))

    def test_validation_free_correction_direction(self):
        args = dict(coefficient_l1_a=2, coefficient_l1_b=2)
        positive = lower_bound([0.25]*1000, omega=0.5, **args)
        negative = lower_bound([0.25]*1000, omega=-0.5, **args)
        zero = lower_bound([0.25]*1000, omega=0, **args)
        self.assertEqual(positive['interaction_upper_bound'], 1/16)
        self.assertEqual(negative['interaction_upper_bound'], 0)
        self.assertEqual(zero['method'], 'zero_overlap')
        self.assertAlmostEqual(negative['lower_bound'], 0.25-2*sqrt(log(20)/2000))

    def test_validation_sign_and_budget(self):
        raw = [[[1, 1], [0, 0], [2, 2]], [[1, 1], [0, 4], [3, -2]]]*10000
        args = dict(coefficient_l1_a=2, coefficient_l1_b=2, validation_raw=raw)
        positive = lower_bound([0.25]*1000, omega=0.5, **args)
        negative = lower_bound([0.25]*1000, omega=-0.5, **args)
        self.assertEqual(positive['validation_count'], 20000)
        self.assertEqual(positive['validation_raw_draws'], 120000)
        self.assertEqual(positive['lambda_interval'][0], 0)
        self.assertAlmostEqual(positive['lambda_interval'][1], sqrt(log(200)/40000))
        self.assertEqual(negative['interaction_upper_bound'], 0)
        self.assertAlmostEqual(positive['evaluation_penalty'], 2*sqrt(log(25)/2000))

    def test_invalid_inputs(self):
        args = dict(omega=0.5, coefficient_l1_a=2, coefficient_l1_b=2)
        for scores in ([], [float('nan')], [2]):
            with self.assertRaises(ValueError):
                lower_bound(scores, **args)
        for raw in ([], [[[1, 2], [3, 4]]], [[[1, 2], [3, 4], [5, float('inf')]]],
                    [[[1e308, 1e308], [0, 0], [1, 1]]]):
            with self.assertRaises(ValueError):
                paired_lambda_samples(raw)
        with self.assertRaises(ValueError):
            lower_bound([0.0], validation_raw=[[[0, 0]]*3],
                        validation_error_probability=0.1, **args)


if __name__ == '__main__':
    unittest.main()
