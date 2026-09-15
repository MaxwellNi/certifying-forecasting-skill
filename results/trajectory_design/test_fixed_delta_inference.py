"""Independent high-precision inversion and difficult numerical boundaries."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import unittest

import numpy as np

from fixed_delta_inference import fixed_delta_hoeffding_pvalue, family_threshold
from trajectory_kernel import corrected_lower


class FixedDeltaInferenceTests(unittest.TestCase):
    def test_against_independent_high_precision_formula(self):
        for M in (30, 100, 300):
            scores, validation = [.25] * M, [0.] * 400
            result = fixed_delta_hoeffding_pvalue(scores, validation, [[1]], [[1]], delta=.01)
            with localcontext() as ctx:
                ctx.prec = 180
                delta = Decimal.from_float(.01)
                margin = Decimal('.25') - (-delta.ln()/Decimal(800)).sqrt()
                expected = min(Decimal(1), delta + (-Decimal(2*M)*margin*margin/Decimal('.25')).exp())
            actual = Decimal.from_float(result['p_value'])
            self.assertGreaterEqual(actual, expected)
            self.assertLessEqual(float(actual - expected), 2*math.ulp(result['p_value']))

    def test_extended_float_input_not_silently_narrowed(self):
        # A long-double input is interpreted as supplied, before any conversion
        # to binary64. The outward final result must enclose its exact formula.
        value = np.longdouble('0.1000000000000000001')
        result = fixed_delta_hoeffding_pvalue([value]*100, None, [[1,0]], [[0,1]], delta=0)
        numerator, denominator = value.as_integer_ratio()
        with localcontext() as ctx:
            ctx.prec = 180
            exact = Decimal(numerator)/Decimal(denominator)
            expected = (-Decimal(200)*exact*exact/Decimal('.25')).exp()
        self.assertGreaterEqual(Decimal.from_float(result['p_value']), expected)

    def test_matches_strict_lower_bound_inversion_away_from_threshold(self):
        for signal in (.1, .2, .25):
            for M in (30, 100, 300):
                scores, validation = [signal]*M, [0.]*400
                result = fixed_delta_hoeffding_pvalue(scores, validation, [[1]], [[1]], delta=.01)
                for alpha in (.015, .05, .1, .3):
                    lower = corrected_lower(scores, validation, [[1]], [[1]], alpha=alpha, delta=.01, rule='hoeffding')
                    self.assertEqual(result['p_value'] < alpha, lower['lower'] > 0)

    def test_smallest_delta_no_overflow_or_underflow(self):
        tiny = math.nextafter(0., 1.)
        out = fixed_delta_hoeffding_pvalue([.25]*1000, [0.]*10000, [[1]], [[1]], delta=tiny)
        self.assertTrue(math.isfinite(out['p_value']))
        self.assertGreater(out['p_value'], tiny)
        self.assertGreater(float(out['validation_radius_upper_decimal']), 0)

    def test_zero_interaction_ignores_no_validation(self):
        result = fixed_delta_hoeffding_pvalue([.25]*100, None, [[1,0]], [[0,1]], delta=0)
        self.assertEqual(result['effective_delta'], 0)
        self.assertEqual(result['validation_count'], 0)
        with localcontext() as ctx:
            ctx.prec = 160
            expected = Decimal(-50).exp()
        self.assertGreaterEqual(Decimal.from_float(result['p_value']), expected)

    def test_tail_never_silently_rounds_to_zero(self):
        result = fixed_delta_hoeffding_pvalue([.25]*3000, None, [[1,0]], [[0,1]], delta=0)
        self.assertEqual(result['p_value'], math.nextafter(0., 1.))

    def test_zero_range_and_nonpositive_signal(self):
        self.assertEqual(fixed_delta_hoeffding_pvalue([0.], None, [[0]], [[1]], delta=0)['p_value'], 1.)
        self.assertEqual(fixed_delta_hoeffding_pvalue([0.]*10, [0.]*20, [[1]], [[1]])['p_value'], 1.)
        self.assertEqual(fixed_delta_hoeffding_pvalue([-.25]*10, None, [[1,0]], [[0,1]])['p_value'], 1.)

    def test_signed_interaction_can_raise_target_center(self):
        result = fixed_delta_hoeffding_pvalue([0.]*100, [-.5]*400, [[1]], [[1]])
        self.assertLess(result['p_value'], .05)

    def test_known_upstream_errors_increase_pvalue(self):
        args = ([.25]*100, [0.]*400, [[1]], [[1]])
        base = fixed_delta_hoeffding_pvalue(*args)['p_value']
        score_error = fixed_delta_hoeffding_pvalue(*args, score_mean_error=.01)['p_value']
        both_error = fixed_delta_hoeffding_pvalue(*args, score_mean_error=.01, interaction_mean_error=.02)['p_value']
        self.assertLess(base, score_error)
        self.assertLess(score_error, both_error)

    def test_support_violation_rejected_even_if_empirical_range_zero(self):
        for scores, val in (([.26], [0]), ([0], [.51])):
            with self.assertRaisesRegex(ValueError, 'support'):
                fixed_delta_hoeffding_pvalue(scores, val, [[1]], [[1]])

    def test_invalid_contracts_rejected(self):
        for kwargs in ({'delta':0}, {'delta':1}, {'delta':float('nan')}, {'score_mean_error':-.1}):
            with self.assertRaises(ValueError):
                fixed_delta_hoeffding_pvalue([0], [0], [[1]], [[1]], **kwargs)
        with self.assertRaises(ValueError):
            fixed_delta_hoeffding_pvalue([], [0], [[1]], [[1]])

    def test_family_threshold_rounds_down(self):
        threshold = family_threshold(.05, 17)
        self.assertLessEqual(Fraction.from_float(threshold), Fraction.from_float(.05)/17)
        for count in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                family_threshold(.05, count)
        with self.assertRaisesRegex(ValueError, 'binary64'):
            family_threshold(math.nextafter(0.,1.), 2)


if __name__ == '__main__':
    unittest.main()
