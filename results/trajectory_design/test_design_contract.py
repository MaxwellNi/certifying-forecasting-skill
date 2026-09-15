"""Meaningful design, target, source-reuse, and information-time checks."""
import copy
from fractions import Fraction
import unittest

from design_contract import (compile_reference_design, disjoint_reference_design,
                             preserved_target, validate_forecast_availability)


def example():
    maps = [{"id": "forecast", "signature": "fixed-forecast-definition"},
            {"id": "baseline", "signature": "fixed-baseline-definition"}]
    refs = [{"id": "shared", "law_id": "declared-trajectory-law"}]
    ft = [{"map_id": "forecast", "reference_id": "shared", "weight": 1},
          {"map_id": "baseline", "reference_id": "shared", "weight": "-1/2"}]
    gt = [{"map_id": "forecast", "reference_id": "shared", "weight": 1},
          {"map_id": "baseline", "reference_id": "shared", "weight": "-1/2"}]
    return maps, maps, refs, ft, gt


class DesignContractTests(unittest.TestCase):
    def test_reused_role_accumulates_exactly(self):
        args = list(example())
        args[3] = [{"map_id": "forecast", "reference_id": "shared", "weight": "1/3"},
                   {"map_id": "forecast", "reference_id": "shared", "weight": "2/3"}]
        design = compile_reference_design(*args)
        self.assertEqual(design['C_exact'], [['1'], ['0']])
        self.assertEqual(design['omega_exact'], [['1', '-1/2'], ['0', '0']])
        self.assertFalse(design['universal_preservation'])
        self.assertFalse(design['independence_verified'])

    def test_disjoint_construction_preserves_target_exactly(self):
        design = compile_reference_design(*example())
        result = disjoint_reference_design(design)
        self.assertTrue(result['universal_preservation'])
        self.assertEqual(result['c_exact'], ['1', '-1/2'])
        self.assertEqual(result['d_exact'], ['1', '-1/2'])
        self.assertTrue(result['target_check']['exact_row_sums_preserved'])
        self.assertEqual(result['C_exact'], [['1', '0'], ['-1/2', '0']])
        self.assertEqual(result['D_exact'], [['0', '1'], ['0', '-1/2']])

    def test_changed_row_sum_is_not_target_preservation(self):
        before = compile_reference_design(*example())
        after = disjoint_reference_design(before)
        after['c_exact'][0] = '2'
        with self.assertRaisesRegex(ValueError, 'row sums'):
            preserved_target(before, after)

    def test_rational_preservation_does_not_hide_floating_target_change(self):
        args = list(example())
        args[2] = [{"id": str(i), "law_id": "law"} for i in range(3)]
        args[3] = [{"map_id": "forecast", "reference_id": str(i), "weight": "1/3"} for i in range(3)]
        args[4] = [{"map_id": "forecast", "reference_id": "0", "weight": 1}]
        before = compile_reference_design(*args)
        self.assertEqual(before['c_exact'][0], '1')
        with self.assertRaisesRegex(ValueError, 'floating coefficient row sums'):
            disjoint_reference_design(before)

    def test_changed_map_or_law_rejected(self):
        before = compile_reference_design(*example())
        for key in ('forecast_maps', 'outcome_maps', 'reference_law_id'):
            after = copy.deepcopy(before)
            if key == 'reference_law_id':
                after[key] = 'other-law'
            else:
                after[key][0]['signature'] = 'changed-map-definition'
            with self.assertRaises(ValueError):
                preserved_target(before, after)

    def test_duplicate_id_and_unknown_source_rejected(self):
        args = list(example())
        args[2] = args[2] * 2
        with self.assertRaises(ValueError):
            compile_reference_design(*args)
        args = list(example())
        args[3][0]['reference_id'] = 'not-declared'
        with self.assertRaises(ValueError):
            compile_reference_design(*args)

    def test_heterogeneous_laws_not_silently_pooled(self):
        args = list(example())
        args[2] += [{"id": "other", "law_id": "different-law"}]
        with self.assertRaisesRegex(ValueError, 'common reference law'):
            compile_reference_design(*args)

    def test_exact_fraction_vs_floating_coefficients_disclosed(self):
        args = list(example())
        args[3][0]['weight'] = Fraction(1, 3)
        result = compile_reference_design(*args)
        self.assertEqual(result['c_exact'][0], '1/3')
        self.assertFalse(result['floating_coefficients_exact'])

    def test_nonfinite_weight_rejected(self):
        for x in (float('nan'), float('inf'), True):
            args = list(example())
            args[3][0]['weight'] = x
            with self.assertRaises(ValueError):
                compile_reference_design(*args)

    @staticmethod
    def forecast():
        return [{"id": "one-hour-forecast", "issue_time": "2014-01-01T14:00:00+08:00",
                 "target_time": "2014-01-01T15:00:00+08:00",
                 "model_available_at": "2013-12-31T12:00:00Z",
                 "features": [{"id": "last-hour-measurement",
                               "observed_at": "2014-01-01T13:00:00+08:00",
                               "available_at": "2014-01-01T14:00:00+08:00"}]}]

    def test_valid_issue_boundary_and_independence_disclaimer(self):
        result = validate_forecast_availability(self.forecast(), ['train-A'], ['audit-B'],
                                                 'Common-law independent draws conditional on external training.')
        self.assertTrue(result['declared_availability_pass'])
        self.assertFalse(result['actual_latency_verified'])
        self.assertFalse(result['independence_verified'])

    def test_future_feature_rejected(self):
        record = self.forecast()
        record[0]['features'][0]['available_at'] = '2014-01-01T14:00:01+08:00'
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            validate_forecast_availability(record, independence_statement='Declared IID.')

    def test_future_target_or_late_model_rejected(self):
        for key, timestamp in (('target_time', '2014-01-01T14:00:00+08:00'),
                               ('model_available_at', '2014-01-01T14:00:01+08:00')):
            record = self.forecast()
            record[0][key] = timestamp
            with self.assertRaises(ValueError):
                validate_forecast_availability(record, independence_statement='Declared IID.')

    def test_naive_time_and_impossible_latency_rejected(self):
        record = self.forecast()
        record[0]['issue_time'] = '2014-01-01T14:00:00'
        with self.assertRaisesRegex(ValueError, 'timezone'):
            validate_forecast_availability(record, independence_statement='Declared IID.')
        record = self.forecast()
        record[0]['features'][0]['available_at'] = '2014-01-01T12:00:00+08:00'
        with self.assertRaisesRegex(ValueError, 'before its observation'):
            validate_forecast_availability(record, independence_statement='Declared IID.')

    def test_known_training_reuse_and_missing_sampling_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_forecast_availability(self.forecast(), ['same'], ['same'], 'Declared IID.')
        with self.assertRaisesRegex(ValueError, 'sampling assumption'):
            validate_forecast_availability(self.forecast())


if __name__ == '__main__':
    unittest.main()
