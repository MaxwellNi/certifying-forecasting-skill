"""Reject unsupported transformations before issuing a source-design result."""
from copy import deepcopy
import unittest
from check_source_design import examples
from source_design import compile_design


class SourceDesignChecks(unittest.TestCase):
    def test_calendar_rows_do_not_establish_source_separation(self):
        d = examples()
        calendar = compile_design(d['calendar_peer_split'])
        source = compile_design(d['source_aligned_peer_split'])
        self.assertEqual(calendar['overlap_coefficient'], '-1')
        self.assertEqual(source['overlap_coefficient'], '0')
        self.assertEqual(calendar['population_coefficient'], source['population_coefficient'])
        self.assertFalse(source['population_reference_centered_for_every_common_law'])

    def test_reject_unsupported_transforms(self):
        for key, value in [('source_weights', [1, 2]), ('standardized', True),
                           ('fit_estimated_from', 'evaluation')]:
            d = deepcopy(examples()['sum_zero_interaction'])
            d['forecast_rows']['2'][key] = value
            with self.assertRaises(ValueError): compile_design(d)
        d['forecast_rows']['2'].pop(key)
        d['standardized'] = True
        with self.assertRaises(ValueError): compile_design(d)

    def test_source_container_and_duplicates(self):
        for sources in ['10', [0, '0'], [True, 0], [[1], 0], [1, 0, 2]]:
            d = deepcopy(examples()['sum_zero_interaction'])
            d['forecast_rows']['2']['sources'] = sources
            with self.assertRaises(ValueError): compile_design(d)

    def test_fixed_weight_and_focal_validation(self):
        for fit in [{'5': '.9'}, {'5': 'NaN'}, {}]:
            d = deepcopy(examples()['sum_zero_interaction'])
            d['forecast_fit'] = fit
            with self.assertRaises(ValueError): compile_design(d)
        d = deepcopy(examples()['sum_zero_interaction'])
        d['forecast_rows']['2']['peers'] = {'0': '1'}
        with self.assertRaises(ValueError): compile_design(d)

    def test_identical_rows_cancel_before_range_calculation(self):
        d = deepcopy(examples()['sum_zero_interaction'])
        d['forecast_rows']['5'] = deepcopy(d['forecast_rows']['2'])
        r = compile_design(d)
        self.assertEqual(r['coefficient_l1_forecast'], '0')
        self.assertEqual(r['overlap_coefficient'], '0')
        self.assertEqual(r['population_coefficient'], '0')


if __name__ == '__main__': unittest.main()
