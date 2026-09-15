"""Independent regression checks for the structural-width diagnostic."""
import copy
import importlib.util
from itertools import product
from pathlib import Path
import unittest

import numpy as np

import exact_checks as exact
import reproduce as study


class StructuralSupportTests(unittest.TestCase):
    def test_all_weak_orderings_and_sharp_endpoints(self):
        result = exact.enumerate_support()
        self.assertEqual(result["shared_baseline_cases"], 2197)
        self.assertEqual([result["kernels"][x]["width"] for x in ("S", "Q", "U")],
                         ["5/4", "5/2", "1/2"])

    def test_against_released_generic_kernel_for_every_weak_ordering(self):
        path = study.default_source() / "reference_kernels/trajectory_kernel.py"
        spec = importlib.util.spec_from_file_location("released_kernel", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cases = list(product(exact.weak_orders(), repeat=3))
        f = np.array([np.column_stack((h, b)) for h, y, b in cases])
        g = np.array([np.column_stack((y, b)) for h, y, b in cases])
        c = np.array([[1, 0], [-1, 0]])
        for fn, exact_fn in ((module.shared_scores, exact.score),
                             (module.validation_scores, exact.correction),
                             (module.full_u_scores, exact.full_u)):
            expected = np.array([float(exact_fn(*case)) for case in cases])
            np.testing.assert_array_equal(fn(f, g, c, c), expected)

    def test_sample_equality_does_not_create_identity(self):
        contract = study.load_contract()
        self.assertEqual(study.structural_widths("persistence", contract), (0., 0., 0.))
        self.assertEqual(study.structural_widths("undeclared_equal_forecast", contract),
                         (1.25, 2.5, .5))
        without_identity = copy.deepcopy(contract)
        without_identity["forecast_equals_baseline"] = []
        self.assertEqual(study.structural_widths("persistence", without_identity), (1.25, 2.5, .5))

    def test_exact_identity_is_zero_for_all_comparisons(self):
        for y, b in product(exact.weak_orders(), repeat=2):
            self.assertEqual(exact.score(b, y, b), 0)
            self.assertEqual(exact.correction(b, y, b), 0)
            self.assertEqual(exact.full_u(b, y, b), 0)


class FixedDrawReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = study.default_source()
        cls.rows, cls.population = study.replay(cls.source, study.load_contract())

    def test_all_88_generic_results_reconstructed_from_recorded_arrays(self):
        result = study.verify_baseline(self.rows["generic"], self.source)
        self.assertEqual(result["rows_checked"], 88)

    def test_centers_and_variances_unchanged_and_every_radius_no_larger(self):
        for old, new in zip(self.rows["generic"], self.rows["structural"]):
            for key in ("candidate", "method", "bound", "raw_center", "interaction_estimate",
                        "corrected_center", "score_variance", "correction_variance"):
                self.assertEqual(old[key], new[key])
            self.assertLessEqual(new["total_radius"], old["total_radius"])
            if new["candidate"] == "persistence":
                self.assertEqual(new["total_radius"], 0.)
                self.assertEqual(new["lower"], 0.)
                self.assertFalse(new["passes"])

    def test_only_stronger_comparator_certification_changes_and_no_selection_changes(self):
        changes = [(new["candidate"], new["method"], new["bound"])
                   for old, new in zip(self.rows["generic"], self.rows["structural"])
                   if old["passes"] != new["passes"]]
        self.assertEqual(changes, [("boosting_7", "complete_pooled_u", "variance")])
        old_gates, new_gates = study.gate_rows(self.rows["generic"]), study.gate_rows(self.rows["structural"])
        self.assertEqual([r["candidate"] for r in old_gates], [r["candidate"] for r in new_gates])
        self.assertFalse(any(r["passes"] for r in self.rows["structural"] if r["method"].startswith("shared")))

    def test_width_scaling_preserves_integer_allocation(self):
        old, new = study.optimal_allocation(73728, 2., 4.), study.optimal_allocation(73728, 1.25, 2.5)
        self.assertEqual((old["M"], old["n"]), (13086, 15852))
        self.assertEqual((new["M"], new["n"]), (old["M"], old["n"]))
        self.assertAlmostEqual(new["range_radius"] / old["range_radius"], .625)

    def test_exact_population_denominator_for_interaction_fraction(self):
        value = next(r for r in self.population if r["candidate"] == "boosting_31")
        self.assertAlmostEqual(value["exact_population_shared_score"], .0099090318388564, places=16)
        self.assertAlmostEqual(value["interaction_over_population_shared_score"], .9140044351603771, places=14)
        self.assertAlmostEqual(value["exact_population_shared_score"],
                               value["true_shared_interaction"] + value["exact_target"], places=16)


if __name__ == "__main__":
    unittest.main()
