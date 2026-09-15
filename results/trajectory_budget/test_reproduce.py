"""Independent producer/formula parity and allocation-contract checks."""
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference_kernels"))
import reproduce as r
from trajectory_kernel import shared_scores, validation_scores, full_u_scores, corrected_lower
from pooled_trajectory_kernel import pooled_u_joint_lower, pooled_u_score, pooled_u_hoeffding_lower


class CostStudyChecks(unittest.TestCase):
    def test_allocation_matches_full_integer_search_including_leftovers(self):
        for budget in range(5, 151):
            observed = r.allocation(budget)
            all_pairs = []
            for n in range(1, budget // 3 + 1):
                for M in range(1, budget // 2 + 1):
                    if 2 * M + 3 * n <= budget:
                        radius = 2 * math.sqrt(-math.log(r.ALPHA-r.DELTA)/(2*M))
                        radius += 4 * math.sqrt(-math.log(r.DELTA)/(2*n))
                        all_pairs.append(radius)
            self.assertAlmostEqual(observed["range_radius"], min(all_pairs), places=14)
        plan = r.allocation(73728)
        self.assertEqual((plan["M"], plan["n"], plan["unused_budget"]), (13086, 15852, 0))

    def test_zero_column_pruning_preserves_every_score_and_correction(self):
        rng = np.random.default_rng(317)
        f, g = rng.integers(5, size=(2, 128, 3, 2))
        old = np.array([[1., 0.], [-1., 0.]])
        new = old[:, :1]
        np.testing.assert_array_equal(shared_scores(f, g, old, old), shared_scores(f[:, :2], g[:, :2], new, new))
        np.testing.assert_array_equal(validation_scores(f, g, old, old), validation_scores(f, g, new, new))
        np.testing.assert_array_equal(old.sum(1), new.sum(1))
        np.testing.assert_array_equal(old @ old.T, new @ new.T)
        changed_f, changed_g = f.copy(), g.copy()
        changed_f[:, 2] = -999
        changed_g[:, 2] = 999
        np.testing.assert_array_equal(shared_scores(f, g, old, old), shared_scores(changed_f, changed_g, old, old))

    def test_explicit_six_role_triple_and_integer_ties(self):
        rng = np.random.default_rng(49)
        for offset in [0, 2**60]:
            f, g = rng.integers(6, size=(2, 11, 2)) + offset
            ix = rng.integers(11, size=(37, 3))
            C = np.array([[1.], [-1.]])
            np.testing.assert_allclose(r.triples(f, g, ix), full_u_scores(f[ix], g[ix], C, C), atol=1e-16, rtol=0)
            theta, gamma, p, q = r.population(f, g)
            self.assertTrue(np.isfinite(theta + gamma))
            self.assertAlmostEqual(r.pooled_center(p, q, ix.reshape(-1)), pooled_u_score(f[ix.reshape(-1)], g[ix.reshape(-1)], C, C), places=15)

    def test_original_all_eight_and_new_allocation_against_producer(self):
        a = np.load(HERE / "inputs/selection_forecasts.npz")
        ix = np.load(HERE / "inputs/selection_indices.npz")
        new = np.load(HERE / "results/active_allocation_indices.npz")
        gates = pd.read_csv(HERE / "inputs/original_selection_gates.csv").set_index("candidate")
        strong = pd.read_csv(HERE / "inputs/original_strong_gates.csv").set_index("candidate")
        result = pd.read_csv(HERE / "results/certificate_all_eight.csv").set_index(["candidate", "method", "bound"])
        oldC = np.array([[1., 0.], [-1., 0.]])
        newC = oldC[:, :1]
        g = np.column_stack([a["target"], a["baseline"]])
        stream = np.r_[ix["evaluation"].reshape(-1), ix["validation"].reshape(-1)]
        np.testing.assert_array_equal(stream, np.r_[new["evaluation"].reshape(-1), new["validation"].reshape(-1)])
        self.assertEqual(len(stream), 73728)
        max_error = 0.
        for k, name in enumerate(a["names"]):
            f = np.column_stack([a["predictions"][:, k], a["baseline"]])
            for method, C, ei, vi in [("shared_original", oldC, ix["evaluation"], ix["validation"]),
                                      ("shared_pruned_allocated", newC, new["evaluation"], new["validation"])]:
                for rule in ["hybrid", "hoeffding"]:
                    expected = corrected_lower(shared_scores(f[ei], g[ei], C, C),
                                               validation_scores(f[vi], g[vi], C, C),
                                               C, C, alpha=r.ALPHA, delta=r.DELTA, rule=rule)
                    got = result.loc[(name, method, rule)]
                    for key, producer_key in [("lower", "lower"), ("corrected_center", "estimate"),
                                               ("evaluation_radius", "sampling_radius"),
                                               ("validation_radius", "validation_radius")]:
                        error = abs(got[key] - expected[producer_key])
                        max_error = max(max_error, error)
                        self.assertLess(error, 1e-12)
            self.assertAlmostEqual(result.loc[(name, "shared_original", "hybrid"), "lower"], gates.loc[name, "corrected_lower"], places=12)
            self.assertAlmostEqual(result.loc[(name, "direct_triple_u", "hybrid"), "lower"], gates.loc[name, "full_u_lower"], places=12)
            for bound, producer in [("variance", pooled_u_joint_lower), ("hoeffding", pooled_u_hoeffding_lower)]:
                expected = producer(f[stream], g[stream], oldC, oldC, alpha=r.ALPHA)
                got = result.loc[(name, "complete_pooled_u", bound)]
                for key, producer_key in [("lower", "lower"), ("corrected_center", "score"), ("evaluation_radius", "radius")]:
                    error = abs(got[key] - expected[producer_key])
                    max_error = max(max_error, error)
                    self.assertLess(error, 1e-12)
            self.assertAlmostEqual(result.loc[(name, "complete_pooled_u", "variance"), "lower"], strong.loc[name, "pooled_lower"], places=12)
        print("Maximum independent/new-producer arithmetic error:", max_error)

    def test_original_bootstrap_endpoints_and_cost_accounting(self):
        original = json.loads((HERE / "inputs/original_utility_diagnostic.json").read_text())["comparisons"][0]
        frame = pd.read_csv(HERE / "results/future_rule_comparisons.csv")
        got = frame[(frame.rule == "shared_pruned_allocated_hybrid") & (frame.comparator == "direct_loss_selection")].iloc[0]
        for key, expected in [("target_weighted_gain", original["target_weighted_absolute_gain"]),
                               ("gain_ci_low", original["target_weighted_absolute_ci"][0]),
                               ("gain_ci_high", original["target_weighted_absolute_ci"][1]),
                               ("equal_week_gain", original["equal_week_absolute_gain"]),
                               ("equal_week_ci_low", original["equal_week_absolute_ci"][0]),
                               ("equal_week_ci_high", original["equal_week_absolute_ci"][1])]:
            self.assertAlmostEqual(got[key], expected, places=11)
        costs = pd.read_csv(HERE / "results/cost_accounting.csv").set_index("method")
        self.assertEqual(costs.loc["shared_original", "discarded_trajectory_draws"], 16384)
        self.assertEqual(costs.loc["shared_original", "used_trajectory_draws"], 57344)
        self.assertEqual(costs.loc["shared_pruned_allocated", "used_trajectory_draws"], 73728)
        self.assertTrue((costs.unique_archive_records_used == 342).all())
        self.assertTrue((costs.distinct_original_hour_rows_used == 2069).all())
        self.assertTrue((costs.new_labels_acquired == 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
