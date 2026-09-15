"""Deterministic synthetic checks; does not inspect any research task outcome."""

import argparse
from fractions import Fraction as F
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from direct_target import (block_empirical_bernstein, full_u,
                           full_u_hoeffding, symmetrized_four_row_kernel, _kernel_bounds)


def a(x, y):
    return F(int(y < x)) + F(int(y == x), 2)


def brute(v, w, c, p):
    n = len(v)
    first = sum((a(v[i], v[j]) * a(w[i], w[k])
                 for i, j, k in permutations(range(n), 3)), F(0)) / (n*(n-1)*(n-2))
    second = sum((a(v[i], v[k]) * a(w[j], w[ell]) / p[c[i]]
                  for i, j, k, ell in permutations(range(n), 4) if c[i] == c[j]), F(0))
    second /= n*(n-1)*(n-2)*(n-3)
    return first, second


def target(rows, masses):
    p = {}
    for row, mass in zip(rows, masses):
        p[row[0]] = p.get(row[0], F(0)) + mass
    ranks = [[sum((mass*a(row[channel], ref[channel]) for ref, mass in zip(rows, masses)), F(0))
              for channel in (1, 2)] for row in rows]
    means = {c: [sum((mass*rank[channel] for row, mass, rank in zip(rows, masses, ranks)
                     if row[0] == c), F(0))/pc for channel in (0, 1)] for c, pc in p.items()}
    theta = sum((mass*(rank[0]-means[row[0]][0])*(rank[1]-means[row[0]][1])
                 for row, mass, rank in zip(rows, masses, ranks)), F(0))
    return theta, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("VERIFICATION.json"))
    parser.add_argument("--skip-timings", action="store_true", help="Skip the optional quadratic runtime measurements.")
    args = parser.parse_args()
    rng = np.random.default_rng(2026091308)
    cases = []
    # Unequal positive probabilities include categories absent in a sample.
    p = {0: F(1, 10), 1: F(3, 10), 2: F(6, 10)}
    for n in range(4, 10):
        for kind in range(12):
            v = rng.integers(0, 4, n).tolist()
            w = rng.integers(-1, 3, n).tolist()
            c = rng.integers(0, 3, n).tolist()
            if kind == 0:
                v = [1] * n
            if kind == 1:
                w = [0] * n
            if kind == 2:
                v, w = [1]*n, [1]*n
            if kind == 3:
                c = [0]*n
            if kind == 4:
                v = [2**60 + k for k in rng.integers(0, 4, n).tolist()]
            if kind == 5:
                # Duplicate all observable fields; these remain separate draws.
                v[1], w[1], c[1] = v[0], w[0], c[0]
            cases.append((v, w, c, p))
    for n in range(4, 10):
        cases.append(([1]*n, list(range(n)), [0]*n, {0: F(1)}))
        cases.append((list(range(n)), [1]*n, [0]*n, {0: F(1)}))
    cases.append(([1,2,1,3], [1,0,2,1], [10,10,700,700], {j: F(1,1000) for j in range(1000)}))
    max_error = 0.0
    zero_cases = 0
    for v, w, c, probs in cases:
        first, second = brute(v, w, c, probs)
        fast = full_u(v, w, c, probs)
        error = max(abs(float(first)-fast["first_term"]), abs(float(second)-fast["second_term"]),
                    abs(float(first-second)-fast["estimate"]))
        max_error = max(max_error, error)
        assert error < 2e-13
        if len(probs) == 1:
            assert first == second
            assert abs(fast["estimate"]) < 2e-13
            zero_cases += 1
    sym_checks, permutation_checks = 0, 0
    for v, w, c, probs in cases[::4]:
        all_subsets = [symmetrized_four_row_kernel([v[i] for i in indices], [w[i] for i in indices],
                                                 [c[i] for i in indices], probs)
                       for indices in combinations(range(len(v)), 4)]
        fast = full_u(v, w, c, probs)["estimate"]
        assert abs(fast - math.fsum(all_subsets)/len(all_subsets)) < 2e-13
        sym_checks += 1
        order = rng.permutation(len(v))
        permuted = full_u([v[i] for i in order], [w[i] for i in order], [c[i] for i in order], probs)
        assert abs(fast-permuted["estimate"]) < 2e-13
        permutation_checks += 1
    populations = [
        ([(0, 0, 0), (0, 1, 2), (1, 1, 1), (1, 2, 2)], [F(1,10), F(2,10), F(3,10), F(4,10)]),
        ([(0, 0, 2), (0, 1, 0), (1, 1, 2), (1, 2, 1)], [F(1,10), F(2,10), F(3,10), F(4,10)]),
        ([(0, 1, 2), (0, 1, 0), (1, 1, 2), (1, 1, 1)], [F(1,10), F(2,10), F(3,10), F(4,10)]),
    ]
    population_checks = []
    for rows, masses in populations:
        theta, probs = target(rows, masses)
        expectation = F(0)
        floating_expectation = []
        for indices in product(range(len(rows)), repeat=4):
            sample = [rows[i] for i in indices]
            weight = math.prod(masses[i] for i in indices)
            v, w, c = [r[1] for r in sample], [r[2] for r in sample], [r[0] for r in sample]
            first, second = brute(v, w, c, probs)
            expectation += weight*(first-second)
            floating_expectation.append(float(weight)*full_u(v, w, c, probs)["estimate"])
        assert expectation == theta
        error = abs(math.fsum(floating_expectation)-float(theta))
        assert error < 2e-13
        population_checks.append({"target": str(theta), "exact_u_expectation": str(expectation),
                                  "floating_expectation_error": error, "samples_enumerated": 256})
    triple_range_checks = 0
    for v in product(range(3), repeat=3):
        for w in product(range(3), repeat=3):
            value = sum((a(v[i],v[j])*a(w[i],w[k]) for i,j,k in permutations(range(3))), F(0))/6
            assert F(1,6) <= value <= F(1,3)
            triple_range_checks += 1
    disjoint_product_checks = 0
    sym_range_checks = 0
    range_p = {0:F(1,10), 1:F(2,10), 2:F(3,10), 3:F(4,10)}
    for case in range(200):
        v, w = rng.integers(0,3,4), rng.integers(0,3,4)
        unweighted = sum((a(v[i],v[k])*a(w[j],w[ell])
                          for i,j,k,ell in permutations(range(4))), F(0))/24
        assert unweighted == F(1,4)
        disjoint_product_checks += 1
        c = rng.integers(0,4,4)
        first, second = brute(v,w,c,range_p)
        assert F(1,6)-F(1,4)/min(range_p.values()) <= first-second <= F(1,3)
        sym_range_checks += 1
    for probs, c in [(range_p, [0]*4), (range_p, [0,1,2,3]), ({0:F(1)}, [0]*4)]:
        lower, upper = _kernel_bounds(probs)
        for w in [[0,1,2,3],[3,2,1,0]]:
            result = symmetrized_four_row_kernel([0,1,2,3],w,c,probs)
            assert lower-2e-13 <= result <= upper+2e-13
    assert abs(symmetrized_four_row_kernel([0,1,2,3],[0,1,2,3],[0,1,2,3],range_p)-1/3) < 2e-13
    assert abs(symmetrized_four_row_kernel([0,1,2,3],[3,2,1,0],[0]*4,range_p)-(1/6-2.5)) < 2e-13
    # Compare numerical confidence formulas to independently written expressions.
    vv = rng.integers(0, 6, 43)
    ww = rng.integers(0, 5, 43)
    cc = rng.choice([0,1,2], size=43)
    alphas = [0.05, 0.01, 0.0002299605781865966, 1e-320]
    bounds = []
    for alpha in alphas:
        hu = full_u_hoeffding(vv, ww, cc, p, alpha)
        expected_h_radius = (8/3)*np.sqrt(-np.log(alpha)/(2*10))
        assert abs(hu["radius"]-expected_h_radius) < 2e-13
        eb = block_empirical_bernstein(vv, ww, cc, p, alpha)
        block_values = np.array([float(brute(vv[j:j+4], ww[j:j+4], cc[j:j+4], p)[0]
                                       -brute(vv[j:j+4], ww[j:j+4], cc[j:j+4], p)[1])
                                 for j in range(0, 40, 4)])
        independent_variance = sum((block_values[i]-block_values[j])**2
                                   for i in range(10) for j in range(i+1,10))/(10*9)
        logfactor = np.log(2)-np.log(alpha)
        expected_eb_radius = np.sqrt(2*independent_variance*logfactor/10)+56*logfactor/81
        assert abs(eb["estimate"]-block_values.mean()) < 2e-13
        assert abs(eb["sample_variance"]-independent_variance) < 2e-13
        assert abs(eb["radius"]-expected_eb_radius) < 2e-13
        bounds.append({"alpha": alpha, "full_u_radius": hu["radius"],
                       "block_empirical_bernstein_radius": eb["radius"]})
    invalid = [
        lambda: full_u([1]*3, [1]*3, [0]*3, {0:1}),
        lambda: full_u([1]*4, [1]*4, [0]*4, {0:0.9}),
        lambda: full_u([1]*4, [1]*4, [2]*4, {0:1}),
        lambda: full_u([np.nan]*4, [1]*4, [0]*4, {0:1}),
        lambda: full_u([1+1j]*4, [1]*4, [0]*4, {0:1}),
        lambda: full_u([-1,2**63+1,2**63+2,0], [0,1,3,2], [0]*4, {0:1}),
        lambda: full_u([0.5,2**60+1,2**60+2,0], [0,1,3,2], [0]*4, {0:1}),
        lambda: full_u([1]*4, [1]*4, [0]*4, {0:1-1e-320,1:1e-320}),
        lambda: full_u_hoeffding([1]*4, [1]*4, [0]*4, {0:1}, 0),
        lambda: block_empirical_bernstein([1]*7, [1]*7, [0]*7, {0:1}),
    ]
    for operation in invalid:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    timings = []
    for n in ([] if args.skip_timings else [128, 256, 512, 1024, 2048, 4096]):
        v = rng.integers(0, 100, n)
        w = rng.integers(0, 100, n)
        c = rng.choice([0,1,2], n, p=[0.1,0.3,0.6])
        elapsed = []
        for repeat in range(3):
            start = time.perf_counter()
            full_u(v, w, c, p)
            elapsed.append(time.perf_counter()-start)
        timings.append({"n": n, "seconds": elapsed, "median_seconds": float(np.median(elapsed))})
    output = {
        "scope": "synthetic finite-sum and inference-formula verification; no new task outcomes or power comparison",
        "seed": 2026091308, "brute_force_cases": len(cases), "max_absolute_error": max_error,
        "zero_cases": zero_cases, "subset_symmetrization_checks": sym_checks,
        "permutation_checks": permutation_checks, "population_expectation_checks": population_checks,
        "triple_range_checks": triple_range_checks,
        "disjoint_pair_product_identity_checks": disjoint_product_checks,
        "symmetrized_kernel_range_checks": sym_range_checks,
        "bound_formula_checks": bounds, "invalid_input_checks": len(invalid),
        "timings": timings, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                               "platform": platform.platform()},
        "passed": True,
    }
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
