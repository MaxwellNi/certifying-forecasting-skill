"""Independent arithmetic and finite-support checks; no coverage by simulation.

Run with --benchmark for wall-time measurements at n=1000, 2000, 4000.
Finite-support expectations use Fraction arithmetic and complete enumeration.
These checks cannot validate conditional IID sampling on any real panel.
"""

import argparse
from fractions import Fraction as F
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path
import time

import numpy as np

from full_u_variance import (
    baseline, full_u_all_delete_one, full_u_empirical_bernstein,
    full_u_joint_bound, interaction_bounds,
)


HERE = Path(__file__).resolve().parent


def exact_a(x, y):
    return F(int(y < x)) + F(int(y == x), 2)


def exact_ordered_u(v, w, c, p):
    """Literal ordered distinct tuples, independent of collision formulas."""
    n = len(v)
    first = sum((exact_a(v[i], v[j]) * exact_a(w[i], w[k])
                 for i, j, k in permutations(range(n), 3)), F())
    second = sum((exact_a(v[i], v[k]) * exact_a(w[j], w[l]) / p[c[i]]
                  for i, j, k, l in permutations(range(n), 4)
                  if c[i] == c[j]), F())
    return first / math.perm(n, 3), second / math.perm(n, 4)


def exact_kernel(states, indices, p):
    rows = [states[i] for i in indices]
    v, w, c = zip(*rows)
    first, second = exact_ordered_u(v, w, c, p)
    return first - second


def exact_support_analysis(name, states, mass, n=5):
    """Exact canonical projections, full-U law and jackknife expectation."""
    q = len(states)
    p = {}
    for row, weight in zip(states, mass):
        p[row[2]] = p.get(row[2], F()) + weight
    tuples = {s: list(product(range(q), repeat=s)) for s in range(5)}
    weights = lambda t: math.prod(mass[i] for i in t)
    h = {t: exact_kernel(states, t, p) for t in tuples[4]}
    g = {}
    for s in range(5):
        for t in tuples[s]:
            g[t] = sum((weights(tail) * h[t + tail]
                        for tail in tuples[4 - s]), F())
    theta = g[()]
    zeta = {}
    for s in range(1, 5):
        z = F()
        for t in tuples[s]:
            projection = sum(((-1)**(s-r) * g[tuple(t[i] for i in subset)]
                              for r in range(s + 1)
                              for subset in combinations(range(s), r)), F())
            z += weights(t) * projection**2
        zeta[s] = z

    def u(t):
        return sum((h[tuple(t[i] for i in subset)]
                    for subset in combinations(range(len(t)), 4)), F()) / math.comb(len(t), 4)

    exact_mean = F()
    exact_variance = F()
    exact_jackknife_mean = F()
    for t in product(range(q), repeat=n):
        probability = weights(t)
        center = u(t)
        deleted = [u(t[:i] + t[i+1:]) for i in range(n)]
        assert sum(deleted, F()) / n == center
        vjack = F(n - 1, n) * sum(((x - center)**2 for x in deleted), F())
        exact_mean += probability * center
        exact_variance += probability * (center - theta)**2
        exact_jackknife_mean += probability * vjack
    variance_formula = sum((F(math.comb(4, s)**2, math.comb(n, s)) * zeta[s]
                            for s in range(1, 5)), F())
    jackknife_formula = sum((F(s * (n-1) * math.comb(4, s)**2,
                              (n-s) * math.comb(n, s)) * zeta[s]
                            for s in range(1, 5)), F())
    efron_n = sum((F(s * math.comb(4, s)**2, math.comb(n, s)) * zeta[s]
                   for s in range(1, 5)), F())
    efron_previous = sum((F(s * math.comb(4, s)**2, math.comb(n-1, s)) * zeta[s]
                          for s in range(1, 5)), F())
    assert exact_mean == theta
    assert exact_variance == variance_formula
    assert exact_jackknife_mean == jackknife_formula
    assert exact_jackknife_mean == F(n-1, n) * efron_previous
    assert efron_n <= exact_jackknife_mean
    return {
        "name": name, "n": n, "sample_outcomes_enumerated": q**n,
        "theta_exact": str(theta),
        "canonical_projection_variances_exact": {str(s): str(zeta[s]) for s in zeta},
        "full_u_variance_exact": str(exact_variance),
        "expected_jackknife_variance_exact": str(exact_jackknife_mean),
        "efron_stein_variance_exact": str(efron_n),
        "all_exact_identities_pass": True,
    }


def run(benchmark=False):
    rng = np.random.default_rng(1283)
    checked_deletions = 0
    max_delete_error = 0.0
    max_center_error = 0.0
    # Literal exact tuples cover tied values, missing categories, n=5,6 and
    # cancellation of repeated-reference contributions in each deletion.
    exact_cases = [
        ([0]*5, [0]*5, [0]*5, {0: F(1)}),
        ([0,0,0,1,1], [0,0,0,1,1], [0]*5, {0: F(1)}),
        ([0,1,0,2,1], [2,1,2,0,0], [0,0,1,1,0], {0:F(2,3),1:F(1,3)}),
        ([0,1,1,2,0,2], [0,1,2,0,1,2], [0,0,1,1,0,1],
         {0:F(1,2),1:F(1,3),2:F(1,6)}),
    ]
    for v, w, c, p in exact_cases:
        out = full_u_all_delete_one(v, w, c, {key:float(value) for key,value in p.items()})
        first, second = exact_ordered_u(v,w,c,p)
        assert abs(out["estimate"] - float(first-second)) < 2e-12
        for i in range(len(v)):
            va,wa,ca = [x[:i]+x[i+1:] for x in (v,w,c)]
            first,second = exact_ordered_u(va,wa,ca,p)
            assert abs(out["delete_one_first"][i]-float(first)) < 2e-12
            assert abs(out["delete_one_second"][i]-float(second)) < 2e-12
            checked_deletions += 1

    random_cases = 0
    for n in (5,6,7,10,30):
        for categories in (1,3):
            for _ in range(20):
                v = rng.integers(0,4,n)
                w = rng.integers(0,3,n)
                c = rng.integers(0,categories,n)
                p = {0:1.0} if categories == 1 else {0:.2,1:.3,2:.5}
                out = full_u_all_delete_one(v,w,c,p)
                original = baseline.full_u(v,w,c,p)
                max_center_error = max(max_center_error, abs(out["estimate"]-original["estimate"]))
                for i in range(n):
                    deleted = baseline.full_u(np.delete(v,i),np.delete(w,i),np.delete(c,i),p)
                    for field,key in (("estimate","delete_one"),
                                      ("first_term","delete_one_first"),
                                      ("second_term","delete_one_second")):
                        error=abs(deleted[field]-out[key][i])
                        max_delete_error = max(max_delete_error,error)
                        assert error < 2e-12
                    checked_deletions += 1
                assert abs(out["delete_average_identity_error"]) < 2e-12
                random_cases += 1

    # An exact integer-order case above 2^53 guards comparison semantics.
    v = np.array([2**60+i for i in (0,1,1,2,4,3)], dtype=np.int64)
    w = np.array([3,1,2,0,4,4], dtype=np.int64)
    large = full_u_all_delete_one(v,w,[0]*6,{0:1.0})
    ordered = full_u_all_delete_one([0,1,1,2,4,3],w,[0]*6,{0:1.0})
    assert np.allclose(large["delete_one"],ordered["delete_one"],atol=0,rtol=0)

    exact_laws = [
        exact_support_analysis("constant", [(0,0,0)], [F(1)]),
        exact_support_analysis("first_projection_degenerate_tied", [(0,0,0),(1,1,0)], [F(1,2),F(1,2)]),
        exact_support_analysis("weak_projection_tied", [(0,0,0),(1,1,0)], [F(49,100),F(51,100)]),
        exact_support_analysis("multicategory_tied", [(0,1,0),(1,0,0),(1,1,1)], [F(1,4),F(1,4),F(1,2)]),
        exact_support_analysis("zero_empirical_variance_obstruction", [(0,1,0),(1,0,0)], [F(999,1000),F(1,1000)]),
    ]
    assert exact_laws[1]["canonical_projection_variances_exact"]["1"] == "0"
    assert exact_laws[1]["full_u_variance_exact"] == str(F(1,128*5*4))
    assert F(exact_laws[4]["theta_exact"]) == -F(999,4000000)
    obstruction_sample = full_u_all_delete_one([0]*5,[1]*5,[0]*5,{0:1.0})
    assert abs(obstruction_sample["estimate"]) < 1e-15
    assert obstruction_sample["jackknife_variance"] < 1e-28
    assert F(999,1000)**5 > F(1,20)

    n=100
    v = rng.binomial(1,.5,n)
    output = full_u_empirical_bernstein(v,v,[0]*n,{0:1.0},alpha=.05)
    count = int(v.sum())
    assert abs(output["estimate"]-count*(n-count)/(4*n*(n-1))) < 2e-12
    M,J = interaction_bounds(n,{0:1.0})
    Mp,Jp = interaction_bounds(n-1,{0:1.0})
    x = math.log(40)
    expected_radius = (math.sqrt(2*output["jackknife_variance"]*x)
                       +x*math.sqrt((n-1)/n*(4*Mp**2+16*Jp**2))
                       +(2*M/3+J)*x)
    assert abs(output["radius"]-expected_radius) < 1e-14
    joint = full_u_joint_bound(v,v,[0]*n,{0:1.0},alpha=.05)
    assert joint["radius"] == min(joint["empirical_bernstein_radius"],joint["hoeffding_radius"])

    # Structural small-n exception and input rejection are operational checks.
    rejected=0
    for arguments in [([0]*4,[0]*4,[0]*4,{0:1.0}),
                      ([0]*5,[0]*5,[0]*5,{0:.5}),
                      ([0]*5,[0]*5,[1]*5,{0:1.0})]:
        try:
            full_u_all_delete_one(*arguments)
        except ValueError:
            rejected += 1
    assert rejected == 3

    timings=[]
    if benchmark:
        for size in (1000,2000,4000):
            v = rng.integers(0,50,size)
            w = rng.integers(0,50,size)
            c = rng.integers(0,3,size)
            start=time.perf_counter()
            fast=full_u_all_delete_one(v,w,c,{0:.2,1:.3,2:.5})
            elapsed=time.perf_counter()-start
            timings.append({"n":size,"all_delete_one_seconds":elapsed,
                            "delete_average_identity_error":fast["delete_average_identity_error"]})

    remainder_examples=[]
    for size in (32000,100000):
        for probabilities in ({0:1.0},{i:.1 for i in range(10)}):
            M,J=interaction_bounds(size,probabilities)
            Mp,Jp=interaction_bounds(size-1,probabilities)
            remainder=x*(math.sqrt((size-1)/size*(4*Mp**2+16*Jp**2))+2*M/3+J)
            lo,hi=baseline._kernel_bounds(probabilities)
            hoeffding=(hi-lo)*math.sqrt(math.log(20)/(2*(size//4)))
            remainder_examples.append({"n":size,"categories":len(probabilities),
                                       "p_min":min(probabilities.values()),
                                       "alpha":.05,"eb_deterministic_remainder":remainder,
                                       "full_u_hoeffding_radius":hoeffding})
    report = {
        "command":"python theory/check_full_u_variance.py" + (" --benchmark" if benchmark else ""),
        "seed":1283,"all_checks_pass":True,
        "literal_fraction_tuple_cases":len(exact_cases),
        "randomized_cases":random_cases,
        "deletions_checked":checked_deletions,
        "max_delete_error":max_delete_error,"max_center_error":max_center_error,
        "exact_finite_support_results":exact_laws,
        "large_integer_order_check_pass":True,"bound_formula_check_pass":True,
        "invalid_inputs_rejected":rejected,
        "remainder_examples":remainder_examples,"timings":timings,
        "limits":"Arithmetic and exact finite-support validation; not a proof by Monte Carlo, not validation of real-panel IID sampling, and not an untouched-task power result.",
    }
    (HERE/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",action="store_true")
    run(parser.parse_args().benchmark)
