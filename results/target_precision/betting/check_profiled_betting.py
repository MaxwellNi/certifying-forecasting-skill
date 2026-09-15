"""Independent convex-profile and exhaustive finite-null checks; no field result."""

from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiled_betting import profiled_betting, profiled_mixture_pvalue


def prepared(cells):
    return {"cells": [{"mass": p, "pairs": len(z), "values": np.array(z, dtype=float)}
                      for p, z in cells],
            "total_raw_rows_charged": 4*sum(len(z) for _, z in cells)}


def exact_two_cell_minimum(cells, fraction):
    """Independent 80-digit derivative bisection on the 1-D null boundary."""
    mp.mp.dps = 80
    b = mp.mpf(str(fraction))
    weights = [mp.mpf(str(p)) for p, _ in cells]
    data = [[mp.mpf(float(z).as_integer_ratio()[0])/float(z).as_integer_ratio()[1]+mp.mpf('.5')
             for z in zz] for _, zz in cells]
    cap = sum(weights)/2
    lo = max(mp.mpf('.25'), (cap-weights[1]*mp.mpf('.75'))/weights[0])
    hi = min(mp.mpf('.75'), (cap-weights[1]*mp.mpf('.25'))/weights[0])

    def fg(eta, x):
        f = sum(mp.log(1-b+b*y/eta) for y in x)
        g = sum(-b*y/(eta*((1-b)*eta+b*y)) for y in x)
        return f, g

    def objective(eta0):
        eta1 = (cap-weights[0]*eta0)/weights[1]
        f0, g0 = fg(eta0, data[0]); f1, g1 = fg(eta1, data[1])
        return f0+f1, g0-weights[0]/weights[1]*g1

    if objective(lo)[1] >= 0:
        return objective(lo)[0]
    if objective(hi)[1] <= 0:
        return objective(hi)[0]
    for _ in range(260):
        mid = (lo+hi)/2
        if objective(mid)[1] > 0:
            hi = mid
        else:
            lo = mid
    return objective((lo+hi)/2)[0]


def main():
    checks = []
    mixed = prepared([(.5, [.1]*100), (.5, [-.1]*100)])
    assert profiled_mixture_pvalue(mixed)["p"] == 1
    assert profiled_mixture_pvalue(prepared([(.4, [.1]*20), (.6, [])]))["p"] == 1
    checks += ["mixed-sign global null retained", "nonbinding missing-mass null returns one"]

    single = profiled_mixture_pvalue(prepared([(1., [.05]*100)]))
    mp.mp.dps = 80
    z = mp.mpf(float(.05).as_integer_ratio()[0])/float(.05).as_integer_ratio()[1]
    exact_e = sum((1+2*mp.mpf(b)*z)**100 for b in ['.1','.3','.5','.7','.9'])/5
    assert mp.mpf(single["p"]) >= 1/exact_e
    assert abs(float(1/exact_e)-single["p"]) < 1e-9
    checks.append("single-stratum closed-form mixture and conservative p rounding")

    rng = np.random.default_rng(913602609)
    max_gap = 0.
    for _ in range(8):
        cells = [(.25, rng.choice([-.5,-.25,0,.125,.5], 17)),
                 (.75, rng.choice([-.5,-.125,0,.25,.5], 23))]
        result = profiled_mixture_pvalue(prepared(cells))
        for profile in result["profiles"]:
            fraction = F(profile["fraction"])
            truth = exact_two_cell_minimum(cells, str(float(fraction)))
            assert mp.mpf(profile["log_e_lower"]) <= truth
            assert truth <= mp.mpf(profile["primal_log_e_upper"])
            max_gap = max(max_gap, profile["certified_optimization_gap_upper"])
    checks.append("40 profiles bracket independent 80-digit constrained minima")

    failed_optimizer = profiled_betting(prepared(cells), max_iterations=1)
    for profile in failed_optimizer["profiles"]:
        truth = exact_two_cell_minimum(cells, str(float(F(profile["fraction"]))))
        assert mp.mpf(profile["log_e_lower"]) <= truth
    checks.append("valid tangent lower evidence with deliberately truncated optimizer")

    # Generic bounded independent stratum streams: a larger class than raw-rank pairs.
    # Constant fractions make wealth invariant to permutations within a stratum,
    # so enumerate binomial counts, not Monte Carlo draws.
    nulls = [(F(1,2), F(3,5), F(2,5)),
             (F(1,4), F(7,10), F(13,30)),
             (F(1,2), F(1,4), F(1,2))]
    alpha_grid = [.01,.05,.1,.2,.5]
    enumeration = []
    q = 6
    for p, mu0, mu1 in nulls:
        assert p*mu0+(1-p)*mu1 <= F(1,2)
        failure = {str(alpha): F(0) for alpha in alpha_grid}
        evidence_expectation = 0.
        for k0 in range(q+1):
            for k1 in range(q+1):
                chance = (math.comb(q,k0)*mu0**k0*(1-mu0)**(q-k0)
                          * math.comb(q,k1)*mu1**k1*(1-mu1)**(q-k1))
                sample = prepared([(float(p), [.5]*k0+[-.5]*(q-k0)),
                                   (float(1-p), [.5]*k1+[-.5]*(q-k1))])
                result = profiled_mixture_pvalue(sample)
                evidence_expectation += float(chance)*math.exp(result["objective_lower"])
                for alpha in alpha_grid:
                    if result["p"] <= alpha:
                        failure[str(alpha)] += chance
        assert evidence_expectation <= 1+1e-12
        assert all(float(failure[str(alpha)]) <= alpha+1e-12 for alpha in alpha_grid)
        enumeration.append({"weights": [str(p),str(1-p)], "shifted_means": [str(mu0),str(mu1)],
                            "pairs_per_stratum": q, "exact_count_states": (q+1)**2,
                            "evidence_expectation": evidence_expectation,
                            "null_rejection_probabilities": {a:float(v) for a,v in failure.items()}})
    checks.append("147 enumerated heterogeneous-null count states obey e-value and p-value guarantees")
    out = {"status":"passed", "checks":checks, "maximum_independent_profile_gap":max_gap,
           "null_enumeration":enumeration,
           "source_sha256":hashlib.sha256(Path(__file__).with_name('profiled_betting.py').read_bytes()).hexdigest(),
           "scope":"Numerical and finite-law checks only; no untouched empirical task or universal power claim."}
    Path(__file__).with_name('profiled_betting_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
