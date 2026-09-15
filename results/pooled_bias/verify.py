"""Deterministic algebra and finite-state checks; not a replacement for proof."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

from pooled_bias_bound import pooled_upper_bound


def stats(a, b):
    return (len(a), sum(a), sum(b), sum(x*x for x in a),
            sum(x*x for x in b), sum(x*y for x, y in zip(a, b)))


def call(p, per_cell, delta, **kwargs):
    return pooled_upper_bound(p, *zip(*per_cell), delta, **kwargs)


def compositions(n, k=4):
    if k == 1:
        yield (n,)
    else:
        for j in range(n+1):
            for rest in compositions(n-j, k-1):
                yield (j, *rest)


def cell_states(n, probabilities, f, g):
    """Exactly enumerate sufficient states of iid four-outcome pairs."""
    outcomes = [(0-f, 0-g), (0-f, 1-g), (1-f, 0-g), (1-f, 1-g)]
    for ns in compositions(n):
        weight = math.factorial(n)
        for count, prob in zip(ns, probabilities):
            weight *= prob**count / math.factorial(count)
        if weight == 0:
            continue
        a, b = [], []
        for count, (av, bv) in zip(ns, outcomes):
            a.extend([av]*count)
            b.extend([bv]*count)
        yield weight, stats(a, b)


def main():
    max_partition_error = 0.0
    for n in range(2, 10):
        # Unequal means, tied values, and arbitrary within-pair dependence.
        a = [((i*7) % 11)/10-0.7 for i in range(n)]
        b = [((i*3+1) % 9)/8-0.2 for i in range(n)]
        result = call([1.0], [stats(a, b)], 0.05)
        k, ell = n//2, n-n//2
        splits = []
        for left in itertools.combinations(range(n), k):
            right = [i for i in range(n) if i not in left]
            aa = sum(a[i] for i in left)/k
            bb = sum(b[i] for i in right)/ell
            splits.append((aa*bb, bb*bb/(4*k), aa*aa/(4*ell)))
        averaged = [sum(v[j] for v in splits)/len(splits) for j in range(3)]
        actual = [result['center'], result['linear_a_proxy'], result['linear_b_proxy']]
        max_partition_error = max(max_partition_error, *(abs(x-y) for x, y in zip(actual, averaged)))
        assert max_partition_error < 1e-12

    specifications = [
        # Positive, zero, and negative within-pair covariances; unequal masses/counts.
        ([1.0], [(20, [.25, .25, .25, .25], .5, .5)]),
        ([1.0], [(20, [.45, .05, .05, .45], .5, .5)]),
        ([1.0], [(20, [.05, .45, .45, .05], .5, .5)]),
        ([1.0], [(20, [.72, .08, .12, .08], .1, .9)]),
        ([1.0], [(20, [.12, .18, .2, .5], .8, .9)]),
        # Large same-direction misspecification exercises nonzero tail failures.
        ([1.0], [(20, [.25, .25, .25, .25], 0., 0.)]),
        ([.2, .8], [(3, [.45, .05, .05, .45], .4, .6),
                   (4, [.1, .4, .4, .1], .8, .2)]),
    ]
    records = []
    max_variance_error = max_mgf = max_square_mgf_ratio = 0.0
    for case, (p, cells) in enumerate(specifications):
        population = []
        for n, q, f, g in cells:
            av = q[2]+q[3]-f
            bv = q[1]+q[3]-g
            siga = (q[2]+q[3])*(1-q[2]-q[3])
            sigb = (q[1]+q[3])*(1-q[1]-q[3])
            sigab = q[3]-(q[2]+q[3])*(q[1]+q[3])
            population.append((av, bv, siga, sigb, sigab))
        truth = sum(w*pop[0]*pop[1] for w, pop in zip(p, population))
        expected_variance = sum(w*w*((bv*bv*sa+av*av*sb+2*av*bv*sab)/cell[0]
                                     +(sa*sb+sab*sab)/(cell[0]*(cell[0]-1)))
                                for w, cell, (av, bv, sa, sb, sab) in zip(p, cells, population))
        total_probability = eu = eu2 = ev_a = ev_b = 0.0
        mgfs = {(which, t): 0.0 for which in ('A', 'B') for t in (-8., -3., -1., 1., 3., 8.)}
        square_mgfs = {(which, alpha): 0.0 for which in ('A', 'B') for alpha in (.25, .5, .9, .99)}
        failures = {(mode, product, delta): 0.0
                    for mode in ('fixed', 'grid', 'mixture', 'square_mgf')
                    for product in ('subgamma', 'exact_mgf') for delta in (.05, .2, .8)}
        product_radii = {}
        zero_stats = [stats([0.]*cell[0], [0.]*cell[0]) for cell in cells]
        for delta in (.05, .2, .8):
            simple = call(p, zero_stats, delta)
            exact = call(p, zero_stats, delta, product_mode='exact_mgf')
            assert 0 < exact['product_radius'] <= simple['product_radius']
            product_radii['subgamma', delta] = simple['product_radius']
            product_radii['exact_mgf', delta] = exact['product_radius']
        num_states = 0
        for combination in itertools.product(*(list(cell_states(*cell)) for cell in cells)):
            num_states += 1
            weight = math.prod(entry[0] for entry in combination)
            sufficient = [entry[1] for entry in combination]
            result = call(p, sufficient, .05)
            u, va, vb = result['center'], result['linear_a_proxy'], result['linear_b_proxy']
            la = sum(w*pop[0]*st[2]/st[0] for w, pop, st in zip(p, population, sufficient))-u
            lb = sum(w*pop[1]*st[1]/st[0] for w, pop, st in zip(p, population, sufficient))-u
            total_probability += weight
            eu += weight*u
            eu2 += weight*u*u
            ev_a += weight*va
            ev_b += weight*vb
            for (which, t) in mgfs:
                ll, vv = (la, va) if which == 'A' else (lb, vb)
                mgfs[which, t] += weight*math.exp(t*ll-t*t*vv/2)
            for (which, alpha) in square_mgfs:
                ll, vv = (la, va) if which == 'A' else (lb, vb)
                if vv == 0:
                    assert abs(ll) < 1e-14
                ratio = ll*ll/vv if vv > 0 else 0.
                square_mgfs[which, alpha] += weight*math.exp(alpha*ratio/2)
            for mode in ('fixed', 'grid', 'mixture', 'square_mgf'):
                for delta in (.05, .2, .8):
                    base = call(p, sufficient, delta, lambda_mode=mode)
                    for product in ('subgamma', 'exact_mgf'):
                        bound = base['upper']-base['product_radius']+product_radii[product, delta]
                        failures[mode, product, delta] += weight*(truth > bound+1e-13)
        assert abs(total_probability-1) < 1e-11
        assert abs(eu-truth) < 1e-12
        actual_variance = eu2-eu*eu
        max_variance_error = max(max_variance_error, abs(actual_variance-expected_variance))
        assert max_variance_error < 1e-12
        max_mgf = max(max_mgf, *mgfs.values())
        assert max_mgf <= 1+1e-12
        max_square_mgf_ratio = max(max_square_mgf_ratio,
                                   *(value*math.sqrt(1-alpha) for (_, alpha), value in square_mgfs.items()))
        assert max_square_mgf_ratio <= 1+1e-12
        for (mode, product, delta), failure in failures.items():
            assert failure <= delta+1e-12
            records.append(dict(case=case, mode=mode, product_mode=product, delta=delta, exact_noncoverage=failure,
                                sufficient_states=num_states, truth=truth))

    empty = call([.4, .6], [stats([], []), stats([.3], [-.2])], .05,
                 absent_upper=[.2, .3])
    assert abs(empty['upper']-.26) < 1e-14
    receipt = dict(
        scope='Deterministic algebra and exact finite-state checks; no empirical generalization claim',
        partition_identity_n=list(range(2, 10)), max_partition_absolute_error=max_partition_error,
        max_variance_absolute_error=max_variance_error, max_checked_linear_mgf=max_mgf,
        max_checked_square_mgf_over_bound=max_square_mgf_ratio,
        square_mgf_alphas=[.25, .5, .9, .99],
        exact_state_results=records, all_checks_pass=True,
    )
    output = Path(__file__).with_name('verification.json')
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'exact_state_results'}, indent=2))
    print(f'Exact coverage configurations: {len(records)}; receipt: {output}')


if __name__ == '__main__':
    main()
