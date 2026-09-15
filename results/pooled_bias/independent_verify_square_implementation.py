"""Independently evaluate the square-normalized bound against scalar formulas."""

from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import importlib.util
import json
import math
import random


def rootu(x):
    lo, hi = 1.0, 2.0
    while hi - 1 - math.log(hi) < 2 * x:
        hi *= 2
    for _ in range(100):
        z = (lo + hi) / 2
        if z - 1 - math.log(z) < 2 * x:
            lo = z
        else:
            hi = z
    return hi


def verify():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('bound_under_test', root / 'pooled_bias_bound.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = random.Random(1309202699)
    maximum = 0.0
    count = zero_cases = all_absent = 0
    roots = {}
    for case in range(160):
        C = 1 + case % 5
        weights = [rng.randrange(0, 10) for _ in range(C)]
        if not sum(weights):
            weights[0] = 1
        ps = [F(w, sum(weights)) for w in weights]
        ns = [rng.choice([0, 1, 2, 3, 4, 7, 16]) for _ in range(C)]
        obs, qs, stats = [], [], []
        for n in ns:
            f, g = F(rng.randrange(9), 8), F(rng.randrange(9), 8)
            A, B = [], []
            for j in range(n):
                x = F(rng.randrange(9), 8)
                y = x if case % 3 == 0 else 1 - x if case % 3 == 1 else F(rng.randrange(9), 8)
                if case % 13 == 0:
                    y = g
                if case % 17 == 0:
                    x = f
                A.append(x - f)
                B.append(y - g)
            obs.append((A, B))
            stats.append((n, sum(A), sum(B), sum(x * x for x in A),
                          sum(y * y for y in B), sum(x * y for x, y in zip(A, B))))
            qs.append(max(f * g, (1 - f) * (1 - g), -f * (1 - g), -(1 - f) * g))
        U = VA = VB = H = F(0)
        Ds = []
        for p, n, (A, B), q in zip(ps, ns, obs, qs):
            if not p:
                continue
            if n < 2:
                H += p * q
                continue
            U += p * sum(A[i] * B[j] for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
            k, ell = n // 2, n - n // 2

            def subset_second(values, m):
                return (sum(x * x for x in values) / (m * n)
                        + F(m - 1, m * n * (n - 1))
                        * sum(values[i] * values[j] for i in range(n) for j in range(n) if i != j))

            VA += p * p * subset_second(B, ell) / (4 * k)
            VB += p * p * subset_second(A, k) / (4 * ell)
            Ds.append(float(p) / (4 * math.sqrt(k * ell)))
        if VA == 0 or VB == 0:
            zero_cases += 1
        if not Ds:
            all_absent += 1
        arr = [list(x) for x in zip(*stats)]
        for dd in [.8, .05, 1e-8, 1e-100]:
            x = math.log(3 / dd)
            u = rootu(x)
            roots[str(dd)] = u
            D, dm = sum(d * d for d in Ds), max(Ds, default=0)
            ra, rb = math.sqrt(u * float(VA)), math.sqrt(u * float(VB))
            r2 = math.sqrt(2 * x * D) + x * dm
            res = mod.pooled_upper_bound(
                list(map(float, ps)), arr[0], *[[float(v) for v in vs] for vs in arr[1:]], dd,
                absent_upper=list(map(float, qs)), lambda_mode='square_mgf')
            expected = {
                'center': float(U), 'linear_a_proxy': float(VA), 'linear_b_proxy': float(VB),
                'linear_a_radius': ra, 'linear_b_radius': rb, 'product_radius': r2,
                'upper': float(U) + ra + rb + r2 + float(H),
            }
            for key, value in expected.items():
                err = abs(res[key] - value)
                maximum = max(maximum, err)
                assert err < 2e-12 * max(1, abs(value)), (case, dd, key, res[key], value)
            count += 1
            exact = mod.pooled_upper_bound(
                list(map(float, ps)), arr[0], *[[float(v) for v in vs] for vs in arr[1:]], dd,
                absent_upper=list(map(float, qs)), lambda_mode='square_mgf', product_mode='exact_mgf')
            assert exact['product_radius'] <= res['product_radius'] + 1e-13
            assert exact['linear_a_radius'] == res['linear_a_radius']
            assert exact['linear_b_radius'] == res['linear_b_radius']
    return {
        'case_seed': 1309202699, 'random_data_configurations': 160,
        'formula_comparisons': count, 'additional_exact_product_calls': count,
        'zero_proxy_configurations': zero_cases, 'all_absent_configurations': all_absent,
        'maximum_formula_absolute_difference': maximum, 'roots_u': roots,
        'code_sha256': hashlib.sha256((root / 'pooled_bias_bound.py').read_bytes()).hexdigest(),
        'theory_sha256': hashlib.sha256((root / 'THEORY.md').read_bytes()).hexdigest(),
        'all_passed': True,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('independent_square_implementation_replay.json'))
    args = parser.parse_args()
    receipt = verify()
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
