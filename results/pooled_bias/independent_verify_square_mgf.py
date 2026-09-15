"""Enumerate finite laws and partitions for the square-normalized bound."""

from fractions import Fraction as F
from itertools import combinations, product
from math import exp, log, sqrt, factorial
from pathlib import Path
import argparse
import json


def comp(n, k=4):
    if k == 1:
        yield (n,)
        return
    for j in range(n + 1):
        for r in comp(n - j, k - 1):
            yield (j,) + r


def cellstates(n, q, f, g):
    outs = [(F(x) - f, F(y) - g) for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    for ns in comp(n):
        prob = F(factorial(n))
        A, B = [], []
        for z, qq, (aa, bb) in zip(ns, q, outs):
            prob *= qq**z / F(factorial(z))
            A.extend([aa] * z)
            B.extend([bb] * z)
        if not prob:
            continue
        k, ell = n // 2, n - n // 2
        sa, sb = sum(A), sum(B)
        U = (sa * sb - sum(a * b for a, b in zip(A, B))) / (n * (n - 1))
        ma, mb = sa / n, sb / n
        va = (mb * mb + F(k, n * ell) * (sum(b * b for b in B) - sb * mb) / (n - 1)) / (4 * k)
        vb = (ma * ma + F(ell, n * k) * (sum(a * a for a in A) - sa * ma) / (n - 1)) / (4 * ell)
        yield prob, (U, ma, mb, va, vb, A, B)


def rootu(x):
    lo, hi = 1.0, 2.0
    while hi - 1 - log(hi) < 2 * x:
        hi *= 2
    for _ in range(90):
        mid = (lo + hi) / 2
        if mid - 1 - log(mid) < 2 * x:
            lo = mid
        else:
            hi = mid
    return hi


def verify():
    laws = [
        ([F(1, 4)] * 4, F(1, 2), F(1, 2)),
        ([F(1, 2), 0, 0, F(1, 2)], F(1, 2), F(1, 2)),
        ([0, F(1, 2), F(1, 2), 0], F(1, 2), F(1, 2)),
        ([F(7, 10), F(1, 10), F(1, 10), F(1, 10)], F(1, 8), F(7, 8)),
        ([F(1, 10), F(1, 10), F(1, 10), F(7, 10)], 0, 0),
        ([F(1, 2), 0, F(1, 2), 0], F(1, 3), 0),
        ([F(1, 2), F(1, 2), 0, 0], 0, F(2, 3)),
        ([1, 0, 0, 0], 0, 0),
    ]
    cases = [([F(1)], [(n,) + law]) for n in [2, 3, 4, 8] for law in laws]
    cases += [([F(1, 5), F(4, 5)], [(2,) + laws[i], (3,) + laws[(i + 3) % len(laws)]]) for i in range(8)]
    records = []
    zero_events = partition_cases = 0
    max_cauchy_gap = F(0)
    max_normalized_mgf_ratio = 0
    for ci, (ps, cells) in enumerate(cases):
        aa = [q[2] + q[3] - f for n, q, f, g in cells]
        bb = [q[1] + q[3] - g for n, q, f, g in cells]
        target = sum(p * a * b for p, a, b in zip(ps, aa, bb))
        D = sum((float(p) / (4 * sqrt((c[0] // 2) * (c[0] - c[0] // 2)))) ** 2 for p, c in zip(ps, cells))
        dm = max(float(p) / (4 * sqrt((c[0] // 2) * (c[0] - c[0] // 2))) for p, c in zip(ps, cells))
        mgf = {(w, t): 0.0 for w in range(2) for t in [.1, .5, .9]}
        fail = {dd: 0.0 for dd in [.05, .2, .8]}
        linear_tail = {(w, dd): 0.0 for w in range(2) for dd in [.05, .2, .8]}
        probtotal, statecount = F(0), 0
        for states in product(*(list(cellstates(*cell)) for cell in cells)):
            prob = F(1)
            for pr, _ in states:
                prob *= pr
            zs = [z for _, z in states]
            U = sum(p * z[0] for p, z in zip(ps, zs))
            VA = sum(p * p * z[3] for p, z in zip(ps, zs))
            VB = sum(p * p * z[4] for p, z in zip(ps, zs))
            LA = sum(p * a * z[2] for p, a, z in zip(ps, aa, zs)) - U
            LB = sum(p * b * z[1] for p, b, z in zip(ps, bb, zs)) - U
            probtotal += prob
            statecount += 1
            for w, (L, V) in enumerate([(LA, VA), (LB, VB)]):
                if V == 0:
                    assert L == 0
                    zero_events += 1
                    Q = F(0)
                else:
                    Q = L * L / V
                for t in [.1, .5, .9]:
                    mgf[w, t] += float(prob) * exp(t * float(Q) / 2)
                for dd in [.05, .2, .8]:
                    linear_tail[w, dd] += float(prob) * (float(Q) > rootu(log(3 / dd)) + 1e-12)
            for dd in [.05, .2, .8]:
                x = log(3 / dd)
                u = rootu(x)
                upper = float(U) + sqrt(u * float(VA)) + sqrt(u * float(VB)) + sqrt(2 * x * D) + x * dm
                fail[dd] += float(prob) * (float(target) > upper + 1e-12)
            if max(c[0] for c in cells) <= 4:
                splits = []
                for z, c in zip(zs, cells):
                    A, B = z[5:7]
                    n, k = c[0], c[0] // 2
                    ell = n - k
                    choices = []
                    for left in combinations(range(n), k):
                        right = [i for i in range(n) if i not in left]
                        choices.append((sum(A[i] for i in left) / k, sum(B[i] for i in right) / ell))
                    splits.append(choices)
                ratios, values, ct = [F(0), F(0)], [F(0)] * 5, 0
                for ss in product(*splits):
                    TS = sum(p * a * b for p, (a, b) in zip(ps, ss))
                    LAS = -sum(p * b * (a - ac) for p, (a, b), ac in zip(ps, ss, aa))
                    LBS = -sum(p * a * (b - bc) for p, (a, b), bc in zip(ps, ss, bb))
                    VAS = sum(p * p * b * b / (4 * (c[0] // 2)) for p, (a, b), c in zip(ps, ss, cells))
                    VBS = sum(p * p * a * a / (4 * (c[0] - c[0] // 2)) for p, (a, b), c in zip(ps, ss, cells))
                    ct += 1
                    for j, value in enumerate([TS, LAS, LBS, VAS, VBS]):
                        values[j] += value
                    for j, (LS, VS) in enumerate([(LAS, VAS), (LBS, VBS)]):
                        if VS == 0:
                            assert LS == 0
                        else:
                            ratios[j] += LS * LS / VS
                assert [v / ct for v in values] == [U, LA, LB, VA, VB]
                for rr, L, V in zip(ratios, [LA, LB], [VA, VB]):
                    lhs = L * L / V if V else F(0)
                    gap = rr / ct - lhs
                    assert gap >= 0
                    max_cauchy_gap = max(max_cauchy_gap, gap)
                partition_cases += 1
        assert probtotal == 1
        for (w, t), value in mgf.items():
            bound = 1 / sqrt(1 - t)
            assert value <= bound + 1e-12
            max_normalized_mgf_ratio = max(max_normalized_mgf_ratio, value / bound)
        for (w, dd), value in linear_tail.items():
            assert value <= dd / 3 + 1e-12
        for dd, value in fail.items():
            assert value <= dd + 1e-12
        records.append({'case': ci, 'cells': len(cells), 'states': statecount,
                        'total_noncoverage': fail,
                        'linear_noncoverage': {str(k): v for k, v in linear_tail.items()},
                        'square_mgf': {str(k): v for k, v in mgf.items()}})
    return {
        'scope': 'Independent Fraction algebra and full sufficient-state enumeration; no producer imports; square-MGF and subgamma product only.',
        'law_configurations': len(cases),
        'total_states': sum(r['states'] for r in records),
        'exact_partition_states': partition_cases,
        'zero_proxy_events_checked': zero_events,
        'max_cauchy_slack': float(max_cauchy_gap),
        'max_square_mgf_ratio_to_bound': max_normalized_mgf_ratio,
        'max_total_noncoverage': max(v for r in records for v in r['total_noncoverage'].values()),
        'all_passed': True, 'records': records,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('independent_square_mgf_replay.json'))
    args = parser.parse_args()
    receipt = verify()
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'records'}, indent=2))
