"""Exact independent checks of rank/reference/temporal composition.

Standard library only; no manuscript/release/package imports.  The raw-array
checks form comparison ranks before temporal residuals.  Their oracle uses
separately calculated one-panel conditional moments and design coefficients.
"""
from fractions import Fraction as F
from itertools import product, combinations, permutations
from pathlib import Path
import hashlib
import json
import time


def cmp(x, y):
    return F(int(y < x)) + F(int(y == x), 2)


def states(laws):
    for choices in product(*[tuple(d.items()) for d in laws]):
        row = tuple(c[0] for c in choices)
        p = F(1)
        for _, weight in choices:
            p *= weight
        yield row, p


def dot(x, y):
    return sum((a*b for a, b in zip(x, y)), F())


def moments(laws):
    """Conditional peer moments at focal entity zero, without rank operators."""
    peers = laws[1:]
    p = {x: [sum((pr*cmp(x, y) for y, pr in law.items()), F())
             for law in peers] for x in laws[0]}
    mu = [sum((pr*p[x][j] for x, pr in laws[0].items()), F())
          for j in range(len(peers))]
    omega = [sum((pr*sum((py*(cmp(x, y)-p[x][j])**2
                                 for y, py in peers[j].items()), F())
                          for x, pr in laws[0].items()), F())
             for j in range(len(peers))]
    K = [[sum((pr*(p[x][j]-mu[j])*(p[x][l]-mu[l])
               for x, pr in laws[0].items()), F())
          for l in range(len(peers))] for j in range(len(peers))]
    return p, mu, omega, K


def source_coefficients(ell, a, b, u, v):
    ns = len(b)
    np = len(u[0])
    cx = [[F() for _ in range(np)] for _ in range(ns)]
    dy = [[b[t]*v[t][j] for j in range(np)] for t in range(ns)]
    for s, t in enumerate(ell):
        if t is not None:
            for j in range(np):
                cx[t][j] += a[s]*u[s][j]
    lam = [sum((cx[t][j]*dy[t][j] for t in range(ns)), F())
           for j in range(np)]
    return cx, dy, lam


def temporal_case(name, laws, ell, a, b, u, v, expected=None):
    """Enumerate every independent raw panel array, then construct ranks."""
    ps, mu, omega, K = moments(laws)
    panels = []
    for row, prob in states(laws):
        comparisons = [cmp(row[0], y) for y in row[1:]]
        panels.append((prob, [dot(w, comparisons) for w in u],
                       [dot(w, comparisons) for w in v],
                       [dot(w, ps[row[0]]) for w in u],
                       [dot(w, ps[row[0]]) for w in v]))
    emp = pop = F()
    for arr in product(panels, repeat=len(b)):
        weight = F(1)
        for row in arr:
            weight *= row[0]
        # Constant forecast rows have rank 1/2, as do their population versions.
        px = sum((a[s]*(F(1, 2) if t is None else arr[t][1][s])
                  for s, t in enumerate(ell)), F())
        qy = sum((b[t]*arr[t][2][t] for t in range(len(b))), F())
        pp = sum((a[s]*(F(1, 2) if t is None else arr[t][3][s])
                  for s, t in enumerate(ell)), F())
        qp = sum((b[t]*arr[t][4][t] for t in range(len(b))), F())
        emp += weight*px*qy
        pop += weight*pp*qp
    cx, dy, lam = source_coefficients(ell, a, b, u, v)
    mean_x = sum((a[s]*(F(1, 2) if t is None else dot(u[s], mu))
                  for s, t in enumerate(ell)), F())
    mean_y = sum((b[t]*dot(v[t], mu) for t in range(len(b))), F())
    pop_oracle = mean_x*mean_y + sum(
        (cx[t][j]*K[j][l]*dy[t][l] for t in range(len(b))
         for j in range(len(mu)) for l in range(len(mu))), F())
    interaction = dot(omega, lam)
    assert pop == pop_oracle, (name, pop, pop_oracle)
    assert emp-pop == interaction, (name, emp-pop, interaction)
    if expected is not None:
        assert (emp, pop) == expected, (name, emp, pop, expected)
    return dict(name=name, raw_arrays=len(panels)**len(b),
                empirical=str(emp), induced_population=str(pop),
                interaction=str(interaction), Lambda=list(map(str, lam)),
                omega=list(map(str, omega)), mean_product=str(mean_x*mean_y))


def graph_check(law):
    """Every entry of a nonuniform four-entity cross-reference covariance."""
    n = 4
    U = [[F(0), F(2, 3), F(1, 3), F(0)],
         [F(1, 4), F(0), F(1, 4), F(1, 2)],
         [F(0), F(1), F(0), F(0)],
         [F(1, 2), F(0), F(1, 2), F(0)]]
    V = [[F(0), F(0), F(1, 2), F(1, 2)],
         [F(1), F(0), F(0), F(0)],
         [F(1, 5), F(2, 5), F(0), F(2, 5)],
         [F(0), F(1, 3), F(2, 3), F(0)]]
    edges = list(combinations(range(n), 2))
    def operators(W):
        L = [[F(int(i == j))-W[i][j] for j in range(n)] for i in range(n)]
        B = [[W[i][q] if i == p else -W[i][p] if i == q else F()
              for p, q in edges] for i in range(n)]
        return L, B
    LU, BU = operators(U)
    LV, BV = operators(V)
    psi = {x: sum((pr*cmp(x, y) for y, pr in law.items()), F())-F(1, 2)
           for x in law}
    varpsi = sum((pr*psi[x]**2 for x, pr in law.items()), F())
    tau = sum((px*py*(cmp(x, y)-F(1, 2)-psi[x]+psi[y])**2
               for x, px in law.items() for y, py in law.items()), F())
    s2 = sum((p*p for p in law.values()), F())
    s3 = sum((p*p*p for p in law.values()), F())
    assert varpsi == (1-s3)/12
    assert tau == (1-3*s2+2*s3)/12
    cov = [[F() for _ in range(n)] for _ in range(n)]
    for row, p in states([law]*n):
        qU = [sum((U[i][j]*(cmp(row[i], row[j])-F(1, 2))
                   for j in range(n)), F()) for i in range(n)]
        qV = [sum((V[i][j]*(cmp(row[i], row[j])-F(1, 2))
                   for j in range(n)), F()) for i in range(n)]
        for i in range(n):
            for j in range(n):
                cov[i][j] += p*qU[i]*qV[j]
    for i in range(n):
        for j in range(n):
            assert cov[i][j] == varpsi*dot(LU[i], LV[j])+tau*dot(BU[i], BV[j])
    return dict(raw_arrays=len(law)**n, v=str(varpsi), tau=str(tau),
                omega=str(varpsi+tau), cross_covariance=[[str(x) for x in r] for r in cov])


def continuous_check():
    """Exact integrals under independent Uniform(0,1), partitioned by order."""
    # Integrate comparisons by all equiprobable relative orders of 3 draws.
    # E[U_(k)] = k/4 and E[U_(k)^2] = k(k+1)/20 are exact beta moments.
    cross = F()
    variance = F()
    for ranks in permutations((1, 2, 3)):
        h1 = F(int(ranks[1] < ranks[0]))-F(1, 2)
        h2 = F(int(ranks[2] < ranks[0]))-F(1, 2)
        cross += h1*h2/6
        variance += h1*h1/6
    assert cross == F(1, 12)
    assert variance-cross == F(1, 6)
    return dict(orderings=6, v=str(cross), omega=str(variance-cross),
                tau=str(variance-2*cross), positive_case=str((variance-cross)/2))


def main(output):
    start = time.monotonic()
    binary = {0: F(1, 2), 1: F(1, 2)}
    ternary = {-2: F(1, 3), 0: F(1, 3), 5: F(1, 3)}
    skewed = {-2: F(1, 2), 0: F(1, 3), 5: F(1, 6)}
    half = [F(1, 2)]*2
    e1, e2 = [F(1), F(0)], [F(0), F(1)]
    cases = []
    for label, law, ep, pp in (
        ('binary_original', binary, -F(3, 32), -F(1, 16)),
        ('ternary_original', ternary, -F(13, 108), -F(2, 27)),
        ('skewed_ternary_original', skewed, -F(1, 9), -F(5, 72))):
        cases.append(temporal_case(label, [law]*3, [None, 0, 1],
            [-F(1, 2), F(1), -F(1, 2)], [-F(1, 2), F(1), -F(1, 2)],
            [half]*3, [half]*3, (ep, pp)))
    # Time-2 score P2-P4, Q2-(Q1+Q3)/2.  P4 uses innovation panel 3.
    for label, vv, expected in (
        ('positive_ternary', [e2, half, e1], F(5, 108)),
        ('negative_ternary', [e1, half, e2], -F(5, 108)),
        ('cancelled_ternary', [e1, half, e1], F())):
        cases.append(temporal_case(label, [ternary]*3, [0, 2],
            [F(1), -F(1)], [-F(1, 2), F(1), -F(1, 2)],
            [e1, e1], vv, (expected, F())))
    # Fully past forecast residual and current/future outcome residual.
    cases.append(temporal_case('directional_ternary', [ternary]*3, [None, 0],
        [-F(1), F(1)], [F(0), F(1), -F(1)], [half, half],
        [e1, half, e2], (F(), F())))
    laws = [{-1: F(1, 3), 0: F(1, 3), 2: F(1, 3)},
            {-1: F(1, 4), 2: F(3, 4)},
            {0: F(2, 3), 3: F(1, 3)},
            {-2: F(1, 2), 0: F(1, 2)}]
    u = [[F(1, 2), F(1, 2), F(0)],
         [F(1, 2), F(0), F(1, 2)]]
    v = [[F(1, 4), F(1, 4), F(1, 2)],
         [F(0), F(2, 3), F(1, 3)],
         [F(1, 2), F(1, 2), F(0)]]
    cases.append(temporal_case('heterogeneous_nonbinary_weighted', laws, [0, 2],
        [F(1), -F(1)], [-F(1, 2), F(1), -F(1, 2)], u, v))
    # Aggregate overlap cancellation is insufficient under heterogeneous laws.
    laws2 = [{0: F(1)}, {-1: F(1, 2), 1: F(1, 2)}, {0: F(1)}]
    cases.append(temporal_case('heterogeneous_scalar_cancellation_fails', laws2,
        [0, 2], [F(1), -F(1)], [-F(1, 2), F(1), -F(1, 2)],
        [e1, e2], [e1, half, e2], (-F(1, 8), F())))
    # Source-aligned peer separation removes reference noise but retains feedback.
    cases.append(temporal_case('disjoint_peer_references', [ternary]*3,
        [None, 0, 1], [-F(1, 2), F(1), -F(1, 2)],
        [-F(1, 2), F(1), -F(1, 2)], [e1]*3, [e2]*3,
        (-F(2, 27), -F(2, 27))))
    cases.append(temporal_case('repeated_source_different_peer_weights', [ternary]*3,
        [0, 0, 2], [F(1), -F(1, 4), -F(3, 4)],
        [-F(1, 4), F(1), -F(3, 4)], [e1, half, e2], [e2, half, e1]))
    result = dict(status='ALL_ASSERTIONS_PASSED',
        temporal_cases=cases, raw_temporal_arrays=sum(x['raw_arrays'] for x in cases),
        graph_cases={label: graph_check(law) for label, law in
                     [('binary', binary), ('ternary', ternary), ('skewed_ternary', skewed)]},
        continuous=continuous_check(), runtime_seconds=time.monotonic()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Exact expectation and covariance checks; no calibration or empirical superiority claim')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    main(args.output)
