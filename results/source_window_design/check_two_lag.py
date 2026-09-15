"""Independent exact checks for ranks of raw two-lag sums.

Standard library only. No manuscript/release imports. All reported equalities
use rational arithmetic; temporal checks enumerate raw entity-time arrays and
form ranks of sums before temporal residuals. Expectations from the theorem
are checked against that separate raw-array construction.
"""
from fractions import Fraction as F
from itertools import product
from math import asin, pi, sqrt, lcm
from pathlib import Path
import hashlib
import json
import time


def h(x, y):
    return F((x > y) - (x < y), 2)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F())


def meanrank(law):
    return {x: sum((p*h(x, y) for y, p in law.items()), F()) for x in law}


def sumlaw(law, weights=(1, 1)):
    out = {}
    for x, y in product(law, repeat=2):
        z = weights[0]*x + weights[1]*y
        out[z] = out.get(z, F()) + law[x]*law[y]
    return out


def constants(law, weights=(1, 1)):
    H = sumlaw(law, weights)
    mf, mh = meanrank(law), meanrank(H)
    kappa, eta = [], []
    for lag in (0, 1):
        kappa.append(sum((law[x]*law[y]*mh[weights[0]*x+weights[1]*y]
                         *mf[(x, y)[lag]] for x, y in product(law, repeat=2)), F()))
        eta.append(sum((law[x]*law[y]*law[u]*law[v]
                       *h(weights[0]*x+weights[1]*y, weights[0]*u+weights[1]*v)
                       *h((x, y)[lag], (u, v)[lag])
                       for x, y, u, v in product(law, repeat=4)), F()))
    lam = [e-k for e, k in zip(eta, kappa)]
    s2 = sum((p*p for p in law.values()), F())
    if weights == (1, 1):
        assert eta[0] == eta[1] == (1-s2*s2)/8
        assert kappa[0] == kappa[1]
    assert min(kappa+lam) >= 0
    return kappa, lam, eta


def temporal(name, law, ns, forecast_times, a, b, u, v, weights=(1, 1)):
    """Form unstandardized ranks from every raw panel, then residual products.

    weights=(recent, older). Forecast at s uses raw columns s-1 and s-2.
    b indexes outcome source time; a and u index forecast_times.
    Integer scaling in the loop preserves exact arithmetic and only avoids
    expensive Fraction additions for every raw panel.
    """
    n = len(v[0])+1
    mf, mh = meanrank(law), meanrank(sumlaw(law, weights))
    panels = tuple(product(law, repeat=n))
    q = [tuple(dot(v[t], [h(row[0], z) for z in row[1:]])
               for row in panels) for t in range(ns)]
    qp = [mf[row[0]] for row in panels]
    xp, xpp = [], []
    for us in u:
        pe, po = [], []
        for recent in panels:
            for older in panels:
                xx = [weights[0]*x+weights[1]*y for x, y in zip(recent, older)]
                pe.append(dot(us, [h(xx[0], z) for z in xx[1:]]))
                po.append(mh[xx[0]])
        xp.append(pe)
        xpp.append(po)
    values = [z for row in q+xp+xpp for z in row]+qp+a+b
    scale = lcm(*(z.denominator for z in values))
    aq = [[int(z*scale) for z in row] for row in q]
    ax = [[int(z*scale) for z in row] for row in xp]
    axp = [[int(z*scale) for z in row] for row in xpp]
    aqp = [int(z*scale) for z in qp]
    aa, bb = [int(z*scale) for z in a], [int(z*scale) for z in b]
    den = lcm(*(p.denominator for p in law.values()))
    panel_mass = []
    for row in panels:
        mass = 1
        for x in row:
            mass *= int(law[x]*den)
        panel_mass.append(mass)
    se = sp = 0
    m = len(panels)
    for raw in product(range(m), repeat=ns):
        mass = 1
        for z in raw:
            mass *= panel_mass[z]
        A = sum(aa[k]*ax[k][raw[s-1]*m+raw[s-2]] for k, s in enumerate(forecast_times))
        A0 = sum(aa[k]*axp[k][raw[s-1]*m+raw[s-2]] for k, s in enumerate(forecast_times))
        B = sum(bb[t]*aq[t][raw[t]] for t in range(ns))
        B0 = sum(bb[t]*aqp[raw[t]] for t in range(ns))
        se += mass*A*B
        sp += mass*A0*B0
    empirical, population = F(se, scale**4*den**(n*ns)), F(sp, scale**4*den**(n*ns))
    T, omega = [F(), F()], [F(), F()]
    for k, s in enumerate(forecast_times):
        for lag, t in enumerate((s-1, s-2)):
            T[lag] += a[k]*b[t]
            omega[lag] += a[k]*b[t]*dot(u[k], v[t])
    kap, lam, _ = constants(law, weights)
    assert population == dot(kap, T), (name, population, dot(kap, T))
    assert empirical-population == dot(lam, omega), (name, empirical-population, dot(lam, omega))
    return dict(name=name, raw_arrays=len(law)**(n*ns), empirical=str(empirical),
                population=str(population), interaction=str(empirical-population),
                T_by_lag=list(map(str, T)), Omega_by_lag=list(map(str, omega)),
                weights=list(weights))


def graph_check(law):
    """Every cross-focal entry, including reversed shared edges, from raw data."""
    U = [[F(), F(2, 3), F(1, 3)], [F(1), F(), F()], [F(1, 2), F(1, 2), F()]]
    V = [[F(), F(), F(1)], [F(1, 4), F(), F(3, 4)], [F(1), F(), F()]]
    kap, lam, _ = constants(law)
    cov = [[F() for _ in range(3)] for _ in range(3)]
    for raw in product(law, repeat=6):
        p = F(1)
        for x in raw:
            p *= law[x]
        yy = raw[:3]
        xx = [raw[i]+raw[3+i] for i in range(3)]
        pr = [sum((U[i][j]*h(xx[i], xx[j]) for j in range(3)), F()) for i in range(3)]
        qr = [sum((V[i][j]*h(yy[i], yy[j]) for j in range(3)), F()) for i in range(3)]
        for i, j in product(range(3), repeat=2):
            cov[i][j] += p*pr[i]*qr[j]
    for i, j in product(range(3), repeat=2):
        li = [F(k == i)-U[i][k] for k in range(3)]
        lj = [F(k == j)-V[j][k] for k in range(3)]
        edge = dot(U[i], V[j]) if i == j else -U[i][j]*V[j][i]
        assert cov[i][j] == kap[0]*dot(li, lj)+(lam[0]-kap[0])*edge
    return dict(raw_arrays=len(law)**6, covariance=[[str(z) for z in row] for row in cov])


def validation_check(law):
    """Six-draw paired-difference validation kernel, directly from raw draws."""
    avg = F()
    lo, hi = F(1), F(-1)
    for x, y, u, v, z, w in product(law, repeat=6):
        R = (h(x+y, u+v)-h(x+y, z+w))*(h(x, u)-h(x, z))/2
        p = law[x]*law[y]*law[u]*law[v]*law[z]*law[w]
        avg += p*R
        lo, hi = min(lo, R), max(hi, R)
    assert avg == constants(law)[1][0]
    assert -F(1, 2) <= lo <= hi <= F(1, 2)
    return dict(raw_arrays=len(law)**6, mean=str(avg), observed_kernel_min=str(lo),
                observed_kernel_max=str(hi))


def uniform_check():
    # E[M_H(x+Y)-1/2] for uniform raw Y, obtained by splitting the
    # triangular sum-CDF integral at y=1-x. Integrate against x-1/2.
    polynomial = [-F(1, 3), F(1, 2), F(1, 2), -F(1, 3)]
    kappa = sum((coefficient*(F(1, k+2)-F(1, 2*(k+1)))
                 for k, coefficient in enumerate(polynomial)), F())
    assert kappa == F(7, 120)
    return dict(kappa=str(kappa), lambda_=str(F(1, 8)-kappa), eta='1/8')


def main(out):
    start = time.monotonic()
    binary = {0: F(1, 2), 1: F(1, 2)}
    ternary = {0: F(1, 3), 1: F(1, 3), 2: F(1, 3)}
    laws = {'binary_half': binary, 'binary_skew': {0: F(9, 10), 1: F(1, 10)},
            'ternary': ternary, 'gapped_ternary': {-2: F(1, 3), 0: F(1, 3), 5: F(1, 3)},
            'skew_ternary': {-2: F(1, 2), 0: F(1, 3), 5: F(1, 6)}}
    cs = {}
    for name, law in laws.items():
        k, lam, eta = constants(law)
        cs[name] = dict(kappa=str(k[0]), lambda_=str(lam[0]), eta=str(eta[0]))
    assert constants(binary)[0][0] == F(3, 64)
    assert constants(ternary)[0][0] == F(13, 243)
    assert constants(ternary)[1][0] == F(14, 243)
    e1, e2, half = [F(1), F()], [F(), F(1)], [F(1, 2)]*2
    cases = []
    # Literal forecast evaluation at r=2; nuisance forecast rank at s=5.
    # Outcome fits use times 0 and 4. All weights are fixed and normalized.
    bb = [-F(1, 2), F(), F(1), F(), -F(1, 2)]
    for name, v0, v4, want in [('positive', e2, e1, F(3, 128)),
                              ('negative', e1, e2, -F(3, 128)),
                              ('signed_cancellation', e1, e1, F())]:
        vv = [v0, half, half, half, v4]
        c = temporal(name, binary, 5, [2, 5], [F(1), -F(1)], bb, [e1, e1], vv)
        assert c['population'] == '0' and c['empirical'] == str(want)
        cases.append(c)
    # Forecast mean fitted earlier, outcome mean fitted later: full source separation.
    c = temporal('directional', binary, 5, [3, 2], [F(1), -F(1)],
                 [F(), F(), F(), F(1), -F(1)], [half]*2, [half]*5)
    assert c['empirical'] == c['population'] == '0'
    cases.append(c)
    # Disjoint peer references preserve a NONZERO population target.
    c = temporal('peers_separated', ternary, 3, [2, 3], [F(1), -F(1)],
                 [-F(1, 2), -F(1, 2), F(1)], [e1]*2, [e2]*3)
    assert c['interaction'] == '0' and c['population'] == '-13/162'
    cases.append(c)
    # The naive unweighted scalar cancels, but unequal component weights do not.
    c = temporal('unequal_lag_scalar_counterexample', binary, 5, [2, 5],
                 [F(1), -F(1)], bb, [e1]*2, [e1]*5, weights=(2, 1))
    assert sum(map(F, c['Omega_by_lag'])) == 0 and c['interaction'] == '1/64'
    assert c['empirical'] == '1/32' and c['population'] == '1/64'
    cases.append(c)
    result = dict(status='ALL_ASSERTIONS_PASSED', temporal_cases=cases,
                  raw_temporal_arrays=sum(c['raw_arrays'] for c in cases),
                  constants=cs, graph={name: graph_check(law) for name, law in laws.items()},
                  validation={name: validation_check(law) for name, law in laws.items()},
                  gaussian=dict(kappa=asin(1/(2*sqrt(2)))/(2*pi),
                                lambda_=(pi/4-asin(1/(2*sqrt(2))))/(2*pi), eta=1/8),
                  uniform_exact=uniform_check(),
                  elapsed_seconds=round(time.monotonic()-start, 3))
    result['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'raw_temporal_arrays', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    main(args.output)
