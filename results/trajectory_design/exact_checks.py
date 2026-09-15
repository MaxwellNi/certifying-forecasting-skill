"""Rational finite-law oracle. Does not import trajectory_kernel or NumPy.

Run directly for the independent algebra checks. All probabilities, midranks,
coefficients, expectations, and equality tests below use Fraction arithmetic.
"""

from fractions import Fraction as F
from itertools import product, permutations
import json


def rank(x, y):
    return F(1) if y < x else F(1, 2) if y == x else F(0)


def dot(x, y):
    return sum((a * b for a, b in zip(x, y)), F(0))


def expectation(law, order, kernel):
    total = F(0)
    for draws in product(law, repeat=order):
        probability = F(1)
        for _, weight in draws:
            probability *= weight
        total += probability * kernel(tuple(point for point, _ in draws))
    return total


def parts(law, fs, gs, C, D):
    c, d = list(map(sum, C)), list(map(sum, D))
    omega = [[dot(row, other) for other in D] for row in C]
    mf = lambda x, f: sum(p * (rank(f(x), f(y)) - F(1, 2)) for y, p in law)
    K, lam = [], []
    for f in fs:
        kr, lr = [], []
        for g in gs:
            k = sum(p * mf(x, f) * mf(x, g) for x, p in law)
            same = expectation(law, 2, lambda z: (rank(f(z[0]), f(z[1])) - F(1, 2)) * (rank(g(z[0]), g(z[1])) - F(1, 2)))
            kr.append(k)
            lr.append(same - k)
        K.append(kr)
        lam.append(lr)
    theta = sum(c[s] * d[t] * K[s][t] for s in range(len(fs)) for t in range(len(gs)))
    gamma = sum(omega[s][t] * lam[s][t] for s in range(len(fs)) for t in range(len(gs)))
    return theta, gamma, K, lam, omega


def shared(z, fs, gs, C, D):
    a = sum(C[s][j] * (rank(f(z[0]), f(z[j + 1])) - F(1, 2)) for s, f in enumerate(fs) for j in range(len(C[0])))
    b = sum(D[t][j] * (rank(g(z[0]), g(z[j + 1])) - F(1, 2)) for t, g in enumerate(gs) for j in range(len(D[0])))
    return a * b


def validator(z, fs, gs, omega):
    return F(1, 2) * sum(
        omega[s][t] * (rank(f(z[0]), f(z[1])) - rank(f(z[0]), f(z[2])))
        * (rank(g(z[0]), g(z[1])) - rank(g(z[0]), g(z[2])))
        for s, f in enumerate(fs) for t, g in enumerate(gs)
    )


def direct(z, fs, gs, C, D, full=True):
    c, d = list(map(sum, C)), list(map(sum, D))
    roles = list(permutations(range(len(z)), 3)) if full else [(0, j, k) for j in range(1, len(z)) for k in range(1, len(z)) if j != k]
    return sum(
        sum(c[s] * (rank(f(z[i]), f(z[j])) - F(1, 2)) for s, f in enumerate(fs))
        * sum(d[t] * (rank(g(z[i]), g(z[k])) - F(1, 2)) for t, g in enumerate(gs))
        for i, j, k in roles
    ) / len(roles)


def run_checks():
    results = {}
    bern = [(0, F(1, 2)), (1, F(1, 2))]
    tern = [(i, F(1, 3)) for i in range(3)]
    copy = [lambda x: x]
    one = [[F(1)]]
    for name, law, expected in (("binary_constants", bern, (F(1, 16), F(1, 16))), ("ternary_constants", tern, (F(2, 27), F(5, 54)))):
        theta, gamma, *_ = parts(law, copy, copy, one, one)
        assert (theta, gamma) == expected
        results[name] = list(map(str, expected))
    theta, gamma, *_ = parts(bern, copy, [lambda x: -x], one, one)
    assert (theta, gamma) == (F(-1, 16), F(-1, 16))
    results["signed_interaction"] = str(gamma)

    law = []
    for z in product((0, 1), repeat=3):
        law.append((z, F(1, 4) * (F(1, 10) if z[2] else F(9, 10))))
    fs = [lambda z: z[0], lambda z: z[2]]
    gs = [lambda z: z[0], lambda z: z[1], lambda z: z[2]]
    C = [[F(1), F(0)], [F(-1), F(0)]]
    D = [[F(1, 2), F(0)], [F(-1, 2), F(-1, 2)], [F(1, 2), F(0)]]
    theta, gamma, _, _, omega = parts(law, fs, gs, C, D)
    assert (theta, gamma) == (F(1, 50), F(1, 50))
    assert omega[0][0] + omega[1][2] == 0
    assert expectation(law, 3, lambda z: shared(z, fs, gs, C, D)) == theta + gamma
    assert expectation(law, 3, lambda z: validator(z, fs, gs, omega)) == gamma
    assert expectation(law, 3, lambda z: direct(z, fs, gs, C, D)) == theta
    results["heterogeneous_sources_scalar_cancel"] = {"theta": str(theta), "gamma": str(gamma), "shared": str(theta + gamma)}

    dependent = [((y, y), p) for y, p in bern]
    theta, gamma, *_ = parts(dependent, [lambda z: z[0]], [lambda z: z[1]], one, one)
    assert (theta, gamma) == (F(1, 16), F(1, 16))
    results["dependent_distinct_coordinate_ids"] = str(gamma)

    nonlinear_law = [((0, 0), F(1, 5)), ((1, 0), F(1, 2)), ((1, 1), F(3, 10))]
    fs = [lambda z: z[0] ^ z[1], lambda z: 2 * z[0] + z[1]]
    gs = [lambda z: z[0] * z[1], lambda z: z[0] - z[1]]
    C = [[F(1, 2), F(1, 3), F(1, 6)], [F(-1), F(0), F(0)]]
    D = [[F(0), F(1), F(0)], [F(1, 2), F(-1, 2), F(0)]]
    for h in (0, 1):
        fh = fs if h == 0 else [lambda z: -(z[0] ^ z[1]), fs[1]]
        theta, gamma, _, _, omega = parts(nonlinear_law, fh, gs, C, D)
        assert expectation(nonlinear_law, 4, lambda z: shared(z, fh, gs, C, D)) == theta + gamma
        assert expectation(nonlinear_law, 3, lambda z: validator(z, fh, gs, omega)) == gamma
        for full in (False, True):
            assert expectation(nonlinear_law, 4, lambda z: direct(z, fh, gs, C, D, full)) == theta
        results[f"external_training_branch_{h}"] = {"theta": str(theta), "gamma": str(gamma)}

    # Each coefficient pair can be isolated; differing binary/ternary ratios
    # force both entries in the full-centering criterion to vanish.
    determinant = F(1, 16) * F(5, 54) - F(1, 16) * F(2, 27)
    assert determinant != 0
    results["universal_centering_determinant"] = str(determinant)
    C0, D0 = [[F(1), F(0)]], [[F(0), F(1)]]
    theta, gamma, *_ = parts(tern, copy, copy, C0, D0)
    assert gamma == 0 and theta == F(2, 27)
    assert expectation(tern, 3, lambda z: shared(z, copy, copy, C0, D0)) == theta
    results["disjoint_reference_target_preserved"] = str(theta)

    contaminated = [(z, F(1, 2)) for z in ((0, 0), (1, 1))]
    actual = sum(p * (rank(z[0], z[1]) - F(1, 2)) ** 2 for z, p in contaminated)
    assert actual == 0 and F(1, 16) + F(1, 16) == F(1, 8)
    results["contaminated_training_actual_vs_naive"] = [str(actual), "1/8"]

    pooled = [(0, F(1, 4)), (1, F(1, 4)), (10, F(1, 4)), (11, F(1, 4))]
    moments = [sum(p * rank(y, ref) for ref, p in pooled) for y, _ in bern]
    second = sum(p * (m - F(1, 2)) ** 2 for (_, p), m in zip(bern, moments))
    mean = sum(p * m for (_, p), m in zip(bern, moments))
    variance = sum(p * (m - mean) ** 2 for (_, p), m in zip(bern, moments))
    assert (second, variance) == (F(5, 64), F(1, 64))
    results["pooled_reference_changed_target"] = list(map(str, (second, variance)))

    # The third finite difference cannot be generated by a <=2-draw kernel.
    theta = lambda t: (1 - t ** 3 - (1 - t) ** 3 / 4) / 12
    third_difference = theta(F(3, 4)) - 3 * theta(F(1, 2)) + 3 * theta(F(1, 4)) - theta(F(0))
    assert third_difference == F(-3, 512)
    results["minimum_generic_unbiased_order_three"] = str(third_difference)

    wrong_validation = expectation(bern, 5, lambda z: F(1, 2) * (rank(z[0], z[1]) - rank(z[0], z[2])) * (rank(z[0], z[3]) - rank(z[0], z[4])))
    assert wrong_validation == 0
    results["four_independent_peers_do_not_validate"] = str(wrong_validation)

    support = set()
    for x, y in product(product(range(3), repeat=3), repeat=2):
        support.add(direct(tuple(zip(x,y)), [lambda z: z[0]], [lambda z: z[1]], one, one))
    assert support == {F(-1,12), F(-1,24), F(0), F(1,24), F(1,12)}
    results["full_u_sharp_729_weak_order_cases"] = list(map(str, sorted(support)))
    return {"status": "PASS", "check_count": len(results), "arithmetic": "fractions.Fraction only", "checks": results}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
