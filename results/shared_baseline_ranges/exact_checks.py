"""Exact, exhaustive support verification for the shared-baseline design.

Only the Python standard library is required. Every scalar triple has exactly
one of the 13 canonical weak orderings generated here, including all ties.
"""
import argparse
from fractions import Fraction
from itertools import permutations, product
import json
from pathlib import Path


def weak_orders(size=3):
    return tuple(x for x in product(range(size), repeat=size)
                 if set(x) == set(range(max(x) + 1)))


def p(x, i, j):
    return Fraction((x[i] > x[j]) - (x[i] < x[j]), 2)


def scalar_u(x, y):
    return sum((p(x, i, j) * p(y, i, k)
                for i, j, k in permutations(range(3))), Fraction()) / 6


def sorted_scalar_formula(x, y):
    """The analytic scalar-kernel formula after sorting the first map."""
    order = sorted(range(3), key=lambda i: x[i])
    x, y = tuple(x[i] for i in order), tuple(y[i] for i in order)
    sign = lambda i, j: (y[i] > y[j]) - (y[i] < y[j])
    if x[0] == x[2]:
        numerator = 0
    elif x[0] == x[1]:
        numerator = -sign(0, 2) - sign(1, 2)
    elif x[1] == x[2]:
        numerator = -sign(0, 1) - sign(0, 2)
    else:
        numerator = -2 * sign(0, 2)
    return Fraction(numerator, 24)


def score(h, y, b):
    return (p(h, 0, 1) - p(b, 0, 1)) * (p(y, 0, 1) - p(b, 0, 1))


def correction(h, y, b):
    a = p(h, 0, 1) - p(h, 0, 2) - p(b, 0, 1) + p(b, 0, 2)
    c = p(y, 0, 1) - p(y, 0, 2) - p(b, 0, 1) + p(b, 0, 2)
    return a * c / 2


def full_u(h, y, b):
    return sum(((p(h, i, j) - p(b, i, j)) *
                (p(y, i, k) - p(b, i, k))
                for i, j, k in permutations(range(3))), Fraction()) / 6


def enumerate_support():
    orders = weak_orders()
    assert len(orders) == 13
    scalar = [scalar_u(x, y) for x, y in product(orders, repeat=2)]
    assert min(scalar) == -Fraction(1, 12)
    assert max(scalar) == Fraction(1, 12)
    for x, y in product(orders, repeat=2):
        assert scalar_u(x, y) == sorted_scalar_formula(x, y)
    for b in orders:
        assert scalar_u(b, b) == (Fraction(1, 12) if len(set(b)) > 1 else 0)
    kernels = {"S": score, "Q": correction, "U": full_u}
    values = {name: [] for name in kernels}
    witnesses = {name: {} for name in kernels}
    for h, y, b in product(orders, repeat=3):
        assert full_u(h, y, b) == (scalar_u(h, y) - scalar_u(h, b)
                                  - scalar_u(b, y) + scalar_u(b, b))
        for name, fn in kernels.items():
            value = fn(h, y, b)
            values[name].append(value)
            witnesses[name].setdefault(value, {"h": h, "Y": y, "b": b})
    expected = {"S": (-Fraction(1, 4), Fraction(1)),
                "Q": (-Fraction(1, 2), Fraction(2)),
                "U": (-Fraction(1, 6), Fraction(1, 3))}
    result = {"weak_order_count": len(orders), "scalar_pair_cases": len(scalar),
              "shared_baseline_cases": len(orders) ** 3,
              "arithmetic": "fractions.Fraction; exact rational comparisons",
              "weak_orders": orders, "kernels": {}}
    for name, vals in values.items():
        low, high = min(vals), max(vals)
        assert (low, high) == expected[name]
        result["kernels"][name] = {
            "minimum": str(low), "maximum": str(high), "width": str(high - low),
            "support": [str(v) for v in sorted(set(vals))],
            "minimum_witness": witnesses[name][low],
            "maximum_witness": witnesses[name][high],
            "minimum_case_count": vals.count(low), "maximum_case_count": vals.count(high)}
    # Known map identity annihilates each kernel for every possible ordering.
    for y, b in product(orders, repeat=2):
        assert score(b, y, b) == correction(b, y, b) == full_u(b, y, b) == 0
    result["declared_identity_cases"] = len(orders) ** 2
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(enumerate_support(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
