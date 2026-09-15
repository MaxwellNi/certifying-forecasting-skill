"""Exact witnesses for reference cost and sequential scope; standard library only.

These checks certify the stated finite arithmetic, not arbitrary-case proofs.
Run: python exact_checks.py
"""

from collections import Counter
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path


def sequential_witness():
    hist = Counter()
    for bits in product((0, 1), repeat=6):
        x, y = bits[:3], bits[3:]
        p01, p02 = F(x[0] - x[1], 2), F(x[0] - x[2], 2)
        q01, q02 = F(y[0] - y[1], 2), F(y[0] - y[2], 2)
        s = p01 * q01
        q = (p01 - p02) * (q01 - q02) / 2
        hist[s - q] += 1
    expectation = sum(r * count for r, count in hist.items()) / 64
    second_moment = sum(r * r * count for r, count in hist.items()) / 64
    e2 = sum((1 + r) ** 2 * count for r, count in hist.items()) / 64
    positive_mass = F(sum(count for r, count in hist.items() if r > 0), 64)
    assert expectation == 0
    assert second_moment == F(1, 64)
    assert e2 == F(65, 64)
    assert positive_mass == F(7, 32)
    assert min(hist) == F(-1, 4) and max(hist) == F(1, 4)
    # P(R>0) is the exact probability of eventual threshold crossing for
    # E_t=(1+R)^t at every fixed threshold greater than one.
    return {
        "number_of_equiprobable_binary_configurations": 64,
        "R_value_counts": {str(r): count for r, count in sorted(hist.items())},
        "E_R": str(expectation),
        "E_R_squared": str(second_moment),
        "E_of_claimed_eprocess_at_t2": str(e2),
        "eventual_rejection_probability": str(positive_mass),
        "nominal_alpha_example": "1/20",
    }


def dot(x, y):
    return sum((a * b for a, b in zip(x, y)), F(0))


def independent_stream_witness():
    """Retain within-round separation of the score pair and Q triple."""
    hist = Counter()
    for bits in product((0, 1), repeat=10):
        x, y = bits[:5], bits[5:]
        s = F((x[0] - x[1]) * (y[0] - y[1]), 4)
        # The Q focal is role 2; binary comparison differences cancel it.
        q = F((x[4] - x[3]) * (y[4] - y[3]), 8)
        hist[s - q] += 1
    mean = sum(r * count for r, count in hist.items()) / 1024
    variance = sum(r * r * count for r, count in hist.items()) / 1024
    e2 = sum((1 + r) ** 2 * count for r, count in hist.items()) / 1024
    positive_mass = F(sum(count for r, count in hist.items() if r > 0), 1024)
    assert mean == 0 and variance == F(5, 256)
    assert e2 == F(261, 256) and positive_mass == F(7, 32)
    return {"number_of_equiprobable_binary_configurations": 1024,
            "independent_evaluation_pair_and_validation_triple_within_round": True,
            "R_value_counts": {str(r): count for r, count in sorted(hist.items())},
            "E_R": str(mean), "E_R_squared": str(variance),
            "E_of_claimed_eprocess_at_t2": str(e2),
            "eventual_rejection_probability": str(positive_mass)}


def omega(c, d):
    return [[dot(row, other) for other in d] for row in c]


def flat_score(c, comparisons):
    return sum((dot(row, comp) for row, comp in zip(c, comparisons)), F(0))


def pruning_witnesses():
    # Two maps per channel, two nonzero-eligible roles; the middle role is
    # exactly unused by both channels. All 3^8 coefficient configurations.
    coefficient_cases = 0
    for values in product((-1, 0, 1), repeat=8):
        c = [[F(values[0]), F(0), F(values[1])],
             [F(values[2]), F(0), F(values[3])]]
        d = [[F(values[4]), F(0), F(values[5])],
             [F(values[6]), F(0), F(values[7])]]
        cp = [[row[0], row[2]] for row in c]
        dp = [[row[0], row[2]] for row in d]
        assert list(map(sum, c)) == list(map(sum, cp))
        assert list(map(sum, d)) == list(map(sum, dp))
        assert omega(c, d) == omega(cp, dp)
        # A concrete admissible comparison array including ties.
        p = [[F(-1, 2), F(1, 2), F(0)], [F(0), F(-1, 2), F(1, 2)]]
        q = [[F(1, 2), F(0), F(-1, 2)], [F(-1, 2), F(1, 2), F(0)]]
        pp = [[row[0], row[2]] for row in p]
        qp = [[row[0], row[2]] for row in q]
        assert flat_score(c, p) == flat_score(cp, pp)
        assert flat_score(d, q) == flat_score(dp, qp)
        assert sum(abs(x) for row in c for x in row) == sum(abs(x) for row in cp for x in row)
        assert sum(abs(x) for row in d for x in row) == sum(abs(x) for row in dp for x in row)
        coefficient_cases += 1
    assert coefficient_cases == 6561
    return {"exact_coefficient_cases": coefficient_cases,
            "row_sums_omega_norms_and_sample_scores_preserved": True}


def main():
    result = {"sequential_counterexample": sequential_witness(),
              "separate_stream_sequential_counterexample": independent_stream_witness(),
              "zero_column_pruning": pruning_witnesses()}
    target = Path(__file__).with_name("exact_checks.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
