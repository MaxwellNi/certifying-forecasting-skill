"""Independent high-precision checks; no new data or benchmark outcomes."""
from decimal import Context, Decimal
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import numpy as np

from finite_support_betting import FiniteSupportBetting
from profiled_betting import profiled_betting, DEFAULT_FRACTIONS


def reference(model, counts):
    ctx = Context(prec=110, Emin=-999999999, Emax=999999999)
    evidence = Decimal(0)
    for b in model.fractions:
        value = Decimal(0)
        for x, count in zip(model.support, counts):
            factor = 1+b*x/(-model.lower)
            argument = ctx.divide(Decimal(factor.numerator), Decimal(factor.denominator))
            value = ctx.add(value, ctx.multiply(Decimal(int(count)), ctx.ln(argument)))
        evidence = ctx.add(evidence, ctx.exp(value))
    evidence = ctx.divide(evidence, Decimal(len(model.fractions)))
    p = min(Decimal(1), ctx.divide(Decimal(1), evidence))
    return evidence, p, ctx.ln(evidence)


def main():
    checked = 0
    max_gap = 0.
    rng = np.random.default_rng(99832)
    for lower in [Fraction(-1, 2), Fraction(-1, 8), Fraction(-1, 12)]:
        model = FiniteSupportBetting([lower, 0, -lower], lower)
        fixtures = [[0, 0, 0], [0, 0, 8192], [8192, 0, 0], [4096, 0, 4096]]
        fixtures += [rng.multinomial(8192, [.2, .5, .3]).tolist() for _ in range(10)]
        for counts in fixtures:
            result = model.evaluate(counts)
            exact_e, exact_p, exact_log = reference(model, counts)
            assert Decimal.from_float(result['p']) >= exact_p
            assert Decimal.from_float(result['log_e_lower']) <= exact_log
            assert Decimal.from_float(result['log_e_upper']) >= exact_log
            assert model.pvalue(counts) == result['p']
            max_gap = max(max_gap, result['log_e_gap_upper'])
            checked += 1

    # Exhaust all 11 fixed-size count outcomes at the symmetric mean-zero null.
    null_model = FiniteSupportBetting([Fraction(-1, 8), Fraction(1, 8)], Fraction(-1, 8))
    mean_e = 0.
    rejection = {a: 0. for a in [.01, .05, .2, .5]}
    for plus in range(11):
        counts = [10-plus, plus]
        exact_e, _, _ = reference(null_model, counts)
        probability = math.comb(10, plus)/2**10
        mean_e += probability*float(exact_e)
        p = null_model.pvalue(counts)
        for alpha in rejection:
            rejection[alpha] += probability*(p <= alpha)
    assert abs(mean_e-1) < 1e-12
    assert all(prob <= alpha for alpha, prob in rejection.items())

    # C=1 profiled comparator has exact minimizer shifted eta=1/2; match it.
    support = [Fraction(-1, 8), Fraction(0), Fraction(1, 8)]
    counts = [10, 30, 110]
    vals = np.repeat([float(v) for v in support], counts)
    prepared = {'cells': [{'mass': 1., 'pairs': len(vals), 'values': vals}],
                'total_raw_rows_charged': 4*len(vals)}
    fast = FiniteSupportBetting(support, Fraction(-1, 2), fractions=DEFAULT_FRACTIONS)
    general = profiled_betting(prepared)
    assert abs(fast.pvalue(counts)-general['p']) < 1e-12

    started = time.perf_counter()
    for _ in range(500):
        null_model.pvalue([490, 510])
    elapsed = time.perf_counter()-started
    out = {'status': 'pass', 'outward_reference_cases': checked,
           'reference_precision': 110, 'max_log_e_gap_upper': max_gap,
           'null_e_expectation': mean_e, 'null_rejection_probabilities': rejection,
           'single_category_profiled_p': general['p'],
           'single_category_finite_support_p': fast.pvalue(counts),
           'five_hundred_scalar_calls_seconds': elapsed,
           'fractions': 64, 'additional_data_rows': 0}
    path = Path(__file__).with_name('finite_support_betting_checks.json')
    path.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
