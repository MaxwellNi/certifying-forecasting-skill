#!/usr/bin/env python3
"""Exact, standalone illustration of reference-induced rank interaction.

Run with Python 3; no data files, dependencies, randomness, or artifact imports.
All probabilities and expectations use fractions.Fraction.
The fixed-focal zero is conditional; the main example's population target is 8/27.
"""
from fractions import Fraction
from itertools import product
import json

SUPPORT = (0, 1, 2)
HALF = Fraction(1, 2)


def mean(values):
    values = tuple(values)
    return sum(values, Fraction(0)) / len(values)


def midrank(value, reference):
    return Fraction(reference < value) + HALF * (reference == value)


def ranks_and_contrasts(focal, reference, maps):
    x, y, z = (midrank(f(focal), f(reference)) for f in maps)
    return x, y, z, x-z, y-z


def evaluate(maps, description):
    """The common law is uniform on SUPPORT for every independent draw."""
    conditional = []
    for focal in SUPPORT:
        rows = [ranks_and_contrasts(focal, ref, maps) for ref in SUPPORT]
        raw_shared = mean(x*y for x, y, z, a, b in rows)
        raw_distinct = mean(l[0]*r[1] for l, r in product(rows, repeat=2))
        shared = mean(a*b for x, y, z, a, b in rows)
        distinct = mean(l[3]*r[4] for l, r in product(rows, repeat=2))
        correction = mean((l[3]-r[3])*(l[4]-r[4])/2
                          for l, r in product(rows, repeat=2))
        assert correction == shared-distinct
        conditional.append({
            'focal': focal,
            'focal_values_h_Y_b': [f(focal) for f in maps],
            'rows': [{'reference': ref, 'reference_values_h_Y_b': [f(ref) for f in maps],
                      'x_y_z_A_B': row} for ref, row in zip(SUPPORT, rows)],
            'raw_uncentered_shared_product': raw_shared,
            'raw_uncentered_distinct_product': raw_distinct,
            'raw_reference_covariance': raw_shared-raw_distinct,
            'centered_raw_shared_product': mean((x-HALF)*(y-HALF) for x,y,z,a,b in rows),
            'centered_raw_distinct_product': mean((l[0]-HALF)*(r[1]-HALF)
                                                 for l,r in product(rows,repeat=2)),
            'contrast_shared_product': shared,
            'contrast_distinct_product': distinct,
            'contrast_correction_mean': correction,
        })
    population = {
        'target': mean(r['contrast_distinct_product'] for r in conditional),
        'expected_shared_score': mean(r['contrast_shared_product'] for r in conditional),
        'reference_interaction': mean(r['contrast_correction_mean'] for r in conditional),
    }
    assert population['expected_shared_score'] == population['target']+population['reference_interaction']
    return {'maps': description, 'conditional_on_U0_equals_1': conditional[1],
            'population_averaged_over_U0': population}


def main():
    examples = {
        'main': evaluate((lambda u:u, lambda u:u, lambda u:2-u), 'h(U)=Y(U)=U; b(U)=2-U'),
        'cancellation': evaluate((lambda u:u, lambda u:u, lambda u:u), 'h(U)=Y(U)=b(U)=U'),
        'opposite_order': evaluate((lambda u:u, lambda u:2-u, lambda u:1), 'h(U)=U; Y(U)=2-U; b(U)=1'),
    }
    focal = examples['main']['conditional_on_U0_equals_1']
    assert focal['raw_uncentered_shared_product'] == Fraction(5,12)
    assert focal['raw_uncentered_distinct_product'] == Fraction(1,4)
    assert focal['contrast_shared_product'] == Fraction(2,3)
    assert focal['contrast_distinct_product'] == 0
    assert examples['main']['population_averaged_over_U0'] == {
        'target': Fraction(8,27), 'expected_shared_score': Fraction(2,3),
        'reference_interaction': Fraction(10,27)}
    cancel = examples['cancellation']['conditional_on_U0_equals_1']
    assert cancel['raw_reference_covariance'] == Fraction(1,6)
    assert cancel['contrast_shared_product'] == cancel['contrast_distinct_product'] == 0
    assert examples['opposite_order']['conditional_on_U0_equals_1']['contrast_correction_mean'] == Fraction(-1,6)
    print(json.dumps({'law': 'Independent U0,U1,U2, each uniform on {0,1,2}',
                      'scope': 'Rank arithmetic, not a forecasting or decision-gain claim',
                      'all_assertions_passed': True, 'examples': examples}, indent=2, default=str))


if __name__ == '__main__':
    main()
