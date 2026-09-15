"""Exact finite-law examples and floor-budget verification for design boundaries."""
from fractions import Fraction as F
from itertools import product
import json

from exact_checks import parts, expectation, shared, validator, direct, run_checks


def checks():
    out = {}
    law = [(point, F(1, 4)) for point in product((0, 1), repeat=2)]
    maps = [lambda z: z[0], lambda z: z[1]]
    C = D = [[F(1)], [F(-1)]]
    theta, gamma, K, lam, omega = parts(law, maps, maps, C, D)
    assert all(len({fn(z) for z, _ in law}) > 1 for fn in maps)
    assert sum(x for row in omega for x in row) == 0
    assert gamma == theta == F(1, 8)
    assert expectation(law, 2, lambda z: shared(z, maps, maps, C, D)) == F(1, 4)
    assert expectation(law, 3, lambda z: validator(z, maps, maps, omega)) == gamma
    assert expectation(law, 3, lambda z: direct(z, maps, maps, C, D)) == theta
    out['all_nonconstant_scalar_sum_failure'] = {
        'theta':str(theta), 'interaction':str(gamma), 'score_mean':'1/4',
        'omega':[[str(x) for x in row] for row in omega],
        'lambda':[[str(x) for x in row] for row in lam],
        'scope':'All-entry scalar sum is zero; active-overlap diagonal sum is not zero.'}

    theta, gamma, _, _, omega = parts(law, maps[:1], maps[1:], [[F(1)]], [[F(1)]])
    assert theta == gamma == 0 and omega == [[F(1)]]
    out['nonzero_operator_can_have_zero_actual_interaction'] = {'omega':'1', 'theta':'0', 'interaction':'0'}

    bern = [(0, F(1, 2)), (1, F(1, 2))]
    duplicate_maps = [lambda x: x, lambda x: x]
    C = [[F(1),F(0)], [F(-1,2),F(0)]]
    D = [[F(0),F(1)], [F(0),F(-1,2)]]
    theta, gamma, *_ = parts(bern, duplicate_maps, duplicate_maps, C, D)
    assert theta == F(1, 64) and gamma == 0
    out['fitted_residual_target_not_full_baseline_adjustment'] = {
        'forecast_equals_outcome_equals_baseline':True, 'fitted_baseline_weight':'1/2',
        'theta_H':str(theta), 'reference_interaction':str(gamma),
        'full_conditional_residual_target':'0'}

    ternary = [(u, F(1, 3)) for u in (-1, 0, 1)]
    nonlinear = [lambda u: u*u, lambda u: u]
    C = [[F(1), F(0)], [F(-1), F(0)]]
    D = [[F(0), F(1)], [F(0), F(-1)]]
    theta, gamma, *_ = parts(ternary, nonlinear, nonlinear, C, D)
    assert theta == F(7,54) and gamma == 0
    assert expectation(ternary, 3, lambda z: direct(z, nonlinear, nonlinear, C, D)) == F(7,54)
    out['nonlinear_baseline_transform_with_unit_coefficients'] = {
        'baseline_values':['-1','0','1'], 'each_probability':'1/3',
        'forecast_and_outcome':'baseline squared', 'row_sums':['1','-1'],
        'fitted_rank_contrasts':['1/2','-1/3','-1/6'],
        'theta_H':str(theta), 'disjoint_reference_interaction':str(gamma),
        'full_conditional_residual_target':'0', 'category_adjusted_rank_target':'0'}

    count = 0
    for M, n, r in product(range(1,101), range(1,21), range(1,11)):
        q = (M*(r+1)+3*n)//3
        assert F(q) > F(2*M,3)
        # Norm ratio <=1 and log ratio <1 imply squared radius ratio <1/6.
        assert F(M, 9*q) < F(1,6)
        count += 1
    out['pooled_radius_floor_inequality'] = {
        'integer_cases':count, 'q':'floor((M*(r+1)+3*n)/3)',
        'verified_exact':'q > 2*M/3 and M/(9*q) < 1/6',
        'scope':'Positive M,n,r. This is a radius bound, not power dominance.'}
    return {'status':'PASS', 'arithmetic':'fractions.Fraction', 'new_check_count':len(out),
            'new_checks':out, 'retained_exact_checks':run_checks()}


if __name__ == '__main__':
    print(json.dumps(checks(), indent=2))
