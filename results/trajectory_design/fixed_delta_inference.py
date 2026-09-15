"""Fixed-validation-budget inversion of the displayed Hoeffding trajectory bound.

The returned p value is rounded outward for the supplied numeric score values and
coefficients. This does not certify upstream floating kernel arithmetic. If those
values approximate exact kernel means, provide justified mean-error allowances.
No data-dependent optimization over delta or family size is performed.
"""

from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import math

import numpy as np

from design_contract import rational


def _matrix(value, name):
    rows = np.asarray(value, dtype=object)
    if rows.ndim != 2 or min(rows.shape) < 1:
        raise ValueError(f"{name} requires a nonempty matrix")
    return [[rational(x, name) for x in row] for row in rows]


def _design_widths(C, D):
    C, D = _matrix(C, "C"), _matrix(D, "D")
    if len(C[0]) != len(D[0]):
        raise ValueError("C and D need the same reference role count")
    ws = sum(abs(x) for row in C for x in row) * sum(abs(x) for row in D for x in row) / 2
    omega = [[sum(x*y for x, y in zip(a, b)) for b in D] for a in C]
    wr = sum(abs(x) for row in omega for x in row)
    return ws, wr


def _values(value, name, width):
    array = np.asarray(value, dtype=object)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    vals = [rational(x, name) for x in array]
    # Actual midrank kernels are enclosed by the symmetric design interval.
    # Reject incompatible inputs rather than accepting a constant outside it.
    if any(abs(x) > width/2 for x in vals):
        raise ValueError(f"{name} lies outside its asserted symmetric kernel support")
    return sum(vals, Fraction()) / len(vals), len(vals)


def _fraction_decimal(value, direction):
    with localcontext() as ctx:
        ctx.prec = 96
        ctx.rounding = direction
        return Decimal(value.numerator) / Decimal(value.denominator)


def _up_float(value):
    result = float(value)
    if math.isinf(result):
        return result
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def fixed_delta_hoeffding_pvalue(score_values, validation_values, C, D, delta=0.01,
                                 score_mean_error=0, interaction_mean_error=0):
    """Valid fixed-delta p value for H0: theta_H <= 0, conditional on training.

    Write G=Sbar-Rbar-h(W_R,n,delta) minus the two supplied mean-error allowances.
    For W_S>0 and G>0, p=min(1,delta+exp(-2*M*G**2/W_S**2)); otherwise p=1.
    Omega=0 removes validation and sets effective delta=0. The generic W_S=0
    formula yields delta if G>0, otherwise 1; valid design scores force G=0 here.

    Conditional IID trajectories, fixed maps/laws, source assignments, and fixed
    delta remain caller assumptions. Known mean-error allowances must cover their
    respective upstream numerical discrepancies. Decimal outward evaluation
    controls this inversion only. Strict scientific guarantees are conditional on
    the assumptions, not inferred from this routine passing its input checks.
    """
    ws, wr = _design_widths(C, D)
    sm, M = _values(score_values, "score_values", ws)
    es = rational(score_mean_error, "score_mean_error")
    er = rational(interaction_mean_error, "interaction_mean_error")
    if min(es, er) < 0:
        raise ValueError("mean-error allowances must be nonnegative")
    if wr:
        delta = rational(delta, "delta")
        if not 0 < delta < 1:
            raise ValueError("a nonzero interaction needs 0 < delta < 1")
        vm, n = _values(validation_values, "validation_values", wr)
    else:
        delta, vm, n = Fraction(0), Fraction(0), 0
        if er:
            raise ValueError("exact zero interaction needs no interaction mean-error allowance")
    gap_exact = sm - vm - es - er
    with localcontext() as ctx:
        ctx.prec = 96
        # All division endpoints are directed. Decimal ln/sqrt/exp are correctly
        # rounded; one adjacent Decimal on the conservative side encloses them.
        if wr:
            delta_lower = _fraction_decimal(delta, ROUND_FLOOR)
            log_upper = (-delta_lower.ln()).next_plus()
            ctx.rounding = ROUND_CEILING
            h_upper = (_fraction_decimal(wr, ROUND_CEILING)
                       * (log_upper / Decimal(2*n)).sqrt().next_plus()).next_plus()
        else:
            h_upper = Decimal(0)
        ctx.rounding = ROUND_FLOOR
        gap_lower = _fraction_decimal(gap_exact, ROUND_FLOOR) - h_upper
        if gap_lower <= 0:
            p_upper, branch = Decimal(1), "nonpositive_margin"
        elif not ws:
            p_upper = _fraction_decimal(delta, ROUND_CEILING)
            branch = "zero_score_width"
        else:
            # Use the lower numerator and upper denominator to get a lower
            # exponent, then an upper exponential tail and upper addition.
            numerator_lower = Decimal(2*M) * gap_lower * gap_lower
            ws_upper = _fraction_decimal(ws, ROUND_CEILING)
            ctx.rounding = ROUND_CEILING
            denominator_upper = ws_upper * ws_upper
            ctx.rounding = ROUND_FLOOR
            exponent_lower = numerator_lower / denominator_upper
            # exp(-800) already lies below the smallest positive binary64. A
            # larger exact exponent may be clipped here only upward in p.
            exponent_lower = min(exponent_lower, Decimal(800))
            tail_upper = (-exponent_lower).exp().next_plus()
            ctx.rounding = ROUND_CEILING
            p_upper = min(Decimal(1), _fraction_decimal(delta, ROUND_CEILING) + tail_upper)
            branch = "fixed_delta_hoeffding"
    return {
        "p_value": min(1.0, _up_float(p_upper)),
        "p_upper_decimal": str(p_upper), "margin_lower_decimal": str(gap_lower),
        "validation_radius_upper_decimal": str(h_upper), "effective_delta": float(delta),
        "score_count": M, "validation_count": n, "branch": branch,
        "score_width_exact": str(ws), "validation_width_exact": str(wr),
        "score_mean_error_exact": str(es), "interaction_mean_error_exact": str(er),
        "arithmetic_scope": "Outward inversion for supplied values; upstream kernel error is separate.",
    }


def family_threshold(alpha, family_size):
    """Downward-rounded Bonferroni threshold; never silently return zero."""
    alpha = rational(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if isinstance(family_size, bool) or not isinstance(family_size, (int, np.integer)) or family_size < 1:
        raise ValueError("family_size must be a positive integer fixed before testing")
    exact = alpha / int(family_size)
    threshold = float(exact)
    if threshold == 0:
        raise ValueError("family threshold is below binary64; use an explicit higher-precision decision")
    if Fraction.from_float(threshold) > exact:
        threshold = math.nextafter(threshold, 0.0)
        if threshold == 0:
            raise ValueError("no positive binary64 threshold is conservative for this family")
    return threshold
