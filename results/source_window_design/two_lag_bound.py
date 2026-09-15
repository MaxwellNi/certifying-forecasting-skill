"""Finite lower bounds for two-lag population-reference score expectations.

The evaluation units must be independent whole panels with common IID raw
source law and fixed comparison coefficients. Each supplied score is A*B.
Validation, when used, consists of fresh independent three-trajectory arrays,
each trajectory containing two IID raw observations from that same law.
This is a bounded-mean calculation, not a dependent-panel inference routine.
"""
from math import fsum, isfinite, log, sqrt
from numbers import Integral, Real


def _raw_number(x):
    # Convert fixed-width integer scalars before addition. In particular,
    # np.int64(max)+np.int64(max) must not wrap before comparison.
    # Native bool is an Integral and is interpreted as the binary value 0/1.
    if isinstance(x, Integral):
        return int(x)
    if not isinstance(x, Real) or not isfinite(x):
        raise ValueError('Raw observations must be finite real numbers.')
    return x


def _finite_float(x):
    try:
        value = float(_raw_number(x))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError('Bound inputs must be representable finite real numbers.') from exc
    if not isfinite(value):
        raise ValueError('Bound inputs must be representable finite real numbers.')
    return value


def _comparison(x, y):
    # Compare original numeric values; converting large integers can create ties.
    return float(x > y)+0.5*float(x == y)


def paired_lambda_samples(raw):
    """Return R from each raw 3-by-2 array (focal, first peer, second peer)."""
    out = []
    for triple in raw:
        if len(triple) != 3 or any(len(trajectory) != 2 for trajectory in triple):
            raise ValueError('Each validation replicate must have shape (3, 2).')
        focal, first, second = [[_raw_number(x) for x in trajectory] for trajectory in triple]
        focal_sum, first_sum, second_sum = [_raw_number(sum(z)) for z in (focal, first, second)]
        sum_difference = (_comparison(focal_sum, first_sum)
                          - _comparison(focal_sum, second_sum))
        component_difference = (_comparison(focal[0], first[0])
                                - _comparison(focal[0], second[0]))
        out.append(0.5*sum_difference*component_difference)
    if not out:
        raise ValueError('Validation requires at least one fresh replicate.')
    return out


def lower_bound(scores, *, omega, coefficient_l1_a, coefficient_l1_b,
                error_probability=0.05, validation_raw=None,
                validation_error_probability=0.01):
    """Return a one-sided bound for theta_0=E[A_0 B_0].

    coefficient_l1_a=sum(abs(c_sj)), and similarly for coefficient_l1_b.
    The fixed equal-weight two-lag design must supply omega=Omega_2. Scores
    must come from independent panels. Coefficients, design, and error budgets
    must be fixed before evaluation. No validation is needed for omega=0.
    With validation_raw=None, the known interval lambda in [0, 1/8] is used.
    """
    values = [_finite_float(s) for s in scores]
    if not values:
        raise ValueError('At least one independent panel score is required.')
    omega, coefficient_l1_a, coefficient_l1_b, error_probability = map(
        _finite_float, (omega, coefficient_l1_a, coefficient_l1_b, error_probability))
    if coefficient_l1_a < 0 or coefficient_l1_b < 0:
        raise ValueError('Coefficient L1 norms must be nonnegative.')
    if not 0 < error_probability < 1:
        raise ValueError('Error probability must be between zero and one.')
    width = 0.5*coefficient_l1_a*coefficient_l1_b
    if not isfinite(width) or (width == 0 and coefficient_l1_a > 0 and coefficient_l1_b > 0):
        raise ValueError('The coefficient range is not representable without overflow or underflow.')
    if any(abs(s) > width/2 for s in values):
        raise ValueError('A score exceeds its declared coefficient range.')
    low, high, n, sample_mean = 0.0, 0.125, 0, None
    evaluation_error = error_probability
    method = 'known_lambda_range'
    if omega == 0:
        method = 'zero_overlap'
    elif validation_raw is not None:
        delta = _finite_float(validation_error_probability)
        if not 0 < delta < error_probability:
            raise ValueError('Validation error must be between zero and total error.')
        rs = paired_lambda_samples(validation_raw)
        n = len(rs)
        sample_mean = fsum(rs)/n
        eps = sqrt((log(2)-log(delta))/(2*n))
        candidate_low, candidate_high = max(0.0, sample_mean-eps), min(0.125, sample_mean+eps)
        if candidate_low <= candidate_high:
            low, high = candidate_low, candidate_high
        # On an empty intersection retain the known valid interval.
        evaluation_error -= delta
        method = 'independent_lambda_validation'
    correction = omega*(high if omega >= 0 else low)
    penalty = width*sqrt(-log(evaluation_error)/(2*len(values)))
    # Average before summation so several large finite scores cannot overflow
    # a representable mean. Reject any unrepresentable derived bound.
    score_mean = fsum(s/len(values) for s in values)
    bound = score_mean-penalty-correction
    if not all(isfinite(x) for x in (penalty, score_mean, correction, bound)):
        raise ValueError('Bound arithmetic overflowed; rescale the fixed score coefficients.')
    return dict(lower_bound=bound,
                score_mean=score_mean, panel_count=len(values),
                score_range_width=width, evaluation_penalty=penalty,
                omega=float(omega), lambda_interval=[low, high],
                interaction_upper_bound=correction, method=method,
                validation_count=n, validation_mean=sample_mean,
                validation_raw_draws=6*n, error_probability=error_probability,
                target='population-reference temporal score expectation')
