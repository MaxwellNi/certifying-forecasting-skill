"""Fixed-grid Hoeffding martingale lower bound for settled hourly loss gain.

See PREDICTABLE_LOSS_THEORY.md for the exact filtration, target and proof.
This module loads no dataset. The caller must supply every frozen calendar hour
in order, including empty hours, and establish predictability of each width.
"""

import math

import numpy as np


DEFAULT_HOURS = 1416
DEFAULT_ALPHA = 0.025
DEFAULT_LAMBDA_GRID = tuple(float(x) for x in np.geomspace(0.01, 100000.0, 32))


def _vector(values, name, allow_nonfinite=False):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_nonfinite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def hourly_gain_bounds(prediction_a, prediction_b, targets):
    """Aggregate one hour's eligible rows; missing target contributes zero gain.

    Inputs a,b are finite delivered predictions in [0,1]. A finite target must
    be the delivered target in [0,1]; any nonfinite target denotes unavailable.
    All arrays contain the same issuance-eligible rows. Bounds and denominator
    depend on those rows and predictions, never on target availability.
    Positive gain means A has smaller squared loss than B.
    """
    a = _vector(prediction_a, "prediction_a")
    b = _vector(prediction_b, "prediction_b")
    y = _vector(targets, "targets", allow_nonfinite=True)
    if len(a) != len(b) or len(a) != len(y):
        raise ValueError("predictions and targets must have equal lengths")
    if np.any((a < 0) | (a > 1)) or np.any((b < 0) | (b > 1)):
        raise ValueError("delivered predictions must lie in [0,1]")
    observed = np.isfinite(y)
    if np.any((y[observed] < 0) | (y[observed] > 1)):
        raise ValueError("finite delivered targets must lie in [0,1]")
    n = len(a)
    if n == 0:
        return {"gain": 0.0, "lower_endpoint": 0.0, "upper_endpoint": 0.0,
                "predictable_width": 0.0, "eligible_rows": 0,
                "observed_targets": 0, "missing_targets": 0}
    difference = b - a
    endpoint_zero = difference * (b + a)
    endpoint_one = difference * (b + a - 2.0)
    lower = math.fsum(np.minimum(endpoint_zero, endpoint_one)) / n
    upper = math.fsum(np.maximum(endpoint_zero, endpoint_one)) / n
    gain = math.fsum(difference[observed] * (b[observed] + a[observed] - 2.0*y[observed])) / n
    width = 2.0 * math.fsum(np.abs(difference)) / n
    return {"gain": gain, "lower_endpoint": lower, "upper_endpoint": upper,
            "predictable_width": width, "eligible_rows": n,
            "observed_targets": int(observed.sum()),
            "missing_targets": int(n - observed.sum())}


def predictable_loss_bound(hourly_gains, predictable_widths, alpha=DEFAULT_ALPHA,
                           lambdas=None, expected_hours=DEFAULT_HOURS):
    """Lower confidence bound for mean E[G_t | prior-settlement filtration].

    Returns mean(G)-min_lambda{log(K/alpha)/lambda + lambda*sum(W^2)/8}/T.
    Every positive lambda, the grid, alpha and T must be fixed before inference.
    Random W_t are allowed when measurable before G_t's settlement. No IID,
    stationarity, MAR, or independence across articles/hours is assumed.

    `expected_hours` defaults to the frozen 1416-hour contract. An alternative
    fixed design must supply its predeclared positive integer T explicitly.
    Numerical checks cannot establish chronology or width predictability.
    """
    gains = _vector(hourly_gains, "hourly_gains")
    widths = _vector(predictable_widths, "predictable_widths")
    if len(gains) != len(widths):
        raise ValueError("hourly_gains and predictable_widths must have equal lengths")
    if isinstance(expected_hours, (bool, np.bool_)) or not isinstance(expected_hours, (int, np.integer)) or expected_hours <= 0:
        raise ValueError("expected_hours must be a positive predeclared integer")
    if len(gains) != expected_hours:
        raise ValueError(f"all {expected_hours} frozen calendar hours are required, including empty hours")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if np.any((widths < 0) | (widths > 2)):
        raise ValueError("loss-gain widths must lie in [0,2]")
    tolerance = 16.0 * np.finfo(float).eps
    if np.any(np.abs(gains) > 1.0 + tolerance):
        raise ValueError("bounded squared-loss gains must lie in [-1,1]")
    if np.any(np.abs(gains) > widths + tolerance):
        raise ValueError("a gain cannot exceed its zero-containing interval width")
    grid = _vector(DEFAULT_LAMBDA_GRID if lambdas is None else lambdas, "lambdas")
    if len(grid) == 0 or np.any(grid <= 0):
        raise ValueError("the predeclared lambda grid must be nonempty and positive")
    if len(np.unique(grid)) != len(grid):
        raise ValueError("the predeclared lambda grid must contain distinct values")
    n = len(gains)
    width_square_sum = math.fsum(float(x) * float(x) for x in widths)
    gain_sum = math.fsum(gains)
    log_penalty = math.log(len(grid)) - math.log(alpha)
    total_radii = log_penalty / grid + grid * (width_square_sum / 8.0)
    if not np.all(np.isfinite(total_radii)):
        raise FloatingPointError("non-finite grid-radius arithmetic")
    chosen = int(np.argmin(total_radii))
    radius = float(total_radii[chosen]) / n
    mean_gain = gain_sum / n
    return {
        "estimate": mean_gain,
        "lower": mean_gain - radius,
        "radius": radius,
        "gain_sum": gain_sum,
        "hours": n,
        "sum_predictable_width_squares": width_square_sum,
        "alpha": alpha,
        "grid_size": len(grid),
        "lambda_grid": [float(x) for x in grid],
        "selected_lambda": float(grid[chosen]),
        "selected_lambda_index": chosen,
        "log_grid_penalty": log_penalty,
        "per_lambda_radii": [float(x)/n for x in total_radii],
        "positive_lower_bound": bool(mean_gain - radius > 0.0),
        "target": "average E[hourly gain | information at the prior hour settlement]",
    }
