"""Fixed-design trajectory rank kernels and classical finite mean bounds.

All maps and coefficients must be fixed by genuinely external training information.
Arrays have shape (independent blocks, trajectories per block, maps). Within one
trajectory arbitrary dependence is allowed. Rank comparisons preserve the input
integer dtype (including integers larger than float64's exact-integer limit).

These are ordinary floating-point implementations of the accompanying exact
mathematics, not interval-arithmetic certificates. The caller supplies the sampling,
training, known support, error allocation, and primitive acquisition-cost contract.
"""

import math
import numbers

import numpy as np


def _map_array(value, name):
    array = np.asarray(value)
    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must have nonempty shape (batch, trajectories, maps)")
    if array.dtype.kind not in "biuf":
        if array.dtype.kind != "O" or not all(
            isinstance(x, numbers.Integral) for x in array.flat
        ):
            raise ValueError(f"{name} must contain finite real numbers or exact integers")
    elif not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _design(C, D):
    C, D = np.asarray(C, dtype=float), np.asarray(D, dtype=float)
    if C.ndim != 2 or D.ndim != 2 or min(*C.shape, *D.shape) == 0:
        raise ValueError("C and D must be nonempty matrices")
    if C.shape[1] != D.shape[1] or not (np.isfinite(C).all() and np.isfinite(D).all()):
        raise ValueError("C and D require a shared peer count and finite coefficients")
    with np.errstate(over="ignore", invalid="ignore"):
        omega = C @ D.T
        c, d = C.sum(axis=1), D.sum(axis=1)
        width_s = 0.5 * np.abs(C).sum() * np.abs(D).sum()
        width_r = np.abs(omega).sum()
        width_u = 0.5 * np.abs(c).sum() * np.abs(d).sum()
    if not all(np.isfinite(x).all() for x in (omega, c, d, width_s, width_r, width_u)):
        raise ValueError("coefficient contractions or enclosing widths overflow")
    return C, D, omega, c, d, float(width_s), float(width_r), float(width_u)


def _inputs(f_maps, g_maps, C, D):
    f, g = _map_array(f_maps, "f_maps"), _map_array(g_maps, "g_maps")
    design = _design(C, D)
    if f.shape[:2] != g.shape[:2]:
        raise ValueError("f_maps and g_maps must share block and trajectory axes")
    if f.shape[2] != design[0].shape[0] or g.shape[2] != design[1].shape[0]:
        raise ValueError("map axes must match coefficient rows")
    return f, g, design


def _centered_rank(x, y):
    # Comparisons happen before conversion: never coerce an exact integer map.
    return (x > y).astype(float) + 0.5 * (x == y) - 0.5


def design_widths(C, D):
    """Known enclosing widths, not assertions that either endpoint is attained."""
    C, D, omega, c, d, ws, wr, wu = _design(C, D)
    return {
        "score_width": ws,
        "validation_width": wr,
        "direct_width": wu,
        "distinct_focal_width": wu,
        "full_u_width": wu / 3.0,
        "peer_count": int(C.shape[1]),
        "omega": omega.tolist(),
        "c": c.tolist(),
        "d": d.tolist(),
    }


def shared_scores(f_maps, g_maps, C, D):
    """Empirical residual products; N=J+1, first trajectory is the focal role."""
    f, g, (C, D, *_) = _inputs(f_maps, g_maps, C, D)
    if f.shape[1] != C.shape[1] + 1:
        raise ValueError("shared scores need one focal trajectory plus J peers")
    p = _centered_rank(f[:, :1, :], f[:, 1:, :])
    q = _centered_rank(g[:, :1, :], g[:, 1:, :])
    return np.einsum("bjs,sj->b", p, C) * np.einsum("bjt,tj->b", q, D)


def validation_scores(f_maps, g_maps, C, D):
    """Three-trajectory unbiased aggregate shared-reference interaction kernel.

    The same two reference trajectories appear in both differences. Its mean is
    <C D.T, Lambda>; using independent reference pairs across factors is invalid.
    """
    f, g, (_, _, omega, *_) = _inputs(f_maps, g_maps, C, D)
    if f.shape[1] != 3:
        raise ValueError("validation scores require exactly three trajectories")
    dp = _centered_rank(f[:, 0], f[:, 1]) - _centered_rank(f[:, 0], f[:, 2])
    dq = _centered_rank(g[:, 0], g[:, 1]) - _centered_rank(g[:, 0], g[:, 2])
    return 0.5 * np.einsum("bs,st,bt->b", dp, omega, dq)


def distinct_focal_scores(f_maps, g_maps, C, D):
    """Same fixed focal role, all ordered distinct reference pairs; N>=3."""
    f, g, (_, _, _, c, d, *_) = _inputs(f_maps, g_maps, C, D)
    J = f.shape[1] - 1
    if J < 2:
        raise ValueError("distinct-reference scores need at least three trajectories")
    a = _centered_rank(f[:, :1], f[:, 1:]) @ c
    b = _centered_rank(g[:, :1], g[:, 1:]) @ d
    return (a.sum(axis=1) * b.sum(axis=1) - (a * b).sum(axis=1)) / (J * (J - 1))


def full_u_scores(f_maps, g_maps, C, D):
    """All focal and distinct-reference roles, degree-three rank U-statistic.

    Each returned value is ONE block observation; its overlapping internal triples
    are not independent observations. Costs O(batch*N*N*(S+T)); mapping cost extra.
    N need not equal the original shared-design J+1.
    """
    f, g, (_, _, _, c, d, *_) = _inputs(f_maps, g_maps, C, D)
    batch, N, _ = f.shape
    if N < 3:
        raise ValueError("full U scores need at least three trajectories")
    a, b = np.zeros((batch, N, N)), np.zeros((batch, N, N))
    for s, weight in enumerate(c):
        if weight:
            a += weight * _centered_rank(f[:, :, None, s], f[:, None, :, s])
    for t, weight in enumerate(d):
        if weight:
            b += weight * _centered_rank(g[:, :, None, t], g[:, None, :, t])
    # Diagonal entries are exactly zero by the midpoint-tie convention.
    numerator = (a.sum(axis=2) * b.sum(axis=2) - (a * b).sum(axis=2)).sum(axis=1)
    return numerator / (N * (N - 1) * (N - 2))


def mean_bound(values, width, alpha=0.05, side="lower", rule="hybrid"):
    """One-sided mean bound for independent values in a common known interval.

    Hoeffding (1963), and Maurer--Pontil (2009), Theorem 11. The latter allows
    independent nonidentical laws. The hybrid allocates alpha/2 to each rule.
    For nonidentical means the target is their arithmetic average. Width must be
    justified before observing these values; their empirical range is insufficient.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("values must be a nonempty finite vector")
    width, alpha = float(width), float(alpha)
    if not math.isfinite(width) or width < 0:
        raise ValueError("width must be finite and nonnegative")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if side not in ("lower", "upper") or rule not in ("hoeffding", "empirical_bernstein", "hybrid"):
        raise ValueError("unknown side or concentration rule")
    n = values.size
    estimate = math.fsum(float(x) / n for x in values)
    span = float(values.max()) - float(values.min())
    if span > width + 32 * np.finfo(float).eps * max(1.0, width):
        raise ValueError("observed span exceeds the asserted enclosing width")
    if width == 0 and np.any(values != values[0]):
        raise ValueError("a zero-width interval requires exactly constant values")
    variance = math.fsum((float(x) - estimate) ** 2 / (n - 1) for x in values) if n > 1 else 0.0
    # Never form alpha/2, alpha/4, or 1/alpha: these can under/overflow.
    log_inverse_alpha = -math.log(alpha)
    h_log = log_inverse_alpha + (math.log(2.0) if rule == "hybrid" and n > 1 else 0.0)
    h_radius = width * math.sqrt(h_log / (2 * n))
    eb_log = log_inverse_alpha + math.log(4.0 if rule == "hybrid" else 2.0)
    eb_radius = (
        math.sqrt(variance) * math.sqrt(2 * eb_log / n)
        + width * (7 * eb_log / (3 * (n - 1)))
    ) if n > 1 else math.inf
    if width == 0:
        radius = 0.0
    elif rule == "empirical_bernstein":
        if n < 2:
            raise ValueError("empirical Bernstein requires at least two blocks")
        radius = eb_radius
    else:
        radius = min(h_radius, eb_radius) if rule == "hybrid" else h_radius
    bound = estimate - radius if side == "lower" else estimate + radius
    return {
        "bound": float(bound), side: float(bound), "estimate": estimate,
        "radius": radius, "sample_variance": variance, "n": int(n),
        "width": width, "alpha": alpha, "side": side, "rule": rule,
    }


def corrected_lower(score_values, validation_values, C, D, alpha=0.05, delta=0.01, rule="hybrid"):
    """Lower bound for the population-reference fitted residual association.

    Independent evaluation blocks cost J+1 trajectories each; independent validation
    blocks cost three each. Coverage is conditional on external training information,
    not on realized validation outcomes. Primitive rows/training/ingestion cost extra.
    """
    widths = design_widths(C, D)
    alpha, delta = float(alpha), float(delta)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if widths["validation_width"] == 0:
        delta = 0.0
        validation = {"estimate": 0.0, "radius": 0.0, "n": 0}
    else:
        if not 0 < delta < alpha:
            raise ValueError("a nonzero interaction needs 0 < delta < alpha")
        validation = mean_bound(validation_values, widths["validation_width"], delta, "upper", rule)
    evaluation = mean_bound(score_values, widths["score_width"], alpha - delta, "lower", rule)
    lower = evaluation["bound"] - validation["estimate"] - validation["radius"]
    return {
        "lower": float(lower), "estimate": evaluation["estimate"] - validation["estimate"],
        "raw_estimate": evaluation["estimate"], "estimated_bias": validation["estimate"],
        "sampling_radius": evaluation["radius"], "validation_radius": validation["radius"],
        "score_count": evaluation["n"], "validation_count": validation["n"],
        "trajectory_draw_calls": (widths["peer_count"] + 1) * evaluation["n"] + 3 * validation["n"],
        "alpha": alpha, "delta": delta, "rule": rule,
    }
