"""Direct residual midrank target estimators from independent raw sample rows.

Positions, not population identities, must be distinct within each U kernel.
The category probabilities must be exact for the conditional sampling law.
This module does not estimate those probabilities or validate independence.
"""

from itertools import permutations
import math
from typing import Mapping

import numpy as np


def _numeric_values(values):
    # NumPy can coerce mixed signed/large unsigned Python integers to float64.
    # Refuse a conversion that changes any supplied integer's exact value.
    original = np.asarray(values, dtype=object)
    converted = np.asarray(values)
    if converted.dtype.kind not in "biuf":
        raise ValueError("v and w must contain real numeric values with an exact supported dtype")
    if not np.all(np.isfinite(converted)):
        raise ValueError("v and w must contain finite numbers")
    if converted.dtype.kind == "f":
        for before, after in zip(original.flat, converted.flat):
            if isinstance(before, (int, np.integer)) and int(after) != int(before):
                raise ValueError("integer input would lose precision when converted to a common numeric dtype")
    return converted


def _inputs(v, w, category, probabilities: Mapping):
    v = _numeric_values(v)
    w = _numeric_values(w)
    labels = list(category)
    if v.ndim != 1 or w.ndim != 1 or len(v) != len(w) or len(v) != len(labels):
        raise ValueError("v, w and category must be equally long one-dimensional arrays")
    keys = list(probabilities)
    p = np.asarray([probabilities[c] for c in keys], dtype=float)
    if not len(p) or not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("all supplied population category probabilities must be positive")
    if np.any(p < 1.0 / np.finfo(float).max):
        raise ValueError("probability reciprocals must be finite in float64 arithmetic")
    if not math.isclose(float(p.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("probabilities must specify the complete positive support and sum to one")
    index = {label: j for j, label in enumerate(keys)}
    try:
        category = np.asarray([index[label] for label in labels], dtype=np.intp)
    except KeyError as exc:
        raise ValueError("every observed category must have a supplied probability") from exc
    return v, w, category, p


def _mid_comparison(focal, references):
    return (references < focal).astype(float) + 0.5 * (references == focal)


def _exclusive_rank_sums(values):
    ordered = np.sort(values)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return left + 0.5 * (right - left) - 0.5


def full_u(v, w, category, probabilities: Mapping):
    """Exact distinct-position U3 minus U4 in O(n^2+C) time and O(n+C) memory.

    An unbiased estimator under the assumptions in THEORY.md. Floating-point
    arithmetic is used; 'exact' describes the full sum, without subsampling.
    """
    v, w, category, p = _inputs(v, w, category, probabilities)
    n = len(v)
    if n < 4:
        raise ValueError("at least four independent sample positions are required")
    # At most n observed categories enter the collision sums. The complete
    # support is needed for probability validation but not inside each pass.
    observed, category = np.unique(category, return_inverse=True)
    p = p[observed]
    ra = _exclusive_rank_sums(v)
    rb = _exclusive_rank_sums(w)
    invp = 1.0 / p
    shared_first = []
    product_second = []
    shared_second = []
    for i in range(n):
        a = _mid_comparison(v[i], v)
        b = _mid_comparison(w[i], w)
        # Exclude the focus itself; retain all other sample positions even
        # when their values, categories or population identities coincide.
        shared_first.append(float(np.dot(a, b) - 0.25))
        same = category == category[i]
        same[i] = False
        product_second.append(float(
            invp[category[i]] * np.sum((ra[i] - a[same]) * (rb[same] - (1.0 - b[same])))
        ))
        # Now i is the shared reference position. The two focus positions
        # can be in any category, but neither may equal this reference.
        acol = 1.0 - a
        bcol = 1.0 - b
        acol[i] = 0.0
        bcol[i] = 0.0
        asum = np.bincount(category, weights=acol, minlength=len(p))
        bsum = np.bincount(category, weights=bcol, minlength=len(p))
        same_focus = np.bincount(category, weights=acol * bcol, minlength=len(p))
        shared_second.append(float(np.dot(invp, asum * bsum - same_focus)))
    first_numerator = float(np.dot(ra, rb)) - math.fsum(shared_first)
    second_numerator = math.fsum(product_second) - math.fsum(shared_second)
    first = first_numerator / (n * (n - 1) * (n - 2))
    second = second_numerator / (n * (n - 1) * (n - 2) * (n - 3))
    return {
        "estimate": first - second,
        "first_term": first,
        "second_term": second,
        "n": n,
        "first_numerator": first_numerator,
        "second_numerator": second_numerator,
    }


def symmetrized_four_row_kernel(v, w, category, probabilities: Mapping):
    """Average the ordered four-position kernel over all 24 permutations."""
    v, w, category, p = _inputs(v, w, category, probabilities)
    if len(v) != 4:
        raise ValueError("exactly four sample positions are required")
    a = np.asarray([_mid_comparison(x, v) for x in v])
    b = np.asarray([_mid_comparison(x, w) for x in w])
    terms = []
    for i, j, k, ell in permutations(range(4)):
        correction = a[i, k] * b[j, ell] / p[category[i]] if category[i] == category[j] else 0.0
        terms.append(a[i, j] * b[i, k] - correction)
    return math.fsum(terms) / 24.0


def _kernel_bounds(probabilities):
    # For the symmetrized kernel, the triple term lies in [1/6,1/3].
    # The unweighted disjoint-pair product averages to exactly 1/4.
    if len(probabilities) == 1:
        return -1.0 / 12.0, 1.0 / 12.0
    return 1.0 / 6.0 - 1.0 / (4.0 * min(probabilities.values())), 1.0 / 3.0


def full_u_hoeffding(v, w, category, probabilities: Mapping, alpha=0.05):
    """Conservative one-sided lower bound for the full U estimator.

    alpha is the error allocated to this bound, not a family error level to
    be reused independently for every candidate.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    result = full_u(v, w, category, probabilities)
    q = result["n"] // 4
    kernel_lower, kernel_upper = _kernel_bounds(probabilities)
    kernel_range = kernel_upper - kernel_lower
    radius = kernel_range * math.sqrt(-math.log(alpha) / (2.0 * q))
    result.update(lower=result["estimate"] - radius, radius=radius,
                  alpha=alpha, independent_blocks=q, kernel_range=kernel_range,
                  kernel_lower=kernel_lower, kernel_upper=kernel_upper)
    return result


def block_empirical_bernstein(v, w, category, probabilities: Mapping, alpha=0.05):
    """Separate finite-sample bound using preassigned disjoint four-row blocks.

    The center is the block average, NOT the full U estimate. The first 4q
    positions are grouped in their given order, which must be chosen without
    inspecting values or labels. Remaining n-4q positions are unused.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    v, w, encoded, p = _inputs(v, w, category, probabilities)
    q = len(v) // 4
    if q < 2:
        raise ValueError("at least eight sample positions are required")
    encoded_probabilities = dict(enumerate(p))
    kernels = np.asarray([
        symmetrized_four_row_kernel(v[4*j:4*j+4], w[4*j:4*j+4],
                                   encoded[4*j:4*j+4], encoded_probabilities)
        for j in range(q)
    ])
    center = float(kernels.mean())
    variance = float(kernels.var(ddof=1))
    kernel_lower, kernel_upper = _kernel_bounds(encoded_probabilities)
    kernel_range = kernel_upper - kernel_lower
    x = math.log(2.0) - math.log(alpha)
    radius = math.sqrt(2.0 * variance * x / q) + 7.0 * kernel_range * x / (3.0 * (q - 1))
    return {"estimate": center, "sample_variance": variance, "radius": radius,
            "lower": center - radius, "alpha": alpha, "independent_blocks": q,
            "used_rows": 4*q, "unused_rows": len(v) - 4*q,
            "kernel_range": kernel_range, "kernel_lower": kernel_lower,
            "kernel_upper": kernel_upper}
