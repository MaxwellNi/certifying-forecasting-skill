"""Finite-sample full-U calibration; see FULL_U_VARIANCE_THEORY.md.

The statistical guarantee uses exact real arithmetic. Computation uses float64,
as does the frozen estimator. No normal approximation or nondegeneracy test is
used. The n inference rows must be conditionally IID with exact known category
probabilities, and the target/predictor and n must be fixed in advance.
"""

from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path

import numpy as np


_BASE_PATH = Path(__file__).resolve().parents[1] / "helpers/direct_target.py"
_SPEC = spec_from_file_location("_direct_target", _BASE_PATH)
baseline = module_from_spec(_SPEC)
_SPEC.loader.exec_module(baseline)


def _falling(n, r):
    return math.prod(range(n - r + 1, n + 1))


def full_u_all_delete_one(v, w, category, probabilities):
    """Full U and every leave-one value in O(n^2+C) time, O(n+C) memory.

    Derivatives of the weighted collision-correct numerators give the sum of
    all tuples containing a specified position. Subtracting this sum deletes
    that position, regardless of ties or repeated population identities.
    """
    v, w, category, p = baseline._inputs(v, w, category, probabilities)
    n = len(v)
    if n < 5:
        raise ValueError("at least five IID sample positions are required")
    observed, category = np.unique(category, return_inverse=True)
    p = p[observed]
    invp = 1.0 / p
    A = baseline._exclusive_rank_sums(v)
    B = baseline._exclusive_rank_sums(w)
    d3 = np.zeros(n)
    dP = np.zeros(n)
    dQ = np.zeros(n)
    dA = np.zeros(n)
    dB = np.zeros(n)
    first_terms = []
    product_terms = []
    collision_terms = []

    for i in range(n):
        a = baseline._mid_comparison(v[i], v)
        b = baseline._mid_comparison(w[i], w)
        a[i] = 0.0
        b[i] = 0.0

        # First numerator: one focus role and two independent reference roles.
        first_focus = float(A[i] * B[i] - np.dot(a, b))
        first_terms.append(first_focus)
        d3[i] += first_focus
        d3 += a * (B[i] - b) + b * (A[i] - a)

        # Product P: reverse derivatives of explicit focus weights and of A,B.
        same = category == category[i]
        same[i] = False
        opposite_b = 1.0 - b[same]
        left = A[i] - a[same]
        right = B[same] - opposite_b
        scale = invp[category[i]]
        terms = scale * left * right
        product_terms.append(float(np.sum(terms)))
        dP[i] += float(np.sum(terms - scale * opposite_b * left))
        dP[same] += terms - scale * a[same] * right
        dA[i] = scale * float(np.sum(right))
        dB[same] += scale * left

        # Q removes the duplicated reference. Its weight has degree two, so
        # the derivative at this reference is twice its collision contribution.
        acol = 1.0 - a
        bcol = 1.0 - b
        acol[i] = 0.0
        bcol[i] = 0.0
        asum = np.bincount(category, weights=acol, minlength=len(p))
        bsum = np.bincount(category, weights=bcol, minlength=len(p))
        diag = np.bincount(category, weights=acol * bcol, minlength=len(p))
        collision = float(np.dot(invp, asum * bsum - diag))
        collision_terms.append(collision)
        dQ[i] += 2.0 * collision
        dQ += invp[category] * (
            acol * (bsum[category] - bcol)
            + bcol * (asum[category] - acol)
        )

    # Propagate A and B derivatives through their row-weighted rank sums.
    for i in range(n):
        a = baseline._mid_comparison(v[i], v)
        b = baseline._mid_comparison(w[i], w)
        a[i] = 0.0
        b[i] = 0.0
        dP += dA[i] * a + dB[i] * b

    N3 = math.fsum(first_terms)
    N4 = math.fsum(product_terms) - math.fsum(collision_terms)
    d4 = dP - dQ
    first = N3 / _falling(n, 3)
    second = N4 / _falling(n, 4)
    center = first - second
    deleted_first = (N3 - d3) / _falling(n - 1, 3)
    deleted_second = (N4 - d4) / _falling(n - 1, 4)
    deleted = deleted_first - deleted_second
    # Average centering is algebraically identical and numerically minimizes
    # the sum of squares. Report deviation from U as an arithmetic diagnostic.
    deleted_mean = float(deleted.mean())
    variance = (n - 1.0) / n * float(np.dot(deleted - deleted_mean, deleted - deleted_mean))
    if not np.all(np.isfinite(deleted)) or not math.isfinite(variance):
        raise FloatingPointError("non-finite numerator or variance arithmetic")
    return {
        "estimate": center,
        "first_term": first,
        "second_term": second,
        "n": n,
        "delete_one": deleted,
        "delete_one_first": deleted_first,
        "delete_one_second": deleted_second,
        "jackknife_variance": variance,
        "delete_average_identity_error": deleted_mean - center,
        "first_incidence_identity_error": float(d3.sum()) - 3.0 * N3,
        "second_incidence_identity_error": float(d4.sum()) - 4.0 * N4,
    }


def interaction_bounds(n, probabilities):
    """Deterministic M_n and J_n upper bounds for this mixed-order U."""
    if n < 4:
        raise ValueError("n must be at least four")
    if len(probabilities) == 1:
        # The complete fourth-order subtraction is the constant 1/4.
        A, B = 0.5, 2.0
    else:
        p_min = min(probabilities.values())
        A, B = 0.5 + 1.0 / p_min, 2.0 + 6.0 / p_min
    return A / n, B / (n - 1)


def full_u_empirical_bernstein(v, w, category, probabilities, alpha=0.05,
                               variance_error_fraction=0.5):
    """One-sided full-U lower bound valid also under first-order degeneracy.

    The error split must be chosen independently of the inference rows. This
    is one candidate's allocated error, not a reusable family error budget.
    The returned radius is a specialization of Maurer--Pontil (2018), with
    an exact full-U leave-one transfer proved in the companion note.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not 0 < variance_error_fraction < 1:
        raise ValueError("variance_error_fraction must lie in (0, 1)")
    result = full_u_all_delete_one(v, w, category, probabilities)
    n = result["n"]
    delta_v = alpha * variance_error_fraction
    delta_t = alpha * (1.0 - variance_error_fraction)
    x_v, x_t = -math.log(delta_v), -math.log(delta_t)
    M_n, J_n = interaction_bounds(n, probabilities)
    M_prev, J_prev = interaction_bounds(n - 1, probabilities)
    rho = (n - 1.0) / n
    sd_error = math.sqrt(rho * (2.0 * M_prev**2 + 8.0 * J_prev**2) * x_v)
    sd_upper = math.sqrt(result["jackknife_variance"]) + sd_error
    stochastic = math.sqrt(2.0 * result["jackknife_variance"] * x_t)
    variance_remainder = math.sqrt(2.0 * x_t) * sd_error
    interaction_remainder = (2.0 * M_n / 3.0 + J_n) * x_t
    radius = stochastic + variance_remainder + interaction_remainder
    result.update(
        lower=result["estimate"] - radius,
        radius=radius,
        alpha=alpha,
        variance_error_fraction=variance_error_fraction,
        stochastic_radius=stochastic,
        variance_estimation_remainder=variance_remainder,
        interaction_remainder=interaction_remainder,
        efron_stein_sd_upper=sd_upper,
        M_bound=M_n,
        J_bound=J_n,
        M_previous_bound=M_prev,
        J_previous_bound=J_prev,
    )
    return result


def full_u_joint_bound(v, w, category, probabilities, alpha=0.05,
                       empirical_error_fraction=0.5):
    """Maximum of EB and Hoeffding lower bounds with a fixed union allocation.

    Report both separately at identical alpha for comparison; only this joint
    allocation licenses selecting the larger bound after seeing the rows.
    """
    if not 0 < empirical_error_fraction < 1:
        raise ValueError("empirical_error_fraction must lie in (0, 1)")
    result = full_u_empirical_bernstein(
        v, w, category, probabilities,
        alpha=alpha * empirical_error_fraction,
    )
    lo, hi = baseline._kernel_bounds(probabilities)
    hoeffding_alpha = alpha * (1.0 - empirical_error_fraction)
    hoeffding_radius = (hi - lo) * math.sqrt(
        -math.log(hoeffding_alpha) / (2.0 * (result["n"] // 4))
    )
    result["empirical_bernstein_radius"] = result["radius"]
    result["hoeffding_radius"] = hoeffding_radius
    result["radius"] = min(result["radius"], hoeffding_radius)
    result["lower"] = result["estimate"] - result["radius"]
    result["alpha"] = alpha
    result["empirical_error_fraction"] = empirical_error_fraction
    return result
