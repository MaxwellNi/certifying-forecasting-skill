"""All-pair, conditional-category U bound obtained by averaging split MGFs.

See THEORY.md in this directory. Inputs are sufficient statistics from the
same independent paired observations, with both residual channels recorded.
Each channel has range length ONE within each category. Dependence between
the two channels in one pair is unrestricted. Fits and exact category masses
are fixed independently of these observations.

This file is a mathematical comparator, not a claim of novel U-statistic theory.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def pooled_upper_bound(
    masses: Sequence[float],
    counts: Sequence[int],
    sum_a: Sequence[float],
    sum_b: Sequence[float],
    sum_a2: Sequence[float],
    sum_b2: Sequence[float],
    sum_ab: Sequence[float],
    delta: float,
    *,
    absent_upper: Sequence[float] | None = None,
    lambda_mode: str = "fixed",
    product_mode: str = "subgamma",
    grid_exponents: Sequence[float] = tuple(range(-8, 9)),
    clip_deterministic: bool = False,
) -> dict[str, float | int | str]:
    """Return a finite-sample (1-delta) upper bound for sum p_c a_c b_c.

    ``counts`` conditions only on focal category assignments; reference
    categories must remain unconditioned. Every observation is an independent
    focal/reference pair. Residual-channel ranges have length one, and lie
    within [-1, 1]. Their within-pair dependence is unrestricted.

    Cells with n<2 use a deterministic product upper bound (default one).
    The grid, mode, and delta must be fixed without inspecting residual values.
    The optional deterministic clipping assumes every supplied absent_upper
    bounds its entire cell, including cells with n>=2.

    Modes:
      fixed: lambda=sqrt(2*log(3/delta)/D), chosen only from counts/masses.
      grid: use a predeclared finite lambda grid around the fixed lambda;
            a log(grid size) penalty permits minimization after observing data.
      mixture: Gaussian-mixture boundary, no tuning grid; see THEORY.md.
      square_mgf: average the split square-normalized MGFs, obtaining
                  r(V)=sqrt(c_x*V), with a deterministic c_x>2x; see THEORY.md.

    product_mode="exact_mgf" numerically chooses a deterministic Chernoff
    tuning from the sharper already-proved product MGF. It uses no residual
    observations and no additional error allocation.

    Input checks cannot establish the sampling or exact-metadata contract.
    """
    length = len(masses)
    arrays = (counts, sum_a, sum_b, sum_a2, sum_b2, sum_ab)
    if length == 0 or any(len(v) != length for v in arrays):
        raise ValueError("All nonempty category arrays must have the same length.")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one.")
    if any(not math.isfinite(p) or p < 0 for p in masses):
        raise ValueError("Known masses must be finite and nonnegative.")
    if not math.isclose(sum(masses), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Known masses must sum to one.")
    if any(not math.isfinite(n) or n < 0 or int(n) != n for n in counts):
        raise ValueError("Counts must be nonnegative integers.")
    if any(not math.isfinite(v) for arr in arrays[1:] for v in arr):
        raise ValueError("Sufficient statistics must be finite.")
    if absent_upper is None:
        absent_upper = [1.0] * length
    if len(absent_upper) != length or any(not math.isfinite(q) for q in absent_upper):
        raise ValueError("A finite deterministic product upper bound is required per cell.")
    if lambda_mode not in ("fixed", "grid", "mixture", "square_mgf"):
        raise ValueError("lambda_mode must be fixed, grid, mixture, or square_mgf.")
    if product_mode not in ("subgamma", "exact_mgf"):
        raise ValueError("product_mode must be subgamma or exact_mgf.")
    if lambda_mode == "grid" and (
        not grid_exponents or any(not math.isfinite(j) for j in grid_exponents)
    ):
        raise ValueError("The predeclared grid must be nonempty and finite.")

    used = [c for c in range(length) if masses[c] > 0 and counts[c] >= 2]
    omitted = [c for c in range(length) if masses[c] > 0 and counts[c] < 2]
    center = variance_a = variance_b = D = dmax = 0.0
    products = []
    for c in used:
        p, n = masses[c], counts[c]
        k, ell = n // 2, n - n // 2
        abar, bbar = sum_a[c] / n, sum_b[c] / n
        raw_ss_a = sum_a2[c] - sum_a[c] * abar
        raw_ss_b = sum_b2[c] - sum_b[c] * bbar
        tolerance = 1e-10 * max(1, n)
        if min(raw_ss_a, raw_ss_b) < -tolerance:
            raise ValueError("Sufficient statistics imply a negative sample variance.")
        if abs(abar) > 1 + 1e-12 or abs(bbar) > 1 + 1e-12:
            raise ValueError("Residual means must lie in [-1, 1].")
        sa2, sb2 = max(raw_ss_a, 0.0) / (n - 1), max(raw_ss_b, 0.0) / (n - 1)
        center += p * (sum_a[c] * sum_b[c] - sum_ab[c]) / (n * (n - 1))
        variance_a += p * p * (bbar * bbar + k * sb2 / (n * ell)) / (4 * k)
        variance_b += p * p * (abar * abar + ell * sa2 / (n * k)) / (4 * ell)
        d = p / (4 * math.sqrt(k * ell))
        D += d * d
        dmax = max(dmax, d)
        products.append(d)

    x_product = math.log(3.0 / delta)
    grid_size = len(grid_exponents) if lambda_mode == "grid" else 1
    x_linear = math.log(3.0 * grid_size / delta)
    lambda0 = math.sqrt(2 * x_product / D) if D > 0 else 0.0
    if D == 0:
        radius_a = radius_b = 0.0
    elif lambda_mode == "fixed":
        radius_a = x_linear / lambda0 + lambda0 * variance_a / 2
        radius_b = x_linear / lambda0 + lambda0 * variance_b / 2
    elif lambda_mode == "grid":
        lambdas = [lambda0 * 2.0**j for j in grid_exponents]
        if any(not math.isfinite(t) or t <= 0 for t in lambdas):
            raise ValueError("Grid produced a nonpositive or nonfinite lambda.")
        radius_a = min(x_linear / t + t * variance_a / 2 for t in lambdas)
        radius_b = min(x_linear / t + t * variance_b / 2 for t in lambdas)
    elif lambda_mode == "mixture":
        rho = D / (2 * x_product)
        radius_a = math.sqrt((variance_a + rho) * (2 * x_linear + math.log1p(variance_a / rho)))
        radius_b = math.sqrt((variance_b + rho) * (2 * x_linear + math.log1p(variance_b / rho)))
    else:
        # Optimize deterministic alpha in (0,1). For u=1/(1-alpha), the
        # stationary equation is u-1-log(u)=2*x and has one root above one.
        lo, hi = 1.0, 2 * x_linear + 2
        while hi - 1 - math.log(hi) < 2 * x_linear:
            hi *= 2
        for _ in range(70):
            u = (lo + hi) / 2
            if u - 1 - math.log(u) < 2 * x_linear:
                lo = u
            else:
                hi = u
        alpha = 1 - 1 / ((lo + hi) / 2)
        # Reevaluate the valid coefficient at the chosen alpha; correctness
        # does not depend on locating the exact optimum numerically.
        coefficient = (2 * x_linear - math.log1p(-alpha)) / alpha
        radius_a = math.sqrt(coefficient * variance_a)
        radius_b = math.sqrt(coefficient * variance_b)
    radius_product = math.sqrt(2 * x_product * D) + x_product * dmax
    if product_mode == "exact_mgf" and D > 0:
        relative_squares = [(d / dmax)**2 for d in products]
        # Normalize z=t*dmax; any 0<z<1 gives a valid deterministic bound.
        lo, hi = 0.0, math.nextafter(1.0, 0.0)
        for _ in range(70):
            z = (lo + hi) / 2
            us = [z * z * ratio for ratio in relative_squares]
            derivative_sign = sum(u/(1-u) + .5*math.log1p(-u) for u in us) - x_product
            if derivative_sign < 0:
                lo = z
            else:
                hi = z
        z = (lo + hi) / 2
        psi = -.5 * sum(math.log1p(-z*z*ratio) for ratio in relative_squares)
        radius_product = min(radius_product, dmax * (x_product + psi) / z)
    allowance = sum(masses[c] * absent_upper[c] for c in omitted)
    upper_unclipped = center + radius_a + radius_b + radius_product + allowance
    deterministic_upper = sum(p * q for p, q in zip(masses, absent_upper))
    upper = min(upper_unclipped, deterministic_upper) if clip_deterministic else upper_unclipped
    return {
        "center": center,
        "linear_a_proxy": variance_a,
        "linear_b_proxy": variance_b,
        "product_variance_proxy": D,
        "product_scale_proxy": dmax,
        "linear_a_radius": radius_a,
        "linear_b_radius": radius_b,
        "product_radius": radius_product,
        "absent_allowance": allowance,
        "upper": upper,
        "upper_unclipped": upper_unclipped,
        "deterministic_upper": deterministic_upper,
        "lambda_mode": lambda_mode,
        "product_mode": product_mode,
        "lambda0": lambda0,
        "grid_size": grid_size,
        "x_linear": x_linear,
        "x_product": x_product,
        "delta": delta,
        "categories_used": len(used),
        "categories_absent": len(omitted),
        "information_contract": "exact category masses; fixed fits; independent pairs with both channels; focal labels conditioned",
    }
