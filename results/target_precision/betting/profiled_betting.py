"""Classical stratified union-of-intersections betting on the SAME pair scores.

Fixed five-fraction mixture; no oracle samples, learned means, or new outcomes.
Evidence is computed from a certified convex-tangent LOWER objective bound,
not the numerical optimizer's primal objective. See PROFILED_BETTING.md.
"""

from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import math
import time

import numpy as np
from scipy.optimize import minimize


DEFAULT_FRACTIONS = ("0.1", "0.3", "0.5", "0.7", "0.9")
LOWER = Fraction(1, 4)
UPPER = Fraction(3, 4)


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("nonfinite value")
        return Fraction.from_float(float(value))
    return Fraction(value)


def _contexts(precision):
    if precision < 30:
        raise ValueError("decimal precision must be at least 30")
    return (Context(prec=precision, rounding=ROUND_FLOOR, Emin=-999999999, Emax=999999999),
            Context(prec=precision, rounding=ROUND_CEILING, Emin=-999999999, Emax=999999999))


def _decimal_fraction(value, context):
    return context.divide(Decimal(value.numerator), Decimal(value.denominator))


def _log_fraction_interval(value, down, up):
    """Outward bounds: decimal ln is correctly rounded to nearest; step out once."""
    if value <= 0:
        raise ValueError("log argument must be positive")
    lo = _decimal_fraction(value, down)
    hi = _decimal_fraction(value, up)
    return down.next_minus(down.ln(lo)), up.next_plus(up.ln(hi))


def _to_float_down(value):
    ans = float(value)
    return math.nextafter(ans, -math.inf) if math.isfinite(ans) else ans


def _to_float_up(value):
    ans = float(value)
    return math.nextafter(ans, math.inf) if math.isfinite(ans) else ans


def _read_prepared(prepared):
    used, missing = [], Fraction(0)
    for cell in prepared["cells"]:
        mass = _fraction(cell["mass"])
        values = np.asarray(cell["values"], dtype=float)
        count = int(cell["pairs"])
        if mass <= 0 or values.ndim != 1 or len(values) != count:
            raise ValueError("positive masses and correct pair counts required")
        if not np.isfinite(values).all() or (np.abs(values) > .5).any():
            raise ValueError("pair scores must be finite and in [-1/2,1/2]")
        if not count:
            missing += mass
            continue
        unique, counts = np.unique(values, return_counts=True)
        # Form X=Z+1/2 as an exact rational; do not round the shifted score.
        support = tuple((_fraction(z) + Fraction(1, 2), int(k))
                        for z, k in zip(unique, counts))
        used.append({"mass": mass, "count": count, "support": support,
                     "x_float": np.array([float(x) for x, _ in support]),
                     "k_float": np.array([k for _, k in support], dtype=float)})
    observed = sum((cell["mass"] for cell in used), Fraction(0))
    return used, missing, observed


def _float_objective(eta, cells, fraction):
    f = 0.0
    g = np.empty(len(cells))
    for k, cell in enumerate(cells):
        den = (1-fraction)*eta[k] + fraction*cell["x_float"]
        f += float(np.dot(cell["k_float"], np.log(den/eta[k])))
        g[k] = -float(np.dot(cell["k_float"], fraction*cell["x_float"]/(eta[k]*den)))
    return f, g


def _feasible_exact_point(eta, masses, cap):
    point = [min(UPPER, max(LOWER, _fraction(float(v)))) for v in eta]
    total = sum((p*x for p, x in zip(masses, point)), Fraction(0))
    lower_total = LOWER*sum(masses, Fraction(0))
    if total > cap:
        factor = (cap-lower_total)/(total-lower_total)
        point = [LOWER+factor*(x-LOWER) for x in point]
    assert all(LOWER <= x <= UPPER for x in point)
    assert sum((p*x for p, x in zip(masses, point)), Fraction(0)) <= cap
    return point


def _tangent_certificate(cells, masses, cap, point, fraction, down, up,
                         downward_tolerance):
    """Certified lower/upper objective interval; all gradient/LP algebra is rational."""
    f_lower = Decimal(0)
    f_upper = Decimal(0)
    gradient = []
    for cell, eta in zip(cells, point):
        g = Fraction(0)
        for x, count in cell["support"]:
            den = (1-fraction)*eta + fraction*x
            lo, hi = _log_fraction_interval(den/eta, down, up)
            f_lower = down.add(f_lower, down.multiply(Decimal(count), lo))
            f_upper = up.add(f_upper, up.multiply(Decimal(count), hi))
            g -= count*fraction*x/(eta*den)
        gradient.append(g)

    # Exact minimizer of the affine tangent over box plus weighted-mean cap.
    # Gradients are nonpositive; allocate the available mass to most negative g/p.
    linear_minimizer = [LOWER for _ in masses]
    remaining = cap-LOWER*sum(masses, Fraction(0))
    for k in sorted(range(len(masses)), key=lambda j: gradient[j]/masses[j]):
        if remaining <= 0 or gradient[k] >= 0:
            break
        allocation = min(remaining, masses[k]*(UPPER-LOWER))
        linear_minimizer[k] += allocation/masses[k]
        remaining -= allocation
    assert sum((p*x for p, x in zip(masses, linear_minimizer)), Fraction(0)) <= cap
    lower = f_lower
    for g, u, x in zip(gradient, linear_minimizer, point):
        lower = down.add(lower, _decimal_fraction(g*(u-x), down))
    lower = down.subtract(lower, downward_tolerance)
    gap = up.subtract(f_upper, lower)
    return lower, f_upper, gap


def profiled_betting(prepared, fractions=DEFAULT_FRACTIONS, *, precision=60,
                     max_iterations=300, optimizer_tolerance=1e-12):
    """Return a valid fixed-sample p-value for the GLOBAL null theta <= 0.

    Fractions and numerical settings must be fixed before looking at inference
    outcomes. Changing fractions after seeing results needs an error argument.
    The input is stratified_reference.prepare(...) and every supplied raw row
    remains charged. Float masses are treated as their exact represented values
    in the numerical certificate; the statistical contract requires the supplied
    weights to identify the target, separately from numerical arithmetic.
    """
    started = time.perf_counter()
    fractions = tuple(_fraction(b) for b in fractions)
    if not fractions or any(not 0 < b < 1 for b in fractions):
        raise ValueError("predeclared fractions must lie strictly between 0 and 1")
    cells, missing, observed = _read_prepared(prepared)
    cap = observed/2 + missing/4
    total_pairs = sum(cell["count"] for cell in cells)
    common = {"method": "classical_profiled_inverse_betting_mixture",
              "fractions": [str(b) for b in fractions],
              "pairs": total_pairs, "used_categories": len(cells),
              "additional_raw_rows": 0,
              "total_raw_rows_charged": prepared["total_raw_rows_charged"],
              "missing_mass": float(missing), "partial_shifted_mean_cap": float(cap),
              "decimal_precision": precision,
              "certificate": "exact-rational tangent LP; outward Decimal logarithms"}
    # The full null imposes no restriction beyond the known stratum mean bounds.
    # Returning one avoids reporting accidental evidence in this uninformative case.
    if not cells or cap >= UPPER*observed:
        return dict(common, p_value=1.0, p=1.0, log_e_lower=0.0,
                    objective_lower=0.0, objective_upper=0.0, objective_gap=0.0, profiles=[],
                    status="no_restriction_on_observed_stratum_means",
                    elapsed_seconds=time.perf_counter()-started)

    down, up = _contexts(precision)
    masses = [cell["mass"] for cell in cells]
    mass_float = np.array([float(p) for p in masses])
    initial = np.full(len(cells), float(cap/observed))
    profiles, lower_logs, upper_logs = [], [], []
    extra_down = Decimal("1e-12")*Decimal(1+total_pairs)
    for fraction in fractions:
        b = float(fraction)
        calls = [0]

        def objective(eta):
            calls[0] += 1
            f, g = _float_objective(eta, cells, b)
            return f/max(1, total_pairs), g/max(1, total_pairs)

        result = minimize(objective, initial, jac=True, method="SLSQP",
                          bounds=[(.25, .75)]*len(cells),
                          constraints=[{"type": "ineq",
                                        "fun": lambda eta: float(cap)-np.dot(mass_float, eta),
                                        "jac": lambda eta: -mass_float}],
                          options={"ftol": optimizer_tolerance, "maxiter": max_iterations})
        candidate = result.x if np.isfinite(result.x).all() else initial
        point = _feasible_exact_point(candidate, masses, cap)
        lower, upper, gap = _tangent_certificate(
            cells, masses, cap, point, fraction, down, up, extra_down)
        lower_logs.append(lower)
        upper_logs.append(upper)
        profiles.append({"fraction": str(fraction),
                         "log_e_lower": _to_float_down(lower),
                         "primal_log_e_upper": _to_float_up(upper),
                         "certified_optimization_gap_upper": _to_float_up(gap),
                         "optimizer_success": bool(result.success),
                         "optimizer_status": str(result.message),
                         "optimizer_iterations": int(result.nit),
                         "objective_evaluations": calls[0],
                         "feasible_mean_vector": [float(x) for x in point]})

    # Mixture of the separately profiled e-values. No independence is required.
    # Outward exp/division makes the reported evidence a lower bound too.
    e_lower = Decimal(0)
    for lower in lower_logs:
        value = down.next_minus(down.exp(lower))
        e_lower = down.add(e_lower, max(Decimal(0), value))
    e_lower = down.divide(e_lower, Decimal(len(lower_logs)))
    if e_lower <= 1:
        pvalue = 1.0
    else:
        p_upper = up.divide(Decimal(1), e_lower)
        pvalue = min(1.0, _to_float_up(p_upper))
    log_e_lower = down.next_minus(down.ln(e_lower)) if e_lower else Decimal("-Infinity")
    e_upper = Decimal(0)
    for upper in upper_logs:
        e_upper = up.add(e_upper, up.next_plus(up.exp(upper)))
    e_upper = up.divide(e_upper, Decimal(len(upper_logs)))
    log_e_upper = up.next_plus(up.ln(e_upper))
    objective_gap = up.subtract(log_e_upper, log_e_lower)
    return dict(common, p_value=pvalue, p=pvalue, log_e_lower=_to_float_down(log_e_lower),
                objective_lower=_to_float_down(log_e_lower),
                objective_upper=_to_float_up(log_e_upper),
                objective_gap=_to_float_up(objective_gap),
                profiles=profiles, status="certified_lower_evidence",
                elapsed_seconds=time.perf_counter()-started)


def profiled_mixture_pvalue(prepared):
    """Fixed primary API: five predeclared fractions, certified mixture p-value."""
    return profiled_betting(prepared)
