"""Explicit source assignments, target preservation, and forecast availability.

Reference IDs name complete trajectory DRAW ROLES, not individual coordinates.
Reusing one role reuses one column. Distinct IDs do not prove independence; the
conditional common-law sampling assumption remains an external requirement.
A fixed archive sampled by independent indices is different from physical IID
observations. Repeated returned archive IDs can still be independent index draws.

Exact rational arithmetic checks design declarations. Floating kernel evaluation
has its separate numerical contract. No metadata validator proves actual feature
latency, stochastic independence, or that a named model was truly frozen.
"""

from datetime import datetime, timezone
from fractions import Fraction
import math
import numbers

import numpy as np


def rational(value, name="coefficient"):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, numbers.Real):
        try:
            numerator, denominator = value.as_integer_ratio()
        except (AttributeError, OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite supported real number") from exc
        return Fraction(int(numerator), int(denominator))
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"invalid {name}") from exc
    raise ValueError(f"unsupported {name}")


def _identifiers(records, name, signature=False):
    if not records:
        raise ValueError(f"{name} cannot be empty")
    out = {}
    for record in records:
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in out:
            raise ValueError(f"{name} require unique nonempty IDs")
        if signature and (not isinstance(record.get("signature"), str) or not record["signature"]):
            raise ValueError("each map requires a fixed signature declaration")
        out[identifier] = len(out)
    return out


def compile_reference_design(forecast_maps, outcome_maps, references, forecast_terms, outcome_terms):
    """Compile weighted source-ID terms into C,D and exact design diagnostics.

    Map records: {id, signature}. Reference records: {id, law_id}. Term records:
    {map_id, reference_id, weight}. Repeated terms for the SAME draw role add;
    they do not create independent copies. Distinct reference laws are rejected
    because this API implements the common-law theorem only.
    """
    fi = _identifiers(forecast_maps, "forecast maps", True)
    gi = _identifiers(outcome_maps, "outcome maps", True)
    ri = _identifiers(references, "reference roles")
    law_ids = {row.get("law_id") for row in references}
    if len(law_ids) != 1 or not all(isinstance(x, str) and x for x in law_ids):
        raise ValueError("the common-law design needs one explicit common reference law")

    def matrix(terms, maps):
        result = [[Fraction(0) for _ in ri] for _ in maps]
        for term in terms:
            if term.get("map_id") not in maps or term.get("reference_id") not in ri:
                raise ValueError("term contains an undeclared map or reference role")
            result[maps[term["map_id"]]][ri[term["reference_id"]]] += rational(term["weight"])
        return result

    C, D = matrix(forecast_terms, fi), matrix(outcome_terms, gi)
    c, d = [sum(row) for row in C], [sum(row) for row in D]
    omega = [[sum(x*y for x, y in zip(a, b)) for b in D] for a in C]
    exact = lambda rows: [[str(x) for x in row] for row in rows]
    Cf, Df = [[float(x) for x in row] for row in C], [[float(x) for x in row] for row in D]
    if not all(math.isfinite(x) for row in Cf + Df for x in row):
        raise ValueError("compiled coefficients exceed floating kernel representation")
    return {
        "forecast_maps": [dict(row) for row in forecast_maps],
        "outcome_maps": [dict(row) for row in outcome_maps],
        "reference_ids": list(ri), "reference_law_id": next(iter(law_ids)),
        "C": Cf, "D": Df, "C_exact": exact(C), "D_exact": exact(D),
        "c_exact": list(map(str, c)), "d_exact": list(map(str, d)),
        "omega_exact": exact(omega),
        "universal_preservation": all(x == 0 for row in omega for x in row),
        "floating_coefficients_exact": all(Fraction.from_float(float(x)) == x for row in C + D for x in row),
        "independence_verified": False,
        "scope": "Declared same-law trajectory draw roles; metadata does not establish independence.",
    }


def preserved_target(before, after):
    """Exact same-map, same-law row-sum comparison; reject a changed target.

    The frozen map signatures and reference law must be the same. A signature
    comparison does not prove equality of executable maps or actual laws.
    """
    for key in ("forecast_maps", "outcome_maps", "reference_law_id"):
        if before[key] != after[key]:
            raise ValueError(f"target declaration changed: {key}")
    for key in ("c_exact", "d_exact"):
        if [rational(x) for x in before[key]] != [rational(x) for x in after[key]]:
            raise ValueError(f"target row sums changed: {key}")
    # The generated float arrays can have different exact row sums after an
    # algebraically valid rational reassignment. Check their actual binary64
    # coefficients as rational numbers too; never hide this target discrepancy.
    for key in ("C", "D"):
        old = [sum((Fraction.from_float(float(x)) for x in row), Fraction()) for row in before[key]]
        new = [sum((Fraction.from_float(float(x)) for x in row), Fraction()) for row in after[key]]
        if old != new:
            raise ValueError(f"floating coefficient row sums changed: {key}")
    return {"same_declared_maps": True, "same_declared_law": True,
            "exact_row_sums_preserved": True, "floating_row_sums_preserved": True,
            "independence_verified": False}


def disjoint_reference_design(design, forecast_reference_id="forecast_reference", outcome_reference_id="outcome_reference"):
    """Two declared independent same-law roles preserve c,d and force Omega=0.

    This is a constructive reassignment of source roles, not reuse of one
    realized observation under two labels. It requires actual independent draws
    from the declared law and does not assert a power or cost advantage.
    """
    if forecast_reference_id == outcome_reference_id:
        raise ValueError("disjoint design requires two different draw-role IDs")
    references = [{"id": identifier, "law_id": design["reference_law_id"]}
                  for identifier in (forecast_reference_id, outcome_reference_id)]
    ft = [{"map_id": row["id"], "reference_id": forecast_reference_id, "weight": weight}
          for row, weight in zip(design["forecast_maps"], design["c_exact"])]
    gt = [{"map_id": row["id"], "reference_id": outcome_reference_id, "weight": weight}
          for row, weight in zip(design["outcome_maps"], design["d_exact"])]
    result = compile_reference_design(design["forecast_maps"], design["outcome_maps"], references, ft, gt)
    result["target_check"] = preserved_target(design, result)
    return result


def _utc_timestamp(value, name):
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} requires an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} requires an explicit timezone")
    return timestamp.astimezone(timezone.utc)


def validate_forecast_availability(forecasts, training_source_ids=(), audit_source_ids=(), independence_statement=""):
    """Validate declared issue/target times, release latency, and training reuse.

    Each forecast has id, issue_time, target_time, model_available_at, and features.
    Each feature has id, observed_at, available_at. Availability at issue time is
    permitted. Future outcomes may be used as outcome maps but never supplied as
    an available forecast feature. Checks concern supplied records only.
    """
    if not forecasts:
        raise ValueError("at least one forecast contract is required")
    if not isinstance(independence_statement, str) or not independence_statement.strip():
        raise ValueError("declare the conditional sampling assumption separately")
    overlap = set(training_source_ids) & set(audit_source_ids)
    if overlap:
        raise ValueError("declared training and audit primitive source IDs overlap")
    seen = set()
    total_features = 0
    for row in forecasts:
        if not row.get("id") or row["id"] in seen:
            raise ValueError("forecast IDs must be unique and nonempty")
        seen.add(row["id"])
        issue = _utc_timestamp(row["issue_time"], "issue_time")
        target = _utc_timestamp(row["target_time"], "target_time")
        available_model = _utc_timestamp(row["model_available_at"], "model_available_at")
        if target <= issue:
            raise ValueError("the forecast target must be strictly after its issue time")
        if available_model > issue:
            raise ValueError("the fitted model was not available at forecast issue time")
        feature_ids = set()
        if not isinstance(row.get("features"), list):
            raise ValueError("features must explicitly list the forecast input records")
        for feature in row["features"]:
            if not feature.get("id") or feature["id"] in feature_ids:
                raise ValueError("feature IDs must be unique and nonempty within a forecast")
            feature_ids.add(feature["id"])
            observed = _utc_timestamp(feature["observed_at"], "observed_at")
            available = _utc_timestamp(feature["available_at"], "available_at")
            if observed > available:
                raise ValueError("a measured feature cannot be available before its observation")
            if available > issue:
                raise ValueError("a forecast feature was unavailable at issue time")
        total_features += len(feature_ids)
    return {
        "forecast_count": len(forecasts), "feature_records_checked": total_features,
        "declared_availability_pass": True, "declared_training_reuse_absent": True,
        "actual_latency_verified": False, "independence_verified": False,
        "sampling_assumption": independence_statement,
    }
