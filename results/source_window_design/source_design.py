"""Trace fixed rank residuals to their raw source times and reference entities.

This is an expectation-design checker. Independence and fixed, outcome-independent
fitting weights are assumptions supplied by the analyst, not inferred from IDs.
"""
from fractions import Fraction
import argparse
import json
from pathlib import Path


def fraction(value):
    result = Fraction(str(value))
    return result


def normalized(weights, name):
    if not isinstance(weights, dict):
        raise ValueError(name + " must be an ID-to-weight object")
    parsed = {str(k): fraction(v) for k, v in weights.items()}
    if not parsed or sum(parsed.values()) != 1:
        raise ValueError(f"{name} must contain fixed weights summing exactly to one")
    return parsed


def expand_rows(rows, evaluation, fit, name):
    """Expand evaluation minus a fixed, normalized mean of declared rows."""
    weights = normalized(fit, name + " fitting weights")
    expanded = {str(evaluation): Fraction(1)}
    for key, value in weights.items():
        expanded[key] = expanded.get(key, Fraction(0)) - value
    result = []
    for key, value in expanded.items():
        if key not in rows:
            raise ValueError(f"Missing {name} row: {key}")
        row = rows[key]
        if not isinstance(row["sources"], (list, tuple)):
            raise ValueError("sources must be a list of raw source IDs")
        if any(isinstance(s, bool) or not isinstance(s, (str, int)) for s in row["sources"]):
            raise ValueError("Raw source IDs must be strings or integers")
        sources = tuple(str(s) for s in row["sources"])
        if not sources:
            # A common constant forecast has centered comparison rank zero.
            continue
        if len(set(sources)) != len(sources):
            raise ValueError("A sum window must use distinct source observations")
        peers = normalized(row["peers"], name + " reference weights")
        if any(w < 0 for w in peers.values()):
            raise ValueError("Reference weights must be nonnegative")
        if value:
            result.append({"sources": sources, "weights": {
                peer: value * weight for peer, weight in peers.items()}})
    return result


def compile_design(design):
    """Return exact population and additional-reference incidence coefficients.

    Forecasts must all copy one source or sum two distinct sources with equal
    weights, up to a common strictly increasing transformation within each row.
    Outcomes copy one source. Each peer ID denotes the same raw entity at every
    source time. The focal entity is excluded from all reference sets.
    """
    required = {"focal", "evaluation", "forecast_fit", "outcome_fit",
                "forecast_rows", "outcome_rows"}
    if set(design) != required:
        raise ValueError("Design fields must be exactly: " + ", ".join(sorted(required)))
    focal = str(design["focal"])
    for side in ("forecast", "outcome"):
        rows = design[side + "_rows"]
        if not isinstance(rows, dict):
            raise ValueError("Rows must be an ID-to-row object")
        for row in rows.values():
            if not isinstance(row, dict):
                raise ValueError("Each row must be an object")
            if set(row) != {"sources", "peers"}:
                raise ValueError("Rows support only sources and peers; unequal lag weights, normalization and learned transforms are unsupported")
        if any(focal in {str(k) for k in row.get("peers", {})} for row in rows.values()):
            raise ValueError("The focal entity cannot be a reference peer")
    a = expand_rows(design["forecast_rows"], design["evaluation"],
                    design["forecast_fit"], "forecast")
    b = expand_rows(design["outcome_rows"], design["evaluation"],
                    design["outcome_fit"], "outcome")
    sizes = {len(t["sources"]) for t in a}
    if sizes and sizes not in ({1}, {2}):
        raise ValueError("Use only single-source copies or only equal two-source sums")
    if any(len(t["sources"]) != 1 for t in b):
        raise ValueError("Each outcome row must use one raw source")
    kind = "equal_two_source_sum" if sizes == {2} else "single_source_copy"
    norms = []
    for terms in (a, b):
        combined = {}
        for term in terms:
            for peer, weight in term["weights"].items():
                key = (tuple(sorted(term["sources"])), peer)
                combined[key] = combined.get(key, Fraction(0)) + weight
        norms.append(sum(map(abs, combined.values()), Fraction(0)))
    population = Fraction(0)
    overlap = Fraction(0)
    matches = []
    for left in a:
        for right in b:
            source = right["sources"][0]
            if source not in left["sources"]:
                continue
            t = sum(left["weights"].values()) * sum(right["weights"].values())
            peers = set(left["weights"]) & set(right["weights"])
            products = {peer: left["weights"][peer] * right["weights"][peer]
                        for peer in sorted(peers)}
            omega = sum(products.values(), Fraction(0))
            population += t
            overlap += omega
            matches.append({"forecast_sources": list(left["sources"]),
                "outcome_source": source, "population_coefficient": str(t),
                "peer_products": {k: str(v) for k, v in products.items()},
                "overlap_coefficient": str(omega)})
    return {"forecast_class": kind, "population_coefficient": str(population),
        "overlap_coefficient": str(overlap),
        "coefficient_l1_forecast": str(norms[0]),
        "coefficient_l1_outcome": str(norms[1]),
        "preserves_population_reference_expectation_for_every_common_law": overlap == 0,
        "population_reference_centered_for_every_common_law": population == 0,
        "matches": matches,
        "meaning": ("E[AB] = v_F T + omega_F Omega" if kind == "single_source_copy"
                    else "E[AB] = kappa_F T + lambda_F Omega"),
        "scope": "Independent common-law raw sources; fixed normalized fitting and reference weights; centered, unstandardized midranks. This output is not a panel p-value or a claim of predictive improvement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_design(json.loads(args.design.read_text()))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
