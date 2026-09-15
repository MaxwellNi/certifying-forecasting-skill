"""Check source tracing against exhaustive products of raw comparison ranks."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
import argparse
import json
from pathlib import Path
import numpy as np
from source_design import compile_design, expand_rows


def row(sources, peer):
    return {"sources": sources, "peers": {str(peer): "1"}}


def examples():
    temporal = {"focal": "0", "evaluation": "2",
        "forecast_fit": {"1": "1"}, "outcome_fit": {"1": "1"},
        "forecast_rows": {"1": row([0], 2), "2": row([1], 1)},
        "outcome_rows": {"1": row([1], 1), "2": row([2], 2)}}
    global_split = deepcopy(temporal)
    for r in global_split["forecast_rows"].values(): r["peers"] = {"1": "1"}
    for r in global_split["outcome_rows"].values(): r["peers"] = {"2": "1"}
    sums = {"focal": "0", "evaluation": "2",
        "forecast_fit": {"5": "1"}, "outcome_fit": {"0": "1/2", "4": "1/2"},
        "forecast_rows": {"2": row([1, 0], 1), "5": row([4, 3], 1)},
        "outcome_rows": {"0": row([0], 2), "2": row([2], 2), "4": row([4], 1)}}
    negative = deepcopy(sums)
    negative["outcome_rows"]["0"]["peers"] = {"1": "1"}
    negative["outcome_rows"]["4"]["peers"] = {"2": "1"}
    zero = deepcopy(sums)
    zero["outcome_rows"]["0"]["peers"] = {"1": "1"}
    return {"calendar_peer_split": temporal, "source_aligned_peer_split": global_split,
            "sum_positive_interaction": sums, "sum_negative_interaction": negative,
            "sum_zero_interaction": zero}


def direct_scores(design, arrays):
    scores = []
    for side in ("forecast", "outcome"):
        terms = expand_rows(design[side + "_rows"], design["evaluation"],
                            design[side + "_fit"], side)
        score = np.zeros(len(arrays))
        for term in terms:
            # Object addition preserves exact integer order even beyond 2**53.
            block = arrays[:, [int(t) for t in term["sources"]], :]
            values = block.astype(object).sum(axis=1)
            focal = values[:, int(design["focal"])]
            for peer, weight in term["weights"].items():
                reference = values[:, int(peer)]
                score += float(weight) * ((reference < focal).astype(float)
                    + .5 * (reference == focal).astype(float) - .5)
        scores.append(score)
    return scores[0] * scores[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    expected = {"calendar_peer_split": ("-1", "-1", "-1/8"),
        "source_aligned_peer_split": ("-1", "0", "-1/16"),
        "sum_positive_interaction": ("0", "1/2", "3/128"),
        "sum_negative_interaction": ("0", "-1/2", "-3/128"),
        "sum_zero_interaction": ("0", "0", "0")}
    checks = []
    for name, design in examples().items():
        compiled = compile_design(design)
        t, omega, target = expected[name]
        assert compiled["population_coefficient"] == t
        assert compiled["overlap_coefficient"] == omega
        times = 5 if name.startswith("sum_") else 3
        arrays = np.array(list(product([0, 1], repeat=times * 3)), dtype=np.int64).reshape(-1, times, 3)
        actual = Fraction(float(direct_scores(design, arrays).mean()))
        assert actual == Fraction(target), (name, actual, target)
        # Increasing common transformations preserve copies, but not arbitrary
        # sums. An affine integer map does preserve both, with exact arithmetic.
        large = 2**54 + 7 * arrays
        assert np.array_equal(direct_scores(design, arrays), direct_scores(design, large))
        checks.append({"design": name, "raw_arrays": len(arrays), "exact_mean": str(actual),
                       "population_coefficient": t, "overlap_coefficient": omega})
        (args.output / (name + ".json")).write_text(json.dumps(design, indent=2) + "\n")
        (args.output / (name + "_audit.json")).write_text(json.dumps(compiled, indent=2) + "\n")
    report = {"status": "PASS", "checks": checks,
        "raw_arrays": sum(c["raw_arrays"] for c in checks),
        "integer_order_check": "All rows repeated after the exact affine map 2**54 + 7*y",
        "sampling": "Exhaustive Bernoulli(1/2) raw source panels, including ties; no Monte Carlo estimation."}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__": main()
