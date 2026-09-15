"""Portable replay of the retained pooled-bias evidence; no model retraining.

The default checks every stored simulation row/summary, scalar sufficient
statistics, both forecast archives, and independently authored finite-state
and formula verifiers. --full additionally regenerates all synthetic draws.
Historical review receipts are preserved. A new receipt records the actual
public sources executed now, without claiming a new independent review.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from pooled_bias_bound import pooled_upper_bound


HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent.parent
KEYS = ["categories", "validation_pairs", "mass", "fit_error", "delta"]
CANDIDATE_KEYS = ["task", "baseline", "candidate", "budget_method"]
ATOL, RTOL = 1e-12, 1e-11


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def compare_json(actual, expected, label="json"):
    """Compare scientific content, including exact structure and count fields."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), label
        return max([0.0] + [compare_json(actual[k], v, f"{label}.{k}")
                            for k, v in expected.items()])
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), label
        return max([0.0] + [compare_json(a, b, f"{label}[{i}]")
                            for i, (a, b) in enumerate(zip(actual, expected))])
    if isinstance(expected, float):
        assert np.isclose(actual, expected, atol=ATOL, rtol=RTOL), (label, actual, expected)
        return abs(actual - expected)
    assert actual == expected, (label, actual, expected)
    return 0.0


def compare_frames(actual, expected, keys, label):
    assert set(actual.columns) == set(expected.columns), (label, "columns")
    assert not actual.duplicated(keys).any(), (label, "duplicate actual keys")
    assert not expected.duplicated(keys).any(), (label, "duplicate stored keys")
    a = actual.sort_values(keys).reset_index(drop=True)[expected.columns]
    b = expected.sort_values(keys).reset_index(drop=True)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    maximum = 0.0
    for col in b:
        av, bv = a[col], b[col]
        assert np.array_equal(av.isna(), bv.isna()), (label, col, "missing values")
        valid = ~bv.isna()
        if pd.api.types.is_numeric_dtype(bv) and not pd.api.types.is_bool_dtype(bv):
            x, y = av[valid].to_numpy(), bv[valid].to_numpy()
            if pd.api.types.is_integer_dtype(bv):
                assert np.array_equal(x, y), (label, col, "integer mismatch")
            else:
                assert np.allclose(x, y, atol=ATOL, rtol=RTOL), (label, col)
            if len(x):
                maximum = max(maximum, float(np.max(np.abs(x - y))))
        else:
            assert np.array_equal(av[valid], bv[valid]), (label, col, "exact values")
    return {"rows": len(b), "columns": len(b.columns), "max_absolute_difference": maximum}


def load_verifier(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # CLI receipts serialize numeric dictionary keys as JSON strings.
    return json.loads(json.dumps(module.verify()))


def verify_independent_sources():
    historical = json.loads((HERE / "independent_math_check.json").read_text())
    finite = load_verifier("independent_verify_square_mgf.py")
    implementation = load_verifier("independent_verify_square_implementation.py")
    finite_difference = compare_json(finite, historical["finite_state_check"], "finite_state")
    old = historical["implementation_check"]
    hash_fields = {"code_sha256", "theory_sha256"}
    formula_difference = compare_json(
        {k: v for k, v in implementation.items() if k not in hash_fields},
        {k: v for k, v in old.items() if k not in hash_fields}, "implementation")
    return {
        "scope": "Reexecution of retained independently authored checks against current public sources; not a new blind or independent review.",
        "historical_receipt_preserved": "independent_math_check.json",
        "historical_receipt_sha256": sha256(HERE / "independent_math_check.json"),
        "source_hash_comparison": {
            k: {"historical": old[k], "current": implementation[k], "matches": old[k] == implementation[k]}
            for k in sorted(hash_fields)
        },
        "provenance_note": "The historical receipt describes the original author-workspace execution. Public portability/prose edits can change hashes. This receipt binds the new execution to the actual public files; the old hashes and original findings are not rewritten.",
        "max_historical_finite_state_difference": finite_difference,
        "max_historical_formula_difference": formula_difference,
        "finite_state_check": finite,
        "implementation_check": implementation,
    }


def scalar_checks():
    examples = json.loads((HERE / "simulation/example_sufficient_statistics.json").read_text())
    maximum, calls = 0.0, 0
    for item in examples:
        for mode in ("fixed", "grid", "mixture"):
            result = pooled_upper_bound(item["masses"], *item["stats"], item["delta"],
                                        absent_upper=item["corner"], lambda_mode=mode)
            for key, expected in [("upper", item["expected"]["pooled_" + mode]),
                                  ("center", item["expected"]["pooled_center"])]:
                maximum = max(maximum, compare_json(result[key], expected, "simulation scalar " + key))
            calls += 1
    stats = json.loads((HERE / "archive/sufficient_statistics.json").read_text())
    candidates = pd.concat([read_csv(HERE / "archive/all_candidates.csv"),
                            read_csv(HERE / "secondary_archive/all_candidates.csv")])
    candidates = candidates.set_index(CANDIDATE_KEYS)
    modes = [("fixed", "subgamma", "pooled_fixed"),
             ("grid", "subgamma", "pooled_grid"),
             ("mixture", "subgamma", "pooled_mixture"),
             ("square_mgf", "subgamma", "pooled_square_subgamma"),
             ("square_mgf", "exact_mgf", "pooled_square_exact_mgf")]
    for item in stats:
        for mode, product, name in modes:
            result = pooled_upper_bound(item["masses"], *item["stats"], item["delta"],
                                        absent_upper=item["corner"], lambda_mode=mode,
                                        product_mode=product)
            key = tuple(item[k] for k in CANDIDATE_KEYS[:-1]) + (name,)
            maximum = max(maximum, compare_json(result["upper"], float(candidates.loc[key, "bias"]),
                                                "archive scalar " + str(key)))
            calls += 1
    return {"simulation_configurations": len(examples), "archive_candidates": len(stats),
            "scalar_bound_calls": calls, "max_absolute_difference": maximum}


def stored_simulation_checks():
    protocol = json.loads((HERE / "protocol.json").read_text())
    rows = read_csv(HERE / "simulation/rows.csv.gz")
    stored_summary = read_csv(HERE / "simulation/summary.csv")
    assert not rows.duplicated(KEYS + ["replicate"]).any()
    expected_settings = (len(protocol["categories"]) * len(protocol["validation_pairs"])
                         * len(protocol["mass_laws"]) * len(protocol["fit_errors"])
                         * len(protocol["deltas"]))
    groups = rows.groupby(KEYS, sort=False)
    assert len(groups) == expected_settings
    summaries = []
    for ident, cell in groups:
        assert np.array_equal(np.sort(cell.replicate), np.arange(protocol["repetitions"]))
        for method in protocol["methods"]:
            slack = cell[method] - cell.true_bias
            summaries.append(dict(zip(KEYS, ident), method=method, repetitions=len(cell),
                                  noncoverage=int((slack < -1e-14).sum()),
                                  median_slack=float(slack.median()), mean_slack=float(slack.mean()),
                                  median_upper=float(cell[method].median())))
    summary_check = compare_frames(pd.DataFrame(summaries), stored_summary, KEYS + ["method"],
                                   "stored simulation summaries")
    # Both error budgets evaluate identical primitive draws, not new replicates.
    draw_keys = KEYS[:-1] + ["replicate"]
    paired = [rows[rows.delta == delta][draw_keys + ["true_bias", "pooled_center"]]
              for delta in protocol["deltas"]]
    paired_check = compare_frames(paired[1], paired[0], draw_keys, "same draws across deltas")
    old = read_csv(PUBLIC / "results/aggregate_bias/simulation/replications.csv.gz")
    actual = rows[rows.delta == .05][draw_keys + ["rectangle", "split_original", "true_bias"]]
    expected = old[draw_keys + ["rectangle_upper", "aggregate_upper", "true_bias"]].rename(
        columns={"rectangle_upper": "rectangle", "aggregate_upper": "split_original"})
    original_check = compare_frames(actual, expected, draw_keys, "original simulation arrays")
    examples = json.loads((HERE / "simulation/example_sufficient_statistics.json").read_text())
    first = rows[rows.replicate == 0].set_index(KEYS)
    example_difference = 0.0
    for item in examples:
        row = first.loc[tuple(item[k] for k in KEYS)]
        for method, value in item["expected"].items():
            example_difference = max(example_difference,
                                     compare_json(float(row[method]), value, "example stored array"))
    assert len(examples) == len(first) == expected_settings
    return {"rows": len(rows), "independent_primitive_draws": len(old),
            "method_setting_cells": len(stored_summary), "summary_reconstruction": summary_check,
            "same_draws_across_deltas": paired_check, "original_array_comparison": original_check,
            "example_rows": len(examples), "max_example_absolute_difference": example_difference}


def run_script(script, *args):
    start = time.monotonic()
    command = [sys.executable, str(HERE / script), *map(str, args)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, cwd=HERE, env=env, text=True, capture_output=True)
    assert result.returncode == 0, (script, result.returncode, result.stdout, result.stderr)
    return {"script": script, "elapsed_seconds": time.monotonic() - start,
            "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}


def archive_checks(temporary):
    primary, secondary = temporary / "archive", temporary / "secondary"
    executions = [run_script("archive_benchmark.py", "--output", primary),
                  run_script("secondary_archive.py", "--primary", primary, "--output", secondary)]
    checks = {}
    for current, stored in [(primary, HERE / "archive"), (secondary, HERE / "secondary_archive")]:
        checks[stored.name] = {
            name: compare_frames(read_csv(current / name), read_csv(stored / name), keys,
                                 stored.name + "/" + name)
            for name, keys in [("all_candidates.csv", CANDIDATE_KEYS),
                               ("summary.csv", ["budget_method"])]
        }
    checks["max_sufficient_statistics_difference"] = compare_json(
        json.loads((primary / "sufficient_statistics.json").read_text()),
        json.loads((HERE / "archive/sufficient_statistics.json").read_text()), "archive statistics")
    checks["executions"] = executions
    return checks


def full_simulation_check(temporary):
    current = temporary / "simulation"
    execution = run_script("benchmark.py", "--output", current)
    checks = {name: compare_frames(read_csv(current / name), read_csv(HERE / "simulation" / name), keys,
                                  "full simulation " + name)
              for name, keys in [("rows.csv.gz", KEYS + ["replicate"]),
                                 ("summary.csv", KEYS + ["method"])]}
    for name in ["example_sufficient_statistics.json", "original_replay.json"]:
        checks[name] = {"max_absolute_difference": compare_json(
            json.loads((current / name).read_text()),
            json.loads((HERE / "simulation" / name).read_text()), name)}
    checks["execution"] = execution
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "PUBLIC_SOURCE_REPLAY.json")
    parser.add_argument("--full", action="store_true", help="Also regenerate the 144,000 primitive synthetic draws.")
    args = parser.parse_args()
    replaceable_receipts = {HERE / "PUBLIC_SOURCE_REPLAY.json", HERE / "PUBLIC_SOURCE_REPLAY_FULL.json"}
    if args.output.suffix.lower() != ".json":
        parser.error("--output must name a JSON replay receipt.")
    if args.output.exists() and args.output.resolve() not in replaceable_receipts:
        parser.error("--output must be new, except for the two standard public replay receipts.")
    started = time.monotonic()
    protected = [p for directory in ("archive", "secondary_archive", "simulation")
                 for p in (HERE / directory).iterdir() if p.is_file()]
    protected += [HERE / "independent_math_check.json", HERE / "verification.json", HERE / "protocol.json"]
    before = {str(p.relative_to(HERE)): sha256(p) for p in protected}
    receipt = {"scope": "Public-source replay of existing mathematical checks and retrospective evidence; no new confirmation, model retraining, or scientific selection.",
               "executed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "full_simulation_regeneration": args.full,
               "numeric_tolerances": {"absolute": ATOL, "relative": RTOL,
                                      "metadata_counts_and_decisions": "exact"},
               "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
               "source_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
               "theory_sha256": sha256(HERE / "THEORY.md")}
    print("Replaying independently authored finite-state and formula checks", flush=True)
    receipt["independent_source_replay"] = verify_independent_sources()
    print("Checking scalar sufficient statistics and stored simulation arrays", flush=True)
    receipt["scalar_checks"] = scalar_checks()
    receipt["stored_simulation_checks"] = stored_simulation_checks()
    with tempfile.TemporaryDirectory(prefix="pooled-bias-replay-") as name:
        temporary = Path(name)
        print("Recomputing primary and secondary forecast archive comparisons", flush=True)
        receipt["archive_replay"] = archive_checks(temporary)
        if args.full:
            print("Regenerating all synthetic draws (optional full replay)", flush=True)
            receipt["full_simulation_replay"] = full_simulation_check(temporary)
    after = {str(p.relative_to(HERE)): sha256(p) for p in protected}
    assert before == after, "Replay altered preserved scientific evidence"
    receipt["preserved_evidence_sha256"] = after
    receipt["all_passed"] = True
    receipt["elapsed_seconds"] = time.monotonic() - started
    # The output is a receipt only; do not permit accidental evidence overwrite.
    assert args.output.resolve() not in {p.resolve() for p in protected}, "Output would replace scientific evidence"
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"all_passed": True, "receipt": str(args.output),
                      "full": args.full, "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
