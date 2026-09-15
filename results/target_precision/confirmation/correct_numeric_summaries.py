#!/usr/bin/env python3
"""Regenerate descriptive summaries without editing or rerunning the frozen study.

Each loss uses the same scalar-vector reduction. Paired gains use the stable
squared-loss identity, with an exact array-equality shortcut and no tolerance.
Original statistical decisions, selected forecasts and primary bounds are inputs,
never recalculated or changed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_tables(out, name, rows):
    write_json(out / (name + ".json"), rows)
    with (out / (name + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def scalar_loss(prediction, target):
    a, y = np.asarray(prediction, float), np.asarray(target, float)
    observed = np.isfinite(y)
    if a.shape != y.shape or a.ndim != 1 or not observed.any() or not np.isfinite(a).all():
        raise ValueError("Expected finite forecast vector and at least one observed target")
    return float(np.mean((a[observed] - y[observed])**2))


def paired_gain(primary, comparator, target):
    """Observed mean loss(comparator)-loss(primary), without subtracting two means."""
    a, b, y = map(lambda z: np.asarray(z, float), (primary, comparator, target))
    observed = np.isfinite(y)
    if a.shape != b.shape or a.shape != y.shape or a.ndim != 1 or not observed.any():
        raise ValueError("Expected aligned vectors and observed targets")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Forecasts must be finite")
    if np.array_equal(a, b):
        return 0.0
    # Algebraically b²-a²-2y(b-a), preserving tiny real differences between
    # nonidentical vectors even when separately rounded mean losses coincide.
    terms = (b[observed] - a[observed]) * (b[observed] + a[observed] - 2*y[observed])
    return math.fsum(float(v) for v in terms) / int(observed.sum())


def hindsight_forecast(baseline, augmentations, failed, target):
    """Fixed-family diagnostic optimum; exact zero-gain ties retain earlier choice."""
    best, index = baseline, -1
    for j in range(augmentations.shape[1]):
        if not failed[j] and paired_gain(augmentations[:, j], best, target) > 0:
            best, index = augmentations[:, j], j
    return best, index


def correct(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out == run or run in out.parents:
        raise ValueError("Corrected summaries must be outside the frozen run")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    before = inventory(run)
    completion = read_json(run / "completion.json")
    if completion.get("status") != "complete" or completion.get("synthetic") is not False:
        raise ValueError("Expected the completed real frozen study")
    freeze = read_json(run / "selector_freeze.json")
    for name, key in [("selectors.json", "selectors_sha256"), ("gate_decisions.json", "gate_decisions_sha256")]:
        if before[name] != freeze[key]:
            raise ValueError("Frozen decisions changed: " + name)
    selection = read_json(run / "selectors.json")
    if before["confirmation_forecasts.npz"] != selection["confirmation_forecasts_sha256"]:
        raise ValueError("Frozen confirmation forecasts changed")
    gates = read_json(run / "gate_decisions.json")
    fit = read_json(run / "fit_receipt.json")
    cfg = read_json(run / "config.json")
    with np.load(run / "confirmation_forecasts.npz", allow_pickle=False) as data:
        ids, baseline, predictions = data["ids"].copy(), data["baseline"].copy(), data["preds"].copy()
    with np.load(run / "sealed/confirmation_labels.npz", allow_pickle=False) as data:
        if not np.array_equal(ids, data["ids"]):
            raise ValueError("Label alignment changed")
        raw = data["raw"].copy()
    y = np.clip(np.log1p(raw) / fit["scale"], 0, 1)
    observed = np.isfinite(y)
    failed = ~np.isfinite(predictions).all(axis=0)
    safe_predictions = np.where(np.isfinite(predictions), predictions, baseline[:, None])
    augmentations = baseline[:, None] + np.asarray(selection["weights"]) * (safe_predictions - baseline[:, None])
    family = selection["family"]
    selected = {}
    for rule, choice in selection["selectors"].items():
        if choice == "persistence_raw":
            selected[rule] = predictions[:, family.index("persistence")]
        elif choice == -1:
            selected[rule] = baseline
        else:
            if failed[int(choice)]:
                raise ValueError("Frozen selector chose a failed candidate")
            selected[rule] = augmentations[:, int(choice)]
    hindsight, hindsight_index = hindsight_forecast(baseline, augmentations, failed, y)
    candidates = []
    for j, name in enumerate(family):
        gain = None if failed[j] else paired_gain(augmentations[:, j], baseline, y)
        candidates.append({"candidate": name, "failed_fit": bool(failed[j]), "weight": selection["weights"][j],
            "raw_candidate_loss": None if failed[j] else scalar_loss(predictions[:, j], y),
            "augmentation_loss": None if failed[j] else scalar_loss(augmentations[:, j], y),
            "gain_vs_baseline": gain,
            "exactly_equal_to_baseline_array": bool(np.array_equal(augmentations[:, j], baseline)),
            "retained_by": [r["method"] for r in gates if r["candidate"] == j and r["retained"]]})
    opportunities = []
    for row in gates:
        j = row["candidate"]
        gain = candidates[j]["gain_vs_baseline"]
        opportunities.append({"method": row["method"], "candidate": family[j], "retained": row["retained"],
            "confirmation_augmentation_gain": gain,
            "exactly_equal_to_baseline_array": candidates[j]["exactly_equal_to_baseline_array"],
            "excluded_useful": bool(gain is not None and not row["retained"] and gain > 0),
            "retained_harmful": bool(gain is not None and row["retained"] and gain < 0)})
    rules = []
    for rule, a in selected.items():
        raw_prediction = np.expm1(a[observed] * fit["scale"])
        raw_log_target = np.log1p(raw[observed])
        rules.append({"rule": rule, "choice": selection["selectors"][rule],
            "bounded_loss": scalar_loss(a, y), "available_labels": int(observed.sum()), "eligible_rows": len(y),
            "gain_vs_baseline_available": paired_gain(a, baseline, y),
            "gain_vs_ungated_available": paired_gain(a, selected["ungated"], y),
            "opportunity_loss_vs_ungated": paired_gain(selected["ungated"], a, y),
            "regret_vs_hindsight_same_family": paired_gain(hindsight, a, y),
            "exactly_equal_to_baseline_array": bool(np.array_equal(a, baseline)),
            "exactly_equal_to_ungated_array": bool(np.array_equal(a, selected["ungated"])),
            "log_mse_unclipped_outcome": float(np.mean((a[observed] * fit["scale"] - raw_log_target)**2)),
            "log_mae_unclipped_outcome": float(np.mean(np.abs(a[observed] * fit["scale"] - raw_log_target))),
            "share_mse_unclipped_outcome": float(np.mean((raw_prediction - raw[observed])**2)),
            "share_mae_unclipped_outcome": float(np.mean(np.abs(raw_prediction - raw[observed])))})
    primary = cfg["confirmation_inference"]["primary"]
    primary_equivalence = {name: bool(np.array_equal(selected[primary], selected[name]))
                          for name in cfg["confirmation_inference"]["primary_comparators"]}
    original_bounds = read_json(run / "confirmation_conditional_bounds.json")
    for name, equal in primary_equivalence.items():
        if equal and (original_bounds[name]["estimate"] != 0 or paired_gain(selected[primary], selected[name], y) != 0):
            raise ValueError("Identical primary forecasts must have zero original and corrected gain")
    out.mkdir(parents=True, exist_ok=True)
    write_tables(out, "confirmation_all_candidates", candidates)
    write_tables(out, "confirmation_opportunity_losses", opportunities)
    write_tables(out, "confirmation_rules", rules)
    after = inventory(run)
    if before != after:
        raise RuntimeError("A frozen input changed during this read-only correction")
    provenance = {"status": "corrected_descriptive_summaries_only", "source_run": run.name,
        "original_files_preserved": True, "changed_gate_decisions": 0, "changed_selectors": 0,
        "changed_primary_forecast_arrays": 0, "changed_primary_bounds": 0,
        "model_refits": 0, "gate_reruns": 0, "additional_independent_confirmations": 0,
        "loss_reduction": "Identical scalar np.mean of the squared error vector for baseline and each candidate/rule",
        "gain_reduction": "Exact array equality returns zero; otherwise math.fsum((b-a)*(b+a-2*y))/observed_count",
        "arbitrary_tolerance": None,
        "hindsight_selection": "Fixed baseline-then-family order, replacing incumbent only for strictly positive stable paired gain",
        "hindsight_candidate_index": hindsight_index, "hindsight_scalar_loss": scalar_loss(hindsight, y),
        "primary_method": primary, "primary_exact_array_equivalence": primary_equivalence,
        "original_run_sha256": before, "script_sha256": digest(__file__),
        "output_sha256": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / "provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "news432_frozen_run")
    parser.add_argument("--output", type=Path, default=HERE / "corrected_summaries")
    args = parser.parse_args()
    result = correct(args.run, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "changed_gate_decisions": 0, "changed_selectors": 0, "changed_primary_bounds": 0}))


if __name__ == "__main__":
    main()
