"""Replay sharp structural widths on the existing fixed trajectory draws.

The baseline archive, learned maps, evaluation/validation allocation, centers,
variance estimates, candidate family and error allocations are unchanged.
Only known support widths change. This is a post-exposure diagnostic, with no
independent confirmation, refitting, new draws or future-loss access.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALPHA, DELTA = .05 / 8, .05 / 16
STATUS = "Post-exposure fixed-draw diagnostic; no independent confirmation."
METHODS = ("shared_original", "shared_pruned_same_counts", "shared_pruned_allocated",
           "direct_triple_u", "complete_pooled_u", "census")


def emit(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_source(package_root=None):
    """Find the sibling public study or an explicitly supplied package root."""
    root = package_root or os.environ.get("TRAJECTORY_PACKAGE_ROOT")
    candidate = Path(root) / "results/trajectory_budget" if root else HERE.parent / "trajectory_budget"
    if (candidate / "inputs/selection_forecasts.npz").is_file():
        return candidate
    raise FileNotFoundError("Supply --source with the released trajectory_budget directory")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_contract(path=HERE / "MAP_CONTRACT.json"):
    contract = json.loads(path.read_text())
    if contract["C"] != [[1, 0], [-1, 0]] or contract["D"] != contract["C"]:
        raise ValueError("This proof supports only the declared one-active-peer contrast")
    if contract["common_baseline_is_same_map"] is not True:
        raise ValueError("The two contrasts must use the same baseline map")
    if contract["maps_left"][1] != contract["maps_right"][1]:
        raise ValueError("The baseline map identifiers differ")
    return contract


def structural_widths(name, contract):
    # No scores, sample ranges or numerical array equalities enter this choice.
    return (0., 0., 0.) if name in contract["forecast_equals_baseline"] else (1.25, 2.5, .5)


def compare(x, y):
    return .5 * (np.greater(x, y).astype(float) - np.less(x, y).astype(float))


def contrasts(h, y, b):
    baseline = compare(b[:, None], b[None, :])
    return (compare(h[:, None], h[None, :]) - baseline,
            compare(y[:, None], y[None, :]) - baseline)


def score_values(p, q, ix):
    return p[ix[:, 0], ix[:, 1]] * q[ix[:, 0], ix[:, 1]]


def correction_values(p, q, ix):
    return .5 * (p[ix[:, 0], ix[:, 1]] - p[ix[:, 0], ix[:, 2]]) * (
        q[ix[:, 0], ix[:, 1]] - q[ix[:, 0], ix[:, 2]])


def triple_values(p, q, ix):
    result = np.zeros(len(ix))
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        result += p[ix[:, i], ix[:, j]] * q[ix[:, i], ix[:, k]]
        result += p[ix[:, i], ix[:, k]] * q[ix[:, i], ix[:, j]]
    return result / 6


def pooled_center(p, q, stream):
    multiplicity = np.bincount(stream, minlength=len(p))
    n = len(stream)
    numerator = multiplicity @ ((p @ multiplicity) * (q @ multiplicity)
                                - (p * q) @ multiplicity)
    return float(numerator / (n * (n - 1) * (n - 2)))


def mean_radius(n, variance, width, alpha, rule):
    if width == 0:
        return 0.
    if rule == "hoeffding" or n == 1:
        return width * math.sqrt(-math.log(alpha) / (2 * n))
    x = math.log(4) - math.log(alpha)
    h = width * math.sqrt((math.log(2) - math.log(alpha)) / (2 * n))
    e = math.sqrt(2 * variance * x / n) + 7 * width * x / (3 * (n - 1))
    return min(h, e)


def complete_radius(n, variance, width, alpha, rule):
    if width == 0:
        return 0.
    if rule == "hoeffding" or n == 1:
        return width * math.sqrt(-math.log(alpha) / (2 * n))
    x = math.log(2) - math.log(alpha)
    h = width * math.sqrt(x / (2 * n))
    e = math.sqrt(2 * variance * x / n) + (
        2 * width / math.sqrt(n * (n - 1)) + width / (3 * n)) * x
    return min(h, e)


def optimal_allocation(budget, ws, wq):
    a, b = ws * math.sqrt(-math.log(ALPHA - DELTA) / 2), wq * math.sqrt(-math.log(DELTA) / 2)
    radius, n, m = min((a / math.sqrt((budget - 3 * n) // 2) + b / math.sqrt(n),
                        n, (budget - 3 * n) // 2)
                       for n in range(1, (budget - 2) // 3 + 1))
    return {"M": m, "n": n, "draws_used": 2 * m + 3 * n, "range_radius": radius}


def gate_rows(rows):
    result = []
    for method in METHODS:
        bounds = ["exact"] if method == "census" else ["variance" if method == "complete_pooled_u" else "hybrid", "hoeffding"]
        for bound in bounds:
            group = [r for r in rows if r["method"] == method and r["bound"] == bound]
            allowed = [r for r in group if r["passes"]]
            chosen = min(allowed, key=lambda r: r["selection_mse"])["candidate"] if allowed else "persistence"
            result.append({"method": method, "bound": bound, "candidate": chosen,
                           "number_passing": len(allowed),
                           "passing_candidates": ";".join(r["candidate"] for r in allowed)})
    return result


def replay(source, contract):
    with np.load(source / "inputs/selection_forecasts.npz") as z:
        names = [str(x) for x in z["names"]]
        predictions, target, baseline = z["predictions"], z["target"], z["baseline"]
    with np.load(source / "inputs/selection_indices.npz") as z:
        evaluation, validation = z["evaluation"], z["validation"]
    with np.load(source / "results/active_allocation_indices.npz") as z:
        active_evaluation, active_validation = z["evaluation"], z["validation"]
    stream = np.concatenate([evaluation.reshape(-1), validation.reshape(-1)])
    assert len(stream) == 73728 and len(target) == 342 and len(names) == 8
    assert (len(active_evaluation), len(active_validation)) == (13086, 15852)
    assert np.array_equal(active_evaluation, stream[:26172].reshape(-1, 2))
    assert np.array_equal(active_validation, stream[26172:].reshape(-1, 3))
    for name in contract["forecast_equals_baseline"]:
        # Check consistency of externally declared identity; never infer it here.
        if name not in names or not np.array_equal(predictions[:, names.index(name)], baseline):
            raise ValueError("The stored arrays contradict the declared map identity")
    rows = {"generic": [], "structural": []}
    all_stats = []
    for column, name in enumerate(names):
        h = predictions[:, column]
        p, q = contrasts(h, target, baseline)
        exact_raw = float(np.mean(p * q))
        theta = float(np.mean(p.mean(1) * q.mean(1)))
        gamma = exact_raw - theta
        uv = triple_values(p, q, stream.reshape(-1, 3))
        pooled = pooled_center(p, q, stream)
        mse = float(np.mean((h - target) ** 2))
        all_stats.append({"candidate": name, "exact_population_shared_score": exact_raw,
                          "exact_target": theta, "true_shared_interaction": gamma,
                          "interaction_over_population_shared_score": gamma / exact_raw if exact_raw else None})
        for method in METHODS:
            if method.startswith("shared"):
                ei, vi = (active_evaluation, active_validation) if method == "shared_pruned_allocated" else (evaluation, validation)
                sv, qv = score_values(p, q, ei), correction_values(p, q, vi)
                raw, correction = float(sv.mean()), float(qv.mean())
                center = raw - correction
                vs, vq = float(np.var(sv, ddof=1)), float(np.var(qv, ddof=1))
                m, n = len(sv), len(qv)
            else:
                raw = center = theta if method == "census" else pooled if method == "complete_pooled_u" else float(uv.mean())
                correction, vq, n = 0., 0., 0
                vs, m = (0., 0) if method == "census" else (float(np.var(uv, ddof=1)), len(uv))
            for policy in rows:
                ws, wq, wu = (2., 4., 2. / 3.) if policy == "generic" else structural_widths(name, contract)
                for rule in (["exact"] if method == "census" else ["hybrid", "hoeffding"]):
                    if method.startswith("shared"):
                        rs = mean_radius(m, vs, ws, ALPHA - DELTA, rule)
                        rq = mean_radius(n, vq, wq, DELTA, rule)
                    elif method == "direct_triple_u":
                        rs, rq = mean_radius(m, vs, wu, ALPHA, rule), 0.
                    elif method == "complete_pooled_u":
                        rs, rq = complete_radius(m, vs, wu, ALPHA, rule), 0.
                    else:
                        rs = rq = 0.
                    lower = center - rs - rq
                    rows[policy].append({"candidate": name, "method": method,
                        "bound": "variance" if method == "complete_pooled_u" and rule == "hybrid" else rule,
                        "selection_mse": mse, "exact_target": theta, "true_shared_interaction": gamma,
                        "raw_center": raw, "interaction_estimate": correction, "corrected_center": center,
                        "evaluation_radius": rs, "validation_radius": rq, "total_radius": rs + rq,
                        "lower": lower, "passes": bool(lower > (1e-14 if method == "census" else 0)),
                        "score_variance": vs, "correction_variance": vq, "alpha": ALPHA,
                        "delta": DELTA if method.startswith("shared") else 0.,
                        "score_width": ws if method.startswith("shared") else wu,
                        "correction_width": wq if method.startswith("shared") else 0.,
                        "width_basis": "generic" if policy == "generic" else "declared_map_identity" if ws == 0 else "shared_baseline_theorem",
                        "analysis_status": STATUS})
    return rows, all_stats


def verify_baseline(rows, source):
    with (source / "results/certificate_all_eight.csv").open() as f:
        old = {(r["candidate"], r["method"], r["bound"]): r for r in csv.DictReader(f)}
    assert len(rows) == len(old) == 88
    max_error = 0.
    numeric = ("selection_mse", "exact_target", "true_shared_interaction", "raw_center",
               "interaction_estimate", "corrected_center", "evaluation_radius", "validation_radius",
               "total_radius", "lower", "score_variance", "correction_variance", "alpha", "delta")
    for row in rows:
        baseline = old[(row["candidate"], row["method"], row["bound"])]
        for key in numeric:
            error = abs(row[key] - float(baseline[key]))
            max_error = max(max_error, error)
            assert math.isclose(row[key], float(baseline[key]), rel_tol=1e-12, abs_tol=1e-14), (row["candidate"], row["method"], key, error)
        assert row["passes"] == (baseline["passes"] == "True")
    return {"rows_checked": len(rows), "numeric_columns_checked": len(numeric), "maximum_absolute_error": max_error}


def main(source, output, contract_path=HERE / "MAP_CONTRACT.json"):
    contract = load_contract(contract_path)
    # The unchanged width-optimal allocation is established before loading scores.
    allocations = {"generic": optimal_allocation(73728, 2., 4.),
                   "shared_baseline": optimal_allocation(73728, 1.25, 2.5)}
    assert [(v["M"], v["n"]) for v in allocations.values()] == [(13086, 15852)] * 2
    rows, population = replay(source, contract)
    baseline_check = verify_baseline(rows["generic"], source)
    differences = []
    for old, new in zip(rows["generic"], rows["structural"]):
        for key in ("candidate", "method", "bound", "raw_center", "interaction_estimate", "corrected_center", "score_variance", "correction_variance"):
            assert old[key] == new[key]
        assert new["total_radius"] <= old["total_radius"] + 1e-15
        differences.append({"candidate": old["candidate"], "method": old["method"], "bound": old["bound"],
            "old_radius": old["total_radius"], "new_radius": new["total_radius"],
            "relative_radius_reduction": 1 - new["total_radius"] / old["total_radius"] if old["total_radius"] else 0.,
            "old_lower": old["lower"], "new_lower": new["lower"],
            "old_passes": old["passes"], "new_passes": new["passes"], "width_basis": new["width_basis"]})
    old_gates, new_gates = gate_rows(rows["generic"]), gate_rows(rows["structural"])
    changed_selections = [dict(old=o, new=n) for o, n in zip(old_gates, new_gates) if o["candidate"] != n["candidate"]]
    changed_certifications = [r for r in differences if r["old_passes"] != r["new_passes"]]
    summary = {"analysis_status": STATUS, "baseline_parity": baseline_check,
        "fixed_allocation": allocations, "new_draws": 0, "new_labels": 0, "new_model_fits": 0,
        "future_forecasts_opened": False, "all_centers_and_variances_identical": True,
        "candidate_family_size": 8, "family_alpha": .05,
        "certification_changes": changed_certifications, "selection_changes": changed_selections,
        "shared_rules_positive_lower_bounds": sum(r["passes"] for r in rows["structural"] if r["method"].startswith("shared")),
        "boosting_31_population": next(r for r in population if r["candidate"] == "boosting_31"),
        "scope": "Tighter valid support bounds and fixed-draw diagnostic only. One stronger-comparator certification can change without changing the selected predictor. No new predictive-utility evidence."}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "generic_replay.csv", rows["generic"])
    write_csv(output / "certificate_all_eight.csv", rows["structural"])
    write_csv(output / "width_gain_comparison.csv", differences)
    write_csv(output / "gate_decisions.csv", new_gates)
    write_csv(output / "population_quantities.csv", population)
    emit(output / "SUMMARY.json", summary)
    emit(output / "ALLOCATION_RULE.json", allocations)
    required = [source / "inputs/selection_forecasts.npz", source / "inputs/selection_indices.npz",
                source / "results/active_allocation_indices.npz", source / "results/certificate_all_eight.csv"]
    preparation = source.parent / "trajectory_forecast/prepare.py"
    emit(output / "REPRODUCTION_MANIFEST.json", {
        "analysis_status": STATUS, "input_sha256": {str(p.relative_to(source)): sha(p) for p in required},
        "code_sha256": {p.name: sha(p) for p in [Path(__file__), HERE / "exact_checks.py", HERE / "test_study.py", contract_path]},
        "map_identity_source": {"path": "results/trajectory_forecast/prepare.py", "sha256": sha(preparation) if preparation.is_file() else None,
            "definition": "NAMES[0]='persistence'; baseline=raw[k][:,0]; prediction column 0 and baseline are the same map by construction."},
        "numpy_version": np.__version__, "map_contract": contract,
        "input_source": "Existing public results/trajectory_budget; input files are read only.",
        "complete_center_algorithm": "Finite-archive multiplicity identity, independently reimplemented; no rank-sweep runtime claim.",
        "future_data": "No confirmation forecasts, future losses, or future bootstrap indices are read."})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--package-root", type=Path,
                        help="Package containing results/trajectory_budget; optional alternative to --source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=HERE / "MAP_CONTRACT.json")
    args = parser.parse_args()
    if args.source and args.package_root:
        parser.error("use either --source or --package-root")
    main(args.source or default_source(args.package_root), args.output, args.contract)
