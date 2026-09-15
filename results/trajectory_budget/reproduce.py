"""Post-exposure, same-draw comparison; allocation uses declared widths only.

This file implements comparison arithmetic independently of reference_kernels/.
No fitting, candidate search, seed search, or evaluation-dependent allocation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALPHA = .05 / 8
DELTA = .05 / 16
WIDTH_S, WIDTH_Q, WIDTH_U = 2., 4., 2. / 3.
STATUS = "Post-exposure diagnostic on the corrected, previously inspected cohorts; not independent confirmation."


def emit(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocation(budget, score_width=WIDTH_S, correction_width=WIDTH_Q,
               alpha=ALPHA, delta=DELTA, evaluation_cost=2):
    """Integer minimizer under evaluation_cost*M+3*n <= budget; width-only.

    For each n use the greatest feasible M. Ties use smaller n. This includes
    allocations leaving one draw unused, rather than assuming exact divisibility.
    """
    if not (0 < delta < alpha < 1) or score_width <= 0 or correction_width <= 0:
        raise ValueError("positive widths and 0 < delta < alpha < 1 required")
    if budget < evaluation_cost + 3:
        raise ValueError("budget must support one evaluation and one validation block")
    a = score_width * math.sqrt(-math.log(alpha - delta) / 2)
    b = correction_width * math.sqrt(-math.log(delta) / 2)
    candidates = []
    for n in range(1, (budget - evaluation_cost) // 3 + 1):
        M = (budget - 3 * n) // evaluation_cost
        candidates.append((a / math.sqrt(M) + b / math.sqrt(n), n, M))
    radius, n, M = min(candidates)
    return {"budget": budget, "M": M, "n": n, "evaluation_cost": evaluation_cost,
            "draws_used": evaluation_cost * M + 3 * n,
            "unused_budget": budget - evaluation_cost * M - 3 * n,
            "range_radius": radius, "alpha": alpha, "delta": delta,
            "score_width": score_width, "correction_width": correction_width,
            "continuous_M_over_n": (3 * a / (evaluation_cost * b)) ** (2 / 3)}


def compare(x, y):
    return .5 * ((x > y).astype(float) - (x < y).astype(float))


def score(f, g, ix):
    """Active one-peer score; a third supplied index is deliberately ignored."""
    p = compare(f[ix[:, 0]], f[ix[:, 1]])
    q = compare(g[ix[:, 0]], g[ix[:, 1]])
    return (p[:, 0] - p[:, 1]) * (q[:, 0] - q[:, 1])


def correction(f, g, ix):
    p = compare(f[ix[:, 0]], f[ix[:, 1]]) - compare(f[ix[:, 0]], f[ix[:, 2]])
    q = compare(g[ix[:, 0]], g[ix[:, 1]]) - compare(g[ix[:, 0]], g[ix[:, 2]])
    return .5 * (p[:, 0] - p[:, 1]) * (q[:, 0] - q[:, 1])


def triples(f, g, ix):
    result = np.zeros(len(ix))
    for focal, first, second in [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                                  (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        p = compare(f[ix[:, focal]], f[ix[:, first]])
        q = compare(g[ix[:, focal]], g[ix[:, second]])
        result += (p[:, 0] - p[:, 1]) * (q[:, 0] - q[:, 1]) / 6
    return result


def population(f, g):
    p = compare(f[:, None, :], f[None, :, :])
    q = compare(g[:, None, :], g[None, :, :])
    p = p[:, :, 0] - p[:, :, 1]
    q = q[:, :, 0] - q[:, :, 1]
    theta = float(np.mean(p.mean(1) * q.mean(1)))
    gamma = float(np.mean(p * q) - theta)
    return theta, gamma, p, q


def pooled_center(p, q, indices):
    """Independent census-multiplicity formula for the repeated iid draw sample.

    This uses the finite accessible archive to reconstruct the same complete U
    center. Its work/clock time must NOT be attributed to the public rank-sweep
    algorithm, whose independent parity is checked by the tests.
    """
    multiplicity = np.bincount(indices, minlength=len(p))
    N = len(indices)
    numerator = np.dot(multiplicity, (p @ multiplicity) * (q @ multiplicity)
                       - (p * q) @ multiplicity)
    return float(numerator / (N * (N - 1) * (N - 2)))


def mean_radius(values, width, alpha, rule):
    N = len(values)
    variance = float(np.var(values, ddof=1)) if N > 1 else 0.
    h = width * math.sqrt(-math.log(alpha) / (2 * N))
    if rule == "hoeffding":
        return h, variance, h, None
    h = width * math.sqrt((math.log(2) - math.log(alpha)) / (2 * N))
    x = math.log(4) - math.log(alpha)
    eb = math.sqrt(2 * variance * x / N) + 7 * width * x / (3 * (N - 1))
    return min(h, eb), variance, h, eb


def pooled_radius(triple_values, alpha, rule):
    n = len(triple_values)
    if rule == "hoeffding":
        return WIDTH_U * math.sqrt(-math.log(alpha) / (2 * n))
    variance = float(np.var(triple_values, ddof=1))
    x = math.log(2) - math.log(alpha)
    h = WIDTH_U * math.sqrt(x / (2 * n))
    v = math.sqrt(2 * variance * x / n) + (
        2 * WIDTH_U / math.sqrt(n * (n - 1)) + WIDTH_U / (3 * n)) * x
    return min(h, v)


def unique_hours(timestamps, indices):
    timestamps = pd.to_datetime(timestamps[np.unique(indices)])
    return len(set((timestamps - pd.Timedelta(hours=lag)).astype(str).tolist()[i]
                   for lag in [0, 1, 2, 3, 6, 12, 24] for i in range(len(timestamps))))


def counts(name, allocated, used, discarded, stamps, M=0, n=0, q=0):
    allocated, used, discarded = map(lambda a: np.asarray(a).reshape(-1),
                                     [allocated, used, discarded])
    return {"method": name, "charged_trajectory_draws": len(allocated),
            "used_trajectory_draws": len(used), "discarded_trajectory_draws": len(discarded),
            "unique_archive_records_accessed": len(np.unique(allocated)),
            "unique_archive_records_used": len(np.unique(used)),
            "unique_records_only_discarded": len(np.setdiff1d(discarded, used)),
            "distinct_original_hour_rows_used": unique_hours(stamps, used),
            "new_labels_acquired": 0, "evaluation_blocks": M,
            "validation_triples": n, "independent_triples": q,
            "cost_scope": "Trajectory draw roles; repeated archive access. Stored maps already fitted."}


def bootstrap_indices(nweeks, replications=2000, block=4, seed=2026091455):
    rng = np.random.default_rng(seed)
    starts = rng.integers(nweeks - block + 1,
                          size=(replications, int(np.ceil(nweeks / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(replications, -1)[:, :nweeks]


def future_evaluation(confirmation, names, choices, output):
    z = np.load(confirmation)
    losses = (z["predictions"] - z["target"][:, None]) ** 2
    weeks = pd.to_datetime(z["timestamps"]).to_period("W-SUN").astype(str)
    frame = pd.DataFrame(losses, columns=names).assign(week=weeks)
    totals = frame.groupby("week", sort=True).sum()
    means = frame.groupby("week", sort=True).mean()
    sizes = frame.groupby("week", sort=True).size().to_numpy()
    indices = bootstrap_indices(len(sizes))
    np.savez_compressed(output / "bootstrap_indices.npz", indices=indices, week_sizes=sizes)
    rows = []
    all_choices = dict(choices)
    all_choices.update({"fixed_" + name: k for k, name in enumerate(names)})
    for rule, k in all_choices.items():
        for comparator, j in [("persistence", 0), ("direct_loss_selection", choices["direct_loss_selection"])]:
            differences = totals.iloc[:, j].to_numpy() - totals.iloc[:, k].to_numpy()
            equal_differences = means.iloc[:, j].to_numpy() - means.iloc[:, k].to_numpy()
            boot = differences[indices].sum(1) / sizes[indices].sum(1)
            boot_relative = differences[indices].sum(1) / totals.iloc[:, j].to_numpy()[indices].sum(1)
            boot_equal = equal_differences[indices].mean(1)
            ci = np.quantile(boot, [.025, .975])
            relative_ci = np.quantile(boot_relative, [.025, .975])
            equal_ci = np.quantile(boot_equal, [.025, .975])
            rows.append({"rule": rule, "candidate": names[k], "comparator": comparator,
                         "comparator_candidate": names[j], "confirmation_records": len(losses),
                         "confirmation_weeks": len(sizes), "mse": float(losses[:, k].mean()),
                         "comparator_mse": float(losses[:, j].mean()),
                         "target_weighted_gain": float((losses[:, j] - losses[:, k]).mean()),
                         "gain_ci_low": ci[0], "gain_ci_high": ci[1],
                         "relative_gain": float(1 - losses[:, k].mean() / losses[:, j].mean()),
                         "relative_ci_low": relative_ci[0], "relative_ci_high": relative_ci[1],
                         "equal_week_gain": float(equal_differences.mean()),
                         "equal_week_ci_low": equal_ci[0], "equal_week_ci_high": equal_ci[1],
                         "identical_predictions": bool(np.array_equal(z["predictions"][:, k], z["predictions"][:, j])),
                         "analysis_status": STATUS})
    pd.DataFrame(rows).to_csv(output / "future_rule_comparisons.csv", index=False)
    candidates = pd.DataFrame({"candidate": names, "mse": losses.mean(0),
                               "mae": np.abs(z["predictions"] - z["target"][:, None]).mean(0)})
    candidates.to_csv(output / "all_eight_future_losses.csv", index=False)
    means.to_csv(output / "future_weekly_mean_losses.csv")
    emit(output / "BOOTSTRAP_PROTOCOL.json", {
        "analysis_status": STATUS, "replications": 2000, "seed": 2026091455,
        "block_length_weeks": 4, "week_definition": "Monday-Sunday pandas W-SUN periods",
        "boundary_scheme": "Noncircular moving blocks. Starts uniform on 0,...,W-4; concatenate ceil(W/4) blocks and truncate last block to W weeks. No wrapping or padding.",
        "pairing": "Same block-index array for every candidate and comparator; paired losses.",
        "target_weighted_statistic": "Sum resampled weekly loss totals / sum resampled weekly record counts.",
        "equal_week_statistic": "Arithmetic mean resampled weekly mean loss differences.",
        "interval": "2.5th and 97.5th percentiles using numpy default linear quantile.",
        "scope": "Descriptive dependent-time-series intervals; neither finite coverage nor independent confirmation is asserted. Pointwise, not family-adjusted.",
        "week_sizes": dict(zip(totals.index, sizes.tolist()))})


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    input_dir = HERE / "inputs"
    # Width-only choice is recorded before loading any forecast/target values.
    plan = allocation(73728)
    emit(output / "ALLOCATION_RULE.json", {
        "analysis_status": STATUS, "allocation": plan,
        "criterion": "Minimize W_S sqrt(log(1/(alpha-delta))/(2M)) + W_Q sqrt(log(1/delta)/(2n)) over integers M,n>=1, 2M+3n<=73728.",
        "inputs_to_allocation": "Budget 73728; declared widths 2 and 4; fixed alpha=.05/8 and delta=.05/16. No score, variance, candidate loss, or evaluation outcome.",
        "draw_repartition": "Flatten recorded evaluation then validation indices in original row-major order. First 2M positions form pairs, remaining 3n positions form validation triples. No permutation, index-value search, or seed search.",
        "independence": "Under the original conditional archive iid index experiment, disjoint draw positions remain independent. Repeated archive IDs are allowed. The new diagnostic and old diagnostic are coupled and are not independent studies.",
        "conditioning": "Same fixed 342-record 2013 archive and same eight models fitted on 2010-2012; no model refit or changed target.",
        "coefficients_original": [[1, 0], [-1, 0]], "coefficients_active": [[1], [-1]],
        "target_preservation": "Both zero second columns are deleted: row sums (1,-1) and Omega=[[1,-1],[-1,1]] stay identical.",
        "original_allocation_unchanged": {"M": 16384, "n": 8192, "J": 2},
        "selection_rule": "Smallest 2013 raw MSE among positive candidate lower bounds; persistence fallback. No evaluation endpoint is used for rule selection."})
    a = np.load(input_dir / "selection_forecasts.npz")
    ix = np.load(input_dir / "selection_indices.npz")
    ev, va = ix["evaluation"], ix["validation"]
    stream = np.concatenate([ev.reshape(-1), va.reshape(-1)])
    M, n = plan["M"], plan["n"]
    new_ev = stream[:2 * M].reshape(M, 2)
    new_va = stream[2 * M:2 * M + 3 * n].reshape(n, 3)
    np.savez_compressed(output / "active_allocation_indices.npz", evaluation=new_ev, validation=new_va)
    assert len(a["target"]) == 342 and len(stream) == 73728
    names = [str(x) for x in a["names"]]
    mse = np.mean((a["predictions"] - a["target"][:, None]) ** 2, axis=0)
    g = np.column_stack([a["target"], a["baseline"]])
    methods = ["shared_original", "shared_pruned_same_counts", "shared_pruned_allocated",
               "direct_triple_u", "complete_pooled_u", "census"]
    cost_rows = [
        counts(methods[0], stream, np.r_[ev[:, :2].reshape(-1), va.reshape(-1)], ev[:, 2], a["timestamps"], len(ev), len(va)),
        counts(methods[1], np.r_[ev[:, :2].reshape(-1), va.reshape(-1)], np.r_[ev[:, :2].reshape(-1), va.reshape(-1)], [], a["timestamps"], len(ev), len(va)),
        counts(methods[2], stream, np.r_[new_ev.reshape(-1), new_va.reshape(-1)], stream[plan["draws_used"]:], a["timestamps"], M, n),
        counts(methods[3], stream, stream, [], a["timestamps"], q=len(stream) // 3),
        counts(methods[4], stream, stream, [], a["timestamps"], q=len(stream) // 3),
        counts(methods[5], np.arange(342), np.arange(342), [], a["timestamps"])]
    cost_rows[-1]["cost_scope"] = "342 accessible archive records read once; exact conditional population target, no Monte Carlo inference."
    pd.DataFrame(cost_rows).to_csv(output / "cost_accounting.csv", index=False)
    rows = []
    for k, name in enumerate(names):
        f = np.column_stack([a["predictions"][:, k], a["baseline"]])
        theta, gamma, p, q = population(f, g)
        uv = triples(f, g, stream.reshape(-1, 3))
        pooled = pooled_center(p, q, stream)
        for method in methods:
            variants = ["exact"] if method == "census" else ["hybrid", "hoeffding"]
            for variant in variants:
                sv_variance, rv_variance = 0., 0.
                if method.startswith("shared"):
                    ei, vi = (new_ev, new_va) if method == "shared_pruned_allocated" else (ev, va)
                    sv, rv = score(f, g, ei), correction(f, g, vi)
                    r_s, sv_variance, _, _ = mean_radius(sv, WIDTH_S, ALPHA - DELTA, variant)
                    r_q, rv_variance, _, _ = mean_radius(rv, WIDTH_Q, DELTA, variant)
                    raw, interaction = float(sv.mean()), float(rv.mean())
                    mean = raw - interaction
                else:
                    interaction, r_q = 0., 0.
                    if method == "direct_triple_u":
                        mean = raw = float(uv.mean())
                        r_s, sv_variance, _, _ = mean_radius(uv, WIDTH_U, ALPHA, variant)
                    elif method == "complete_pooled_u":
                        mean = raw = pooled
                        r_s = pooled_radius(uv, ALPHA, variant)
                        sv_variance = float(np.var(uv, ddof=1))
                    else:
                        mean = raw = theta
                        r_s = 0.
                lower = mean - r_s - r_q
                rows.append({"candidate": name, "method": method,
                             "bound": "variance" if method == "complete_pooled_u" and variant == "hybrid" else variant,
                             "selection_mse": float(mse[k]), "exact_target": theta,
                             "true_shared_interaction": gamma, "raw_center": raw,
                             "interaction_estimate": interaction, "corrected_center": mean,
                             "evaluation_radius": r_s, "validation_radius": r_q,
                             "total_radius": r_s + r_q, "lower": lower,
                             "passes": bool(lower > (1e-14 if method == "census" else 0)),
                             "score_variance": sv_variance, "correction_variance": rv_variance,
                             "alpha": ALPHA, "delta": DELTA if method.startswith("shared") else 0.,
                             "analysis_status": STATUS})
    result = pd.DataFrame(rows)
    result.to_csv(output / "certificate_all_eight.csv", index=False)
    choices = {"persistence": 0, "direct_loss_selection": int(np.argmin(mse))}
    selection_rows = []
    for (method, bound), group in result.groupby(["method", "bound"], sort=False):
        allowed = group[group.passes]
        name = str(allowed.loc[allowed.selection_mse.idxmin(), "candidate"]) if len(allowed) else names[0]
        key = method + "_" + bound
        choices[key] = names.index(name)
        selection_rows.append({"rule": key, "candidate": name, "number_passing": len(allowed),
                               "passing_candidates": ";".join(allowed.candidate),
                               "selection_mse": float(mse[names.index(name)])})
    pd.DataFrame(selection_rows).to_csv(output / "gate_decisions.csv", index=False)
    emit(output / "RULE_CHOICES_BEFORE_REPLAYING_FUTURE.json", {
        "analysis_status": STATUS, "choices": {rule: names[k] for rule, k in choices.items()},
        "mechanical_order": "Choices written before this script opens stored 2014 arrays. This does not undo prior exposure.",
        "selection_information": "Only 2013 scores/losses; exact same fixed models, targets, and candidate family."})
    future_evaluation(input_dir / "confirmation_forecasts.npz", names, choices, output)
    # Figure-ready coordinates preserve full numerical precision in source tables.
    focus = result[result.candidate == "boosting_31"].copy()
    for column in ["exact_target", "raw_center", "interaction_estimate", "corrected_center",
                   "evaluation_radius", "validation_radius", "total_radius", "lower"]:
        focus[column + "_times_1000"] = focus[column] * 1000
    focus.to_csv(output / "figure_boosting_certificate_coordinates.csv", index=False)
    grid = []
    for budget in [4608, 9216, 18432, 36864, 73728, 147456, 294912, 589824]:
        allocated = allocation(budget)
        allocated["pooled_hoeffding_radius"] = WIDTH_U * math.sqrt(-math.log(ALPHA) / (2 * (budget // 3)))
        grid.append(allocated)
    pd.DataFrame(grid).to_csv(output / "declared_width_budget_curve.csv", index=False)
    emit(output / "REPRODUCTION_MANIFEST.json", {
        "analysis_status": STATUS, "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "inputs": {p.name: sha(p) for p in sorted(input_dir.iterdir()) if p.is_file()},
        "code": {str(p.relative_to(HERE)): sha(p) for p in [Path(__file__), HERE / "test_reproduce.py", *sorted((HERE / "reference_kernels").glob("*.py"))]},
        "scientific_outputs": "All CSV and JSON numeric results deterministic; NPZ archive byte timestamps are not scientific data.",
        "no_new_labels": True, "number_candidates": 8, "family_alpha": .05,
        "preserved_width_scope": "Generic declared widths 2,4,2/3 are held fixed across candidates including persistence. No empirical-range substitution or post-outcome map-specific tightening."})
    print(json.dumps({"allocation": plan, "decisions": selection_rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
