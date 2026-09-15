#!/usr/bin/env python3
"""Reproduce the fixed UCI432 study from the licensed source ZIP.

Every stage retrains from source. This is a reproduction of an already completed
task, not another independent confirmation. All labels are machine-ingested
before separate stage files are used. No network or approval receipt is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
STAGES = ("training", "calibration", "selection", "confirmation")
FAMILY = ("rich_ridge", "rich_hgb", "rich_extra_trees", "persistence", "category_copy", "hash_noise")
METHODS = ("full_u_joint", "stratified_hybrid", "stratified_hoeffding", "stratified_profiled_betting",
           "aggregate_reference_u", "aggregate_reference_betting", "pooled_reference_u",
           "pooled_reference_betting", "direct_loss")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def receipt(out, event, **fields):
    with (Path(out) / "access_receipts.jsonl").open("a") as f:
        f.write(json.dumps(jsonable({"utc": utc(), "event": event, **fields}), allow_nan=False) + "\n")


def read_config(path=HERE / "config.json"):
    cfg = load(path)
    if tuple(cfg["family"]) != FAMILY or tuple(cfg["methods"]) != METHODS:
        raise ValueError("Executable family/method order differs from config")
    if cfg["sampling"]["raw_draw_budget"] != 32768:
        raise ValueError("Only the predeclared single 32768-draw budget is supported")
    for left, right in zip(STAGES[:-1], STAGES[1:]):
        end = pd.Timestamp(cfg["chronology"][left][1])
        start = pd.Timestamp(cfg["chronology"][right][0])
        if start - end != pd.Timedelta(hours=48):
            raise ValueError("Stages must have the exact frozen 48-hour embargo")
    a, b = map(pd.Timestamp, cfg["chronology"]["confirmation"])
    if b - a != pd.Timedelta(hours=1416):
        raise ValueError("Expected the fixed 1416-hour confirmation grid")
    return cfg


def start_run(out, cfg_path, synthetic=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output must be new and empty; preserve earlier evidence")
    out.mkdir(parents=True, exist_ok=True)
    cfg = read_config(cfg_path)
    dump(out / "config.json", cfg)
    dump(out / "run_identity.json", {"created_utc": utc(), "synthetic": synthetic,
         "config_sha256": sha(cfg_path), "experiment_sha256": sha(__file__),
         "python": sys.version, "versions": {m: importlib.import_module(m).__version__
         for m in ("numpy", "pandas", "scipy", "sklearn")}, "reproduction_of_completed_task": True})
    return cfg


def copy_source(source, out, cfg):
    """Validate unchanged source bytes before ingesting any source records."""
    source=Path(source)
    if sha(source)!=cfg["source"]["archive_sha256"]:
        raise ValueError("Source ZIP SHA-256 does not match the documented dataset")
    dest=Path(out)/"source.zip"
    import shutil
    shutil.copyfile(source,dest)
    receipt(out,"source_copied",sha256=sha(dest),bytes=dest.stat().st_size,
            local_raw_archive_materializes_confirmation_labels=True)
    return dest


def deduplicate(frame, key):
    if frame[key].isna().any():
        frame = frame.loc[frame[key].notna()].copy()
    before = len(frame)
    frame = frame.drop_duplicates().copy()
    conflict = frame[key].duplicated(keep=False)
    info = {"input_rows_after_nonnull_id": before, "identical_duplicates_removed": before - len(frame),
            "conflicting_ids": int(frame.loc[conflict, key].nunique()),
            "conflicting_rows_excluded": int(conflict.sum())}
    return frame.loc[~conflict].copy(), info


def social_early_and_labels(social):
    """Future values can affect label availability but never issuance eligibility."""
    social = social.loc[social.IDLink.notna(), ["IDLink", "TS1", "TS2", "TS3", "TS144"]].copy()
    for c in ("TS1", "TS2", "TS3", "TS144"):
        social[c] = pd.to_numeric(social[c], errors="coerce")
        social.loc[~np.isfinite(social[c]) | (social[c] < 0), c] = np.nan
    early, info = deduplicate(social[["IDLink", "TS1", "TS2", "TS3"]], "IDLink")
    label_group = social.groupby("IDLink", sort=False)["TS144"]
    conflicting_labels = label_group.nunique(dropna=False) > 1
    # nth(0) preserves a genuinely missing first value; first() would skip it.
    labels = social.drop_duplicates("IDLink", keep="first").set_index("IDLink")["TS144"]
    labels.loc[conflicting_labels[conflicting_labels].index] = np.nan
    early["TS144"] = early.IDLink.map(labels)
    early_ids = set(early.IDLink)
    info["target_conflicting_ids_retained_eligible_if_TS3_available"] = int(
        sum(i in early_ids for i in conflicting_labels[conflicting_labels].index))
    info["conflict_basis"] = "IDLink and normalized TS1/TS2/TS3 only; TS144 conflicts make label unavailable"
    return early, info


def ingest_frames(news, social, out, cfg):
    """Outcome-aware sealing process; no outcome summaries leave this function."""
    news = news.copy()
    social = social.copy()
    news.columns = news.columns.str.strip()
    social.columns = social.columns.str.strip()
    required_news = {"IDLink", "Title", "Headline", "Source", "Topic", "PublishDate"}
    required_social = {"IDLink", "TS1", "TS2", "TS3", "TS144"}
    if not required_news <= set(news.columns) or not required_social <= set(social.columns):
        raise ValueError("Unexpected provider schema")
    nnews, nsocial = len(news), len(social)
    news["IDLink"] = news["IDLink"].astype("string").str.strip()
    social["IDLink"] = social["IDLink"].astype("string").str.strip()
    news.loc[news.IDLink == "", "IDLink"] = pd.NA
    social.loc[social.IDLink == "", "IDLink"] = pd.NA
    # Never let TS144, intermediate future slices, or final metadata popularity
    # change issuance eligibility or the denominator of a predictable range.
    news, nd = deduplicate(news[list(sorted(required_news))], "IDLink")
    social, sd = social_early_and_labels(social)
    news = news.loc[news.Topic.astype("string").str.strip().str.casefold() == cfg["source"]["topic"]].copy()
    frame = news.merge(social, on="IDLink", how="inner", validate="one_to_one")
    dates = pd.to_datetime(frame.PublishDate, errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        raise ValueError("Unexpected timezone-aware source; no speculative conversion permitted")
    frame["PublishDate"] = dates
    valid_date = frame.PublishDate.notna()
    invalid_date = int((~valid_date).sum())
    frame = frame.loc[valid_date].copy()
    for c in ("TS1", "TS2", "TS3", "TS144"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
        frame.loc[~np.isfinite(frame[c]) | (frame[c] < 0), c] = np.nan
    no_ts3 = int(frame.TS3.isna().sum())
    frame = frame.loc[frame.TS3.notna()].sort_values(["PublishDate", "IDLink"], kind="stable").reset_index(drop=True)
    sealed = Path(out) / "sealed"
    sealed.mkdir()
    count = {}
    all_features = []
    for stage in STAGES:
        start, end = map(pd.Timestamp, cfg["chronology"][stage])
        part = frame.loc[(frame.PublishDate >= start) & (frame.PublishDate < end)].copy()
        labels = part.pop("TS144").to_numpy(dtype=float)
        available = np.isfinite(labels)
        ids = part.IDLink.astype(str).to_numpy(dtype=str)
        np.savez_compressed(sealed / (stage + "_labels.npz"), ids=ids, raw=labels)
        os.chmod(sealed / (stage + "_labels.npz"), 0o600)
        part["stage"] = stage
        part["label_available"] = available
        all_features.append(part)
        count[stage] = {"eligible": len(part), "available_labels": int(available.sum()),
                        "missing_TS1": int(part.TS1.isna().sum()), "missing_TS2": int(part.TS2.isna().sum())}
    features = pd.concat(all_features, ignore_index=True)
    features.to_csv(Path(out) / "features.csv", index=False)
    ledger = {"source_rows": {"news": nnews, "social": nsocial}, "duplicates": {"news": nd, "social": sd},
              "invalid_publication_dates": invalid_date, "ineligible_missing_TS3": no_ts3, "stages": count,
              "label_availability_census_reads": nsocial, "full_source_label_ingestion": True,
              "scope": "eligibility and availability, no numerical outcome summaries"}
    dump(Path(out) / "preparation_metadata.json", ledger)
    receipt(out, "raw_source_sealing", source_rows=ledger["source_rows"],
            all_stage_label_values_machine_ingested=True, label_values_exposed_to_model_stages=False,
            label_files={s: sha(sealed / (s + "_labels.npz")) for s in STAGES})
    return features


def ingest_zip(path, out, cfg):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for member in cfg["source"]["members"]:
            if member not in names:
                raise ValueError("Frozen source member absent: " + member)
        with z.open(cfg["source"]["members"][0]) as f:
            news = pd.read_csv(f, dtype={"IDLink": "string"}, low_memory=False)
        with z.open(cfg["source"]["members"][1]) as f:
            social = pd.read_csv(f, dtype={"IDLink": "string"}, low_memory=False)
    return ingest_frames(news, social, out, cfg)


def read_labels(out, stage, ids, cfg):
    path = Path(out) / "sealed" / (stage + "_labels.npz")
    receipt(out, "model_stage_label_access", stage=stage, path=path.name, sha256=sha(path))
    with np.load(path, allow_pickle=False) as a:
        if not np.array_equal(a["ids"], np.asarray(ids, dtype=str)):
            raise ValueError("Label/feature ID alignment changed")
        y = a["raw"].copy()
    minimum = cfg["eligibility"]["minimum_observed_labels"][stage]
    if int(np.isfinite(y).sum()) < minimum:
        raise ValueError("Frozen feasibility threshold failed in " + stage)
    return y


def numeric_features(frame, rich=False):
    a = np.log1p(frame[["TS1", "TS2", "TS3"]].to_numpy(dtype=float))
    dt = pd.DatetimeIndex(frame.PublishDate)
    hour = dt.hour.to_numpy() + dt.minute.to_numpy() / 60 + dt.second.to_numpy() / 3600
    weekday = dt.weekday.to_numpy()
    columns = [a[:, 0], a[:, 1], a[:, 2], a[:, 1] - a[:, 0], a[:, 2] - a[:, 1],
               np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
               np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)]
    if rich:
        title = frame.Title.fillna("").astype(str)
        headline = frame.Headline.fillna("").astype(str)
        columns.extend([title.str.len().to_numpy(), headline.str.len().to_numpy(),
                        title.str.split().str.len().to_numpy(), headline.str.split().str.len().to_numpy()])
    return np.column_stack(columns).astype(float)


def fit_transform_features(frame, training, rich=False):
    raw = numeric_features(frame, rich)
    tr = raw[training]
    medians = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0 for x in tr.T])
    missing = ~np.isfinite(raw)
    imputed = np.where(missing, medians, raw)
    means = imputed[training].mean(axis=0)
    scales = imputed[training].std(axis=0)
    scales[scales == 0] = 1
    matrix = np.column_stack([(imputed - means) / scales, missing.astype(float)])
    meta = {"medians": medians, "means": means, "scales": scales, "all_numeric_missing_indicators": True}
    if rich:
        source = frame.Source.fillna("").astype(str)
        counts = source[training & (source.to_numpy() != "")].value_counts()
        levels = sorted(counts.index, key=lambda x: (-counts[x], x))[:32]
        mapped = np.array([levels.index(x) if x in levels else len(levels) for x in source])
        matrix = np.column_stack([matrix, np.eye(len(levels) + 1)[mapped]])
        meta["source_levels"] = levels
        meta["other_source_column"] = len(levels)
    return matrix, meta


def delivered(raw, scale):
    return np.clip(np.log1p(raw) / scale, 0, 1)


def fit_models(frame, out, cfg):
    start = time.perf_counter()
    train = (frame.stage == "training").to_numpy()
    raw = read_labels(out, "training", frame.loc[train, "IDLink"], cfg)
    valid = np.isfinite(raw)
    scale = max(1.0, float(np.quantile(np.log1p(raw[valid]), 0.99, method="linear")))
    target = delivered(raw[valid], scale)
    xb, baseprep = fit_transform_features(frame, train, False)
    xr, richprep = fit_transform_features(frame, train, True)
    fitted_rows = np.flatnonzero(train)[valid]
    baseline = np.clip(Ridge(alpha=10, solver="svd").fit(xb[fitted_rows], target).predict(xb), 0, 1)
    cuts = np.quantile(baseline[train], [0.25, 0.5, 0.75], method="linear")
    categories = np.searchsorted(cuts, baseline, side="left")
    preds = np.full((len(frame), 6), np.nan)
    models = [Ridge(alpha=10, solver="svd"),
              HistGradientBoostingRegressor(loss="squared_error", max_iter=200, max_leaf_nodes=15,
                  learning_rate=0.1, min_samples_leaf=20, l2_regularization=1,
                  early_stopping=False, random_state=cfg["seed"]),
              ExtraTreesRegressor(n_estimators=128, max_depth=12, min_samples_leaf=5,
                  max_features=1.0, bootstrap=False, n_jobs=1, random_state=cfg["seed"])]
    errors, timing = {}, {}
    for j, model in enumerate(models):
        t = time.perf_counter()
        try:
            model.fit(xr[fitted_rows], target)
            preds[:, j] = np.clip(model.predict(xr), 0, 1)
        except Exception as exc:
            errors[FAMILY[j]] = repr(exc)
        timing[FAMILY[j]] = time.perf_counter() - t
    preds[:, 3] = delivered(frame.TS3.to_numpy(dtype=float), scale)
    preds[:, 4] = categories / 3
    preds[:, 5] = np.array([int.from_bytes(hashlib.sha256((bytes((110,101,119,115,52,51,50,45,118,57,58))+str(i).encode())).digest()[:8], "big") / 2**64
                             for i in frame.IDLink])
    for stage in STAGES:
        keep = (frame.stage == stage).to_numpy()
        np.savez_compressed(Path(out) / (stage + "_forecasts.npz"),
             ids=frame.loc[keep, "IDLink"].astype(str).to_numpy(dtype=str),
             dates=frame.loc[keep, "PublishDate"].astype(str).to_numpy(dtype=str),
             baseline=baseline[keep], preds=preds[keep], categories=categories[keep])
    dump(Path(out) / "fit_receipt.json", {"scale": scale, "category_breakpoints": cuts,
         "training_labels_used": int(valid.sum()), "preprocessing_training_features": int(train.sum()),
         "training_rows_with_missing_labels": int((~valid).sum()), "baseline_preprocessing": baseprep,
         "rich_preprocessing": richprep, "errors": errors, "learned_predictor_fit_attempts": 4,
         "fit_wall_seconds": time.perf_counter() - start, "model_wall_seconds": timing,
         "forecasts_sha256": {s: sha(Path(out) / (s + "_forecasts.npz")) for s in STAGES},
         "peak_resident_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    receipt(out, "all_stage_forecasts_frozen", confirmation_labels_opened_by_model=False,
            confirmation_forecasts_sha256=sha(Path(out) / "confirmation_forecasts.npz"))


def read_forecasts(out, stage):
    with np.load(Path(out) / (stage + "_forecasts.npz"), allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def calibrate(out, cfg):
    d = read_forecasts(out, "calibration")
    raw = read_labels(out, "calibration", d["ids"], cfg)
    y = delivered(raw, load(Path(out) / "fit_receipt.json")["scale"])
    valid = np.isfinite(y)
    grid = np.asarray(cfg["augmentation"]["weight_grid"])
    weights, losses = [], []
    for j in range(6):
        if not np.isfinite(d["preds"][:, j]).all():
            weights.append(0.0)
            losses.append(float("inf"))
            continue
        blends = d["baseline"][:, None] + grid * (d["preds"][:, j] - d["baseline"])[:, None]
        loss = ((blends[valid] - y[valid, None])**2).mean(axis=0)
        index = int(np.argmin(loss))
        weights.append(float(grid[index]))
        losses.append(float(loss[index]))
    base_loss = float(np.mean((d["baseline"][valid] - y[valid])**2))
    best = min(range(6), key=lambda j: (losses[j], j))
    choice = best if losses[best] < base_loss else -1
    dump(Path(out) / "calibration.json", {"weights": weights, "bounded_losses": losses,
         "baseline_loss": base_loss, "calibration_choice": choice, "labels_used": int(valid.sum())})
    receipt(out, "augmentation_weights_frozen", sha256=sha(Path(out) / "calibration.json"))


def augment(d, weights):
    preds = np.where(np.isfinite(d["preds"]), d["preds"], d["baseline"][:, None])
    return d["baseline"][:, None] + np.asarray(weights) * (preds - d["baseline"][:, None])


def by_rejections(p, q=0.05):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    critical = q * np.arange(1, len(p) + 1) / (len(p) * np.sum(1 / np.arange(1, len(p) + 1)))
    passing = np.flatnonzero(p[order] <= critical)
    result = np.zeros(len(p), dtype=bool)
    if len(passing):
        result[order[:passing[-1] + 1]] = True
    return result


def choose(losses, base_loss, allowed):
    choices = [j for j in range(6) if allowed[j] and np.isfinite(losses[j])]
    if not choices:
        return -1
    best = min(choices, key=lambda j: (losses[j], j))
    return best if losses[best] < base_loss else -1


def exact_theta(preds, y, categories):
    n = len(y)
    v = (rankdata(y, method="average") - 0.5) / n - 0.5
    values = []
    for j in range(preds.shape[1]):
        if not np.isfinite(preds[:, j]).all():
            values.append(float("nan"))
            continue
        u = (rankdata(preds[:, j], method="average") - 0.5) / n - 0.5
        theta = 0.0
        for c in np.unique(categories):
            keep = categories == c
            theta += float(np.sum((u[keep] - u[keep].mean()) * (v[keep] - v[keep].mean()))) / n
        values.append(theta)
    return np.asarray(values)


def select(out, cfg, audit_fn=None):
    d = read_forecasts(out, "selection")
    raw = read_labels(out, "selection", d["ids"], cfg)
    yall = delivered(raw, load(Path(out) / "fit_receipt.json")["scale"])
    valid = np.isfinite(yall)
    y = yall[valid]
    failed = ~np.isfinite(d["preds"]).all(axis=0)
    cal = load(Path(out) / "calibration.json")
    blends = augment(d, cal["weights"])[valid]
    preds = np.where(np.isfinite(d["preds"][valid]), d["preds"][valid], d["baseline"][valid, None])
    if audit_fn is None:
        sys.path.insert(0, str(HERE.parent))
        receipt(out, "gate_module_preimport_hash", path="study_gates.py",
                sha256=sha(HERE.parent / "study_gates.py"))
        audit_fn = importlib.import_module("study_gates").audit_candidates
    receipt(out, "selection_gate_queries_begin", nominal_shared_iid_draws=32768,
            observed_archive_rows=len(y), exact_archive_rank_target_computed=False)
    rows = audit_fn(preds=preds, y=y, categories=d["categories"][valid], baseline=d["baseline"][valid],
                    blends=blends, seed=cfg["seed"], family=6)
    if isinstance(rows, tuple):
        rows, gate_meta = rows
        dump(Path(out) / "gate_sampling_receipt.json", gate_meta)
    canonical = {}
    for source in rows:
        row = dict(source)
        method = row["method"]
        j = row.get("candidate", row.get("candidate_index"))
        j = FAMILY.index(j) if isinstance(j, str) and j in FAMILY else int(j)
        if method not in METHODS or j not in range(6) or (method, j) in canonical:
            raise ValueError("Unexpected/duplicate method-candidate gate output")
        p = float(row["p"])
        if failed[j]:
            p = 1.0
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid gate p-value")
        row.update(candidate=j, candidate_name=FAMILY[j], p=p, failed_fit=bool(failed[j]))
        if failed[j]:
            row["failed_fit_audit_input"] = "baseline placeholder; diagnostic estimates invalid; p forced to one"
            for key in ("estimate", "lower", "radius", "bias"):
                if key in row:
                    row[key] = None
        canonical[method, j] = row
    if set(canonical) != {(m, j) for m in METHODS for j in range(6)}:
        raise ValueError("Incomplete gate family; no silent comparator removal")
    losses = np.mean((blends - y[:, None])**2, axis=0)
    losses[failed] = np.inf
    base_loss = float(np.mean((d["baseline"][valid] - y)**2))
    selectors = {"baseline": -1, "ungated": choose(losses, base_loss, ~failed),
                 "calibration_only": cal["calibration_choice"], "persistence": "persistence_raw"}
    flat = []
    for method in METHODS:
        rejected = by_rejections([canonical[method, j]["p"] for j in range(6)], cfg["selection"]["within_method_q"])
        selectors[method] = choose(losses, base_loss, rejected & ~failed)
        for j in range(6):
            row = canonical[method, j]
            row.update(retained=bool(rejected[j]), selection_bounded_loss=losses[j])
            flat.append(row)
    dump(Path(out) / "gate_decisions.json", flat)
    dump(Path(out) / "selectors.json", {"selectors": selectors, "selection_losses": losses,
        "baseline_loss": base_loss, "weights": cal["weights"], "family": FAMILY,
        "calibration_sha256": sha(Path(out) / "calibration.json"),
        "confirmation_forecasts_sha256": sha(Path(out) / "confirmation_forecasts.npz")})
    freeze = {"utc": utc(), "selectors_sha256": sha(Path(out) / "selectors.json"),
              "gate_decisions_sha256": sha(Path(out) / "gate_decisions.json"),
              "confirmation_labels_opened_by_model": False, "exact_selection_theta_computed": False}
    dump(Path(out) / "selector_freeze.json", freeze)
    receipt(out, "all_selectors_frozen", **freeze)
    # Deliberately after durable selector and gate-decision hashes.
    theta = exact_theta(d["preds"][valid], y, d["categories"][valid])
    dump(Path(out) / "selection_exact_diagnostics.json", {"theta": dict(zip(FAMILY, theta)),
        "category_counts": np.bincount(d["categories"][valid], minlength=4),
        "labels_in_exact_rank_census": len(y), "selectors_sha256": freeze["selectors_sha256"],
        "normalization": "midrank=(average_rank-.5)/n-.5; covariance residualized by exact category means"})
    receipt(out, "selection_exact_rank_diagnostic", after_all_selector_decisions=True, labels=len(y))


def utility_hours(dates, primary, comparator, y, start, end):
    """G includes zero gain for unavailable labels; W uses only frozen forecasts."""
    grid = pd.date_range(start, end, freq="h", inclusive="left")
    dt = pd.DatetimeIndex(dates)
    positions = ((dt.floor("h") - pd.Timestamp(start)) / pd.Timedelta(hours=1)).astype(int)
    if np.any(positions < 0) or np.any(positions >= len(grid)):
        raise ValueError("Confirmation publication outside fixed grid")
    gain = np.where(np.isfinite(y), (comparator - y)**2 - (primary - y)**2, 0.0)
    count = np.bincount(positions, minlength=len(grid))
    gain_sum = np.bincount(positions, weights=gain, minlength=len(grid))
    width_sum = np.bincount(positions, weights=2 * np.abs(primary - comparator), minlength=len(grid))
    hourly = np.divide(gain_sum, count, out=np.zeros(len(grid)), where=count > 0)
    widths = np.divide(width_sum, count, out=np.zeros(len(grid)), where=count > 0)
    return grid, count, hourly, widths, gain


def predictable_grid_bound(gains, widths, alpha):
    """Delegate calibration to the separately proved and verified module."""
    sys.path.insert(0, str(HERE.parent / "theory"))
    result = importlib.import_module("predictable_loss").predictable_loss_bound(
        gains, widths, alpha=alpha, expected_hours=1416)
    result.update(lower_bound=result["lower"], observed_hourly_mean_gain=result["estimate"],
                  assumptions_verified_from_archive=False,
                  scope="same-period conditional-average settled-hour gain under declared availability/filtration assumptions")
    return result


def selected_forecast(choice, d, blends):
    if choice == "persistence_raw":
        return d["preds"][:, 3]
    return d["baseline"] if choice == -1 else blends[:, int(choice)]


def confirm(out, cfg):
    freeze = load(Path(out) / "selector_freeze.json")
    if sha(Path(out) / "selectors.json") != freeze["selectors_sha256"] or sha(Path(out) / "gate_decisions.json") != freeze["gate_decisions_sha256"]:
        raise ValueError("Frozen selector/gate decisions changed")
    selection = load(Path(out) / "selectors.json")
    if sha(Path(out) / "confirmation_forecasts.npz") != selection["confirmation_forecasts_sha256"]:
        raise ValueError("Frozen confirmation forecasts changed")
    d = read_forecasts(out, "confirmation")
    raw = read_labels(out, "confirmation", d["ids"], cfg)
    scale = load(Path(out) / "fit_receipt.json")["scale"]
    y = delivered(raw, scale)
    available = np.isfinite(y)
    blends = augment(d, selection["weights"])
    forecast = {name: selected_forecast(choice, d, blends) for name, choice in selection["selectors"].items()}
    available_losses = {name: float(np.mean((a[available] - y[available])**2)) for name, a in forecast.items()}
    candidate_losses = np.mean((blends[available] - y[available, None])**2, axis=0)
    failed = ~np.isfinite(d["preds"]).all(axis=0)
    candidate_losses[failed] = np.inf
    hindsight_loss = min(available_losses["baseline"], float(np.min(candidate_losses)))
    rows = []
    for name, a in forecast.items():
        losses = (a[available] - y[available])**2
        rawpred = np.expm1(a[available] * scale)
        lograw = np.log1p(raw[available])
        rows.append({"rule": name, "choice": selection["selectors"][name],
            "bounded_loss": float(losses.mean()), "available_labels": int(available.sum()), "eligible_rows": len(raw),
            "gain_vs_baseline_available": available_losses["baseline"] - float(losses.mean()),
            "gain_vs_ungated_available": available_losses["ungated"] - float(losses.mean()),
            "opportunity_loss_vs_ungated": float(losses.mean()) - available_losses["ungated"],
            "regret_vs_hindsight_same_family": float(losses.mean()) - hindsight_loss,
            "log_mse_unclipped_outcome": float(np.mean((a[available] * scale - lograw)**2)),
            "log_mae_unclipped_outcome": float(np.mean(np.abs(a[available] * scale - lograw))),
            "share_mse_unclipped_outcome": float(np.mean((rawpred - raw[available])**2)),
            "share_mae_unclipped_outcome": float(np.mean(np.abs(rawpred - raw[available])))})
    pd.DataFrame(rows).to_csv(Path(out) / "confirmation_rules.csv", index=False)
    decisions = load(Path(out) / "gate_decisions.json")
    candidates = []
    for j, name in enumerate(FAMILY):
        raw_loss = float(np.mean((d["preds"][available, j] - y[available])**2)) if not failed[j] else None
        candidates.append({"candidate": name, "failed_fit": bool(failed[j]), "raw_candidate_loss": raw_loss,
            "augmentation_loss": candidate_losses[j], "gain_vs_baseline": available_losses["baseline"] - candidate_losses[j],
            "retained_by": [r["method"] for r in decisions if r["candidate"] == j and r["retained"]]})
    dump(Path(out) / "confirmation_all_candidates.json", candidates)
    opportunity = []
    for r in decisions:
        j = r["candidate"]
        gain = available_losses["baseline"] - candidate_losses[j]
        opportunity.append({"method": r["method"], "candidate": FAMILY[j], "retained": r["retained"],
            "confirmation_augmentation_gain": gain, "excluded_useful": bool(not r["retained"] and gain > 0),
            "retained_harmful": bool(r["retained"] and gain < 0)})
    dump(Path(out) / "confirmation_opportunity_losses.json", opportunity)
    contrasts, bounds, article_frames, day_frames, hour_frames = [], {}, [], [], []
    primary_method = cfg["confirmation_inference"]["primary"]
    a = forecast[primary_method]
    primary = cfg["confirmation_inference"]["primary_comparators"]
    for name, b in forecast.items():
        if name == primary_method:
            continue
        grid, count, hourly, widths, gain = utility_hours(d["dates"], a, b, y, *cfg["chronology"]["confirmation"])
        contrasts.append({"primary": primary_method, "comparator": name,
            "mean_gain_available_articles": float(gain.sum() / available.sum()),
            "mean_gain_all_eligible_articles": float(gain.mean()),
            "mean_gain_fixed_publication_hours": float(hourly.mean()),
            "formal_primary_contrast": name in primary})
        af = pd.DataFrame({"IDLink": d["ids"], "publication": d["dates"], "available": available,
                           "comparator": name, "gain_unavailable_zero": gain})
        article_frames.append(af)
        day = pd.to_datetime(af.publication).dt.floor("D")
        daily = af.assign(day=day).groupby("day").agg(eligible=("available", "size"),
                    available=("available", "sum"), gain_sum=("gain_unavailable_zero", "sum")).reset_index()
        daily["comparator"] = name
        daily["mean_gain_eligible"] = daily.gain_sum / daily.eligible
        day_frames.append(daily)
        hour_frames.append(pd.DataFrame({"publication_hour": grid, "settlement_hour_end": grid + pd.Timedelta(hours=49),
             "comparator": name, "eligible": count, "mean_gain": hourly, "predictable_width": widths}))
        if name in primary:
            bounds[name] = predictable_grid_bound(hourly, widths, cfg["confirmation_inference"]["alpha_each"])
    pd.DataFrame(contrasts).to_csv(Path(out) / "confirmation_contrasts.csv", index=False)
    pd.concat(article_frames).to_csv(Path(out) / "confirmation_article_gains.csv", index=False)
    pd.concat(day_frames).to_csv(Path(out) / "confirmation_day_gains.csv", index=False)
    pd.concat(hour_frames).to_csv(Path(out) / "confirmation_hour_gains.csv", index=False)
    dump(Path(out) / "confirmation_conditional_bounds.json", bounds)
    success = all(bounds[c]["lower_bound"] > 0 and available_losses[primary_method] < available_losses[c] for c in primary)
    dump(Path(out) / "confirmation_adjudication.json", {"observed_primary_success_under_stated_conditional_assumptions": success,
         "operational_future_gain_established": False, "label_acquisition_saving_established": False,
         "independent_dataset_tasks": 0, "reproduction_of_completed_task": True,
         "synthetic": load(Path(out) / "run_identity.json")["synthetic"], "confirmation_hours": 1416,
         "primary_method": primary_method, "primary_comparators": primary,
         "target_clip_fraction_available": float(np.mean(np.log1p(raw[available]) > scale)),
         "formal_target": cfg["confirmation_inference"]["target"],
         "caveat": cfg["confirmation_inference"]["availability_assumption"],
         "selectors_sha256": freeze["selectors_sha256"]})
    metadata = load(Path(out) / "preparation_metadata.json")
    dump(Path(out) / "cost_ledger.json", {"source": metadata["source_rows"], "stages": metadata["stages"],
        "all_archive_labels_machine_ingested": True, "label_availability_census_reads": metadata["label_availability_census_reads"],
        "selection_loss_census_labels": metadata["stages"]["selection"]["available_labels"],
        "post_decision_exact_rank_census_labels": metadata["stages"]["selection"]["available_labels"],
        "shared_gate_draw_calls": 32768, "fit_receipt": "fit_receipt.json",
        "distinct_gate_labels": "see gate_sampling_receipt.json or per-method gate_decisions metadata",
        "draw_calls_are_not_new_future_observations": True, "actual_label_acquisition_saving": False})
    receipt(out, "confirmation_completed", once_after_selectors_frozen=True)


def run_phases(out, cfg, frame, audit_fn=None):
    def timeout_handler(signum, frame):
        raise TimeoutError("Frozen phase wall-time ceiling exceeded")
    signal.signal(signal.SIGALRM, timeout_handler)
    for name, fn in (("fit", lambda: fit_models(frame, out, cfg)), ("calibrate", lambda: calibrate(out, cfg)),
                     ("select", lambda: select(out, cfg, audit_fn)), ("confirm", lambda: confirm(out, cfg))):
        start = time.perf_counter()
        receipt(out, "phase_begin", phase=name)
        signal.alarm(cfg["limits"]["phase_wall_seconds"])
        with threadpool_limits(limits=1):
            fn()
        signal.alarm(0)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if peak > cfg["limits"]["resident_memory_gib"] * 1024:
            raise MemoryError("Frozen phase resident-memory ceiling exceeded")
        receipt(out, "phase_complete", phase=name, wall_seconds=time.perf_counter() - start, peak_resident_mib=peak)


def synthetic_frames(cfg, seed=17):
    """Finite artificial fixture, with ties/missingness and no source-data use."""
    rng = np.random.default_rng(seed)
    news, social = [], []
    identity = 0
    for stage in STAGES:
        n = max(cfg["eligibility"]["minimum_observed_labels"][stage] + 20, 80)
        a, b = map(pd.Timestamp, cfg["chronology"][stage])
        offsets = np.linspace(0, int((b - a).total_seconds()) - 1, n).astype(int)
        for offset in offsets:
            identity += 1
            early = int(rng.integers(0, 20))
            news.append({"IDLink": str(identity), "Title": "synthetic title " + "a" * (identity % 7),
                         "Headline": "synthetic headline", "Source": "source_" + str(identity % 4),
                         "Topic": "economy", "PublishDate": str(a + pd.Timedelta(seconds=int(offset)))})
            social.append({"IDLink": str(identity), "TS1": -1 if identity % 7 == 0 else early,
                           "TS2": early + 1, "TS3": early + 2,
                           "TS144": -1 if identity % 101 == 0 else int(3 * early + rng.integers(0, 8))})
    return pd.DataFrame(news), pd.DataFrame(social)


def synthetic_audit(**kwargs):
    """Testing stub only; reports no scientific p-values or benchmark results."""
    return [{"method": m, "candidate": j, "p": 1.0, "synthetic_gate_stub": True}
            for m in METHODS for j in range(kwargs["family"])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "synthetic", "synthetic-real-gates"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--source-zip", type=Path, default=HERE.parent / "data/news432.zip")
    args = parser.parse_args(argv)
    synthetic = args.mode != "run"
    cfg = start_run(args.output, args.config, synthetic)
    try:
        if synthetic:
            news, social = synthetic_frames(cfg)
            frame = ingest_frames(news, social, args.output, cfg)
        else:
            source = copy_source(args.source_zip, args.output, cfg)
            frame = ingest_zip(source, args.output, cfg)
        run_phases(args.output, cfg, frame, synthetic_audit if args.mode == "synthetic" else None)
        dump(args.output / "completion.json", {"status": "complete", "utc": utc(), "synthetic": synthetic,
             "scientific_confirmation_result": False, "reproduction_of_completed_task": True})
    except Exception as exc:
        signal.alarm(0)
        dump(args.output / "failure.json", {"status": "failure_preserved_no_replacement", "utc": utc(),
             "error": repr(exc), "traceback": traceback.format_exc(), "synthetic": synthetic})
        raise
    print(json.dumps({"output": str(args.output), "status": "complete", "synthetic": synthetic}))


if __name__ == "__main__":
    main()
