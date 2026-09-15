"""Report a fixed paired contrast from the distributed matched experiment.

The interval is an additional summary of previously inspected outcomes, not a
new experiment. It follows the exact paired-interval convention already used
in results/certificate_factorial: jointly cover the two discordant probabilities
with two 97.5% Clopper-Pearson intervals, then subtract their endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import beta

BASE = Path(__file__).resolve().parent


def interval(count: int, total: int, alpha: float) -> list[float]:
    """Two-sided exact interval for one binomial event probability."""
    return [0.0 if count == 0 else float(beta.ppf(alpha / 2, count, total-count+1)),
            1.0 if count == total else float(beta.ppf(1-alpha / 2, count+1, total-count))]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate() -> dict:
    rows_path = BASE / 'recorded/replications.csv.gz'
    rows = pd.read_csv(rows_path)
    setting = dict(design='eight_rare', training_rows=8192, validation_pairs=8192,
                   groups=100, peers=64, total_observations=30976)
    keep = pd.Series(True, index=rows.index)
    for name, value in setting.items():
        keep &= rows[name].eq(value)
    rows = rows.loc[keep]
    if rows.duplicated(['signal', 'replication', 'method']).any():
        raise ValueError('Duplicate paired observations in the selected setting.')
    first, second = 'signed_variance', 'independent_betting'
    positive = rows.loc[rows.signal.eq(.25)].pivot(
        index='replication', columns='method', values='reject')[[first, second]]
    if len(positive) != 300 or positive.isna().any().any():
        raise ValueError('Expected 300 complete paired repetitions.')
    before = positive[first].to_numpy(dtype=bool)
    after = positive[second].to_numpy(dtype=bool)
    total = len(positive)
    plus = int((after & ~before).sum())
    minus = int((before & ~after).sum())
    lp, up = interval(plus, total, .025)
    lm, um = interval(minus, total, .025)
    null = rows.loc[rows.signal.eq(0)].pivot(
        index='replication', columns='method', values='reject')[[first, second]]
    if len(null) != total or null.isna().any().any():
        raise ValueError('Expected complete paired zero-signal repetitions.')
    return dict(
        analysis_scope='Additional summary of previously inspected archived outcomes; no new experiment and no claim that this paired interval was prespecified for the matched comparison.',
        setting=setting, signal=.25, nominal_level=.05, replications=total,
        first_method=first, second_method=second,
        first_rejections=int(before.sum()), second_rejections=int(after.sum()),
        first_only=minus, second_only=plus, both=int((before & after).sum()),
        neither=int((~before & ~after).sum()), difference=(plus-minus)/total,
        paired_95_interval=[max(-1., lp-um), min(1., up-lm)],
        interval_method='Two two-sided 97.5% Clopper-Pearson intervals for discordant probabilities; subtract endpoints. The union bound gives at least95% pointwise coverage.',
        null_same_setting_signal_zero={method: dict(
            rejections=int(null[method].sum()), replications=total,
            pointwise_95_interval=interval(int(null[method].sum()), total, .05))
            for method in [first, second]},
        interpretation='Pointwise fixed-setting comparison, without simultaneous-grid coverage or a claim of general superiority. Equal nominal levels and observed zero null rejections do not establish equal actual error rates.',
        source_sha256={
            'paired_summary.py': sha256(Path(__file__)),
            'recorded/replications.csv.gz': sha256(rows_path),
            'recorded/protocol.json': sha256(BASE / 'recorded/protocol.json')})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON file; existing files are never overwritten.')
    args = parser.parse_args()
    result = calculate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Paired summary written to {args.output}')


if __name__ == '__main__':
    main()
