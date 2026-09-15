# Zero-column pruning at the actual trajectory cost

This package adds a **post-exposure diagnostic** to the corrected Beijing study.
It uses exactly the original 73,728 independently sampled archive-draw positions,
the same 342-record 2013 archive, the same eight trained models, the same fitted
rank target, and the same candidate alpha=0.05/8 and correction delta=0.05/16.
It does not create a new independent confirmation or change the recorded study.

The main new finding is that removing a known-unused reference role and allocating
its budget improves the shared certificate, but does not change any shared gate
decision. All pure-Hoeffding methods also abstain. Positive triple/pooled lower
bounds use their hybrid/variance implementation. They cannot be attributed to
the pure range-only radius comparison.

## Declared allocation and its scope

The original coefficient matrices on both sides are `[[1,0],[-1,0]]`. The second
reference column is identically zero in both matrices. Deleting it preserves
every retained-block score, the row sums, the overlap matrix, the interaction
correction and the fitted target. Its active evaluation cost is therefore two
whole trajectories per block, rather than three.

Let `a = W_S sqrt(log(1/(alpha-delta))/2)` and
`b = W_Q sqrt(log(1/delta)/2)`. Before loading forecast or target values, the new
script minimizes `a/sqrt(M) + b/sqrt(n)` over positive integer allocations with
`2M+3n <= 73728`. It uses the same declared enclosing widths `W_S=2`, `W_Q=4`
for all candidates, including persistence, and does not optimize observed
variances, lower bounds, gates, future losses, or seeds. The continuous ratio
is `M/n=(3a/(2b))^(2/3)`. Exact integer enumeration includes possible unused
remainders and chooses **M=13,086, n=15,852**, using all 73,728 draws.

The sample coupling is deterministic: flatten the original evaluation indices
followed by validation indices in their recorded order, use the first `2M`
positions as evaluation pairs and the remaining `3n` as correction triples.
Disjoint positions are independent under the original conditional iid index
experiment. Repeated archive IDs are allowed. The original and new diagnostic
outputs are coupled; repartitioning does not provide independent studies.
The rule is outcome-independent in its inputs, but was designed after earlier
outcomes had been inspected, so this is explicitly a post-exposure comparison.

## Costs and certificates

| Shared allocation | Charged draws | Draw positions affecting the estimator | Ignored draw positions | M | n |
|---|---:|---:|---:|---:|---:|
| Recorded two-peer construction | 73,728 | 57,344 | 16,384 | 16,384 | 8,192 |
| Delete zero columns, preserve counts | 57,344 | 57,344 | 0 | 16,384 | 8,192 |
| Delete zero columns, width-only allocation | 73,728 | 73,728 | 0 | 13,086 | 15,852 |

“Ignored” means that the draw position has zero influence on the estimator; the
original implementation still formed comparisons with that role before its zero
coefficient contraction. Every row above accesses all **342** archive records
and **2,069** distinct original hourly rows; no record occurs only in the ignored
roles. All methods acquire **zero new labels**. Triple and pooled U use all
73,728 original draw positions, with **24,576** independent disjoint triples for
their concentration bounds. Pooled overlapping triples are not counted as
independent observations. Exact census reads the 342 already-accessible archive
records once and has no conditional-archive sampling uncertainty. The original
2010–2012 training sample remains 23,006 hourly records; fitting is not included
in the draw budget. Direct-loss selection reads all 342 selection outcomes;
fixed-candidate choices need no selection outcomes. Every decision is evaluated
on the same 350 future records.

For boosting_31, values below are in units of 10^-3. All lower bounds use
unrounded values. The exact archive target is **0.852133** in these units.

| Method | Bound | Raw center | Interaction | Evaluation radius | Correction radius | Lower bound |
|---|---|---:|---:|---:|---:|---:|
| Recorded shared | hybrid | 8.529663 | 8.148193 | 4.804982 | 10.943040 | -15.366552 |
| Pruned, reallocated shared | hybrid | 10.354577 | 8.476848 | 5.886686 | 6.239438 | -10.248395 |
| Triple U | hybrid | 0.869751 | 0 | 0.755408 | 0 | 0.114343 |
| Complete pooled U | variance | 0.847358 | 0 | 0.692417 | 0 | 0.154940 |
| Recorded shared | pure Hoeffding | 8.529663 | 8.148193 | 26.535650 | 75.054154 | -101.208334 |
| Pruned, reallocated shared | pure Hoeffding | 10.354577 | 8.476848 | 29.691781 | 53.954500 | -81.768552 |
| Triple U | pure Hoeffding | 0.869751 | 0 | 6.774287 | 0 | -5.904536 |
| Complete pooled U | pure Hoeffding | 0.847358 | 0 | 6.774287 | 0 | -5.926930 |
| Exact census | exact | 0.852133 | 0 | 0 | 0 | 0.852133 |

Pruning without reallocation preserves the original certificate exactly while
removing 22.22% of the charged draw positions. Reallocation reduces the shared
pure-Hoeffding total radius from 0.101590 to 0.083646 (**17.66%**), and its hybrid
radius from 0.015748 to 0.012126 (**23.00%**). The center also changes because
the draw roles change; the lower-bound difference is not entirely a radius
effect. The generic range widths remain conservative, particularly for the
structurally zero persistence contrast. This comparison establishes the result
for these declared widths and this archive, not an optimal hybrid allocation
or an impossibility result for all conceivable bounds.

The independent reconstruction of pooled U uses archive multiplicities. It
reproduces the public rank-sweep center and bounds, but its runtime must not be
reported as the runtime of that public algorithm. The old recorded census and
pooled timings remain in `inputs/original_strong_gates.csv`; this diagnostic
makes no new hardware-independent timing claim.

## Decisions and future losses

Every shared gate retains 0/8 and falls back to persistence. Triple U with the
hybrid rule and pooled U with the variance rule each retain ridge_1, ridge_100,
and boosting_31, then select boosting_31 by its lowest selection MSE. Exact
census finds positive fitted rank targets for all seven augmentations and also
selects boosting_31. Every pure-Hoeffding gate retains 0/8.

The ungated/direct-loss selector chooses boosting_31 (future MSE **295.9390**).
Persistence and all shared gates have future MSE **288.3200**. Their point gain
over direct-loss selection is **7.6190**, or **2.5745%**, with the descriptive
target-weighted paired interval **[-36.4804, 51.3807]**. Their gain over choosing
persistence directly is exactly zero. The equal-week gain is **6.5342**, with
interval **[-35.0218, 48.6297]**. Neither interval establishes a stable or unique
gating advantage.

All eight fixed candidates are included in `all_eight_future_losses.csv` and
`future_rule_comparisons.csv`, including the two blend candidates with lower
point future MSE than persistence: blend_boosting_7 **283.4075** and
blend_boosting_31 **282.8837**. These observed rankings were not used to choose a
new deployment rule. Four of seven positive census rank targets have adverse
point future MSE differences against persistence. A positive fitted rank target
therefore does not by itself promise future loss improvement.

Intervals use **2,000** paired noncircular moving-block bootstrap replicates,
block length **four weeks**, seed **2026091455**. The 53 Monday–Sunday weeks
have 3–7 observations. Starts are uniform among `0,...,W-4`, blocks do not wrap,
and concatenated blocks are truncated to exactly W weeks. The target-weighted
statistic divides resampled weekly loss totals by resampled record counts;
the equal-week statistic averages weekly mean loss differences. The same
indices are used for every comparator. These are pointwise descriptive
percentile intervals for the dependent physical series, with no finite
coverage guarantee and no multiplicity adjustment. Full indices, weekly loss
means, and boundary details are released.

## Reproduction and verification

From this directory:

```sh
python reproduce.py --output /tmp/trajectory-budget-replay
python -m unittest discover -s . -p 'test_reproduce.py' -v
```

The output directory must not exist. Runtime needs Python, numpy and pandas;
the independent verification additionally imports the two included unchanged
reference-kernel snapshots. The inputs are small exact copies from the released
corrected study. `results/REPRODUCTION_MANIFEST.json` records file hashes and
versions. NPZ byte timestamps are not scientific results; arrays are compared
numerically. No source download or model refit is needed for this diagnostic;
the larger existing artifact retains the unchanged licensed source and fitting
code. See `DATA_LICENSE.md` and `PROVENANCE.json` for provenance.

Five targeted tests verify integer allocation against exhaustive search over
all feasible small M,n pairs; pointwise zero-column deletion including adversarial
changes to unused draws; explicit six-role triple and pooled formulas with ties
and integers beyond binary64's exact range; all eight old and new certificates
against the independently implemented producer; and original bootstrap endpoint
and cost parity. All pass; maximum new-formula/producer difference is
**9.94e-17**. `TESTS.txt` retains the execution receipt.

For a reader figure, use `FIGURE_SPEC.md` and the CSV coordinates in `results/`.
