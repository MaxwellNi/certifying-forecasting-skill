# Final-target inference and forecast decisions

This standalone module supplies complete-U variance calibration, stratified
reference inference, strong classical betting comparators, the complete binary
benchmark, and a source-to-prediction reproduction of the Facebook economy-news
study. It preserves the negative comparative results: no unique method advantage
or incremental forecast-selection benefit was established.

Use Python 3.12 on Linux. Commands below run from this directory and write to
new empty output directories. The pinned packages match the recorded study
versions; numerical linear algebra may still vary across platforms.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s confirmation/tests
.venv/bin/python theory/check_full_u_variance.py
.venv/bin/python checks/check_stratified.py
.venv/bin/python betting/check_profiled_betting.py
.venv/bin/python theory/check_predictable_loss.py
.venv/bin/python theory/binary_checks.py
.venv/bin/python checks/check_gate_composition.py
.venv/bin/python theory/binary_benchmark.py --output runs/binary
.venv/bin/python confirmation/experiment.py run --source-zip data/news432.zip --output runs/news432
.venv/bin/python confirmation/correct_numeric_summaries.py --run runs/news432 --output runs/news432_corrected
.venv/bin/python compare_reproduction.py --run runs/news432
```

The news command reads the source ZIP, reconstructs features and stage labels,
retrains Ridge, histogram gradient boosting and ExtraTrees, recalibrates all
six augmentations, computes all nine complete gate families, selects every
rule and evaluates the fixed confirmation period. It does not load the saved
reference forecasts. This repeats an already completed task and is not an
additional independent confirmation. The original pipeline ingested all source
labels before its stage-specific access sequence; no external blind custody or
external preregistration is claimed.

The full news run can take several minutes, with a 40-minute phase ceiling and
an 8 GiB resident-memory limit. Complete-U deletion is O(n²+C) time and O(n+C)
working memory. The binary runner exploits the same estimator's finite-support
count identities and repeats 8,000 paired samples (13 methods, 208 summaries).
Timings are platform dependent. The formula checks are finite algebra and
numerical checks, not an exhaustive proof of validity on every distribution.

## Statistical contract and method attribution

| Component | Conditions and target | Attribution and limit |
|---|---|---|
| Full U | Conditionally IID raw rows, exact category probabilities, fixed target and sample size; global-midrank covariance | Explicit deletion computation; classical weak-interaction concentration, including higher-order terms |
| Stratified reference pairs | Independent focal and global-reference streams; condition on focal category labels; exact masses and omitted-mass allowance | Classical covariance identity and bounded-sum concentration; no conditional-sampling oracle assumed |
| Profiled betting | Same independent scores and complete composite mean null allowing positive/negative strata | Classical stratified inverse betting; optimizer values alone are insufficient, so evidence uses a conservative lower bound |
| Archive study | Independent replacement draws conditional on the fixed labeled archive | Association beyond four declared categories, not beyond the full continuous baseline or a future population |
| Hourly loss bound | Fixed predictors, eligibility and 1,416-hour calendar; stated settlement-time predictability and [0,1] targets | Classical bounded martingale argument and fixed tuning grid; not unknown-future or issuance-conditioned risk |

The complete proofs are in `theory/FULL_U_VARIANCE_THEORY.md`,
`STRATIFIED_THEORY.md`, `betting/PROFILED_BETTING.md` and
`theory/PREDICTABLE_LOSS_THEORY.md`. Classical sources are cited there.
These implementations do not establish a new universal concentration result,
optimal rate, or unique capability over the strongest same-information method.
Methods must be fixed before evaluation or combined with a valid joint error
budget. BY cannot repair invalid marginal p-values. Non-retention does not
establish absence of useful prediction information.

## Recorded results, including failures

`results/binary/` contains every 208 summary cell, 1,248 pointwise paired
intervals and raw paired replicate arrays. In the rare perfectly correlated
binary law, V=W~Bernoulli(.01), the correlation is one and the absolute midrank
covariance is .002475. At 2,048 rows, full-U joint detects 0/500, classical
four-row betting 486/500 and binary-pair betting 498/500. Their small mutual
difference is unresolved by the reported interval. At 32,768 rows, full-U
variance improves its own range bound (500/500 versus 478/500). This is one
preselected scene, not universal dominance or a 16-fold optimal complexity result.
The balanced null, negative and degenerate cases remain in the complete table.
Monte Carlo intervals are pointwise, not simultaneous over all comparisons.

`results/development/` preserves the complete 192-row comparison on 32 previously
inspected candidates, including every nonpositive target and each BY family.
Stratified hybrid retains 14/32; profiled betting retains 16/32 and includes all
14. There are 25 positive exact targets. These are development diagnostics,
not independent-task power estimates. The data required to regenerate those
older archive predictions belong to the surrounding public artifact.

`results/news432/` contains the original full source-derived stage arrays and
all nine six-candidate gate families. Training/calibration/selection/confirmation
have 2,706/931/1,324/1,110 observed-label rows. The main method, stratified betting,
pooled reference betting and ungated selection deliver elementwise identical
HGB augmentations with weight .45. All three predefined incremental gains are
zero, failing the primary success criterion. The lower bound's tiny negative
grid penalty is not an observed loss. The combination improves the specified
clipped loss by .1581% against Ridge, but untransformed share-count MSE worsens
from 29,677.37 to 33,481.15 (12.82%); untruncated log MSE also worsens slightly.
These results do not establish additional benefit from gating.

All selection labels have already been ingested, so exact archive census is a
legal comparator and chooses the same augmentation. There is no demonstrated
label-acquisition saving. Positive archive association can accompany later harm;
a fixed noise realization can also have a positive finite-archive target.
The fixed confirmation grid includes 347 nonempty and 1,069 empty hours.

`results/news432_corrected/` gives descriptive summaries corrected for floating
summation-order artifacts. Original arrays and decisions are unchanged. The
correction uses elementwise forecast equality for exact-zero claims and preserves
real tiny positive differences. The original numerical tables remain available
for comparison; use corrected tables for descriptive interpretation.

## Reproduction identity and licensing

`compare_reproduction.py` checks stage arrays, all gate decisions, selectors and
primary loss comparisons against the saved reference run; it reports differences
without treating a repeat as new scientific evidence. Floating arithmetic is
reported separately from discrete decisions. `SHA256SUMS.json` identifies all
included files; hashes establish byte identity, not statistical assumptions or
unseen-data custody. No historical model from other tasks is retrained here.

Code is under the included MIT license. The unchanged UCI source ZIP and derived
source data retain CC BY 4.0 attribution; see `data/DATA_LICENSE.md`.
