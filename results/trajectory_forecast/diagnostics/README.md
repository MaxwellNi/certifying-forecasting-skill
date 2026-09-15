# Trajectory study: independent numerical and timing check

This numerical check reimplements the fixed rank kernels, all diagnostic cells, final
gates, finite-population target, complete pooled center, and descriptive loss
intervals without importing the study's code. It does not choose new models,
budgets, seeds, endpoints, or successful cells. The original and corrected
cohorts remain in the public artifact. The corrected cohort was already
inspected; none of this is a new independent confirmation.

Run `python results/trajectory_forecast/diagnostics/reproduce.py --output /tmp/trajectory-diagnostics` with NumPy, pandas and SciPy from the repository root.
The script reads the bundled study arrays and source CSV. It does not retrain
models; use the separate study preparation command for source-to-model reproduction.

## What was verified

`INDEPENDENT_CHECK.json` pins the input array and recorded-diagnostic hashes.
All 9,600 original diagnostic rows were independently recomputed: maximum
absolute discrepancy is 1.12e-16. Eight final split/triple gates and eight
complete pooled centers/bounds agree within 1.0e-16. Pooled centers are computed
with a finite-archive multiplicity contraction, independently of the released
rank/Fenwick implementation.

For all 692 selection/evaluation records, target values, persistence forecasts,
and all 15 stored numeric features agree exactly with the original attributed
CSV at their timestamp. Wind direction is a sixteenth logical feature available
in the raw CSV; it is not stored in the numeric NPZ feature matrix.

The source timestamp audit confirms lag alignment, not real-time publication
latency. Previous-hour weather and pollutant measurements are assumed available
at issue; the source does not certify when each measurement was published.

## Exact application mapping and costs

The corrected daily issue hour is 14:00 and the target hour is 15:00. Training
uses all eligible hours in 2010–2012 (23,006 records). The selection archive has
342 eligible days in 2013 and evaluation has 350 eligible days in 2014. Each
trajectory uses raw hourly offsets 0, -1, -2, -3, -6, -12 and -24 from the target:
seven used hourly records spanning 24 hours. Weather/wind are at offset -1;
the target-hour calendar is known at issue. The union over selection inputs
contains 2,069 distinct raw hourly records. This is different from either the
342 daily archived trajectories or the repeated draw count.

For each candidate, f=(candidate, persistence) and g=(target, persistence).
The maps are learned/fixed before archive draws. Both channel row sums are
c=d=(1,-1). The fixed coefficient matrices are:

```
C = [[ 1, 0], [-1, 0]]
D_shared   = [[ 1, 0], [-1, 0]]
D_separate = [[ 0, 1], [ 0,-1]]
D_crossed  = [[ 0, 1], [-1, 0]]
```

All three have the same row sums and therefore the same fitted rank target.
The matrix interactions differ. Each selected candidate uses M=16,384 blocks,
two references per block, and n=8,192 independent validation triples. The total
is 73,728 draws, at candidate alpha=.05/8 and delta=alpha/2. The same index
arrays are shared across candidates for paired comparisons; 8 times 73,728 is
not the number of independent newly acquired labels. All 342 existing rows are
accessed, with no new label acquisition. The originally reported 17 coordinates
is 16 logical features plus one target, not instrumented memory reads.

The actual split/triple bounds use the predeclared hybrid minimum with explicit
union allocation between Hoeffding and empirical Bernstein. They are not the
numerical output of the main paper's simpler range-only formula. For boosting
with 31 leaves, the range-only split lower bound is -0.101208. The Hoeffding
component of the hybrid, with its additional half-budget allocation, is
-0.107139;
the published hybrid lower bound is -0.0153666.

`certificate_decomposition.csv` reports all eight candidates: raw mean,
interaction estimate, both radii, corrected center/lower, triple and pooled
centers/radii, exact target, counts and original same-task timings. Census time
is 0.0103–0.0128 seconds per candidate, pooled time 0.430–0.439 seconds on the
recorded execution. These timing values are recorded from the original execution.
No acquisition or computation saving is claimed for resampling this small
fully available archive.

## Coverage and detection, all cells retained

`coverage_decomposition_all_9600.csv` and `coverage_all_32_cells.csv` contain
every original corrected diagnostic draw and summary. Diagnostics use M=1,024,
n=512, alpha=.05, delta=.025. Four cells have target zero and 28 positive target;
targets range from 0 to .000948129. All 32 cells have 0/300 noncoverage and also
0/300 detection. This small-budget study is conservative, not evidence of
high detection power. The two-sided 95% exact upper bound for each zero count
is .0122209747. Shared seeds make designs paired; cells are not pooled as
independent replications. A one-sided lower interval is unbounded above; the
tables correctly report kernel range widths and finite lower-bound radii,
not a finite total confidence-interval width.

## Two loss estimands and their matching descriptive intervals

`utility_weighted_and_equal_week.json` distinguishes target-weighted losses
from equally weighted weekly means. Both use the same paired four-week
noncircular moving-block bootstrap, 2,000 draws and seed 2026091455. The original
endpoint was the equal-week mean; the target-weighted ratio interval is an
additional reporting check after the outcomes were inspected.

| Comparison | Point gain | Matching descriptive 95% interval |
|---|---:|---:|
| Gate vs ungated, target-weighted MSE difference | 7.619001 | [-36.480381, 51.380742] |
| Gate vs ungated, target-weighted relative MSE gain | 2.574518% | [-18.888745%, 11.998060%] |
| Gate vs ungated, equal-week MSE difference | 6.534213 | [-35.021838, 48.629715] |
| Gate vs persistence baseline | 0 | [0, 0], identical predictions |

Raw-unit MSE is 288.32 for persistence and 295.939001 for the 31-leaf boosting
candidate. All intervals are descriptive; no finite validity for the physical
time series is asserted. The earlier interval was in squared PM2.5 units,
not percentage points. The positive point difference is retained, but stable
incremental decision benefit is not established. An interval spanning zero
does not prove absence of benefit or impossibility of measuring such benefit.

The primary dataset [UCI Beijing PM2.5 page](https://archive.ics.uci.edu/dataset/381/beijing%2Bpm2%2B5%2Bdata)
identifies the hourly 2010–2014 observations and CC BY 4.0 attribution. It does
not supply a real-time publication-latency guarantee.
