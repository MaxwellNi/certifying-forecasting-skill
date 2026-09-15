# Shared-baseline support study

This directory derives sharp support intervals and replays the existing public
trajectory budget experiment using those intervals. It preserves every draw,
allocation, center, sample variance, model and error budget. Both the shared
correction and the stronger U comparators receive the same structural information.
General-width results are preserved separately.

The valid widths are S: 5/4, Q: 5/2, symmetric U: 1/2. A declared identity
between the persistence forecast map and the baseline gives exact zero kernels
and widths. `MAP_CONTRACT.json` records this identity from the original map
definition; equality of observed arrays alone never triggers the identity rule.

Run from the public package root (Python 3 and NumPy):

```sh
python -B results/shared_baseline_ranges/exact_checks.py
OPENBLAS_NUM_THREADS=1 python -B -m unittest discover -s results/shared_baseline_ranges -p 'test_study.py' -v
OPENBLAS_NUM_THREADS=1 python -B results/shared_baseline_ranges/reproduce.py --output /tmp/shared-baseline-replay
```

Use a new output directory. `--source` can specify the released `trajectory_budget`
directory explicitly, or use --package-root to supply the public package root.
Tests can set TRAJECTORY_PACKAGE_ROOT when run outside the public layout.
The -B flag suppresses Python bytecode caches. Once this study is copied alongside `trajectory_budget` in
the public results directory, the default source lookup still works. No training
or confirmation forecasts are loaded. The replay reads only the existing 342-row
selection archive, original draws, active allocated draws and prior certificates.

- `proof.md`: theorem, analytic proof, attainable endpoints and claim limits.
- `exact_checks.py`: exact rational enumeration of all 2,197 weak-order triples.
- `test_study.py`: exact-kernel parity, original-result parity, unchanged allocation,
  explicit-identity handling and certification/selection outcomes.
- `results/exact_support.json`: full exact supports and endpoint witnesses.
- `results/fixed_draw/certificate_all_eight.csv`: new structural certificates.
- `results/fixed_draw/width_gain_comparison.csv`: old/new radius and lower-bound pairs.
- `results/fixed_draw/population_quantities.csv`: population shared score, target,
  interaction and correctly matched interaction fraction for every candidate.
- `results/fixed_draw/SUMMARY.json`: compact check results and outcome changes.
- `results/fixed_draw/REPRODUCTION_MANIFEST.json`: input and code hashes.

The allocated shared hybrid radius for boosting_31 falls 20.9189% to
0.009589475071547636. No shared certificate becomes positive. Complete U newly
certifies boosting_7, while every method's selected predictor is unchanged.
These are descriptive improvements on the same already exposed data, not new
independent confirmation or forecasting utility.

## Using the bounds

The existing public `mean_bound(values, width, ...)` already accepts a justified
width, so no change to the generic matrix-kernel API is necessary. The specialized
shared-baseline caller uses `structural_widths(name, contract)` to choose
widths for the existing mean-bound formulas; use the symmetric-U width in the
complete-U radius as well. Keep the generic `design_widths(C,D)` unchanged:
coefficient matrices alone cannot certify that the two baseline maps are the
same map. An unrestricted optional width override would transfer proof obligations
to callers and is unnecessary for this bounded application.
