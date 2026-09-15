# Direct estimation from sampled forecasts and outcomes

This comparator estimates residual rank association directly from raw forecasts, outcomes, and a fixed finite control category. It uses all available sampled rows, including rows used by other methods to fit nuisance functions. [THEORY.md](THEORY.md) gives the target, the sampling assumptions, the complete estimator, and its confidence bounds.

The complete estimator runs in O(n²+C) time with O(n+C) additional memory. It accounts for overlapping comparison positions, tied values, and repeated draws of the same archive row. Exact category probabilities must be known for the fixed sampling population.

The saved comparison uses all 32 previously inspected forecasting candidates and the original 32,768 raw draw positions per candidate:

| Confidence rule | Candidates retained |
|---|---:|
| Complete U statistic with a Hoeffding bound | 0/32 |
| Independent four-row blocks with an empirical Bernstein bound | 0/32 |

Each task, baseline, and method has a separate complete BY family of eight candidates. The complete results are in [archive/all_candidates.csv](archive/all_candidates.csv). Both bounds are conservative and depend on the smallest category probability, which is as low as 1/4,362 in these archives. These results do not establish superiority over a more tightly calibrated direct estimator. They are retrospective comparisons, and the original cross-task confirmation result remains 0/2.

## Reproduce the checks

From this directory, using the repository's Python environment:

```sh
python reproduce.py --output /tmp/direct_target_checks
```

This checks the complete estimator against exact rational sums on small samples, verifies its range and confidence formulas, and independently recomputes all 262,144 four-row block values, all 64 result rows' inferential calculations, and all eight BY families. It does not repeat the 32 larger quadratic complete-U calculations. To regenerate those estimates as well, write to a separate new directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python compare_archive.py --output /tmp/direct_target_complete --workers 4
```

The complete-U and block rules use different centers. Their uncertainty formulas must not be mixed. The recorded arithmetic times were a median 13.484 seconds for the complete estimator and 0.832 seconds for the block rule per candidate, using four concurrent processes with one BLAS thread each. These timings do not establish a speed advantage over another method.

`protocol.json` records the fixed comparison choices, and `input_hashes.json` binds the code and source archives. The forecasting data and draw indices are supplied by the adjacent `forecast_confirmation` directory; its source citations and licenses apply. No forecast model was reselected or refitted for this comparison.
