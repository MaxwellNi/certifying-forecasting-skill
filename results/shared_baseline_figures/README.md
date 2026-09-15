# Reference correction and shared-baseline bounds

This is the main-paper comparison figure. The vector PDF is 7.1 by 2.6 inches;
ordinary labels use 9 TeX points, with IEEEtran's Nimbus Roman text and Computer
Modern math. Fonts are embedded Type 1. The PDF has no raster content.

```sh
python scripts/figures/make_trajectory_comparison.py --output /tmp/trajectory-figure
```

The input is `../shared_baseline_ranges/results/fixed_draw/certificate_all_eight.csv`.
The original and reallocated designs use the same 73,728 draw positions.
Both use the shared-baseline ranges, as does the pooled comparator. The left
panel shows original shared scores, their corrected centers and exact archive
targets. The middle panel shows original and reallocated shared hybrid bounds;
the right uses a separate linear scale for pooled variance bounds and exact
targets. Diamonds are population targets under the empirical archive law,
not random lower bounds. Four pooled bounds are positive; no shared bound is
positive. Persistence has zero width because its forecast equals the baseline
as a declared map identity.

Ridge labels give the penalty strength, and boosting labels give the maximum
leaf count. Blend models average a candidate with persistence before the common
nonnegative clipping operation. This is a diagnostic on previously inspected
data; selected predictors and future-loss evidence are unchanged.

`trajectory_comparison_coordinates.csv` contains all 56 plotted positions.
Every horizontal coordinate is exactly 1,000 times its unrounded source value,
and vertical positions are fixed candidate rows. No jitter is used. All 8
candidates are present; redundant pooled and census markers are omitted from
the full-scale panel because the near-zero panel displays them clearly.
The receipt checks coordinates, source hashes, fonts, size and text bounds.
The 216-dpi rendering was visually checked for signs, labels, legends, clipping
and overlap. Standard plotting/TeX packages and the licensed underscore support
file in `../trajectory_budget_figures/tex/` are required.
