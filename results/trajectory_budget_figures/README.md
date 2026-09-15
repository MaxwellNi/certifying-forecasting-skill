# Learned-trajectory comparison figure

`fig_trajectory_comparison.pdf` is the general-width comparison figure, sized 7.1 by
2.6 inches. Ordinary text is 9 TeX points. IEEEtran supplies Nimbus Roman text
with Computer Modern math; every font is embedded Type 1. The PDF contains no
raster images. PNG and SVG files support inspection and reuse.

From the artifact root:

```sh
python scripts/figures/make_general_width_comparison.py
```

An alternate output directory can be supplied with `--output`; the script
resolves data and supporting paths relative to the artifact root. It needs
numpy, pandas, matplotlib, PyMuPDF, pdflatex, IEEEtran and PGF. The unchanged
`tex/underscore.sty` dependency is included with its author/copyright and
LPPL 1.2-or-later license notice because Matplotlib's PGF backend loads it.
Normal TeX search paths and caller-supplied `TEXINPUTS` remain active.

The figure reads the complete eight-candidate numerical table from
`../trajectory_budget/results/certificate_all_eight.csv`. Its exact source
copy is `trajectory_comparison_source.csv`; all 72 plotted point coordinates
are in `trajectory_comparison_coordinates.csv`. Every horizontal value is
exactly 1,000 times its unrounded source value. Every candidate uses the same
integer vertical row across panels; there is no jitter or broken axis.

The left panel connects each original shared score to its interaction-corrected
center and marks the exact archive target. The right panel compares shared
hybrid, pruned/reallocated shared hybrid, and pooled variance lower bounds at
the same 73,728 draw cost. A separately labeled near-zero scale shows pooled
lower bounds and exact targets. Shared/pruned lower bounds remain visible on
the full scale; they are outside the near-zero range.

The three pooled positive bounds use the variance implementation. All shared
and all pure-Hoeffding lower bounds are nonpositive. The census diamonds represent the
exact archive target. This is a post-exposure diagnostic of the corrected,
previously inspected archive, not an independent confirmation or a guarantee
of future forecast-loss improvement.

`TRAJECTORY_FIGURE_RECEIPT.json` records coordinate checks, input/generator/PDF
hashes, fonts, exact size and absence of off-page text. The rendered PNG was
visually inspected at 216 dpi: candidate rows, legends, signs, zero lines and
the labeled separate scale are legible, with no clipping or overlap.
