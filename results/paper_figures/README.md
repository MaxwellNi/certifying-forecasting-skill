# Paper figures

The diagram and numerical figures use IEEEtran's unmodified default Times text
and Computer Modern mathematical symbols. Text is embedded in vector PDFs;
labels use 9pt type at the stated paper inclusion size. Every numerical point
and interval comes from the recorded studies. Figure 3b uses no jitter, and
MovieLens comparisons concern cluster order.

From the repository root:

```sh
python scripts/figures/make_ieee_figures.py --package-root . --output /tmp/paper-figures
python scripts/figures/make_reference_design.py --output /tmp/paper-figures/reference_design.pdf --svg
```

Dependencies: matplotlib, pandas, NumPy, and a LaTeX installation with IEEEtran,
PGF, TikZ, geometry, Times text fonts, amsmath, amssymb, amsfonts and underscore.
The optional SVG export requires PyMuPDF; glyphs are outlined for portable
viewing. Temporary TeX/PGF files are generated in the chosen output directory
or a temporary build folder. No manuscript typesetting sources are included
in this code package.
