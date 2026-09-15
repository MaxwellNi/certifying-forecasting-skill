# Same observations, different fitting direction

The main-paper figure displays every retail and ratings model on identical
middle-fold observations. These two complete domains offer readable labels
and contrasting audit changes. The [all-domain figure](../publication_figure_assets/fig_public_directional.pdf)
and the accompanying CSV preserve all 41 models, including non-retained and
undefined results. No candidate family or numerical result has been changed.

Each line joins a model's complementary and directional statistic. The zero
line is a sign reference, not a significance threshold. The original bin
assignments, guards and uncertainty choices remain fixed. Ratings folds order
user clusters, not dates. These are sensitivity screens.

From the package root:

```sh
python scripts/figures/make_directional_model_figure.py \
  --input results/directional_comparison/model_comparison.csv \
  --output-dir /tmp/forecast-focused-figure --focus
```

Omit `--focus` to regenerate the all-domain version. The receipt records the
source values and undefined markers. Horizontal position directly represents
the statistic; coincident methods remain at the same coordinate.
