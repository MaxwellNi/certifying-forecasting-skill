# Follow a forecast back to its original observations

Different rows can contain the same information. A forecast recorded on Tuesday
may reuse Monday's outcome, which also appears in a fitted baseline. Splitting
rows or changing reference products within each day does not necessarily remove
that reuse.

The paper gives an exact expectation criterion for two forecast classes: a copy
of one historical observation and the rank of an equally weighted sum of two
historical observations. Comparisons may contain ties. The fits and reference
weights are fixed, and the raw source observations are independent with a common
distribution. The result follows the original source time and reference entity,
including their reuse through fitted means.

## What the two coefficients mean

`T` describes the association already present with population references.
`Omega` describes the additional association from sharing finite references.
Both are signed sums; they are not counts of overlapping observations.

For two-source sums, the expected score is

$$E[AB]=\kappa_F T+\lambda_F\Omega.$$

The constants are given in the main paper and the [complete derivation](DERIVATION.md).
For Bernoulli outcomes with probability one half, both constants equal `3/64`.
The original forecast values are added **before** ranking.

| Design | T | Omega | Exact mean under Bernoulli(1/2) |
| --- | ---: | ---: | ---: |
| Two-source sum, positive reference interaction | 0 | 1/2 | 3/128 |
| Two-source sum, negative reference interaction | 0 | -1/2 | -3/128 |
| Two-source sum, cancelling reference interaction | 0 | 0 | 0 |
| Lag copy, peers split only by calendar row | -1 | -1 | -1/8 |
| Lag copy, peers split consistently across raw sources | -1 | 0 | -1/16 |

The last two rows are useful together: source-aligned separation removes the
extra reference term but preserves the nonzero population target. It does not
make the whole score centered, create independent entity scores, or establish
calibration for an observed forecasting panel.

## Run a source-design check

From the package root:

```sh
python results/source_window_design/source_design.py \
  --design results/source_window_design/examples/sum_positive_interaction.json \
  --output /tmp/source-design-result.json
```

Use a new output path. The JSON result records every matched source window and
peer product, the exact rational coefficients, and coefficient norms for a
bounded-mean calculation. Identical source terms are combined before computing
the norms. Source and peer IDs refer to raw observations, even when those
observations appear in several calendar rows.

The input has exactly six fields: `focal`, `evaluation`, `forecast_fit`,
`outcome_fit`, `forecast_rows`, and `outcome_rows`. Row maps contain `sources`
(an array of raw source IDs) and `peers` (reference IDs with nonnegative weights
summing to one). Fitting maps give fixed row weights summing to one. Rational
strings such as `"1/2"` avoid rounding. An empty forecast source array denotes a
common constant forecast whose centered comparison is zero.

The checker rejects unsupported fields, mixed one/two-source forecast classes,
unequal-weight declarations, normalization declarations, focal references, and
malformed source arrays. The input does not fit a model or infer independence.
The analyst must justify fixed weights and the common independent source law.
The criterion does not cover arbitrary learned regressions or random tied-rank
standardization. Uniform complete-peer ranks with fixed size and no ties have
a deterministic scale; this special case is derived in the paper.

## Reproduce the exact calculations

```sh
python results/source_window_design/check_source_design.py \
  --output /tmp/source-design-enumeration
python results/source_window_design/check_two_lag.py \
  --output /tmp/two-lag-enumeration.json
python -m unittest discover -s results/source_window_design -p 'test_*.py' -v
```

The first checker enumerates 99,328 raw binary panels and computes fitted ranks
from the declared rows. It repeats the calculation after a large integer affine
transformation to check that rounding does not create ties. A separate checker
uses exact rational arithmetic on 183,523 raw temporal arrays, five discrete
laws, all entries of selected cross-entity covariance matrices, and the
six-observation validation kernel. It also demonstrates why unequal lag weights
cannot be reduced to an unweighted overlap scalar.

These are exact design checks, not new forecasting tasks or empirical power
comparisons. [two_lag_bound.py](two_lag_bound.py) supplies a classical one-sided
bound for the population-reference temporal score from independent whole
panels. Optional fresh validation uses six raw observations per replicate to
estimate the reference constant. With no validation, its known range provides
a conservative alternative. Count every raw panel observation and every
validation observation; shared entities within a panel are not additional
independent replicates. Gaussian rank moments and bounded-mean concentration
are classical ingredients, not new claims of this implementation.
