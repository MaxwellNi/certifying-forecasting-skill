# Forecast trajectories: design validity and prediction utility

This study connects a learned forecast to the whole-trajectory design identity.
The conditional population is the fixed 2013 Beijing PM2.5 archive. Independent
replacement index draws implement the theorem; consecutive days are not assumed
independent. Eight models trained on 2010–2012 are selected on 2013 and evaluated
on 2014. The target is fitted population-midrank association, not full conditional
covariance or a guarantee of future prediction improvement.

The original protocol specifies origin 14:00 and target 15:00. The first code
used target 14:00. Both executions are retained. The corrected target-15 run
was performed after the initial results were viewed; it is not a new untouched
confirmation. Read [execution corrections](EXECUTION_DEVIATIONS.md) before using
the results. Model settings, chronology, and budgets were unchanged.

At the corrected hour, the signed shared-reference gate falls back to persistence.
Its MSE is 288.32 versus 295.939 for ungated selection and both classical U gates.
The target-weighted MSE point difference is 7.619, or 2.57%. Its descriptive
paired four-week bootstrap interval is [-36.48, 51.38] in squared PM2.5 units.
The original equal-week estimate is 6.534 with interval [-35.02, 48.63].
Both include zero; neither establishes stable extra gating benefit, and the
gated prediction is identical to the persistence baseline. Full-U recombination is a subsequent stronger
comparison using every original draw. An exact census uses all 342 accessible
selection records and makes label-saving claims inappropriate.

## Reproduce

From this directory, with the repository's full requirements installed:

```sh
python prepare.py --output /tmp/trajectory-inputs --target-hour 15
python experiment.py --data /tmp/trajectory-inputs --output /tmp/trajectory-results
python strong_comparison.py --data /tmp/trajectory-inputs --recorded /tmp/trajectory-results --output /tmp/trajectory-pooled
```

Use target hour 14 to reproduce the initial cohort. Output directories must not
exist. The unchanged licensed source is under `data/`; stored predictions and
all 9,600 coverage rows per cohort are under `data/` and `recorded/`. Floating
point arrays are compared numerically; archive byte timestamps and wall-clock
execution times are not scientific outputs. Raw MSE weights individual records;
bootstrap intervals describe equally weighted weeks and do not establish finite
coverage for the physical time series. Timestamp-based feature availability
assumes previous-hour observations are available at the forecast origin.

See the [trajectory identity and kernels](../trajectory_design/DERIVATION.md).

## Certificate terms and matching loss intervals

[Independent numerical reconstruction](diagnostics/README.md) reproduces all
32 diagnostic cells and the final eight-candidate gates without importing the
producer kernels. It reports raw means, signed interactions, both uncertainty
terms, complete pooled centers, and target-weighted and equal-week loss
estimands separately. The additional weighted interval is a reporting check
on the corrected, already inspected evaluation, not a new confirmation.
