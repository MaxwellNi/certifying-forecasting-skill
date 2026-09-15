# Certifying forecasting skill

A rank-based forecast comparison can contain association created by its own
reference observations. This package identifies which reference assignments
preserve the association between specified rank contrasts, measures the
interaction from sharing references, and provides conditional finite-sample
bounds. Complete U-statistics estimate the same target directly. The bounds
require independently sampled whole trajectories after training fixes the
maps; dependence within a trajectory is allowed.

Read the [paper](paper.pdf), *When a High Score Is an Illusion: Certifying
Genuine versus Repackaged Forecasting Skill*, or start with the
[method-to-evidence map](REPRODUCTION_MAP.md). The
[repository](https://github.com/MaxwellNi/certifying-forecasting-skill)
also includes [additional results and reproduction details](technical-details.pdf).

Start with the [three-value example](examples/reference_rank/README.md) for an exact,
executable explanation of how reference assignments change a rank score. It also
shows when baseline subtraction cancels the interaction.

## Choose the appropriate analysis

| Your question | Entry point | What the output establishes |
|---|---|---|
| Are whole trajectories independently sampled after maps and weights are fixed? | [Reference design and numerical inversion](results/trajectory_design/README.md) | A specified fitted rank target, reference interaction and conditional finite bound; declared time and source contracts are checked separately |
| Does a forecast add residual association under independent categorical sampling? | [Complete-family command and worked example](examples/aggregate_reference_family/README.md) | A signed learning allowance, full-U score, lower bound and complete-family BY decision, under the stated independent sampling design and exact category masses |
| Are exact category probabilities available without forecast or outcome ranks? | [Aggregate validation](results/aggregate_bias/README.md) | A finite-sample upper bound for total learning bias, using independent validation streams and the stated category metadata |
| Can known marginal rank means improve the learning allowance? | [Joint rank-mean bound](results/joint_bias/README.md) | A conservative upper bias certificate using the same validation event |
| How do audit choices change an observed panel comparison? | [Panel command](methods/panel_audit.md) and the example below | Target-labelled statistics, abstentions and full-family decisions; a screen unless its assumptions are established |
| Does selecting an augmentation improve subsequent predictions? | [Chronological forecast study](results/forecast_confirmation/README.md) | Fixed-model and fixed-rule confirmation losses, with costs and dependence-sensitive uncertainty stated separately |

An association certificate and a future prediction gain answer different
questions. Every reported family includes copies, undefined scores and
non-retained candidates. The [data guide](DATA_ACCESS.md) states which sources
can be redistributed. The [correction note](ERRATA.md) records the identity
and target-unit qualifications for the restricted historical results.

## Three questions to answer

1. **Where can the apparent association come from?** Shared rank references,
   fitting errors and temporal feedback have distinct contributions. The paper
   states which sampling design permits each correction.
2. **What does the certificate establish?** The trajectory branch targets the
   specified fitted population-rank contrast. The category branch targets residual
   association beyond the declared categories. Each requires its stated sampling
   design; complete baseline adjustment and future forecast improvement are
   separate claims.
3. **Which rule should I use?** Choose the information and sampling design before
   evaluation. Compare shared references with complete pooled U when the same
   whole-trajectory draws can be recombined for the same fitted target. Unknown
   category masses use rectangle bounds; exact masses permit aggregate validation.
   Classical betting is a strong alternative. Complete labeled archives permit
   direct census.

## Learned forecasts and whole trajectories

Start with the [trajectory design](results/trajectory_design/README.md) and its
executable contracts. A matrix identifies possible reference interaction;
a nonzero matrix does not by itself establish bias for the supplied maps and
law. An observable kernel estimates the signed contribution. Preserve the
maps, reference law and coefficient row sums when comparing reference assignments.
Confidence bounds require
conditionally IID whole-trajectory draws with maps and coefficients fixed by
independent training. When all sampled trajectories can be recombined, use the
complete pooled U-statistic as a direct comparator for the same fitted target.
The [Beijing study](results/trajectory_forecast/README.md) supplies source data,
full retraining, both documented hour cohorts, and complete-U and census
comparisons. Prediction improvement and extra audit-gating benefit are reported
separately. The [certificate decomposition and loss intervals](results/trajectory_forecast/diagnostics/README.md)
show every candidate, all 32 diagnostic cells and both target-weighted and
equal-week loss differences. Stable extra gating benefit is not established.

The [shared-baseline study](results/shared_baseline_ranges/README.md) uses the
known common baseline to sharpen both shared-reference and U bounds. On the
same recorded draws, Boosting 31's allocated hybrid radius falls from 0.01213
to 0.00959 (20.9%). Shared rules certify none of eight candidates; complete U
certifies four instead of three. Every rule selects the same predictor as
before. These are inspected-data comparisons, with no new confirmation or
extra prediction gain. The [figure](results/shared_baseline_figures/README.md)
shows all eight candidates, both score allocations and exact archive targets.

The [finite-law calibration](results/shared_baseline_calibration/README.md)
checks the final sharp-width rules under a protocol frozen before simulation.
Six laws, two budgets and four fixed maps give 3,600 replications. Every
law/budget/procedure cell has 0/300 family noncoverage events; its pointwise
95% upper limit is 1.222%, not zero. At 12,288 draws, both direct U variants
certify all weak and near-degenerate positive cases, while shared correction
certifies none. The original zero-target cases have zero reference interaction. A separately
frozen [biased-null check](results/shared_baseline_calibration/biased_null/README.md)
adds a nonidentity zero target with positive expected interaction 1/32; all
12 additional procedure cells have 0/300 false certifications and family
noncoverage events. These checks do not establish exhaustive null calibration
or independent application benefit.

The [general-width comparison](results/trajectory_budget/README.md) preserves
the original certificates, allocation and full future-loss comparisons.
Removing an unused role and reallocating the same 73,728 draw positions lowers
the generic shared hybrid radius by 23.0%; this is separate from the 20.9%
structural-width improvement. The shared selector returns persistence (future
MSE 288.320), compared with 295.939 for direct-loss selection. Its descriptive
MSE-gain interval is [-36.480, 51.381]; gain over choosing persistence directly
is zero.

The [three-draw explanation](results/trajectory_design/estimability.md) gives the
unknown-law information restriction under which unbiased correction requires
three draws, with lower-order exceptions. It is an estimability statement,
not a claim of optimal confidence bounds or prediction gains.

The trajectory p-value API inverts the pure-Hoeffding bound with a fixed
validation allocation. Hybrid lower bounds are evaluated at their declared
fixed levels; the API does not provide a hybrid p-value inversion.

The [paper figures](results/paper_figures/README.md) use the manuscript's actual
LaTeX text and mathematical fonts. Every recorded point and interval is preserved.

## Independent categorical samples

The [independent-data diagram and numerical figures](results/reader_figures/README.md)
show the validation and evaluation roles, every recorded ablation budget and
absolute validation slack. With exact category masses, the aggregate bound
estimates total learning bias directly. The complete-family interface requires
both an explicit `--exact-category-masses` declaration and a nonempty
`--mass-source`; its Python API requires `exact_masses_declared=True`.
Estimated frequencies do not satisfy the exact-mass premise. The declaration
and source description record the caller's assertion, without verifying it.

The [final-target module](results/target_precision/README.md) provides the complete
variance and betting comparisons, including the rare perfectly correlated case,
and a source-to-prediction news-study reproduction. In that study, the
stratified hybrid, aggregate betting, pooled-reference betting, profiled betting
and direct-loss rules choose the same augmentation as ungated selection. Its
primary clipped loss improves over Ridge by 0.158%, while untransformed MSE
worsens by 12.82%. The other four gates return the baseline. All outcomes are retained.

## Follow the original observations

Different fitting rows can reuse the same earlier outcome. The
[source-window guide](results/source_window_design/README.md) traces each
comparison through its raw source times and reference entities. It handles
lag copies and ranks of equal two-lag sums, with an exact distinction between
the population target and additional reference interaction. The guide includes
executable JSON designs, tied-outcome examples and explicit unsupported cases.
The [one-lag derivation and experiments](results/reference_time_composition/README.md)
remain available in full. These design checks assume fixed coefficients and
independent common-law raw sources; observational-panel inference is a separate
requirement.

## Install and run

Use Python 3.12 in a fresh environment. Rebuilding the final paper figures also
requires a LaTeX installation with IEEEtran, TikZ, PGF, the standard Times text
fonts, AMS mathematical packages and the `underscore` package.
`--skip-figures` omits typesetting.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-artifacts-lock.txt
python reproduce_artifacts.py --output /tmp/forecasting-skill-results
```

This command verifies recorded arithmetic, designated primitive replays and
figure generation. It does not retrain the original 41 predictors. Use
`--skip-figures` for arithmetic checks alone. Outputs go to a new directory;
the distributed evidence is preserved. The lock file records all installed packages in the checked Python 3.12 Linux
CPU environment. `requirements-artifacts.txt` lists the direct dependencies.
This smaller installation supports aggregate replay and the trajectory tests.
The full root test suite also imports forecasting models; install
`requirements-lock-py312-linux-cpu.txt` before running that suite, as shown below.

To check the trajectory contracts and independently reconstruct the same-budget
comparison from its bundled draws and predictions:

```sh
python results/trajectory_design/contract_exact_checks.py
python results/trajectory_budget/reproduce.py --output /tmp/trajectory-budget-replay
python results/shared_baseline_ranges/reproduce.py --output /tmp/shared-baseline-replay
python results/shared_baseline_calibration/study.py --verify /tmp/sharp-calibration-replay
```

The second command writes all eight general-width candidate certificates,
decisions, costs and future-loss comparisons. The third sharpens known support
widths on the same draws, preserves the centers and verifies the decision changes. It uses the fixed recorded
forecasts; the Beijing study documents the separate source-to-model retraining.

To regenerate the final companion figures, historical comparison plots and
complete 41-model table in the same IEEE fonts:

```sh
python scripts/figures/make_companion_figures.py --output /tmp/companion-figures
python scripts/figures/make_public_display_figures.py --output /tmp/display-figures
python scripts/figures/make_trajectory_comparison.py --output /tmp/trajectory-comparison-figures
```

These commands preserve the recorded observations and write vector PDFs,
display coordinates and checks to the specified output directories.

To inspect a supplied panel:

```sh
python scripts/analysis/forecast_audit_cli.py \
  --input examples/forecast_audit_cli/synthetic_forecasts.csv \
  --output-dir /tmp/forecasting-skill-example \
  --frequency cluster --lag 0 --beta 2 --ladder 2,3,4
```

Run `python scripts/analysis/forecast_audit_cli.py --help` for the input columns
and analysis arguments. Each output identifies its target, represented controls,
sampling assumptions and validity status. The command does not infer those
assumptions from data. The worked example is synthetic and outcome-free copy
checks do not calibrate an otherwise invalid p-value.

For the categorical reference example:

```sh
python scripts/analysis/categorical_reference_audit.py \
  --fits examples/categorical_reference/fits.csv \
  --validation examples/categorical_reference/validation.csv \
  --evaluation examples/categorical_reference/evaluation.csv \
  --certificate signed_variance \
  --output /tmp/reference-certificate.json
```

Choose the rule before evaluation. The small example illustrates the input
format and returns no positive lower bound. For exact category probabilities,
the [complete-family example](examples/aggregate_reference_family/README.md) joins
validation, the full-U score and BY in one command. It documents every input
column and counts training, validation and evaluation observations. A separate
`python examples/aggregate_validation.py` example isolates the bias allowance.
Choose the sampling bound and family before viewing evaluation results.

## Reproduce the principal comparisons

| Evidence | Files and commands | Interpretation |
|---|---|---|
| Learned maps and whole-trajectory references | [Reference design](results/trajectory_design/README.md), [Beijing source-to-forecast study](results/trajectory_forecast/README.md), [same-budget diagnostic](results/trajectory_budget/README.md) | Target-preserving reference changes and signed interaction correction, with all eight candidates, matched draw costs and subsequent loss. Shared certificates retain none; the positive pooled bounds use the variance implementation. |
| Reference correction with learned controls | [48-cell study](results/learned_references/README.md) | At 8,192 training rows, 400 groups and 64 observations per group, matched null rejections change from 402 to 41 of 1,000. Small training samples remain difficult. |
| Learning allowance versus sampling precision | [Four-method factorial study](results/certificate_factorial/README.md) | At eight categories and 30,976 observations, signed versus absolute variance bounds detect 435 versus 274 of 1,000; the paired pointwise 95% gain interval is [0.131, 0.189]. |
| Strong classical comparators | [Five-method comparison](results/reference_certificate_efficiency/README.md), [bounded-mean betting](results/matched_betting/README.md) | Identical primitives, full costs and all outcomes. The proposed certificate does not dominate the classical alternatives. |
| Aggregate learning bias | [Validation and matched comparisons](results/aggregate_bias/README.md) | In a post-exposure diagnostic on 32 existing candidates, matched known-probability rectangle and aggregate betting rules retain 3 and 13 candidates across four separate families. A 72-setting simulation checks bound coverage and width; 10 of 72 settings have smaller median slack under the rectangle. |
| Is aggregate improvement unique to splitting? | [Fully averaged second-order comparison](results/pooled_bias/README.md) | With the same exact masses, fits and 32,768 raw draws, the fixed-tuning pooled comparator matches 9 U and 13 betting retentions. These are retrospective diagnostics. |
| Can the final rank target be estimated directly? | [Four-observation target comparator](results/direct_target/README.md) | For iid draws with known category masses: exact unbiased identity and collision-aware computation, with ties. Two conservative bounds retain 0/32 inspected candidates; this does not establish superiority over every direct-target method. |
| Joint rank-mean information | [Bias-bound study](results/joint_bias/README.md) | All 32 inspected allowances tighten, by median 1.24%; no reference decision changes. The LP gives an outer bound, not generally the attainable maximum. |
| Exact archive certification | [Electricity archive](results/public_archive_certificate/README.md) | A positive lower bound for a fixed archive and four declared categories; all seven matched methods retain the forecast. |
| Two forecast confirmations | [Appliances and Metro](results/forecast_confirmation/README.md) | Ridge augmentation reduces bounded loss by 16.95% and 73.87%; known-forecast gates equal ungated selection. The reference gates fall back to baseline. |
| Temporal feedback and classical moments | [Independent-entity study](results/equal_entity_directional/README.md), [full-family comparison](results/inference_validation/fod_comparison/README.md) | Exact raw-score bias identities, separate inference conditions, paired calibration and power |
| All 41 public predictors | [Focused reader figure](results/reader_figures/fig_public_directional_focus.pdf), [all-domain figure](results/publication_figure_assets/fig_public_directional.pdf), [model roster](results/public_score_link/public_model_roster.md) | Every model, with audit changes on fixed observations; sensitivity does not establish calibrated discoveries. |
| Retail changes and rental decisions | [Fit-change decomposition](results/retail_fit_changes/README.md), [Seoul confirmation](results/seoul_confirmation/README.md), [gate diagnosis](results/seoul_gate_diagnosis/README.md) | Exact observed decomposition, subsequent loss and all selected or excluded candidates |

Retrain the two new public forecast studies from their bundled source ZIPs:

```sh
python results/forecast_confirmation/reproduce.py --output /tmp/forecast-retraining
```

It fits models in the installed environment, checks all 78 saved arrays and
all non-runtime scientific table fields, and invokes an independent verifier.
Allow up to 40 minutes per phase. The separate Seoul study uses its own pinned
model environment; follow its README or pass `--confirmation-python` to the
package reproduction command. No incompatible serialized models from the two
new tasks need to be loaded.

Synthetic regeneration and study-specific verification commands are in
[REPRODUCTION_MAP.md](REPRODUCTION_MAP.md). All distributed unit suites can
be run from the repository root:

```sh
python -m pip install -r requirements-lock-py312-linux-cpu.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest discover -s results/public_archive_certificate -p 'test_*.py' -v
python -m unittest discover -s results/forecast_confirmation -p 'test_*.py' -v
python -m unittest discover -s examples/aggregate_reference_family -p 'test_*.py' -v
python -m unittest discover -s results/target_precision/confirmation/tests -p 'test_*.py' -v
python -m unittest discover -s results/reference_time_composition -p 'test_*.py' -v
python -m unittest discover -s results/source_window_design -p 'test_*.py' -v
python -m unittest discover -s results/trajectory_design -p 'test_*.py' -v
python -m unittest discover -s results/trajectory_budget -p 'test_*.py' -v
python -m unittest discover -s results/shared_baseline_ranges -p 'test_*.py' -v
python -m unittest discover -s results/shared_baseline_calibration -p 'test_*.py' -v
```

The recorded run passes 304 distinct tests across these suites in an installed
Python 3.12.11 environment. It covers all distributed `test_*.py` files, including
eight exact-mass family tests, five general-width budget tests and nine
shared-baseline support tests and eight finite-law calibration tests. These tests
check the implemented calculations and contracts; sampling assumptions and
independent source-to-model retraining have their own evidence requirements.

## Contents and scope

`results/` contains all declared cells, including adverse outcomes and clearly
identified exploratory extensions. `scripts/analysis/` contains audit and
forecasting implementations; `scripts/figures/` contains deterministic plotting
code. `data_reference/` records provider sources and checksums. `MANIFEST.json`
identifies every distributed file. The paper and companion are supplied as
PDFs; document typesetting sources are separate from this code package.

The four original application panels are distributed as aggregates. The
Appliances, Metro and Seoul studies include their attributed public source ZIPs
under CC BY 4.0; the separate electricity study includes an attributed derived
archive. Restricted trading files contain only model-level and simulation
aggregates. Private observations, identifiers, predictions, residuals, source
loaders and model weights are excluded. Aggregate replay cannot reconstruct
restricted training or establish historical data and checkpoint availability.

Locally fixed protocols are distinguished from exploratory analyses and from
external preregistration. Hashes establish byte identity, not unobserved research
history. Forecast confirmation is distinct from fixed-archive inference; no
label-acquisition saving is established by a fully materialized archive. The
software uses the [MIT license](LICENSE); data and dependencies retain their
own terms.
