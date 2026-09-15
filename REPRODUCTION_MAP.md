# Paper, code and evidence

Run commands from the package root after installing `requirements-artifacts.txt`.
Use a fresh output directory. The main paper contains the core statements and
proofs; the technical companion holds expanded experiments and protocols.

| Main item | Recorded evidence | Reproduce or verify |
|---|---|---|
| Figure 1: categorical and whole-trajectory designs | `results/reader_figures/fig_reference_certificate.pdf` and `.svg`; Propositions 1 and 3, and Lemma 1 | The schematic identifies independent data roles, categorical validation metadata and the trajectory interaction. The complete worked command is in `examples/aggregate_reference_family/README.md`. |
| Figure 2: reference sampling and fitted means | `results/design_validation/peer_sampling.csv`; `results/learned_references/summary.csv` | `python scripts/figures/make_ieee_figures.py --output /tmp/paper-figures` (Figures 2–3 and the companion directional plot) |
| Figure 3: learning allowance and validation slack | All four eight-category budgets in `results/certificate_factorial/` and all 72 settings in `results/aggregate_bias/` | `python scripts/figures/make_ieee_figures.py --output /tmp/paper-figures` |
| Companion: all 20 retail and ratings models on unchanged middle folds | `results/directional_comparison/model_comparison.csv`; current plot in `results/reader_figures/fig_public_directional_focus.pdf` | `python scripts/figures/make_ieee_figures.py --output /tmp/paper-figures` |
| Figure 4: reference correction and same-budget lower bounds for all eight Beijing candidates | `results/shared_baseline_ranges/results/fixed_draw/certificate_all_eight.csv`; `results/shared_baseline_figures/` | `python scripts/figures/make_trajectory_comparison.py --output /tmp/trajectory-figure` |
| Table I: available information, sampling and interpretation | Main definitions, theorems and complete audit procedure | This is a logical scope map, not an experimental estimate or an automatic assumption check. |
| Companion: source coefficients and the overlap sign | `results/source_window_design/` and its exact finite-law checks | The displayed coefficients hold the marginal sum fixed while changing the overlap term; see the source-window README. |
| Table III: trajectory score, interaction and radii | `results/shared_baseline_ranges/results/fixed_draw/certificate_all_eight.csv` | `python results/shared_baseline_ranges/reproduce.py --output /tmp/shared-baseline-replay` reconstructs all 88 rows and compares the original and sharp support bounds on identical centers. The older 9,600 generic-width diagnostic rows remain under `results/trajectory_forecast/diagnostics/`. Bounds use unrounded values. |
| Companion: final sharp-width finite-law calibration | `results/shared_baseline_calibration/PROTOCOL.json`; all 86,400 candidate/procedure rows and 21,600 family rows | `python results/shared_baseline_calibration/study.py --verify /tmp/sharp-calibration-replay` regenerates all 3,600 prescribed finite-law replications and checks exact output bytes; `verify_results.py` independently recounts decisions. The separate `biased_null/experiment.py --verify FRESH_OUTPUT` plus `biased_null/independent_verify.py --results FRESH_OUTPUT --output FRESH_OUTPUT/CHECK.json` verify the additional 600 replications with zero target and positive shared-reference bias. The original and addendum protocols remain separate; neither establishes an independent application or exhaustive null calibration. |
| Companion: exact feedback expectations and paired null results | `results/equal_entity_directional/summary.csv`; `results/equal_entity_directional/verification.json` | `python scripts/analysis/equal_entity_directional.py --output /tmp/directional-study --repetitions 500 --seed 914273` |
| Companion: complete-family temporal comparison | `results/inference_validation/fod_comparison/` and `results/inference_validation/raw_family_study/` | `python scripts/analysis/verify_inference_studies.py --output /tmp/inference-check` checks stored family decisions, supplied-null calibration ranks and the matched classical comparison. Full regeneration uses the study commands below. |
| Companion: scalar methods on common draws | `results/canonical_beta2/replications.csv`; `results/canonical_beta2/summary.json` | `python reproduce_artifacts.py --output /tmp/paper-results` recounts every cell. Generator and canonical comparator dependencies are documented beside these results. |
| Estimated conditional means and ties | `results/learned_references/replications.csv`; protocol and 48-cell summary | `python scripts/analysis/learned_reference_study.py --output /tmp/learned-study --workers 4` |
| Complete correlated families, K = 10 and 11 | `results/inference_validation/raw_family_study/` | `python scripts/analysis/independent_entity_family.py --output /tmp/family-study` regenerates the independent calibration bank and evaluation families. |
| Matched classical FOD moment | `results/inference_validation/fod_comparison/` | `python scripts/analysis/fod_comparison.py --verify results/inference_validation/fod_comparison`; use `--output /tmp/fod-study` for full regeneration. |
| Near-copy controls and weak alternatives | `results/inference_validation/near_copy_weak_study/` | `python scripts/analysis/near_copy_adjustment.py --output /tmp/near-copy-study` retains the original settings and the documented exploratory extension. |
| DLinear worked mean/SE/T/p and full-family BY | `results/spline_panel/public_all_models.csv`; `results/directional_comparison/model_comparison.csv` | `python reproduce_artifacts.py --output /tmp/worked-results` checks aggregate arithmetic and full-family adjustments. The worked example uses complementary all-fold fitting and exponent two. |
| Retail identity changes and mean/SE decomposition | `results/directional_comparison/model_comparison.csv` | The same-support figure command above exports all displayed coordinates and underlying means, standard errors, support and fallbacks. |
| Restricted stored-output provenance | `ERRATA.md`; `results/restricted_temporal_aggregate/`; `results/design_validation/` | Only model and simulation aggregates are distributed. Aggregate replay does not reconstruct restricted observations or establish corrected retraining. |

The package-level `reproduce_artifacts.py` combines aggregate checks and figure
reconstruction. `scripts/analysis/verify_inference_studies.py --output /tmp/inference-check`
recounts the learned-reference, complete-family and near-copy studies, checks
normal/t tails, independently checks selected BY decisions and calibration ranks,
and verifies exact synthetic expectations. Analytic floating-point comparisons
allow library-level rounding differences; hypothesis decisions and integer counts
remain exact. The focused tests independently enumerate small tied laws and
literal reference products, and verify temporal supports and family calculations.

For the current paper font and layout, rebuild Figures 2–3 and the companion directional plot with
`python scripts/figures/make_ieee_figures.py --package-root . --output /tmp/paper-figures`.
Rebuild Figure 4 with `python scripts/figures/make_trajectory_comparison.py --output /tmp/trajectory-figure`.
The earlier figure commands above reproduce the same evidence in alternative layouts.
See `results/paper_figures/README.md` for TeX dependencies.

The final companion layout is rebuilt with
`python scripts/figures/make_companion_figures.py --output /tmp/companion-figures`.
The remaining historical comparison plots and complete 41-model table use
`python scripts/figures/make_public_display_figures.py --output /tmp/display-figures`.
Both use the IEEEtran default text and mathematical fonts. The earlier
study-specific commands retain their numerical checks and alternative layouts.

The older scalar/resolution, oracle-rank and original all-fold public figures
remain in `results/publication_figure_assets/` and the technical companion.
They use their own clearly identified configurations; their coordinates should
not be substituted for the companion same-middle-support comparison.

There are three different levels of reproducibility:

1. **Aggregate replay:** recount released numbers and redraw figures without provider data.
2. **Synthetic regeneration:** rerun the fully specified mechanisms from independent random streams.
3. **Forecast retraining:** obtain provider inputs and follow the documented model workflows. This is separate from the default replay; restricted original training is not publicly reproducible.

The package includes the licensed Seoul source archive and the derived
electricity archive used by the finite-archive study. Original observations for
the four-domain model comparisons and all restricted observations are excluded.
See `DATA_ACCESS.md` for study-specific access and reuse conditions, and
`MANIFEST.json` for the exact distributed files and hashes.

## Reference variance, finite bounds and held-out loss

| Main result | Code and evidence | Verification scope |
|---|---|---|
| Theorem 1, full covariance-null projection | `results/reference_covariance/reference_covariance_checks.py` and `results.json` | Rebuilds exact finite-support projection moments and six simulation cells. Run on a copy to preserve recorded results. |
| Proposition 3, observable categorical bound | `scripts/analysis/categorical_reference_audit.py`; `methods/categorical_references.md` | General numeric scores, ties, category validation, and complete observed bound. Input checks do not establish iid sampling. |
| Both categorical simulation studies | `scripts/analysis/verify_categorical_studies.py --base results/category_reference --output /tmp/categorical-checks.json` | Rebuilds all training/validation primitives, all 216,000 stored-row calculations and 20 complete evaluation replications. It does not regenerate all evaluation draws. |
| Companion: rental held-out gains | `results/seoul_confirmation/README.md` | Source-data refitting is separate from the fast aggregate, frozen-prediction and timing verifiers. |
| Companion: retail fit-change decomposition | `results/retail_fit_changes/README.md` | Two models, five resolutions, 17 weeks and complete covariance; provider prediction cohorts are separate inputs. |

The categorical study producers use a Bernoulli-specific simulation shortcut.
The public CSV certificate instead uses the general sorted comparison kernel
in `scripts/analysis/peer_rank_products.py`, checked against literal ordered
triples including ties. Synthetic targets and ideal conditional means are
used only for study diagnostics; the executable certificate does not receive
them.

## Signed and variance-sensitive reference certificates

| Item | Recorded evidence | Command and scope |
|---|---|---|
| Signed categorical bias and full-U variance penalty | `results/reference_certificate_efficiency/DERIVATION.md` | `scripts/analysis/categorical_reference_audit.py --certificate signed_range` or `--certificate signed_variance`; rule and input order fixed before evaluation. The default remains `absolute_range`. |
| Five matched synthetic methods | `results/reference_certificate_efficiency/replications.csv.gz` and 500-cell `summary.csv` | `python scripts/analysis/reference_certificate_efficiency_study.py --output /tmp/reference-efficiency --replications 1000 --workers 4` regenerates all 500,000 rows. All training, validation and evaluation observations count toward cost. |
| Companion: five-method same-budget efficiency | The complete matched-method summary above | `python scripts/figures/make_efficiency_figure.py --summary results/reference_certificate_efficiency/summary.csv --output-dir /tmp/efficiency-figure` writes `fig_certificate_efficiency.pdf`, `.svg`, `.png`, and the complete plotted records in `.json`. Distinct markers identify all five methods; coincident values are plotted without offsets. |
| Independent implementation check | Portable `scripts/analysis/verify_reference_certificate_efficiency.py` | `python scripts/analysis/verify_reference_certificate_efficiency.py --output /tmp/efficiency-check` checks every stored row/cell and regenerates eight complete primitive replications (4,000 rows), using no producer imports. |
| Hardened independent-triple API | `protocol_original.json`, `protocol_hardened_replay.json`, `provenance.json` beside the results | Both complete historical runs have identical compressed result and summary bytes. The portable code rejects an independent-triple mean inconsistent with those same triples. The permissive historical helper is not distributed. |
| Prespecified family failure budget | `family_validation_delta` in `scripts/analysis/reference_certificate_efficiency.py` | `--family-size K` reserves rho of the first BY threshold; it does not apply BY or establish optimal allocation. Stored study results retain their original delta. |
| Fixed public archive application | `results/public_archive_certificate/README.md` | `python results/public_archive_certificate/verify.py` checks the sealed archive, independent replacement draws, all 21 certificate rows and census comparisons without writing or downloading. Full source reconstruction is separate. |
| Seoul gate decision and opportunity loss | `results/seoul_gate_diagnosis/README.md`, all six CSVs | Read-only aggregate verification: `python scripts/analysis/verify_seoul_gate_tables.py`. Full replay: `python results/seoul_gate_diagnosis/diagnose_gate.py --study-dir results/seoul_confirmation` using that study's separately pinned environment. This is exposed-data diagnosis, not a newly selected or confirmed gate. |

The default artifact reproduction runs the study-specific independent checks
in addition to every retained aggregate check. Neither a source hash nor a complete
arithmetic replay establishes optimality, absence of earlier exploration, or
general forecasting utility.

## Factorial attribution, joint information and chronological confirmation

| Main item | Evidence | Reproduce or verify |
|---|---|---|
| Section III: target definitions | Declared targets and recurrent error terms, defined where used | Definitions, not experimental estimates |
| Figure 3(a): matched learning comparison at all four recorded budgets | `results/certificate_factorial/`, all 400,000 factorial rows | `python results/certificate_factorial/verify.py --output /tmp/factorial-check`; redraw with `python scripts/figures/make_ieee_figures.py --output /tmp/paper-figures` |
| Main comparison and companion table: matched bounded-mean betting | `results/matched_betting/`, all 48,000 method rows | `python results/matched_betting/verify.py`; full regeneration uses its `reproduce.py --output /tmp/betting-replay` |
| Companion: joint rank-mean allowance | `results/joint_bias/`, exact dual certificates and all 32 inspected candidates | `python results/joint_bias/verify.py --output /tmp/joint-check`; full replay uses `run.py --output /tmp/joint-replay` |
| Table II: all primary Ridge confirmation rules | `results/forecast_confirmation/confirmation_results.csv`, with source ZIPs, forecast arrays and sampling indices | `python results/forecast_confirmation/verify.py --output /tmp/confirmation-check`; source retraining uses `reproduce.py --output /tmp/confirmation-retraining` |
| Companion: all 32 rules and four temporal loss bounds | Complete candidate, selected-loss, weekly and conditional-gain tables in the same directory | The confirmation verifier reconstructs every record. The bounds concern the same-period conditional average, not a future population or raw-unit risk. |

The factorial and joint studies inspect already observed simulation or archive
results. Their protocols and original hashes are distinguished from portable
code adaptations. The two public-task rules were fixed before source download;
the mechanically corrected computation is disclosed and is not relabelled as
a newly preregistered experiment. The new full retraining uses provider ZIPs
and does not deserialize the original scikit-learn 1.6 models.

## Aggregate learning-bias validation

[results/aggregate_bias/](results/aggregate_bias/README.md) contains the known-category-mass bound, all 32 exposed-archive candidates under four allowance choices (including the exact-bias diagnostic), with the same 32,768 sampled observations, and 144,000 validation simulation replications. Its default reproduction rebuilds 128 diagnostic rows and independently checks both U/betting BY decisions; the full simulation uses `--regenerate-simulation`. The original chronological forecasting confirmation is unchanged.

Lemma 1 and Equations (11)–(12) define the aggregate allowance. A minimal
synthetic independent-pair example is available with
`python examples/aggregate_validation.py`. It computes a learning-bias bound,
not a complete association certificate or a forecast-retention decision.

Figure 3(b) displays all 72 validation settings from `results/aggregate_bias/`.
Run `python scripts/figures/make_ieee_figures.py --output /tmp/paper-figures`
to regenerate the vector figure and every plotted coordinate. It displays
absolute median slack; the earlier ratio plot remains in `results/aggregate_bias/`. The complete
older three-panel factorial figure remains in the companion and public files.


## Complete reference analysis and matched direct comparisons

- `examples/aggregate_reference_family/README.md`: exact-mass input format,
  independent samples, preselected sampling bounds, full-family BY and costs.
  `python examples/aggregate_reference_family/test_example.py` checks this
  executable composition, including exact numeric order preservation.
- `results/pooled_bias/README.md`: fully averaged second-order validation,
  all retrospective same-budget comparisons and independent formula replay.
- `results/direct_target/README.md`: direct four-observation target identity,
  a collision-aware quadratic implementation and two conservative calibrations.
  Neither calibration retains an inspected archive candidate. Their outcome
  does not establish a general advantage over direct-target approaches.
- `results/reader_figures/fig_public_directional_focus.pdf`: companion
  plot of all 20 retail and ratings models. `results/focused_directional/` preserves
  the earlier rendering; the full 41-model data and original figure remain available.

The two comparator directories are part of the package reproduction command.
They preserve the original confirmation results. Their archive comparisons
were developed after inspecting those tasks and are not new confirmations.

## Final target and independent data streams

- [Inference module](results/target_precision/README.md): exact full-U identities,
  deletion variance, stratified reference bounds, classical profiled betting,
  all 8,000 binary repetitions and the source-to-prediction news reproduction.
- [Reader figures](results/reader_figures/README.md): all four existing ablation
  budgets with pointwise exact intervals, and all 72 absolute slack pairs.
- [Reference-design checks](results/reference_design/README.md): independent exact
  enumeration of the cross-time rank example. The weighted-reference identity
  and proof are in the technical companion.

The earlier temporal-design timeline and the complete 41-model plot remain in
the companion. `python scripts/analysis/panel_design_checks.py` checks
the temporal identities; `scripts/analysis/independent_entity_audit.py`
runs that procedure on supplied data. Omit `--focus` from the model-figure
command to plot the full model roster. These are separate from main Figure 1.

## Reference reuse across source times

[Source-time composition](results/reference_time_composition/README.md) contains
the exact overlap identity, heterogeneous-peer qualifications, exhaustive raw-array
checks, a finite-sample audit and all 4,800 generated design repetitions.
The main paper contains the general trajectory characterization and proof; the
companion contains the scalar source-time lemma and extensions. These simulations test a developed mechanism, not unseen forecast utility.

## Source windows and nonlinear ranks

The [source-window guide](results/source_window_design/README.md) implements
the companion source-window lemma, including ranks of equal two-lag sums, exact
design coefficients, finite-law constants and independent whole-panel bounds.
Its two independent enumerations check raw arrays rather than replacing
forecasts by sums of their ranks. The unequal-lag counterexample is retained.

## Learned trajectories

The [trajectory design](results/trajectory_design/README.md) contains exact identities,
finite bounds, and the complete pooled-U comparator. The [forecast study](results/trajectory_forecast/README.md)
includes raw public data, retraining commands, all coverage rows, corrected
hour alignment, subsequent strong comparisons, and paired loss uncertainty.

## Reference contracts and trajectory evidence

`results/trajectory_design/design_contract.py` compiles declared reference roles
and verifies map, law and row-sum preservation. Forecast availability checks
validate supplied timestamps, not factual independence or publication latency.
`fixed_delta_inference.py` inverts the stated range-only bound with upward
rounding and fixed validation allocation. It does not invert the separate hybrid
bound. Run unittest discovery in that folder for all 43 tests.

`python results/trajectory_forecast/diagnostics/reproduce.py --output /tmp/trajectory-diagnostics`
independently reconstructs all 9,600 diagnostic rows, eight final gates and
matching paired loss intervals from bundled arrays. Complete source retraining
uses the study preparation command documented in its README.

Recorded protocol documents retain their experiment-local table labels; the mapping above uses the current manuscript numbering.

The shared-baseline range theorem and exact endpoint checks are in
`results/shared_baseline_ranges/proof.md` and `exact_checks.py`. The primary
trajectory figure uses its fixed-draw structural bounds; the separate
general-width plot and every original numeric result remain available.
