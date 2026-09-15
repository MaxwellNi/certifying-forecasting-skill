# Aggregate reference certificate for a complete model family

This CLI combines the split aggregate learning-bias allowance,
the existing full-U reference score and range or variance sampling bound,
and Benjamini--Yekutieli adjustment over every declared model. It accepts
exact category probabilities, independently trained fits, raw validation
pairs, and independent evaluation groups. It is a runnable composition of
the existing methods; it does not use the later pooled comparator.

Run from the public artifact root. The example uses the NumPy, pandas and
SciPy dependencies already supplied by `requirements-core.txt`.

```sh
python examples/aggregate_reference_family/generate_demo.py --output /tmp/aggregate-family-input
python examples/aggregate_reference_family.py \
  --masses /tmp/aggregate-family-input/masses.csv \
  --fits /tmp/aggregate-family-input/fits.csv \
  --validation /tmp/aggregate-family-input/validation.csv \
  --evaluation /tmp/aggregate-family-input/evaluation.csv \
  --training-calls 512 \
  --mass-source 'Exact two-category Bernoulli sampling design' \
  --exact-category-masses \
  --family-level 0.05 --validation-fraction 0.1 \
  --sampling-bound range --output /tmp/aggregate-family-output
```

The CLI requires `--exact-category-masses` and a nonempty `--mass-source`.
Python callers must pass the literal boolean `exact_masses_declared=True`.
Missing, false, unknown, string, or numeric declarations are rejected.
This is an intentional interface change: earlier callers must add the
explicit declaration. Observed validation frequencies are estimates and
cannot justify it; use a method that budgets mass uncertainty instead.
The receipt records `exact_masses_declared: true` while retaining
`exact_mass_provenance_verified: false`: an assertion is not verification.

Both commands require new output directories. They preserve existing outputs.
The generator uses a fixed seed, learns category means from 512 separate
training draws, then generates 4,096 new validation pairs and two independent
evaluation groups of 3,072 rows. Category masses are exactly `(0.5,0.5)` by
design. They are supplied independently of the observed category counts.

The three declared synthetic models are a channel coupled to the outcome,
a conditionally independent channel, and a constant channel. The saved
`expected/results.csv` retains the coupled member; the other two have
p-value one. All three remain in BY. This is a synthetic wiring demonstration,
not a new power study, forecasting confirmation, or method-selection exercise.
`expected/receipt.json` retains inputs, helper/source hashes, allocations and
raw-call accounting. Generate the inputs with the documented default seed
to compare them with those hashes.

## Input files

| File | Required columns | Meaning |
|---|---|---|
| `masses.csv` | `category,probability` | One row per declared category, including any unobserved category; exact nonnegative masses sum to one. |
| `fits.csv` | `model,category,forecast_mean,outcome_mean` | Every model has one pair of fixed fitted conditional marginal-midrank means in `[0,1]` per category. Model order declares the complete family. |
| `validation.csv` | `pair,role,stream,category,outcome,forecast__MODEL,...` | Each pair ID has exactly one `focal` and one `reference` row and one shared `A` or `B` stream label. Supply a forecast column for every declared model. |
| `evaluation.csv` | `group,category,outcome,forecast__MODEL,...` | All evaluation groups have the same size `N>=3`. Supply every declared forecast column. |

Extra ordinary columns, such as a source population ID, are ignored. Extra
`forecast__...` columns are rejected so an undeclared candidate cannot be
silently omitted. Required values must be finite and present. Category and
model labels are strings. Pair IDs identify draw occurrences, not source
population entities.
An observed category declared to have exactly zero probability is rejected
as a contradiction of the supplied design. Unobserved zero-mass categories
may remain declared.

Raw forecast/outcome CSV fields are parsed before numeric type inference.
Exact decimal values, including integers larger than `2^53`, receive common
order codes across validation and evaluation. This preserves every observed
comparison and tie before calling the existing binary64 rank helpers; it
does not refit a nuisance or change a statistic. Nonfinite values and numeric
lexemes outside the parser's supported range are rejected. Already rounded
floating-point input cannot recover digits lost before it reached the CLI.

The validation stream labels must be chosen before examining values. Stream
A estimates forecast-comparison residual means; stream B estimates
outcome-comparison residual means. Both members cost a raw observation.
Only focal category assignments define the validation strata. Reference
categories retain their common marginal sampling law: the CLI does not
match or stratify references by their labels. A category missing in either
stream contributes its deterministic product-corner allowance.

Evaluation uses every row in the existing distinct-reference full-U mean.
For `--sampling-bound variance`, consecutive disjoint triples in the supplied
within-group row order estimate kernel variance; their six role assignments
are averaged. The order must be fixed independently of the values. At least
two such triples are required in total. Leftover rows still enter the full
mean. Choose `range` or `variance` before evaluation; picking the more
favorable reported result after evaluation requires another validity argument.

## Error allocation and outputs

For `K` supplied models, BY family level `q`, and validation fraction `rho`,
the CLI fixes

```text
H_K = sum_{j=1}^K 1/j
alpha_first = q/(K H_K)
delta = rho * alpha_first
```

Each model receives the existing aggregate upper allowance at that delta.
The range construction computes

```text
p = min(1, delta + exp(-2 J * max(mean-bias_upper,0)^2 / R^2))
J = number_of_groups * floor(rows_per_group/3)
```

Here `R` is the fitted product's global category-corner range, as returned by
the existing certificate helper; it is not assumed to equal one. The optional
variance construction uses that helper's proved variance-bound inversion.
The same complete set of p-values then receives BY adjustment. Family
dependence from shared observations does not require removing candidates.

`results.csv` contains every model's bias center and three radii, omitted-cell
allowance, full-U mean, p-value, BY adjusted p-value and retention flag. It
also reports the sampling radius and lower bound at the conservative **first**
BY threshold. That lower bound is a readable marginal diagnostic; the final
retention flag uses BY's full step-up calculation.

`receipt.json` records the family, allocation, counts, metadata source,
sampling construction, input and implementation hashes, and raw-call costs.
For shared training and evaluation across this family, the total is

```text
training_calls + 2 * validation_pairs + groups * rows_per_group
```

The default demonstration counts `512 + 8192 + 6144 = 14848` raw calls,
once for the shared family data. `--training-calls` is a supplied provenance
count; the CLI does not refit models or verify it. `--metadata-rows` reports
separate control-metadata access and defaults to zero for the declared
synthetic design. Complete metadata are not automatically free in another
application; document their source through `--mass-source`.
One raw observation here supplies its category, outcome, and all declared
forecast channels. Additional forecast-computation or acquisition costs
outside that observation unit are not measured by this count.

Repeated population IDs are counted as repeated draw calls. Independent
sampling with replacement can legitimately repeat an ID; the CLI neither
deduplicates such rows nor treats duplication or uniqueness as evidence
that the required independence holds. Reusing the same random reference
across units requires a different dependence argument.

The finite guarantees require independent training, independent validation
pairs, and fresh iid observations within independent evaluation groups from
the same law, together with exact masses and predeclared choices. Format
checks cannot establish these assumptions. The receipt says so explicitly.
Retention concerns marginal-midrank residual covariance, and does not itself
establish a raw-unit forecasting gain. No power guarantee is inferred from
an observed bias upper allowance alone.

## Focused checks

```sh
python examples/aggregate_reference_family/test_example.py
```

The checks compare full-U scores with brute-force ordered-triple enumeration
including ties; verify BY and complete-family handling; preserve repeated-ID
draw accounting and evaluation leftovers; check that reference labels do not
define strata; exercise missing-category corners, malformed pair rejection,
the optional variance construction, a configurable family allocation, and
CLI output provenance/overwrite protection. They validate the wiring, not
the input sampling law or a new scientific claim.
Additional numeric checks preserve the results after large monotone integer
shifts, through both the API and CSV CLI, and reject unsupported extreme
numeric literals explicitly.
