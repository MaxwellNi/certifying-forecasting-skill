# Fully averaged bias comparisons

These comparisons let every method use the same exact category probabilities,
fixed fits and independent validation pairs, including both channels of every
pair. They distinguish a classical fully averaged second-order estimator from
the finite confidence bound placed around it. Lower estimator variance does
not imply a uniformly narrower bound.

The mathematical conditions and all tuning modes are in [THEORY.md](THEORY.md).
The companion paper supplies the same-target interpretation. Modes are evaluated
separately; the most favorable realized bound is not selected as a certificate.

## Results and scope

All 72 previously used synthetic settings were regenerated using their original
144,000 draws and evaluated at two validation error budgets. Thus there are
288,000 rows, not 288,000 independent draws. The 864 method-setting cells cover
six methods. They concern bias-bound coverage and width, not full-pipeline
power or independent forecast confirmation.

The archive comparison holds all 32 previously examined candidates fixed in
four separately adjusted eight-member families, at 32,768 raw calls per
candidate and method. Known-probability rectangles retain 1 candidate under
pooled-U sampling and 3 under betting; the split aggregate and both pooled
fixed-tuning and Gaussian-mixture rules retain 9 and 13. This improvement over
rectangles is not unique to splitting. Original confirmation outcomes remain
unchanged. Complete candidate rows are supplied.

The later square-normalized comparison is secondary development. Its subgamma
product bound retains 9/13; using a sharper product bound gives 10/13. The
latter changes both pooling/calibration and product-radius components.
It is not an independent confirmation or a proof of dominance.

## Reproduce

From this directory, using the package Python 3.12 dependencies:

```sh
python reproduce.py
python reproduce.py --full --output PUBLIC_SOURCE_REPLAY_FULL.json
```

The default command reruns the independently authored finite-state and scalar
formula checks, compares scalar bounds with sufficient statistics, and checks
the keys, replicate counts, and summary consistency of all 288,000 stored rows
and 864 simulation summaries. It verifies identical centers across the two
error budgets, compares the original rectangle/split arrays at `delta=0.05`,
and recomputes both pooled forecast archive comparisons. The optional `--full` also regenerates all
144,000 primitive synthetic draws and compares every resulting row and summary.
Both commands preserve the stored scientific files and write a replay receipt.
An alternative `--output` must name a new JSON file; only the two standard
public replay receipts can be overwritten. Imports create no bytecode files.
Floating-point comparisons use absolute tolerance `1e-12` and relative tolerance
`1e-11`; metadata, counts, and retention decisions must agree exactly.

`independent_math_check.json` contains the recorded check results and original
source identities. Its explanatory scope text is clarified for readers; the
numerical findings and original source hashes are preserved. The package's
`SOURCE_OVERRIDES.json` records the original and distributed file hashes.
`PUBLIC_SOURCE_REPLAY.json` and `PUBLIC_SOURCE_REPLAY_FULL.json` retain their
recorded execution inputs, including the original receipt hash. This distribution
preserves those prior receipts. To preserve them when rerunning, give `--output`
a new filename; the default path replaces its designated receipt. A new execution
binds its output to the files actually used. These commands reproduce the
mathematical checks and recorded comparisons, without generating new forecasting evidence.

Individual regeneration commands are also available:

```sh
python archive_benchmark.py --output /tmp/pooled-archive
python secondary_archive.py --primary /tmp/pooled-archive --output /tmp/pooled-secondary
python benchmark.py --output /tmp/pooled-simulation
python independent_verify_square_mgf.py --output /tmp/pooled-mgf.json
python independent_verify_square_implementation.py --output /tmp/pooled-formulas.json
```

Each directory output path must be new. These commands reconstruct archive results or
all synthetic draws; they do not retrain the original forecast models.
The package-level reproduction command runs the archive calculations and
checks stored simulation summaries. Full simulation regeneration is explicit.
The distributed tables retain all unsuccessful configurations. Public paths
are adapted for this standalone package; original numerical evidence is unchanged.
