# A zero target with positive reference bias

This separately frozen addendum checks a nonidentity candidate with exact zero rank-contrast target and positive shared-reference bias. It was designed after inspecting the original six-law calibration and frozen before generating any of its own Monte Carlo outcomes. Its 12 procedure cells are separate from the original study's 72 cells.

Let `B` be uniform on `{0,1,2,3}`, `h=1{B=3}` and `Y=1{B>=2}`. For the common-baseline contrasts `(h,B)` and `(Y,B)`, the exact target is `theta_h=0`, while `E[S]=Gamma=1/32`. The left population-rank contrast is `(1/4,0,-1/4,0)`, so the null is not obtained by declaring `h=B`. The symmetric triple kernel for h varies between `-1/24` and `1/12` under this law. [DERIVATION.md](DERIVATION.md) supplies the rational calculation and [EXACT_ORACLE.json](EXACT_ORACLE.json) records an independent check of all 256 candidate/ordered-triple cases.

The four candidates remain `(B,h,-h,0)`, with targets `(0,0,1/32,1/64)`. Thus B and h are the true nulls; the reverse and constant candidates are positive for this rank-contrast target. All maps here are functions of B. The target and null labels must not be interpreted as conditional independence or information beyond the baseline.

The [protocol](PROTOCOL.json) was frozen at **2026-09-15 13:06:09.629131 UTC**, after the exact oracle and before simulation. Its SHA-256 is `1c3c13a0b36f424684b9effa6147db9a327ec2d27af52076b2f9c3aea4ecd75a`. The [freeze receipt](FROZEN_BEFORE_SIMULATION.json) binds the derivation, oracle, independent verifier, simulation wrapper and reused method source. This is local chronological evidence, not an externally timestamped registration. The law was selected by exact finite construction; no Monte Carlo outcomes or seed search were used to select it.

The simulation uses the original final shared-correction, disjoint-triple-U and complete-U rules, each with both range and hybrid/variance bounds. `study.py` is a byte-identical copy of the original frozen implementation, with digest `754132bc8a1b067e2f151c390af97191822df01eda23bc5f94d47348a5ca950d`. The wrapper changes the law, status and predeclared seed only. The family level remains 0.05, the candidate level 0.0125, and the shared score/correction split 0.00625 each. The structural widths remain `5/4`, `5/2` and `1/2`, with zero width only for the declared identity candidate. Additional law-specific widths are not used.

Each method receives the same raw stream within a replication. Budgets 3,072 and 12,288 use shared allocations `(M,n)=(546,660)` and `(2181,2642)`, respectively, including every evaluation and correction draw. Both U methods receive all draws; their effective independent triple counts are 1,024 and 4,096. Training cost is zero because the maps are fixed. Complete U uses the complete center and the prespecified disjoint-triple variance, with the original finite error allocation.

Exactly 300 replications were run at each budget: **600 replications, 4,608,000 draw positions, 14,400 candidate rows, 3,600 family rows and 48 summary rows**. Every one of the 12 law/budget/procedure cells has **0/300 family noncoverage events and 0/300 false certifications**. Candidate h has 0/300 certifications for every method, bound and budget. The pointwise two-sided 95% Clopper-Pearson upper limit for each 0/300 count is **1.222097%**, not zero. These intervals are not simultaneous over cells or procedures, and these procedures cannot be selected after inspecting their outcomes without further error allocation.

The independent recomputation also reports the raw score and correction means for h as diagnostics:

| Budget | Mean shared score across replications | Mean correction across replications | Exact value of each expectation |
|---|---:|---:|---:|
| 3,072 | 0.031245 | 0.031284 | 1/32 = 0.03125 |
| 12,288 | 0.031278 | 0.031149 | 1/32 = 0.03125 |

All candidates and both bounds remain in [summary.csv](results/summary.csv), including the weak shared-rule power. The shared hybrid certifies the positive reverse candidate 1/300 times at the larger budget and 0/300 at the smaller budget; it never certifies the positive constant candidate. Both direct-U hybrid/variance rules certify both positive candidates 300/300 times at both budgets. Under the pure range rule, both direct-U methods certify the reverse candidate 300/300 at both budgets and the constant candidate 0/300 at the smaller budget and 300/300 at the larger budget. The 300/300 pointwise two-sided 95% lower limit is 98.777903%, not guaranteed power one.

[INDEPENDENT_CHECK.json](INDEPENDENT_CHECK.json) confirms all targets, centers, radii, decisions, family events and summary statistics using a separate implementation that imports neither the simulation wrapper nor the reused methods. It rebuilds the seed streams, constructs comparisons from raw values, and obtains complete U by empirical-midrank contraction with forbidden equal-reference pairs removed. All 14,400 candidate rows match with maximum numerical difference zero in this environment. [REPLAY_CHECK.json](REPLAY_CHECK.json) separately confirms byte-identical replay of all three result CSVs. Replays are verification of the original 600 replications, not additional observations.

To reproduce from this directory into a fresh output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiment.py --verify reproduced_results
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python independent_verify.py --results reproduced_results --output reproduced_results/INDEPENDENT_CHECK.json
```

NumPy and SciPy are required. The directory is self-contained and can be moved without editing frozen files. The standalone `study.py` is the unchanged method dependency; `experiment.py` is the entry point for this addendum. The independent oracle is recomputed during verification; its original receipt is preserved.

This adds calibration evidence for one analytically chosen biased-null law and closes that specific gap in the original grid. It does not provide exhaustive null calibration, a new application, prediction-loss gains, a physical-panel sampling guarantee, a unique advantage for complete U, or a coverage theorem proved by simulation.
