# Finite-law calibration of common-baseline bounds

This study checks the implemented sharp-width rules on six prescribed finite laws. It holds the candidate family, error allocation, observation costs, random seeds and reporting rules fixed before generating simulation outcomes. Every result, including failures to detect positive targets, is retained.

All zero-target cells in this grid have zero expected reference interaction. The grid therefore does not stress a zero target with a positive shared-reference bias. A [separately frozen addendum](biased_null/README.md) addresses one such law: its nonidentity candidate has target zero and expected shared score 1/32. All 12 additional procedure cells have 0/300 false certifications and family noncoverage events. These additional 600 replications are separate from the original protocol and do not establish exhaustive null calibration.

The study concerns the fitted rank-contrast target in the paper. It does not test whether a forecast supplies new information beyond its baseline, establish validity for dependent physical panels, or measure prediction loss. The informative-baseline example makes this distinction explicit: a deterministic transformation of the baseline has a positive fitted rank-contrast target.

## Design

Each law supplies IID draws of `(h, Y, b)`, where `b` is the common baseline. The four candidate maps are `b`, `h`, `-h`, and the constant zero map. Their left contrasts are paired with the right contrast `(Y, b)`. The baseline candidate is declared to be the same function on both sides, so its kernels and radii are zero. Accidental equality in an observed sample never establishes this identity.

- Six laws: a tied null, a weak positive association, a strong positive association, a near-degenerate null, a near-degenerate positive association, and an informative nonlinear baseline.
- Two total observation budgets: 3,072 and 12,288 IID draw positions.
- Three constructions: shared score minus an independent interaction estimate, the mean of disjoint symmetric triple kernels, and the complete degree-three U-statistic using all available draws.
- Two preselected bounds for each construction: a range bound and the fixed-level hybrid or variance-aware bound.
- Four fixed candidates, family level 0.05 and candidate level 0.0125. The shared construction splits the candidate level equally between its score and correction bounds.
- 300 replications for every law and budget, using the exact seed derivation in [PROTOCOL.json](PROTOCOL.json).

All methods receive the same raw draw stream. Shared-score allocation minimizes the declared range radius before seeing outcomes. Complete U may recombine every draw. It uses `floor(N / 3)` as the effective independent triple count, never `N choose 3`. Both direct U methods estimate the same target as the corrected shared score.

The fixed common-baseline widths are 5/4, 5/2 and 1/2 for the score, correction and symmetric U kernel. Some individual finite laws admit tighter support-specific widths. Those additional widths are deliberately outside this check of the final common-baseline rule, for every comparator. This study makes no claim that its rules attain globally optimal efficiency for these laws.

Exact rational targets come from population midranks, independently of the implemented kernels. Literal role enumeration checks the score/correction identity and the expectation of the symmetric kernel. Complete-U calculations use empirical state multiplicities; parity with the released rank-sweep routine is tested. Runtime here is a property of this finite-support reproducer, not a benchmark for a general forecasting algorithm.

## Results

There are 3,600 independent simulation replications, 27,648,000 draw positions, 86,400 candidate-level result rows and 21,600 family-level rows. Every one of the 72 law/budget/procedure cells had zero family noncoverage events and zero false certifications in 300 replications. The pointwise two-sided 95% Clopper-Pearson upper limit for 0/300 is **1.222%**, not zero. These intervals are not simultaneous across cells. Each procedure has its own predeclared family error allocation; the study does not justify choosing among procedures after observing outcomes.

The table reports certifications of `h` under the fixed-level hybrid or variance-aware rule. Every denominator is 300. Pure-range results and all other candidates remain in [summary.csv](results/summary.csv).

| Law | Exact target | Budget | Shared correction | Disjoint triple U | Complete U |
|---|---:|---:|---:|---:|---:|
| Tied null | 0 | 3,072 | 0 | 0 | 0 |
| Tied null | 0 | 12,288 | 0 | 0 | 0 |
| Tied weak association | 1/135 | 3,072 | 0 | 0 | 0 |
| Tied weak association | 1/135 | 12,288 | 0 | 300 | 300 |
| Tied strong association | 2/27 | 3,072 | 0 | 300 | 300 |
| Tied strong association | 2/27 | 12,288 | 300 | 300 | 300 |
| Near-degenerate null | 0 | 3,072 | 0 | 0 | 0 |
| Near-degenerate null | 0 | 12,288 | 0 | 0 | 0 |
| Near-degenerate positive | 9801/2000000 | 3,072 | 0 | 0 | 0 |
| Near-degenerate positive | 9801/2000000 | 12,288 | 0 | 300 | 300 |
| Informative nonlinear baseline | 7/54 | 3,072 | 4 | 300 | 300 |
| Informative nonlinear baseline | 7/54 | 12,288 | 300 | 300 | 300 |

For 300/300 certifications, the pointwise two-sided 95% lower confidence limit is 98.778%, not guaranteed power one. Both direct U constructions show nontrivial power on the weak and near-degenerate positive laws at the larger budget. The shared correction remains conservative in those cases. Neither a unique advantage of complete U nor an improvement in future prediction loss follows from these results.

## Reproduction

NumPy and SciPy are required for simulation. The tests also use the sibling `trajectory_budget` and `shared_baseline_ranges` modules in the released package.

From this directory, reproduce the saved result arrays into a new output directory:

```bash
python study.py --verify reproduced_results
python verify_results.py --results reproduced_results --output reproduced_results/CHECK_RESULTS.json
python -m unittest -v test_study.py
```

The verifier compares all three CSV files byte for byte. It does not add repetitions or change seeds. To run the tests from a separate checkout, set `CERTIFYING_PACKAGE_ROOT` to the root of the released package. The protocol stores the exact simulation-source digest and refuses to run if that source or protocol changes.

[Eight tests](TESTS.txt) check closed-form targets, literal kernel expectations, ties, complete-U distinct-index arithmetic, parity with the released routines, budget allocation, radius formulas and the map-identity guard. A separate [result recount](CHECK_RESULTS.json) checks all stored decisions and summaries, and [deterministic replay](REPLAY_CHECK.json) checks all generated arrays.

Simulation cannot prove a coverage theorem. These checks confirm the specified implementation on the prescribed IID laws at the stated Monte Carlo precision. They do not replace a new independent application or establish a physical-panel sampling model.
