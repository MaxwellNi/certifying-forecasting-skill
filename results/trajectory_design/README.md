# Reference designs, target preservation, and forecast availability

The trajectory identity separates a specified fitted rank association from the interaction introduced by reused references. Read [the derivation](DERIVATION_AND_IMPLEMENTATION.md) for its target, sampling assumptions, source-design construction, and the matched-budget comparison.

`design_contract.py` turns declared map and reference draw IDs into coefficient matrices. It checks that a redesigned reference assignment keeps the same maps, reference law and coefficient row sums. The forecast validator checks declared feature-release and model-availability times against the forecast issue time. These checks do not establish actual independence, actual release latency, or that a fitted model was truly frozen.

`fixed_delta_inference.py` inverts the pure-Hoeffding bound using a fixed validation allocation. Its p value is rounded upward for supplied numeric values. This numerical guarantee covers the inversion only; floating kernel errors require a separate justified mean-error allowance. The routine also provides a downward-rounded threshold for a fixed Bonferroni family. It does not invert the separate empirical-Bernstein/hybrid bound or select an allocation after outcomes.

`trajectory_kernel.py` and `pooled_trajectory_kernel.py` retain the established block and complete pooled kernels. The latter uses all trajectories at a computational cost of O((S+T+ST)N log N) with S forecast-side and T outcome-side maps. Both keep integer ordering and ties before floating coefficient contractions.

Run all tests from this folder:

```sh
python -m unittest discover -s . -p 'test_*.py' -v
python contract_exact_checks.py
```

The compiled design uses reference IDs as whole-trajectory **draw roles**. One role used by several map terms remains one column. Different IDs do not establish statistical independence, and relabeling a single realized record does not create an independent draw. Sampling a fixed archive by independent random indices is a distinct, conditional experiment that can return repeated archive records.

The complete pooled estimator is the default comparison when the same raw draws and target can be reused legally. Its stated range-only confidence radius is strictly smaller than the split evaluation radius under the positive matched-budget conditions in the derivation. This does not imply samplewise lower-bound, power, or future prediction-loss dominance.
