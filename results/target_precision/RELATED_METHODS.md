# Relation to classical methods

The complete-U bound applies weak-interaction concentration developed by
[Maurer and Pontil](https://proceedings.mlr.press/v75/maurer18a.html).
The stratified estimator specializes the independent-copy covariance identity
and bounded-sum inference. The profiled betting code implements the composite
mean-null construction discussed in `betting/PROFILED_BETTING.md`; optimizer
output is accompanied by a conservative numerical evidence bound.

For temporal forecast comparison, [Choe and Ramdas, Comparing Sequential
Forecasters](https://arxiv.org/abs/2110.00115) already provide confidence
sequences and e-processes for bounded forecast-score differences and a weak
average null. The fixed-grid predictable-loss calculation here is a classical
finite-calendar construction under its stated settlement-time information.
It is not the first sequential forecast comparison method, and it does not
certify arbitrary future or issuance-conditioned risk.

For binary variables, population midrank covariance is ordinary covariance
scaled by one quarter. In the rare perfectly correlated law, the pair kernel
K=(V1-V2)(W1-W2)/8 has variance theta/8-theta². The complete different-reference
mean also reduces to the complete covariance pair mean. Differences in tested
detection rates therefore concern information, independent evidence blocks
and calibration, not an inherent failure of every complete-U estimator.

A nondegenerate balanced-binary submodel gives a valid worst-case 1/theta²
sample requirement. The rare-event path changes variance with the target;
a first-order variance/target² scale can explain its earlier detection.
That local comparison does not remove finite-sample range terms or higher-order
remainders at degenerate boundaries and is not a universal optimality claim.
