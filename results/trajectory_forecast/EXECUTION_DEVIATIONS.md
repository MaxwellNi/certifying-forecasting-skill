# Execution corrections and comparator additions

The locally recorded protocol specifies forecast origin hour 14, hence target hour 15.
The first implementation instead filtered target hour 14. Its predictions,
choices and results are preserved in `data/initial/` and `recorded/initial/`. The mismatch was
identified by independent code review after the first confirmation losses had
been inspected. The corrected run uses target hour 15 with every training,
model, budget and selection rule unchanged. It is a documented implementation
correction, not a newly unseen confirmation and not a choice between outcomes.

The first comparator averages fully symmetrized kernels within disjoint triples.
It does not recombine observations across these triples. A stronger subsequent
comparison pools all acquired evaluation and validation trajectories in the
complete degree-three U-statistic, with a classical joint mean/variance bound.
This analysis is added after first-run results. It cannot replace the original
protocol's recorded result without identifying the change. Exact census values
and timing are also reported because all selection outcomes are accessible.
Neither resampling nor these timing measurements demonstrates label savings.

Selection seed 2026091499 also occurs in diagnostic replication 99. It was fixed
before either execution, so the individual bound remains valid; selection is
not an independent new simulation stream. The 17-coordinate cost is a logical
16-input-plus-target record size, not instrumented memory reads.

The UCI source supplies observation timestamps but not publication latency.
Historical availability assumes the previous-hour measurements are available at
the next forecast origin. Confirmation bootstrap intervals summarize equal-week
paired losses; primary raw-record MSE and its relative gains have a different
weighting and are labeled separately.
