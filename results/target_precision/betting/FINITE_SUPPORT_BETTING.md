# Fixed-grid finite-support betting

This is a computational specialization of classical product betting. It is not
a new statistical capability. The default is the 64 fractions returned by
`numpy.geomspace(1e-4, .99, 64)`, fixed before inference outcomes.

Initialize `FiniteSupportBetting(support, lower, fractions=None, precision=40)`
once for the predeclared finite score support and its known negative lower bound
`lower=a`. Call `.pvalue(counts)` for a conservative scalar p-value, or
`.evaluate(counts)` for p and outward log-evidence bounds. Counts follow the
initial support order. Fractions and bounds supplied as floats identify their
exact represented dyadic values; use `Fraction(-1,12)` for a mathematical -1/12.

For independent scores S_i with E[S_i] <= 0 and S_i >= a < 0, each factor
`1 + b S_i/(-a)` is positive for fixed 0 < b < 1 and has expectation at most one.
The product E_b therefore has expectation at most one. The fixed mixture
E = mean_b E_b also has expectation at most one, and min(1,1/E) is a valid
p-value by Markov's inequality. The same argument allows predictable conditional
mean nulls, but this benchmark uses IID disjoint block scores. The null must
bound each relevant conditional mean; a global weighted null over heterogeneous
strata instead needs the separately implemented profiled construction.

Products depend on the scores only through their finite-support counts. For a
single category, the generic same-pair score uses a=-1/2 and factors 1+2bZ. A
binary-law-specific score bound a=-1/8 supplies factors 1+8bZ. A binary disjoint
direct-target kernel with known bound a=-1/12 supplies factors 1+12bK. Each tighter
bound is an explicit restriction of the data-generating model, available to all
methods in that restricted benchmark; observing a small empirical range does
not justify it. No overlapping U-statistic terms may be treated as independent
factors.

The implementation precomputes outward Decimal intervals for the logarithm of
every exact rational factor. Per-replicate products become directed rounded dot
products of the count vector and these lower logarithms. It subtracts an extra
(1+number of scores)*1e-12 tolerance, then exponentiates, sums, and divides
outward to obtain a lower bound on E. The reciprocal rounds up, including the
conversion to binary float. Returning one when every computed lower log-product
is nonpositive is conservative. `.evaluate` additionally computes an outward
upper evidence bound for diagnosis. Optimization is absent.

Precomputation time is exposed as `precompute_seconds`; it should be charged
once for every distinct initialized betting design, with all per-replicate
evaluation time also charged. Count aggregation and score construction belong
to the benchmark runner's measured work. The helper requests no data and uses
no extra raw rows. `check_finite_support_betting.py` compares bounds with an
independent 110-digit evaluation, exhausts a symmetric finite null experiment,
and checks equality with the single-category profiled comparator.
