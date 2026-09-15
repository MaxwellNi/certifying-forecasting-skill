# Classical stratified betting on the same independent pair scores

Date: 2026-09-13. This is a retrospectively developed comparator specification, fixed before any new task retrieval or outcomes. It is a specialization of existing stratified betting, with a conservative numerical certificate. It makes no priority or universal power claim.

## Published method being specialized

Spertus, Sridhar and Stark express a global weighted-mean null as a union of intersection nulls, combine stratum betting processes by multiplication, then minimize evidence over nuisance means. Their inverse bets give a convex profiling problem; alternative approximately Kelly bets can improve efficiency but cost more computation. Their current version is v5, August 25, 2026. The present implementation fixes the existing inverse-bet class and uses the stratified construction's independent pair streams. [Sequential stratified inference for the mean, Sections 2.3, 3.2, 4.2.2 and 5](https://arxiv.org/html/2409.06680).

Ordinary bounded-mean betting assumes the observations have a common conditional mean. A global signed average across strata does not imply every stratum has nonpositive mean. Thus applying a zero-mean betting process separately to each stratum and multiplying it is invalid for the stated global null. [Waudby-Smith and Ramdas, Section 2](https://academic.oup.com/jrsssb/article/86/1/1/7043257).

## Same-data contract

Use exactly `stratified_reference.prepare(...)`. Condition only on the focal category labels and the fixed training/design information, as in `STRATIFIED_THEORY.md`. The usable categories S have q_c independent pair scores Z_cj in [-1/2,1/2], each with conditional mean theta_c in [-1/4,1/4]. Distinct scores use disjoint focal/reference positions. All original N raw rows remain charged, including unused rows and coordinates. The betting comparator acquires zero extra raw rows and never resamples pair scores.

Let m=sum_(c outside S) p_c, H=m/4, P_S=sum_(c in S)p_c and X_cj=Z_cj+1/2. The shifted stratum means eta_c=theta_c+1/2 lie in [1/4,3/4]. Under the full target null theta=sum p_c theta_c<=0, the partial target obeys theta_S<=H. Therefore the true shifted mean vector belongs to

```
E_0 = {eta : 1/4<=eta_c<=3/4,
       sum_(c in S) p_c eta_c <= P_S/2+H}.
```

This includes mixed positive and negative theta_c. When the cap is at least (3/4)P_S, the full null imposes no additional restriction on the observed stratum means beyond their known bounds; the implementation conservatively returns p=1.

## Fixed-fraction evidence and the global null

For a fraction b in (0,1), define

```
E_b(eta) = product_(c in S) product_(j=1..q_c)
           [1-b+b X_cj/eta_c],
f_b(eta) = log E_b(eta).
```

At the true vector eta*, each factor has conditional mean one. Since all pair scores are independent under the conditioned design and b is fixed, E[E_b(eta*)]=1. The factors are positive because b<1, X>=0 and eta>=1/4.

Under the full null eta* is in E_0, so

```
e_b = inf_(eta in E_0) E_b(eta) <= E_b(eta*),
E[e_b] <= 1.
```

Fix the five fractions B={1/10,3/10,1/2,7/10,9/10} before outcomes. Linearity of expectation gives

```
e_mix = (1/5) sum_(b in B) e_b,    E[e_mix] <= 1,
p_bet = min(1,1/e_mix).
```

Markov's inequality gives P(p_bet<=alpha)<=alpha. No independence between the five e-values is needed. This is a mixture of five separately profiled e-values. Profiling a mixture before minimizing could give stronger evidence, but that different construction is not implemented. Choosing fractions after looking at outcomes is not covered.

Conditional validity implies unconditional validity by integration over focal labels. No inference is conditional on reference labels. The implementation is a fixed-sample comparator. Although the predecessor framework can allow sequential operation, optional stopping of the focal/reference allocation is not asserted here.

## Convex optimization and the direction of numerical error

For a fixed score x>=0 and fraction b, let d=(1-b)eta+b x. Then

```
log(1-b+b x/eta) = log(d)-log(eta),
derivative = -b x/(eta*d),
second derivative = 1/eta^2-(1-b)^2/d^2 >= 0.
```

Thus f_b is convex, separable and nonincreasing in the coordinates. SLSQP with analytic gradients is used only to locate a promising point. A feasible primal objective is an **upper** bound on inf f_b; treating it as exact would overstate evidence and can produce anti-conservative p-values.

For any point eta_0, convexity supplies the global tangent inequality

```
f_b(eta) >= f_b(eta_0) + grad f_b(eta_0) dot (eta-eta_0).
```

Minimize this affine expression over E_0. Because each gradient is nonpositive, the exact linear minimizer starts every coordinate at 1/4 and spends the remaining weighted mass on coordinates in increasing order of g_c/p_c, filling each to 3/4 before proceeding. All but at most one filled coordinate are at endpoints. The resulting lower objective bound is valid regardless of the accuracy or success flag of the numerical optimizer.

The implementation repairs the candidate to a feasible point by moving it toward the lower corner if needed. It computes the gradients, the feasible repair, all ordering comparisons, and the linear allocation with exact rational arithmetic. Repeated score values are compressed first; this does not change any product.

Each logarithm is bounded outward using 60-digit Decimal arithmetic: divide the exact positive rational argument with downward/upward rounding, evaluate the correctly rounded Decimal logarithm, then step to the adjacent representable Decimal in the outward direction. Weighted summation uses the matching directed rounding. An extra downward tolerance of (1+B_pairs)*10^(-12) is subtracted from each lower objective. This extra tolerance is a conservative implementation choice; validity already follows from the outward bounds and exact rational tangent algebra.

The lower exponential evidence and its five-way average also use outward rounding. The reported p-value is rounded upward, including the final Decimal-to-float conversion. The p-value therefore uses a lower bound on the mathematical mixture evidence. Per-fraction feasible objectives provide upper evidence bounds and an optimization-gap diagnostic. A truncated or failed optimizer can widen that gap but cannot increase certified evidence above the true profiled value.

Float inputs are treated as their exact represented rational values by this numerical certificate. Supplying correct population masses and an appropriate sampling law is a separate statistical requirement. The numeric certificate does not turn estimated probabilities into exact metadata or validate a real observational panel.

## Interface, cost, and limitations

`profiled_mixture_pvalue(prepared)` is the fixed primary API. It returns `p`, `objective_lower`, `objective_upper`, `objective_gap`, component diagnostics, all charged raw rows, zero additional rows, and elapsed computation time. Here the objective fields bound the log of the **mixture of separately profiled** e-values.

The five fractions are fixed in source. For every fraction the code solves one convex constrained optimization, then computes one exact tangent linear program. Score compression costs O(B_pairs log B_pairs) with the current per-cell sort. Objective and derivative evaluations are linear in the number of distinct scores across cells; the actual midrank pair score has at most seven values. The exact tangent allocation costs O(|S| log |S|) rational comparisons plus arithmetic whose bit cost depends on the supplied floating representations. Report elapsed time and optimizer evaluations rather than claiming the numerical optimization is free.

The fixed positive fractions need not be powerful against arbitrarily small alternatives and are not asserted to be a strongest possible betting strategy. A richer fraction mixture or predictable approximately Kelly bets may improve sensitivity but would be another predeclared comparator with its own computational certificate. The stratified hybrid Hoeffding/empirical-Bernstein method remains useful alongside this betting comparator.

The exact pair scores do not use category-constant fitted nuisance means: such offsets cancel pointwise from both differences. Thus this comparator has the same scores whether or not a method calls them learned. No learning benefit or unique capability can be attributed to these offsets.

## Checks

`check_profiled_betting.py` checks the single-stratum closed form, mixed-sign null preservation, missing-mass behavior, 40 profile intervals against independent 80-digit constrained minimization, and a deliberately truncated optimizer. It also enumerates 147 count states across three heterogeneous bounded null laws and checks the e-value expectation and p-value rejection probabilities. These are generic bounded-stream tests; they are not field outcomes or evidence of broad empirical power. Recorded results and source hash are in `profiled_betting_checks.json`.
