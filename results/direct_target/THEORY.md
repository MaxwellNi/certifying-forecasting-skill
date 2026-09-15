# Direct estimation of the residual midrank target

This note derives a complete finite-sum estimator and conservative confidence bounds for the residual midrank target. The associated archive comparison uses previously inspected forecasting data.

## Target and sampling conditions

Condition on fixed training information H. Let the raw rows X_i=(V_i,W_i,C_i), i=1,...,n, be independent and identically distributed from the same conditional distribution P_H. The finite category probabilities p_c=P_H(C=c)>0 must be known exactly, including categories absent from the observed sample. Sample size and any disjoint blocking must be fixed without inspecting the inference rows. The predictor, target, category definitions, and any candidate selection must already be fixed or handled by a separate valid selection argument.

Define a(x,y)=1{y<x}+0.5 1{y=x}, R_V=E[a(V,V')|V,H], and R_W analogously, with independent references from P_H. The target is

    theta = E[R_V R_W|H] - sum_c p_c E[R_V|C=c,H] E[R_W|C=c,H].

This is the population midrank residual covariance for finite C. It is not automatically the conditional target given a richer continuous control, a standardized empirical panel rank statistic, or the distribution mixed over different trained predictors. In general, conditioning additionally on all observed category labels destroys the identities below. Empirical category frequencies cannot replace p without a new error analysis.

Independent sampling positions can contain the same population ID and identical observed values. Such positions remain separate in all sums. Conversely, deduplicating with-replacement draws or reusing dependent historical rows does not satisfy the stated argument. A finite population sampled independently with replacement is covered; exhaustive sampling without replacement is a different design.

## Unbiased estimator

Write (n)_r=n(n-1)...(n-r+1), a_ij=a(V_i,V_j), and b_ij=a(W_i,W_j). The complete distinct-position estimator is

    U = U3 - U4,
    U3 = (n)_3^{-1} sum_{i,j,k all distinct} a_ij b_ik,
    U4 = (n)_4^{-1} sum_{i,j,k,l all distinct}
                         1{C_i=C_j} a_ik b_jl / p_{C_i}.

For U3, the two references are independent given focus row i and H, so its expectation is E[R_V R_W|H]. For U4, the two focus rows and their two references are mutually independent. Matching a category has probability p_c^2; division by p_c leaves p_c times the product of its two conditional mean ranks. Thus E[U|H]=theta for n>=4.

Equivalently, symmetrize the four-position kernel

    h(X1,X2,X3,X4) = a_12 b_13 - 1{C1=C2} a_13 b_24 / p_{C1}

over all 24 permutations, obtaining h_s. The complete fourth-order U statistic with kernel h_s equals U3-U4: each ordered triple in the first term appears n-3 times in the ordered quadruple sum. This equality is important for inference because the many kernel evaluations overlap.

## Collision-correct quadratic algorithm

Let

    A_i = sum_{k != i} a_ik,    B_i = sum_{k != i} b_ik.

The first numerator is

    N3 = sum_i A_i B_i - sum_i sum_{k != i} a_ik b_ik.

The subtraction removes exactly the collision in which the two reference positions coincide. For the second term define

    P = sum_{i != j, C_i=C_j} (A_i-a_ij)(B_j-b_ji)/p_{C_i}.

For fixed distinct focus positions i,j, the product permits reference positions k,l from the common set excluding i,j. It includes both the desired k!=l terms and a shared-reference collision. Let, for every possible shared reference k,

    S_A(c,k) = sum_{i != k, C_i=c} a_ik,
    S_B(c,k) = sum_{j != k, C_j=c} b_jk,
    D(c,k)   = sum_{i != k, C_i=c} a_ik b_ik.

Then the total shared-reference contribution is

    Q = sum_k sum_c [S_A(c,k) S_B(c,k) - D(c,k)] / p_c.

The diagonal subtraction D removes the prohibited focus collision i=j. Therefore N4=P-Q and U=N3/(n)_3-N4/(n)_4. Every prohibited position collision has been excluded; equal observed values and repeated population identities have not been removed.

The implementation computes A_i and B_i by sorted midrank counts with the self-comparison 1/2 removed. Each subsequent pass compares one position with all n values and uses category sums. It compresses the observed category index to at most n categories before these passes. Total time is O(n^2+C) and additional memory is O(n+C), where C is the size of the supplied positive population support. It never stores an n-by-n comparison matrix. Safe integer arrays retain their exact order above 2^53. Mixed numeric input that would silently change any integer when converted to a common floating dtype is rejected; the caller must supply a supported exact representation. Extremely small probabilities whose reciprocal cannot be represented in float64 are also rejected. The returned estimator uses floating-point summation; the word complete describes the full mathematical sum, not exact rational machine arithmetic.

## A tighter deterministic kernel range

The symmetrized first term is an average of symmetrized three-position products. For strictly ordered V, sorting its three positions leaves six possible strict W orders. The six means are 1/3, 1/3, 1/3, 1/6, 1/6, 1/6. For ties, break V ties and W ties independently and uniformly. The expectation of each product of the two comparison bits is precisely the product of its original midrank comparisons. The tied case is therefore a convex combination of strict-order cases. The first term always lies in [1/6,1/3].

For the unweighted disjoint-pair product a_13 b_24, partition the 24 permutations into orbits obtained by swapping positions 1 and 3 and swapping positions 2 and 4. Each orbit of four products sums to

    (a_13+a_31)(b_24+b_42) = 1.

Thus its permutation average is exactly 1/4, including ties. The category-weighted second term is nonnegative and at most 1/(4 p_min), because each category weight is between zero and 1/p_min. Consequently

    1/6 - 1/(4 p_min) <= h_s <= 1/3,
    R = 1/6 + 1/(4 p_min)

is a valid range width. These endpoints are attainable when the positive support has at least four categories: use four distinct categories for the upper endpoint, and one least-probable category with oppositely ordered values for the lower endpoint. With fewer categories the range remains valid but need not be sharp. For a single category, the second term is identically 1/4 and h_s lies in [-1/12,1/12], so R=1/6.

This improves the earlier valid but loose width 1+1/p_min. It is an algebraic constant improvement, not a claim of improved statistical rate or a priority claim about concentration theory.

## Two valid, deliberately conservative inference baselines

### Complete U statistic

Let q=floor(n/4). For a uniformly random permutation independent of the data, partition its first 4q positions into q disjoint four-row blocks and average h_s over the blocks. Given H, this block average is an average of q independent bounded variables, each with mean theta and range width R. Conditional on the observed rows, its average over all permutations is the complete U statistic, including when n is not divisible by four.

Jensen's inequality and the bounded-variable exponential moment bound therefore give

    P(U - theta >= t | H) <= exp(-2 q t^2/R^2).

Hence U - R sqrt(log(1/alpha)/(2q)) is a one-sided lower confidence bound with error at most alpha. The function `full_u_hoeffding` implements this formula with the tighter range above. It neither treats binomial(n,4) kernels as independent nor requires a nondegenerate first projection. Small p_min and small q can still make the bound uninformative.

### Separate variance-sensitive block estimator

For one grouping into disjoint four-row blocks chosen without inspecting their values or labels, let Y_1,...,Y_q be the symmetrized kernel values, q>=2. Let S_Y^2 be their usual sample variance with denominator q-1. Applying Theorem 4 of [Maurer and Pontil (2009)](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf) to the appropriately normalized negative block values gives the lower bound

    mean(Y) - sqrt(2 S_Y^2 log(2/alpha)/q)
            - 7 R log(2/alpha)/(3(q-1)).

The independent observations here are q block values, not the overlapping U kernels. This is implemented in `block_empirical_bernstein`. Its center is mean(Y); combining the same sample variance with the complete U center has not been justified. The block rule responds to observed variance, but its remainder still depends on R and its block count is smaller than the raw row count. It is not presented as a calibrated competitive or strongest comparator.

Selecting between both bounds after seeing them requires an appropriate joint error allocation. For a fixed candidate family, candidate-specific error levels or valid marginal p-values must be combined under an appropriate family rule. Using alpha=0.05 independently for every candidate does not provide a 0.05 family guarantee. The current code does not implement adaptive stopping, family selection, or simultaneous confidence intervals.

## What this closes and what remains open

The exact fast algorithm removes a computational objection to a raw-row comparator. The inference bounds are valid under their stated conditional IID design and tolerate ties and degeneracy, with no claim of competitive power. The synthetic checks verify arithmetic, exact finite-population expectation, collision removal, range lemmas, and formula implementation. They do not independently validate IID sampling or known p on a real panel.

A practical variance method for the complete statistic must account for its full Hoeffding decomposition and overlapping kernels. An ordinary first-projection normal approximation is not justified uniformly near degeneracy. Degeneracy of a separately estimated learning-bias term does not establish degeneracy of this final-target kernel. A useful comparator still needs defensible finite-sample or other clearly justified calibration, identical target and information access, full raw-observation and computation costs, and fixed-rule evaluation on untouched tasks. This note supplies no such new task result and no evidence of unique advantage for the paper's method.
