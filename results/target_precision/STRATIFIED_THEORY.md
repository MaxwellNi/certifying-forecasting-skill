# A direct certificate with within-category focal pairs

This is a classical stratified covariance construction used as a stronger
same-information comparator. It is not presented as a new covariance identity,
new concentration theorem, general panel guarantee, or priority claim.

Fix training information, the predictor, the positive category support and exact
probabilities p_c. All N raw rows (C,V,W) are independent draws from that fixed
law. The target is theta = sum_c p_c Cov(R_V,R_W | C=c), where R_V and R_W
are population midranks under independent global reference draws.

The first F=floor(N/2) positions are focal rows. The remaining positions are
independent global references. Condition only on the focal category labels.
Within each category, pair focal positions in arrival order; q_c=floor(n_c/2).
For every pair, allocate two separate global reference rows, one for each rank
margin. Allocation and pairing depend only on focal labels, never reference
labels or observed values. There are enough references because 2 sum q_c <= F.
All N supplied rows are charged, including unused rows and unused coordinates.

For each pair of focal rows X1,X2 in c, and global reference rows X3,X4, set

    Z = (a(V1,V3)-a(V2,V3)) (a(W1,W4)-a(W2,W4)) / 2,
    a(x,y) = 1{y<x} + 1{y=x}/2.

Conditional on the focal rows, independent marginal reference roles make the
expected product equal to their population-rank difference product. Taking
expectations over two independent conditional focal rows yields
E[Z | C1=C2=c] = Cov(R_V,R_W | c) = theta_c. This holds with ties and at zero
variance; no first-projection normal approximation is used. Z lies in [-1/2,1/2]
and theta_c lies in [-1/4,1/4]. The latter follows from bounded-variable variance
and Cauchy–Schwarz. Conditional on all focal labels, the resulting Z values are
independent, though their means need not be identical across categories.

Let O={c:q_c>0}, m=sum_(c outside O) p_c and

    T = sum_(c in O) p_c mean(Z_c),
    K = sum_(c in O) p_c^2/q_c,       H = m/4.

T estimates the observed-category partial target. The omitted contribution has
absolute value at most H. Weighted Hoeffding therefore gives the one-sided bound

    L_H(alpha) = T - H - sqrt(K log(1/alpha)/2),
    P(L_H(alpha)>theta | focal labels, training) <= alpha.

Integration over the focal labels preserves the bound. The result accounts for
unseen and singly observed categories by m; replacing their contribution by
zero without H is invalid. The bound avoids a worst-case reciprocal p_min but
does not avoid sparse-category or sample-cost penalties.

For B=sum q_c >=2, let Y_ci=p_c Z_ci/q_c, let S_Y^2 be the ordinary sample
variance of the B weighted observations, and let w_max=max_(c in O) p_c/q_c.
Maurer–Pontil (2009), Theorem 11, covers independent nonidentically distributed
bounded variables. Apply it to 1/2-Y_ci/w_max and rescale the mean by B*w_max:

    L_E(alpha) = T - H
       - sqrt(2 B S_Y^2 log(2/alpha))
       - 7 B w_max log(2/alpha)/(3(B-1)).

This estimates variability of the final target estimator rather than one
learning-bias component. Between-category mean heterogeneity can make it
conservative. It is not a variance estimate for the overlapping complete U
statistic, whose center is different. A fixed hybrid uses
max{L_H(alpha/2), L_E(alpha/2)}; the union bound gives error at most alpha.
With fewer than two pairs, the empirical-Bernstein rule returns the deterministic
lower limit -1/4. Both lower bounds may be clipped to [-1/4,1/4] without losing
coverage. The cellwise empirical-Bernstein implementation is a secondary control:
its conditional error allocations sum to alpha and each selected pair of cell
bounds splits that cell's allocation.

Monotone inversion gives a marginal one-sided p value for theta<=0. Use the
entire fixed candidate family in BY or another justified rule. Arbitrary
selection of pairing, category definition, estimator or stopping time after
observing results is not covered.

For alpha,eta in (0,1), a sufficient conditional power >=1-eta condition for
theta>=theta0 is

    theta0 > m/2 + sqrt(K/2) [sqrt(log(1/alpha))+sqrt(log(1/eta))].

The m/2 term is twice H: the partial-target mean may be theta-H, and the lower
bound subtracts another H. If each focal count is even and exactly F p_c, all
categories are represented and K=2/F, or K=4/N when N=2F. This is a conditional
allocation statement, not an assertion that random counts always balance.

Subtracting any learned category-constant means cancels pointwise in the focal
differences. Thus learning such means cannot improve this estimator. A learning
benefit would require a different estimator, richer control information, or a
different experimental design, and its costs and target must be compared anew.

Primary source: [Maurer and Pontil, Empirical Bernstein Bounds and Sample
Variance Penalization, COLT 2009, Theorem 11](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).
The covariance identity and stratification argument are elementary classical
constructions; observed success over a weaker implementation is not proof of
originality over the strongest legal comparator.
