# Pooled comparator to split aggregate bias validation

Date: 2026-09-13. This is a proof and implementation note produced after
inspection of the split construction, not a preregistered new-method claim.
The derivation and implementations are included in this package.

## Conclusion and scope

The split aggregate allowance has a direct all-pair comparator with the same
information contract and the same balanced exact-fit expected-radius upper
bound. Its center is the fully averaged within-category order-two U statistic.
It uses both channels of every available validation pair, permits arbitrary
dependence between the channels of one pair, and requires no enumeration or
random choice of splits in implementation. Averaging stronger,
square-normalized split MGFs gives an adaptive radius proportional to the
square root of an observed proxy, with a constant-factor cost and no
tuning-grid or sample-size logarithm. This `square_mgf` mode is the
strongest simple adaptive comparator established in this note. Fixed-tuning,
finite-grid and Gaussian-mixture versions are retained as separately valid
comparators and as a record of the derivation. The square-normalized mode
and exact-product-MGF tightening were developed after the initial three
linear modes and are explicitly secondary retrospective extensions.

The construction averages the split **exponential inequalities**, using
Jensen's inequality. It does not average already optimized confidence bounds
without an error-budget argument. The classical second-order estimator,
its degenerate variance, and the averaging argument provide no basis here for
claiming new general inference theory. No priority claim has been verified.

## 1. Contract and estimator

Condition on independent training, fixed fits f_c,g_c in [0,1], and all focal
category assignments. Category c then has n_c independent paired observations
(A_ci,B_ci), with A_ci in [-f_c,1-f_c], B_ci in [-g_c,1-g_c]. Each channel has
range length one. The two channels within one pair may be dependent. Pairs
are independent across both indices and categories. Reference observations
and their categories remain unconditioned. Exact p_c are provided by the
same metadata/design access offered to the split construction.

Let a_c=E A_ci, b_c=E B_ci, and target beta=sum_c p_c a_c b_c. For
S={c:p_c>0,n_c>=2}, write beta_S for this sum restricted to S and set

    k_c=floor(n_c/2), l_c=n_c-k_c,
    U_c = [(sum_i A_ci)(sum_i B_ci)-sum_i A_ci B_ci]/[n_c(n_c-1)],
    U = sum_{c in S} p_c U_c.

Thus U_c averages A_ci B_cj over ordered i!=j; equivalently it averages the
symmetric kernel (A_ci B_cj+A_cj B_ci)/2 over unordered pairs. Independence
of distinct pairs gives E U=beta_S. The diagonal sum is essential:
E[A_ci B_ci]=a_c b_c+Cov(A_ci,B_ci), so the all-data product of means is
biased by Cov(A_ci,B_ci)/n_c.

This **conditional-count, per-cell-normalized estimator** should not be
confused with the unconditional iid inverse-probability kernel
1{C_i=C_j}(A_i B_j+A_j B_i)/(2p_Ci). The latter is unbiased before conditioning
on category counts, but its conditional expectation weights cell c by
n_c(n_c-1)/[N(N-1)p_c], rather than p_c. The conditional-count contract used
here requires the displayed U_c weights.

Cells with n_c<2 contribute deterministic allowance

    H = sum_{c not in S} p_c q_c,
    q_c = max{f_c*g_c, -f_c*(1-g_c), -(1-f_c)*g_c, (1-f_c)*(1-g_c)}.

Zero-mass cells contribute zero. This omission rule is determined entirely
by the conditioning variables. Any deterministic valid q_c can replace the
corner bound. The full upper bound may also be clipped at sum p_c q_c.

## 2. Observable variance proxies

For c in S, let Abar_c,Bbar_c be full-category empirical means, and let
s_Ac^2,s_Bc^2 be unbiased sample variances, with denominator n_c-1. Define

    V_A = sum p_c^2/(4k_c) * [Bbar_c^2 + k_c*s_Bc^2/(n_c*l_c)],
    V_B = sum p_c^2/(4l_c) * [Abar_c^2 + l_c*s_Ac^2/(n_c*k_c)],
    d_c = p_c/(4 sqrt(k_c*l_c)),
    D = sum d_c^2, dmax = max d_c.

Empty sums and maxima are zero. These quantities use only six sufficient
statistics per cell: n, sum A, sum B, sum A^2, sum B^2, sum AB.

The fundamental exponential inequalities are

    E exp[t L_A - t^2 V_A/2] <= 1,
    E exp[t L_B - t^2 V_B/2] <= 1,                 (all fixed real t)

where the unobserved errors and canonical remainder satisfy

    L_A = sum p_c a_c Bbar_c - U,
    L_B = sum p_c b_c Abar_c - U,
    R = U - sum p_c(a_c Bbar_c+b_c Abar_c) + beta_S,
    beta_S-U = L_A+L_B+R.

The product remainder also satisfies

    log E exp(t R) <= t^2 D/[2(1-t*dmax)],         (0<t<1/dmax)
    P{R>sqrt(2xD)+x*dmax} <= exp(-x).             (x>0)

All expectations and probabilities in this note condition on the stated
contract. There is no conditioning on residual values in the displayed
unconditional exponential inequalities.

## 3. Implementable upper bounds

Let x=log(3/delta), r_2=sqrt(2xD)+x*dmax. If S is empty, H is the upper bound.
Otherwise each construction below gives

    P{beta > U+r_A+r_B+r_2+H} <= delta.

An optional deterministic tightening uses the sharper product MGF from
the proof rather than its sub-gamma relaxation:

    r_2,exact = inf_{0<t<1/dmax}
       {x-(1/2)sum_c log(1-t^2*d_c^2)}/t.

The optimizer depends only on conditioned counts, exact masses, and delta.
Thus this optimization incurs no data-adaptation penalty. It is implementable
by one-dimensional bisection: with z=t*dmax and u_c=z^2*d_c^2/dmax^2,
the derivative changes sign where

    sum_c [u_c/(1-u_c)+(1/2)log(1-u_c)] = x.

The left side is strictly increasing from zero to infinity as z runs from
zero to one. Any interior z yields a valid bound, even without finding
the exact optimizer. The optional `product_mode="exact_mgf"` uses this
bisection and retains the smaller of that bound and the sub-gamma radius.
Both choices are functions of conditioning information, so this minimum
requires no additional error budget. The default remains `subgamma` for
direct comparison with the existing split constant. This tightening changes
constants, not the rate or source of the method.

### Fixed tuning, with the original exact-fit expected-radius guarantee

Choose lambda_0=sqrt(2x/D) from counts and masses only, and use

    r_A=x/lambda_0 + lambda_0*V_A/2,
    r_B=x/lambda_0 + lambda_0*V_B/2.

Any other deterministic positive lambda is valid. The stated choice is
tuned for small fit errors. It is not uniformly efficient for large fit
errors because its linear radii grow proportionally to V rather than sqrt(V).
Its radius can stay bounded away from zero as N grows with C and nonzero
fit errors fixed. It should not be advertised as a consistent all-fit
adaptive validator.

### Finite tuning grid, with explicit adaptation penalty

Declare J positive lambdas without inspecting channel observations; they
may depend on counts and exact masses. Put x_L=log(3J/delta) and use

    r_A=min_j [x_L/lambda_j+lambda_j*V_A/2],
    r_B=min_j [x_L/lambda_j+lambda_j*V_B/2].

The implementation's default grid is lambda_j=lambda_0*2^j for
j=-8,-7,...,8. The product radius still uses x=log(3/delta). Each of the J
linear events has failure probability delta/(3J), and the two groups plus
the product event have total failure probability delta.

For context only, the unpenalized algebraic optimum would be sqrt(2x_L V).
If a factor-two geometric grid brackets its optimal lambda, the grid costs
at most 3/(2 sqrt(2)) times this quantity: a nearest grid point differs by
at most sqrt(2), and (r+1/r)/2<=3/(2 sqrt(2)). This statement requires the
optimum to lie in the grid's covered interval; it is not asserted outside.
The fixed-width 17-point grid also has endpoint limitations as sample size
grows. A count-dependent widening grid can remove those limitations,
with its explicit log J penalty retained.

More generally, a predeclared weighted grid with weights w_j>0 summing to
one can replace x_L by log(3/(delta*w_j)) separately for each j. Choosing
weights or grid points after looking at the residual values is unauthorized.

### Gaussian mixture, with no finite tuning grid

Choose any deterministic rho>0. For either (L,V)=(L_A,V_A) or (L_B,V_B),
integrating the fundamental inequality against t~Normal(0,1/rho) gives

    E [sqrt(rho/(rho+V))*exp(L^2/[2(rho+V)])] <= 1.

Thus the observable two-sided inequality

    |L| <= sqrt((V+rho)*[2x+log(1+V/rho)])

fails with probability at most exp(-x). Its upper side suffices here. Use
this expression as each linear radius. The implementation fixes
rho=D/(2x) from the count information. This method adapts to V without a
grid, at the cost of the explicit logarithm. No data-dependent optimization
of rho is permitted unless another mixture or error budget is supplied.

One can also integrate over positive t only and numerically invert the
resulting half-normal mixture for a somewhat sharper one-sided boundary.
That extension is not implemented or empirically claimed here.

### Square-normalized MGF: adaptive radius without a grid

This secondary extension uses the conditional split structure more strongly
than the unnormalized fundamental inequalities alone. For one split s,
conditional on its B stream, L_A,s/sqrt(V_A,s) is centered sub-Gaussian
with proxy one. If V_A,s=0 then L_A,s=0; take the ratio to be zero. For
0<alpha<1, Gaussian integration therefore gives

    E exp[alpha*L_A,s^2/(2V_A,s)] <= (1-alpha)^(-1/2).

By the Cauchy--Schwarz inequality for the finite partition average,

    L_A^2/V_A = (E_s L_A,s)^2/(E_s V_A,s)
                <= E_s[L_A,s^2/V_A,s].

Apply the increasing convex exponential, then Jensen, then expectation
over the data. The result is

    E exp[alpha*L_A^2/(2V_A)] <= (1-alpha)^(-1/2).

The same argument applies to B. If V_A=0, all split proxies are zero,
so L_A=0 and the convention again holds. Markov's inequality gives the
upper-tail-valid radius

    r_A = sqrt({2V_A/alpha}*[x-(1/2)log(1-alpha)]),
    r_B = sqrt({2V_B/alpha}*[x-(1/2)log(1-alpha)]).

Alpha may be optimized from x alone, without residual data. There is a
unique u_x>1 satisfying

    u_x-1-log(u_x)=2x.

The optimizing alpha is 1-1/u_x and the optimal radius is simply

    r_A=sqrt(u_x*V_A), r_B=sqrt(u_x*V_B).

This is implemented as `lambda_mode="square_mgf"`; the common argument
name is retained for API compatibility, although this mode chooses alpha.
At delta=.05, x=log(60), u_x is approximately 11.64, whereas the
unpenalized split coefficient is 2x approximately 8.19. The factor in
each linear radius is about 1.19. No universal realized-width dominance
follows. The bound does adapt proportionally to sqrt(V) at every fit quality
and sample size, without a logarithmic cost depending on V/D or grid length.

This does not retroactively justify optimizing a random V inside an arbitrary
fixed-t MGF. Its proof explicitly uses the stronger split conditional
normalization and the convex quadratic perspective under averaging.

An optional joint-linear variant follows from Cauchy and Holder:

    E exp{alpha*(L_A+L_B)^2/[4(V_A+V_B)]} <= (1-alpha)^(-1/2).

To see this, first bound (L_A+L_B)^2/[2(V_A+V_B)] by
(L_A^2/V_A+L_B^2/V_B)/2, then use Cauchy on the exponentials.
Thus with y=log(2/delta), one may replace r_A+r_B by
sqrt(2*u_y*(V_A+V_B)) and use product radius at y, spending delta/2 on
each of these two tails. It can improve constants when the two proxies
are comparable; it can be worse when they differ substantially. This
joint variant is proved but not implemented or selected after seeing data.

## 4. Proof of the averaged inequalities

For each category independently, average uniformly over all choices of a
left subset L_c of size k_c, with right complement R_c of size l_c. This is
an auxiliary finite average, not an extra randomization requirement in the
algorithm. Split selection depends only on the conditioned category labels.
For one partition s define

    a_s,c = mean_{i in L_c} A_ci,
    b_s,c = mean_{j in R_c} B_cj,
    e_A,s,c=a_s,c-a_c, e_B,s,c=b_s,c-b_c,
    T_s=sum p_c a_s,c b_s,c,
    L_A,s=-sum p_c b_s,c e_A,s,c,
    L_B,s=-sum p_c a_s,c e_B,s,c,
    R_s=sum p_c e_A,s,c e_B,s,c.

Then beta_S-T_s=L_A,s+L_B,s+R_s. Although both channels are observed in
each pair, this split expression uses A only on left and B only on right,
so its two streams are independent. Conditional on the right stream,
Hoeffding's lemma gives

    E exp[t L_A,s - (t^2/2) V_A,s] <= 1,
    V_A,s=sum p_c^2 b_s,c^2/(4k_c).

This remains true after averaging over the right stream. The symmetric
statement has V_B,s=sum p_c^2 a_s,c^2/(4l_c). Denote uniform partition
averaging by E_s. For i!=j,

    P_s{i in L_c,j in R_c}=k_c*l_c/[n_c(n_c-1)],

so E_s T_s=U, E_s a_s,c=Abar_c and E_s b_s,c=Bbar_c. Consequently
E_s L_A,s=L_A, E_s L_B,s=L_B, and E_s R_s=R.

For a simple random subset of size l from n fixed values, a direct
indicator expansion yields

    E_s[subset_mean^2] = full_mean^2 + (n-l)*s^2/(n*l).

Hence E_s V_A,s=V_A and E_s V_B,s=V_B. Jensen's inequality, pointwise in
the whole observed data, now gives

    exp[t L_A - (t^2/2)V_A]
      <= E_s exp[t L_A,s - (t^2/2)V_A,s].

Take expectation over the data and use the split conditional-MGF result
inside the finite average. This proves the first fundamental inequality;
the second follows identically. This is the exact step that makes the
observable proxy legitimate without pretending an estimated mean is fixed.

For the remainder, each split's errors in category c are independent
centered sub-Gaussian variables with proxies 1/(4k_c),1/(4l_c).
For such X,Y with proxies s_X^2,s_Y^2, conditioning on X and integrating
a standard normal auxiliary variable gives

    E exp(t XY) <= (1-t^2*s_X^2*s_Y^2)^(-1/2).

Independence across categories therefore gives

    log E exp(t R_s) <= -(1/2)sum_c log(1-t^2*d_c^2)
                       <= t^2 D/[2(1-t*dmax)].

The last step follows from -log(1-u^2)<=u^2/(1-|u|), |u|<1.
All partitions have the same counts and thus the same d_c. Jensen again
gives exp(tR)<=E_s exp(tR_s), proving the claimed remainder MGF. Its usual
Chernoff optimization yields the displayed sub-gamma tail. Markov's
inequality at fixed lambda, a union bound over a fixed grid, or the shown
Gaussian integration supplies the three alternative linear radii.
A final union bound over the two linear terms and the remainder proves the
result. Events need not be independent. Add deterministic omitted-cell
allowances, then remove conditioning if desired.

## 5. Exact-fit radius and variance

At a_c=b_c=0, unbiasedness of the sample variance and of a random subset
mean give

    E[Bbar_c^2+k_c*s_Bc^2/(n_c*l_c)] = sigma_Bc^2/l_c,
    E[Abar_c^2+l_c*s_Ac^2/(n_c*k_c)] = sigma_Ac^2/k_c.

Since each marginal has range one, sigma_Ac^2,sigma_Bc^2<=1/4. Thus
E V_A<=D and E V_B<=D. The fixed-lambda construction therefore has

    E[r_A+r_B+r_2] <= 3sqrt(2xD)+x*dmax.

For balanced p_c=1/C and n_c=N/C even, where N is the **total** number of
validation pairs, this is

    [3sqrt(C*x/2)+x/2]/N.

This is exactly the split construction's displayed bound after setting its per-stream count
m=N/2. The result compares proved expected-radius envelopes; it does not
assert pointwise width dominance or universal test-power dominance.
The mixture boundary is also second-order: its radius function r(V) is
concave for V>=0 (differentiation leaves a negative numerator), so for
rho=D/(2x),

    E r(V_A), E r(V_B)
      <= sqrt(D*(1+1/(2x))*[2x+log(1+2x)]).

The fixed finite grid retains the same N^(-1) rate at fixed C,delta; its
additional log J is constant for the declared 17-point grid. Balanced-count
results do not silently extend to rare/missing cells: H and actual counts
must be retained in those settings.

For the square-normalized mode, Jensen directly gives

    E[r_A+r_B+r_2] <= [2sqrt(u_x)+sqrt(2x)]sqrt(D)+x*dmax.

This is the same second-order rate, with an explicit constant cost in the
linear terms. More generally, without exact fits,

    E V_A = sum p_c^2/(4k_c) * [b_c^2+sigma_Bc^2/l_c],
    E V_B = sum p_c^2/(4l_c) * [a_c^2+sigma_Ac^2/k_c].

Thus the square-normalized mode has the usual root-N linear-error rate
and second-order residual rate. For balanced counts its linear radii obey

    E r_A <= sqrt{u_x*[sum p_c b_c^2/(2N)+C/(4N^2)]},
    E r_B <= sqrt{u_x*[sum p_c a_c^2/(2N)+C/(4N^2)]}.

This is a finite-sample expected-radius statement; it does not establish
a high-probability width or power comparison.

For comparison, write sigma_AB,c=Cov(A_ci,B_ci). Direct Hoeffding
decomposition of the symmetric within-cell kernel gives the exact variance

    Var(U) = sum_c p_c^2 * {
       [b_c^2*sigma_Ac^2+a_c^2*sigma_Bc^2+2a_c*b_c*sigma_AB,c]/n_c
       +[sigma_Ac^2*sigma_Bc^2+sigma_AB,c^2]/[n_c(n_c-1)] }.

Indeed the first projection is
(b_c(A_ci-a_c)+a_c(B_ci-b_c))/2, and the canonical second projection is
[(A_ci-a_c)(B_cj-b_c)+(A_cj-a_c)(B_ci-b_c)]/2. Orthogonality across
projections and distinct pairs yields the formula. At exact fits,

    Var(U) <= sum_c p_c^2/[8 n_c(n_c-1)].

An oriented size-(k_c,l_c) split center has variance

    sum_c p_c^2 * [b_c^2*sigma_Ac^2/k_c
                 +a_c^2*sigma_Bc^2/l_c
                 +sigma_Ac^2*sigma_Bc^2/(k_c*l_c)].

U is the conditional average of such split centers, and variance cannot
increase under that averaging. At exact fits and even n_c, the per-cell
pooled/split variance ratio, when both marginal variances are nonzero, is

    n_c/[4(n_c-1)] * (1+Corr(A_ci,B_ci)^2).

It lies between n_c/[4(n_c-1)] and n_c/[2(n_c-1)]. The off-diagonal center
removes within-pair covariance **bias**, while covariance still affects
its variance. The Jensen concentration proxy for R is conservative: it
retains the split MGF proxy and does not realize every possible variance
improvement in its confidence radius.

## 6. Fair comparisons and preserved limitations

Use the same actual independent focal/reference pairs and the same exact
category masses for all candidates. The required comparison separates:

1. Original global split at the globally declared time/index half.
2. One split balanced within focal category, chosen without residual values.
3. This pooled estimator and fixed-lambda bound.
4. The predeclared-grid or mixture pooled bound.
5. A full-data simultaneous rectangle using the same exact masses.

This separates a count-allocation benefit from a pooling benefit. Within
category balancing is permissible under the conditional-label contract.
Both-channel pooled methods use no additional raw observations, outcomes,
or category metadata. A split sum with a smaller radius on one realization
does not establish an estimator optimum; the pooled center has smaller
variance, but the observable finite-sample radii need not dominate.

Several tempting alternatives do not establish the required guarantee:

* The same-unit average of A_i B_i estimates a_c b_c+sigma_AB,c, not a_c b_c.
* Averaging individually valid upper bounds at the same delta is not in
  general valid at delta. For target zero, let bound 1 be -1 on an event
  of probability delta and epsilon elsewhere, and bound 2 be -1 on a
  disjoint event of probability delta and epsilon elsewhere, with
  0<epsilon<1. Each is valid at delta, while their average fails on the
  union, of probability 2delta. A finite collection may be averaged with
  a union-bound error allocation; this is valid but pays log(number of
  splits) if every split uses a conventional exponential tail. Averaging
  the exponentials first is the alternative proved here.
* Substituting the realized V into sqrt(2xV) by optimizing a fixed-t MGF
  after seeing data is unsupported. The fixed lambda, finite grid, or
  mixture above supplies the missing uniformity.
* A generic bounded-kernel Hoeffding inequality does not adapt to the
  zero first projection and ordinarily has an N^(-1/2) radius, losing the
  second-order exact-fit behavior.
* Variance alone does not make a deployable adaptive bound: it involves
  unknown means and covariances. If exact fits were known in advance, a
  one-sided Chebyshev/Cantelli bound could use the displayed null variance,
  but this cannot validate unknown learning bias near, rather than exactly
  at, the null. The null-only bound must not be presented as an all-fit
  finite-sample validator.
* With one arbitrary paired observation there is no unbiased estimator
  of the product of its marginal means. For a distribution assigning
  probability t to (1,1) and 1-t to (0,0), an estimator's expectation is
  affine in t, whereas the product is t^2. This observation motivates the
  n>=2 requirement for an unbiased center; it does not prove that all
  single-observation confidence bounds are uninformative.

No direct exact-variance exponential inequality improving all constants
over the Jensen bound has been proved here. No numeric claim of new power,
coverage, or public-domain generalization is made by this note. Empirical
comparison and independent proof checks remain distinct tasks.

## 7. Implementation and attribution

The split construction is documented in `../aggregate_bias/THEORY.md`.
The fully averaged center and projection decomposition use classical
U-statistic algebra. The present comparison supplies explicit finite
partition averaging and observable proxy formulas; it makes no priority
claim about second-order estimation or concentration theory.

`pooled_bias_bound.py` implements `pooled_upper_bound`. The modes and grids
must be chosen before inspecting inference values. Choosing a random best
mode requires a shared error allocation. The accompanying study keeps
primary comparisons separate from later square-normalized extensions.
