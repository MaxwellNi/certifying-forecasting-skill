# Ranks of raw two-lag sums: exact source-window interaction

The equal-weight two-lag sum admits an exact scalar source-window criterion, despite having no fixed linear representation in single-lag comparisons. The extension is finite-sample, covers ties, and separates the population-reference temporal target from reference reuse. Its distributional ingredients are classical bivariate rank covariances. The useful incremental result is their explicit reduction for this forecast class, the source-window design mapping, and sharp cancellation criteria.

The raw-array checker is `check_two_lag.py`; its frozen receipt is `two_lag_verification.json`. `two_lag_bound.py` implements the finite bound and raw six-draw validation kernel, with separate checks in `test_two_lag_bound.py`.

## 1. Objects and fixed-design mapping

Let every raw observation $Y_{it}$ be independent with common law $F$, and define

$$
D_s=\{s-1,s-2\},\quad X_{is}=Y_{i,s-1}+Y_{i,s-2},\quad H=F*F.
$$

No moment assumption on $F$ is needed because every comparison is bounded. Write

$$
a(x,y)=\mathbf1\{y<x\}+\tfrac12\mathbf1\{y=x\},\qquad
M_G(x)=\mathbb E_{Z\sim G}a(x,Z).
$$

For a fixed focal entity $i$ and peers $j\ne i$, let

$$
P_{sj}=a(X_{is},X_{js})-\tfrac12,\qquad
Q_{tj}=a(Y_{it},Y_{jt})-\tfrac12,
\quad A=\sum_{s,j}c_{sj}P_{sj},\quad B=\sum_{t,j}d_{tj}Q_{tj}.
$$

For temporal mean fits, $c_{sj}=\alpha_su_{sj}$, $d_{tj}=\beta_tv_{tj}$, where $\alpha_s=\mathbf1\{s=r\}-f_{rs}$, $\beta_t=\mathbf1\{t=r\}-g_{rt}$, the fit weights each sum to one, and each peer vector sums to one. These quantities must be fixed independently of the evaluated raw array. Normalization cancels deterministic rank constants. No claim is made for learned random residual coefficients or random sample-scale denominators.

The induced population-reference counterparts replace $P_{sj}$ by $p_s=M_H(X_{is})-1/2$ and $Q_{tj}$ by $q_t=M_F(Y_{it})-1/2$. Thus the forecast target uses $F*F$, not $F$. Both transformed means are zero, including under ties.

Define

$$
T_2=\sum_s\sum_{t\in D_s}(\sum_jc_{sj})(\sum_jd_{tj}),\qquad
\Omega_2=\sum_s\sum_{t\in D_s}\sum_jc_{sj}d_{tj}.
$$

The compiler needs the two actual source times of each raw sum. It must not replace the sum comparison by the sum of its component comparisons. For example, focal-minus-peer component differences $(2,-1)$ and $(1,-2)$ have identical comparison signs and opposite sum signs.

## 2. Exact pair and score expectations

For $Y_1,Y_2\stackrel{\mathrm{iid}}\sim F$, put

$$
\kappa_F=\operatorname{Cov}\{M_H(Y_1+Y_2),M_F(Y_1)\},\qquad
s_2=\Pr(Y_1=Y_2),\quad
\eta_F=\frac{1-s_2^2}{8},\quad \lambda_F=\eta_F-\kappa_F.
$$

Then the complete same-focal pair operator is

$$
\mathbb E[P_{sj}Q_{tk}]
=\mathbf1\{t\in D_s\}\{\kappa_F+\mathbf1\{j=k\}\lambda_F\}.
\tag{1}
$$

Consequently,

$$
\mathbb E[A_0B_0]=\kappa_FT_2,\qquad
\mathbb E[AB]=\kappa_FT_2+\lambda_F\Omega_2.
\tag{2}
$$

To prove (1), disjoint source times give independent centered comparisons. At an overlapping time and distinct peers, condition on both focal components: the two peers are independent and the resulting product mean is $\kappa_F$. With the same peer, introduce independent symmetric differences $D_1=Y_1-Y'_1,D_2=Y_2-Y'_2$. The covariance is

$$
\eta_F=\tfrac14\mathbb E[\operatorname{sgn}(D_1+D_2)\operatorname{sgn}(D_1)].
$$

Symmetry of the components gives

$$
2\mathbb E[\operatorname{sgn}(D_1+D_2)\operatorname{sgn}(D_1)]
=\mathbb E[\operatorname{sgn}(D_1+D_2)(\operatorname{sgn}D_1+\operatorname{sgn}D_2)].
$$

The integrand equals two for nonzero differences of the same sign, one when exactly one difference is zero, and zero otherwise. Since $\Pr(D_k=0)=s_2$, the expectation is $1-s_2^2$. This proves the closed form for $\eta_F$, with no assumptions on tie sizes or on spacing between atoms. Expanding the finite sums proves (2).

Both $\kappa_F$ and $\lambda_F$ are nonnegative. For $\kappa_F$, condition the first transform on $Y_1$; the conditional mean and $M_F(Y_1)$ are nondecreasing functions. For $\lambda_F$, condition on focal values $x_1,x_2$, average out the second peer component, and take the covariance in the first peer component $U$ of

$$
U\mapsto\mathbb E_V a(x_1+x_2,U+V),\qquad U\mapsto a(x_1,U).
$$

Both are nonincreasing. Their covariance is nonnegative by the independent-copy identity $\operatorname{Cov}(f(U),g(U))=\tfrac12\mathbb E[(f(U)-f(U'))(g(U)-g(U'))]$. In particular $0\le\lambda_F\le\eta_F\le1/8$.

## 3. Computable tied and continuous constants

For a finite law with $m$ ordered atoms, construct the convolution masses $H$, sort its at most $m(m+1)/2$ distinct sums, and compute its mid-CDF by prefix summation. Then

$$
\kappa_F=\sum_{a,b}p_ap_b\{M_H(y_a+y_b)-\tfrac12\}\{M_F(y_a)-\tfrac12\}.
\tag{3}
$$

Together with $\lambda_F=(1-(\sum_a p_a^2)^2)/8-\kappa_F$, this calculates the constants in $O(m^2\log m)$ time, with exact rational arithmetic when atom probabilities and values are rational. There is no opaque conditional-covariance oracle to supply. The independent checker instead obtains the same-peer moment from four raw draws and verifies (3) against temporal arrays.

| Law | $\kappa_F$ | $\lambda_F$ | $\eta_F$ |
|---|---:|---:|---:|
| Bernoulli($p$), $q=1-p$ | $pq(1-pq)/4$ | $pq(1-pq)/4$ | $pq(1-pq)/2$ |
| Bernoulli$(1/2)$ | $3/64$ | $3/64$ | $3/32$ |
| Uniform on $\{0,1,2\}$ | $13/243$ | $14/243$ | $1/9$ |
| Uniform on $\{-2,0,5\}$ | $13/243$ | $14/243$ | $1/9$ |
| Uniform$(0,1)$ | $7/120$ | $1/15$ | $1/8$ |
| Nondegenerate Gaussian | $\arcsin(1/(2\sqrt2))/(2\pi)$ | $1/8-\kappa_F$ | $1/8$ |

For binary laws, $a(Y,Y')-1/2=(Y-Y')/2$ after mapping the ordered support to $\{0,1\}$. Antisymmetry gives $\eta_F=2\kappa_F$; evaluating the convolution mid-CDF gives the displayed expression. For uniform $F$, integrating the triangular CDF gives

$$
\mathbb E[M_H(x+Y)-1/2]
=-x^3/3+x^2/2+x/2-1/3,
$$

whose product with $x-1/2$ integrates to $7/120$. For Gaussian $F$, $(Y_1+Y_2,Y_1)$ has correlation $\rho=1/\sqrt2$. The classical Gaussian orthant calculation gives $\eta=\arcsin\rho/(2\pi)$, while using two independent reference observations gives correlation $\rho/2$ and $\kappa=\arcsin(\rho/2)/(2\pi)$.

## 4. Sharp design conditions and explicit constructions

For a fixed design, the following are equivalent uniformly over all source laws:

- Population-reference expectation preservation: $\mathbb E[AB]=\mathbb E[A_0B_0]$ for every $F$ if and only if $\Omega_2=0$.
- Population-reference centering: $\mathbb E[A_0B_0]=0$ for every $F$ if and only if $T_2=0$.
- Empirical centering: $\mathbb E[AB]=0$ for every $F$ if and only if $T_2=\Omega_2=0$.

Sufficiency follows from (2). Binary $F$ proves necessity for the first two statements. For the third, binary $F$ requires $T_2+\Omega_2=0$, and uniform ternary $F$ requires $13T_2+14\Omega_2=0$; hence both vanish. These are criteria over all $F$. For one known $F$, the single equality $\kappa_FT_2+\lambda_F\Omega_2=0$ can hold accidentally with nonzero coefficients.

Source separation is sufficient, but not necessary. Full separation between forecast residual source windows and outcome residual source times forces both coefficients to zero. Using different peers whenever source times match forces only $\Omega_2=0$. Signed cancellation can also force either coefficient to zero without disjointness.

For an exact example, evaluate the rank score

$$
S=(P_{i2}-P_{i5})\{Q_{i2}-(Q_{i0}+Q_{i4})/2\}.
$$

The forecast ranks are ranks of $Y_{i1}+Y_{i0}$ and $Y_{i4}+Y_{i3}$. The total source coefficient is $T_2=-1/2+1/2=0$. Give both forecast ranks peer 1, $Q_{i0}$ peer 2, and $Q_{i4}$ peer 1. Then $\Omega_2=1/2$, so $\mathbb ES=\lambda_F/2$. Swapping the two outcome references gives $-\lambda_F/2$; assigning peer 1 to both gives zero despite source reuse. All preserve their common-law population-reference target. For binary $F$, the three means are $3/128,-3/128,0$.

The directional example $A=P_{i3}-P_{i2}$, $B=Q_{i3}-Q_{i4}$ uses forecast sources $\{0,1,2\}$ and outcome sources $\{3,4\}$, so both expectations vanish. In contrast, $A=P_{i2}-P_{i3}$, $B=Q_{i2}-(Q_{i0}+Q_{i1})/2$, with disjoint peer sets, has $\Omega_2=0$, $T_2=-3/2$, and ternary expectation $-13/162$. Peer separation preserves this nonzero population target.

## 5. Where one overlap scalar fails

For an equal-weight sum against a raw one-period outcome, one scalar really does suffice: all nonzero overlap pairs share the same distributional constant. It would be incorrect to claim failure of every scalar extension here.

For $X_{is}=w_1Y_{i,s-1}+w_2Y_{i,s-2}$ with fixed positive unequal weights, use two component constants and two overlap coefficients:

$$
\mathbb E[A_0B_0]=\sum_{\ell=1}^2\kappa_{F,\ell}T_\ell,\qquad
\mathbb E[AB]-\mathbb E[A_0B_0]=\sum_{\ell=1}^2\lambda_{F,\ell}\Omega_\ell.
$$

In the Gaussian class, $\rho_\ell=w_\ell/(w_1^2+w_2^2)^{1/2}$, $\kappa_\ell=\arcsin(\rho_\ell/2)/(2\pi)$, and $\lambda_\ell=[\arcsin\rho_\ell-\arcsin(\rho_\ell/2)]/(2\pi)$. Thus an unweighted total $\Omega_1+\Omega_2$ loses relevant information.

The preceding $P_2-P_5$ design with a single common peer, binary $F$, and weights $(2,1)$ supplies a rational counterexample. Its lag overlaps are $(\Omega_1,\Omega_2)=(1/2,-1/2)$, and $(\kappa_1,\kappa_2)=(\lambda_1,\lambda_2)=(1/16,1/32)$. The naive total overlap is zero, but the reference interaction is $1/64$, population expectation $1/64$, and empirical expectation $1/32$. These values are checked directly on raw arrays. Different laws across source times or peers likewise require indexed constants. Ranks of sums on both score sides also admit multiple overlap sizes and generally require more than one constant.

## 6. Cross-focal covariance operator

At an overlapping component, regard $(X_i,Y_i)$ as independent identically distributed bivariate entity vectors. For a row-stochastic zero-diagonal reference matrix $U$, let $L_U=I-U$ and let $B_U$ orient the comparison edges as in the existing companion. The usual two-term Hoeffding projection gives

$$
\operatorname{Cov}(p_s^U,q_t^V)
=\mathbf1\{t\in D_s\}\{\kappa_FL_UL_V^T+(\lambda_F-\kappa_F)B_UB_V^T\}.
\tag{4}
$$

Node cross-covariance is $\kappa_F$; same-edge degenerate cross-covariance is $\eta_F-2\kappa_F=\lambda_F-\kappa_F$; distinct edges have zero covariance by conditional degeneracy. For distinct focal entities $i,l$, the entry is

$$
\kappa_F(e_i-U_i)^T(e_l-V_l)
-(\lambda_F-\kappa_F)U_{il}V_{li}.
$$

Thus raw entity independence does not imply rank-score independence. The checker verifies all nine entries of unequal-reference three-entity matrices for five source laws, rather than only same-focal entries.

For fixed $N$, complete peers, no ties, and variance divisor $N$, the empirical rank variance is deterministically $(N+1)/\{12(N-1)\}$. At a matched source component the standardized same-focal expectation is

$$
\frac{12\{(N-1)\kappa_F+\lambda_F\}}{N+1}
=\frac{(N-2)\rho_S+3\tau}{N+1},
$$

where the final equality uses continuous bivariate rank coefficients. This is the classical expected sample Spearman formula, not a new standardization theorem. Random tie-dependent denominators are outside (2).

## 7. Finite inference from independent whole panels

Let $S_g=A_gB_g$, $g=1,\ldots,M$, be independent whole-panel replicates from the same $F$ and fixed design. The estimand is $\theta_0=\mathbb E[A_0B_0]$, so $\mathbb ES_g=\theta_0+\Omega_2\lambda_F$. It need not be a forecasting-gain estimand or zero under a particular temporal fit.

For validation, take independent draws $Z=(Z_1,Z_2), Z'=(Z'_1,Z'_2), Z''=(Z''_1,Z''_2)$, all six coordinates from $F$, and define

$$
R=\tfrac12\{a(Z_1+Z_2,Z'_1+Z'_2)-a(Z_1+Z_2,Z''_1+Z''_2)\}
\{a(Z_1,Z'_1)-a(Z_1,Z''_1)\}.
$$

Conditional independence of $Z',Z''$ given $Z$ gives $\mathbb ER=\lambda_F$; the second peer comparison must reuse its two-component trajectory across the two factors. The kernel range is $[-1/2,1/2]$, and its expectation lies in $[0,1/8]$. For $n$ fresh validation triples of trajectories and budgets $0<\delta<\xi<1$, set

$$
\epsilon_\lambda=\sqrt{\frac{\log(2/\delta)}{2n}},\qquad
[l,u]=[\overline R-\epsilon_\lambda,\overline R+\epsilon_\lambda]\cap[0,1/8].
$$

If the intersection is empty, return the full interval $[0,1/8]$; this makes the procedure defined on every sample. A valid upper bound for the score range width is

$$
R_S=\tfrac12\Big(\sum_{s,j}|c_{sj}|\Big)\Big(\sum_{t,j}|d_{tj}|\Big),
$$

For nonnegative normalized fit and peer weights, with the evaluated row excluded from each fit, this gives the valid upper bound $R_S=2$. It does not assert that both range endpoints are attained. The lower bound

$$
L=\overline S-R_S\sqrt{\frac{\log\{1/(\xi-\delta)\}}{2M}}
-\Omega_2\begin{cases}u,&\Omega_2\ge0,\\l,&\Omega_2<0\end{cases}
\tag{5}
$$

satisfies $\Pr(\theta_0\ge L)\ge1-\xi$. This follows directly from two classical Hoeffding bounds and a union bound. No distributional superiority or observational-panel calibration follows. The raw-array checker verifies the validation-kernel expectation separately for five tied laws.

Validation can be omitted when $\Omega_2=0$. A validation-free conservative alternative is $\overline S-R_S\sqrt{\log(1/\xi)/(2M)}-\max(\Omega_2,0)/8$, using the proved range of $\lambda_F$. With complete peers, $\Omega_2=T_2/(N-1)$, so this maximum correction decreases with peer count for fixed temporal design.

For budgeting, let a whole panel cost $c_M$ raw draws and one validation replicate cost six. Ignoring interval clipping, the two estimation penalties are $a/\sqrt M+b/\sqrt n$, where $a=R_S\sqrt{\log\{1/(\xi-\delta)\}/2}$ and $b=|\Omega_2|\sqrt{\log(2/\delta)/2}$. For $a,b>0$, under $c_MM+6n=C$ the optimal continuous allocation has $n/M=(bc_M/(6a))^{2/3}$. This exposes the required data cost; it does not promise sensitivity to arbitrarily small targets.

## 8. Verification and literature boundary

Run `python check_two_lag.py --output /tmp/two-lag-results.json` from this directory, or run the file by path. It needs only the Python standard library. The recorded run passes 183,523 complete raw temporal arrays. Additional exact checks cover four-draw distributional constants, all cross-focal covariance entries, and six-draw validation kernels for binary, skew binary, ternary, gapped ternary, and skew ternary laws. Integer-scaled accumulation retains exact fractions and prevents rounding-created ties. The checker imports no paper or release implementation.

Run the separate API checks with `python -m unittest discover -s . -p test_two_lag_bound.py -v` from this directory. All nine API tests pass, including the optional NumPy regression test in the tested environment. They check exact validation moments, both endpoints of the validation kernel, large-integer ordering and true NumPy `int64` maximum-plus-maximum sums, sign-correct bias subtraction, error-budget allocation, and rejection of malformed observations, overflowing source sums, unrepresentable coefficient ranges, and nonfinite derived bounds. Integral scalar inputs are converted to Python integers before any source summation; no floating conversion creates ties. These are deterministic implementation checks, not claims of empirical coverage or power.

The public source pages were checked on 2026-09-14. The closest primary references establish that the bivariate rank moments and projection calculation are classical:

- Hoeffding, W. (1948), *A Class of Statistics with Asymptotically Normal Distribution*, Annals of Mathematical Statistics 19(3), 293–325. Its rank-statistic examples and projection machinery are the foundation for (4). [Primary publisher record](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-19/issue-3/A-Class-of-Statistics-with-Asymptotically-Normal-Distribution/10.1214/aoms/1177730196.full).
- Moran, P. A. P. (1948), *Rank correlation and product-moment correlation*, Biometrika 35(1–2), 203–206. The Gaussian rank-correlation and expected-sample-Spearman calculation belongs here. The publisher record was accessible; its full PDF requires access. [Primary publisher record](https://academic.oup.com/biomet/article-abstract/35/1-2/203/179031).
- Croux, C. and Dehon, C. (2010), *Influence functions of the Spearman and Kendall correlation measures*, Statistical Methods & Applications 19, 497–515. This primary paper states the classical Gaussian relationships and supplies proofs in its appendix. [Open-access publisher article](https://link.springer.com/article/10.1007/s10260-010-0142-z), [verified open publisher PDF, pp. 499–500, equations (1)–(5)](https://link.springer.com/content/pdf/10.1007/s10260-010-0142-z.pdf).

The derivation does not claim to have discovered Gaussian arcsine moments, covariance decomposition, or bounded-mean concentration. Its concrete increment is the tie-aware additive-source identity $\eta_F=(1-s_2^2)/8$, computable convolution constants, source-window mapping for a forecast class excluded by the old lag-copy representation, sharp uniform design criteria, and verified failures of collapsing unequal-lag contributions. The additive identity is presented as a derived specialization. This bounded literature check does not establish priority for that specialization.
