# Conditional source designs with externally learned transformations

This extension gives a usable conditional experiment, not a new concentration
inequality. The important change is to treat an entire trajectory as the raw
sampling unit. A model trained on genuinely separate information can be an
arbitrary fixed measurable map of that trajectory. Within-trajectory dependence,
unequal lag weights, ties, and source-specific marginal laws are allowed. The
extra reference term is estimated by one explicit aggregate kernel rather than
left as an unknown collection of constants.

## 1. Information, sampling unit, and target

Let H_train be training information. Conditional on H_train, assume all
evaluation and validation trajectories are independent draws from a common law
P_H. A sufficient special case is that H_train is independent of those draws
and their law is a fixed P. A chronological split or distinct row number alone
does not imply this condition: raw source reuse must be tracked, and actual
stochastic independence still needs a sampling/design assumption. A fixed
archive sampled with replacement supplies conditional IID indices given the
archive; it does not make the original chronological records IID or establish
future forecasting performance. In this archive experiment the conditioning
information includes the fixed archive as well as training. Independent index
draws may return repeated archive records; distinct returned IDs are not needed
for conditional independence of the random draws.

H_train can fix arbitrary scalar measurable maps f_s(Z), s=1,...,S, and
g_t(Z), t=1,...,T, their finite source windows, peer weights, residual-fit
coefficients, evaluation sizes, error budgets, and a finite candidate family.
For example f_s can be an externally fitted nonlinear forecasting model at a
specified origin, while g_t is an outcome at another time in the same
trajectory. Within-trajectory dependence and unequal coordinate marginals are
unrestricted in the primary theorem. Predictions must obey the separate
historical feature-availability contract if given a forecasting interpretation.

For J peers define H_train-measurable matrices C in R^(S x J) and D in
R^(T x J), with arbitrary finite signed entries. Set c=C1, d=D1, and
Omega=C D^T. For one focal trajectory Z_0 and J independent peers Z_j, let

    p_sj = a(f_s(Z_0),f_s(Z_j)) - 1/2,
    q_tj = a(g_t(Z_0),g_t(Z_j)) - 1/2,
    a(x,y)=1{y<x}+(1/2)1{y=x},
    A=sum_sj C_sj p_sj,  B=sum_tj D_tj q_tj,  S_block=A B.

Normalized temporal mean fits have C_sj=alpha_s u_sj,
alpha_s=1{s=r}-fit_weight_rs, and analogously for D. Reference vectors sum
to one; these cancel deterministic rank offsets. The broader matrix theorem
also allows fixed signed comparison constructions directly. Constant maps
have p or q identically zero.

Write M_f_s(x)=E[a(x,f_s(Z'))|H_train], and likewise M_g_t. The conditional
population-reference target is

    theta_H = E[(sum_s c_s [M_f_s(f_s(Z_0))-1/2])
                (sum_t d_t [M_g_t(g_t(Z_0))-1/2]) | H_train].       (1)

This is the fitted population-reference residual association for the stated
trajectory law and fixed learned design. It is not automatically the paper's
other theta_infinity, full continuous-control covariance, a conditional
independence parameter, or expected future loss gain. Training randomness
makes theta_H random before training. A conditional confidence statement is
about this trained rule's target, not about E_H theta_H unless another
argument is supplied.

## 2. Exact conditional operator

Suppress H_train in notation, but take every expectation below conditionally.
Define

    K_st = E[(M_f_s(f_s(Z_0))-1/2)(M_g_t(g_t(Z_0))-1/2)],
    Lambda_st = E Cov(a(f_s(Z_0),f_s(Z_1)),
                      a(g_t(Z_0),g_t(Z_1)) | Z_0,H_train).

Then, exactly including ties,

    E[p_sj q_tk | H_train] = K_st + 1{j=k} Lambda_st,
    theta_H = c^T K d,
    E[S_block | H_train] = theta_H + <Omega,Lambda>.             (2)

For j != k, condition on Z_0: the two reference trajectories are independent,
so the conditional product equals the product of the conditional means. For
j=k, the conditional covariance is the additional term. Integrating and
expanding proves (2). No linear representation of f_s in its raw coordinates
is assumed. H_train can choose the design arbitrarily as long as the stated
conditional product experiment remains true almost surely.

Neither K_st nor Lambda_st need be nonnegative. For f(Z)=Y and g(Z)=-Y with
Y~Bernoulli(1/2), both equal -1/16. The nonnegative scalar Lambda in the
equal-weight monotone-lag theorem must not be reused without its assumptions.
The safe bound |Lambda_st|<=1/4 follows from conditional covariance of two
range-one variables. Under the common-reference-law contract, Cauchy-Schwarz
and the classical midrank conditional-variance formula improve it to 1/6:
E Var(a(X,X')|X)=(2-3 sum p_atom^2+sum p_atom^3)/12<=1/6.

When the raw coordinates are conditionally independent with possibly
source-specific marginals F_b,H, and f_s/g_t depend on source sets D_s/E_t,
both K_st and Lambda_st vanish if D_s and E_t are disjoint. This gives an
active overlap mask G_st=1{D_s intersects E_t}. Formula (2) can then use
Omega_active=G elementwise-multiplied by Omega for the correction. If internal
source dependence is allowed, disjoint coordinate names do NOT imply this
zero: use the full matrix Omega unless another conditional independence
statement has been established.

## 3. Sharp universal design criteria, and their limits

Fix H_train and the coefficient matrices. Over all common trajectory laws and
all measurable maps (constant maps included), the following are equivalent:

    preservation: E[S_block]=theta_H for every maps/law
                  iff Omega=0 entrywise;                       (3)
    empirical centering: E[S_block]=0 for every maps/law
                  iff c d^T=0 and Omega=0 entrywise.             (4)

For necessity of (3), isolate any chosen pair (s,t): set f_s=g_t=Y for a
Bernoulli(1/2) component and make every other map constant. Only the selected
Lambda_st=1/16 remains, so Omega_st must vanish. For (4), the same isolation
with binary and uniform ternary Y gives (K,Lambda)=(1/16,1/16) and
(2/27,5/54). Their ratios differ, forcing c_s d_t=Omega_st=0. Sufficiency
is (2). The outer product c d^T=0 means c=0 or d=0: one population residual
is identically zero. Demanding uniform centering over every possible learned
map and dependence law therefore removes every nontrivial target. Preservation
is the useful audit goal; centering requires a substantively declared null.

For fixed source windows and the smaller class of conditionally independent
coordinate laws, (3)-(4) require the corresponding entries to vanish only on
the active overlap graph. Necessity isolates a shared coordinate of any active
pair; sufficiency uses the disjoint-source zeros. Under a restricted fixed map
class further equalities between Lambda entries can permit signed scalar
cancellation, as in the previous equal-weight two-lag theorem. That scalar
criterion is not universal for the expanded class.

A usable sufficient construction is to allocate same-law reference trajectories
to disjoint forecast and outcome pools before audit draws. Then C D^T=0 while
keeping c and d unchanged; (1) is preserved exactly. With ordinary nonnegative
peer weights and nonzero temporal coefficients, entrywise orthogonality means
the relevant reference supports are disjoint. The construction must preserve
each side's declared reference law; replacing grouped references by a pooled
law changes the target even when the new reference draws are independent.

### Exact examples

* **Source-specific independent marginals defeat scalar cancellation.** Let
  Z=(Y_1,Y_2,Y_3) have independent components with probabilities (1/2,1/2,1/10).
  Take f=(Y_1,Y_3), g=(Y_1,Y_2,Y_3),
  C=((1,0),(-1,0)),
  D=((1/2,0),(-1/2,-1/2),(1/2,0)). Both temporal coefficient sums are zero.
  The active overlaps are Omega_11=1/2 and Omega_23=-1/2, summing to zero.
  Yet Lambda_11=1/16 and Lambda_23=9/400, hence interaction=1/50,
  population target=1/50, and empirical mean=1/25. Equal overlap counts do
  not equate different source-law constants.
* **Dependent coordinates defeat source-name separation.** Z=(Y,Y), with
  Y~Bernoulli(1/2), has distinct coordinate names. Using f(Z)=Z_1 and
  g(Z)=Z_2 with one common peer gives K=Lambda=1/16. A compiler that treats
  different coordinate IDs as stochastic independence incorrectly reports zero.
* **External training is allowed; contaminated training is not.** An independent
  training coin can choose either a nonlinear map or a sign/peer design. Each
  conditional identity holds, and their weighted average gives the unconditional
  identity. By contrast let H=1{Y_0=Y_1} use the actual focal/reference Bernoulli
  draws. Conditional on H=1 both individual marginals remain Bernoulli(1/2),
  but their comparison is zero. Plugging independent-reference constants gives
  1/8 for its squared score instead of the actual zero. Distinct row labels
  cannot repair reuse of the same primitive source information.
* **Pooling changes the reference target.** For a focal Y~Bernoulli(1/2), a
  same-group reference gives E[(M(Y)-1/2)^2]=1/16. A half/half pool of that
  law and its shift by ten gives 5/64. Even with exact mean recentering the
  induced variance changes from 1/16 to 1/64. Reference independence does
  not license switching these laws.

### Multiple evaluated rows: aggregate products, not residuals

Take a common dictionary of trajectory maps and fixed row weights v_r, with
rowwise matrices C_r,D_r. For S_total=sum_r v_r A_r B_r define
T_total=sum_r v_r c_r d_r^T and Omega_total=sum_r v_r C_r D_r^T.
Then E[S_total|H]=<T_total,K>+<Omega_total,Lambda>. This is a weighted sum
of the stated row targets, not the product of averaged residuals. The same
entry-isolation arguments prove uniform preservation iff Omega_total=0,
and uniform centering iff T_total=Omega_total=0. In the latter case the
aggregate population target is identically zero, though it need not be
represented by a single outer product.

The validator below uses Omega_total with the SAME three trajectories.
Its width is ||Omega_total||_1; a score width is
sum_r |v_r| ||C_r||_1 ||D_r||_1/2. The block APIs can be called per row
on the same mapped block, followed by the declared weighted sum and a
mean_bound with the aggregate width; do not treat rows sharing trajectories
as independent. Full-U comparison uses the sum of row U kernels with width
sum_r |v_r| ||c_r||_1 ||d_r||_1/6 (or the no-larger entrywise
||T_total||_1/6 bound). All draw and primitive coordinate costs count the
union needed by the complete aggregate.

## 4. A single computable aggregate validation kernel

For three independent whole trajectories Z_0,Z_1,Z_2 from P_H, define vectors

    dp_s = a(f_s(Z_0),f_s(Z_1))-a(f_s(Z_0),f_s(Z_2)),
    dq_t = a(g_t(Z_0),g_t(Z_1))-a(g_t(Z_0),g_t(Z_2)),
    R = (1/2) dp^T Omega dq.                                    (5)

The conditional independent-copy covariance identity gives

    E[R | H_train] = <Omega,Lambda> =: gamma_H.                  (6)

The *same* Z_1 and Z_2 must be used in both factors; replacing them by four
independent peers makes the validation expectation zero. Formula (5) handles
arbitrary peer count J and all signed rows at once. It does not need a law
oracle or separate confidence events for individual K/Lambda entries.

A valid validation range width is W_R=sum_st |Omega_st| because dp,dq lie
in [-1,1] and R lies in [-W_R/2,W_R/2]. A tighter computable box bound is
max_{x in {-1,1}^S,y in {-1,1}^T}|x^T Omega y|, obtained by finite enumeration
for small designs. These are enclosing widths, not attained-range claims.
Under the independently justified overlap-mask branch, Omega can be replaced
by Omega_active throughout (5)-(6). The safe parameter clipping interval
[-sum|Omega|/6,sum|Omega|/6] can additionally be used in the common-law case;
it is not needed for validity.

## 5. Conservative and variance-aware block bounds

Let M evaluation blocks each supply focal+J peers, and n fresh validation
triples supply R_1,...,R_n. These blocks are conditionally independent given
H_train. The score width is

    W_S=(1/2)||C||_1 ||D||_1,                                  (7)

where the norms sum absolute entries. Coefficients, M,n, and total alpha with
0<delta<alpha<1 are fixed given H_train before either stream is observed.
For values X_1,...,X_m with known enclosing width W and m>=1, define

    h(X,W,epsilon)=W sqrt(log(1/epsilon)/(2m)).

For m>=2 let v_X=sum(X_i-Xbar)^2/(m-1), and define the classical
Maurer-Pontil empirical-Bernstein one-sided radius

    e(X,W,epsilon)=sqrt(2 v_X log(2/epsilon)/m)
                         +7W log(2/epsilon)/(3(m-1)).            (8)

Theorem 11 of Maurer-Pontil (2009) supports independent, not necessarily
identically distributed bounded variables. Use either preselected radius, or
the jointly allocated hybrid

    r(X,W,epsilon)=min{h(X,W,epsilon/2),e(X,W,epsilon/2)}.         (9)

For m=1 the hybrid falls back to h(X,W,epsilon). Degenerate W=0 contributes
radius zero. A minimum of unadjusted h/e at the same epsilon is not justified.

With r denoting the selected valid rule, set

    U_gamma=Rbar+r(R,W_R,delta),
    L=Sbar-r(S,W_S,alpha-delta)-U_gamma.                         (10)

Conditional on H_train, the validation upper event fails with probability
at most delta and the evaluation lower event fails with probability at most
alpha-delta. On their intersection L<=theta_H by (2) and (6), including
negative gamma_H. Therefore P(L<=theta_H | H_train)>=1-alpha. Averaging
also gives unconditional coverage for the random trained-rule target.

H_val denotes the separate validation information. Evaluation concentration
can be conditioned on H_train together with H_val because its blocks are
independent. The validation coverage statement is conditional on H_train,
not generally on H_val. Do not silently merge the two information sets and
claim conditional-on-validation 1-alpha coverage. For example, a mean-zero
Rademacher validation average with n=10 and delta=.05 has a negative
Hoeffding upper bound on the positive-probability event that all ten values
are -1; conditional coverage on that event is zero. The union argument in
(10) remains valid at the stated level. Validation-selected designs need a
simultaneous argument or an additional independent design-selection split.

If Omega=0, gamma_H is exactly zero: omit validation and use alpha for the
score mean. A validation-free conservative alternative subtracts
sum|Omega|/6 from Sbar-r(S,W_S,alpha) using the common-law bound above.
If blocks have H-fixed different designs or laws, use their individually
identified targets and widths: Hoeffding gives radius
sqrt{log(1/epsilon) sum_g W_g^2/(2M^2)} for their average. A shared single
validation law cannot estimate different block-law interactions; matched
validation or an explicitly weighted mixture design is required.

## 6. Strong same-information comparator and the minimal generic kernel order

The proposed validation branch has no unique ability to identify theta_H.
For J>=2, use the *same* focal+J peer block and set

    a_j=sum_s c_s p_sj,     b_j=sum_t d_t q_tj,
    D_focal=[(sum_j a_j)(sum_j b_j)-sum_j a_j b_j]/[J(J-1)].       (11)

Conditional independence of distinct references proves E[D_focal|H]=theta_H.
It reassigns original peer weights while preserving c and d, the only peer
weights entering the target. Its enclosing width is
W_D=(1/2)||c||_1 ||d||_1<=W_S. It uses no validation and no unknown constants.

A stronger complete within-block comparator uses every trajectory as a
focal. For N>=3 define a_ij=sum_s c_s[a(f_s(Z_i),f_s(Z_j))-1/2], and b_ij
analogously. Then

    U_N = sum_i [(sum_{j!=i}a_ij)(sum_{k!=i}b_ik)
                         -sum_{j!=i}a_ij b_ij] / [N(N-1)(N-2)]  (12)

is the complete degree-three U-statistic for theta_H. Its sharper enclosing
width is W_U=||c||_1 ||d||_1/6, one third of the fixed-focal width W_D.
To prove this including ties, first isolate one scalar map pair and let
s_ij=sgn(f_i-f_j), t_ij=sgn(g_i-g_j). On three trajectories its symmetric
kernel is sum_{i,j,k distinct}s_ij t_ik/24. Order f_1<=f_2<=f_3. If the three
f values are distinct, cancellation reduces the numerator to -2t_13.
If f_1=f_2<f_3 it is -(t_13+t_23); if f_1<f_2=f_3 it is
-(t_12+t_13); if all are tied it is zero. Thus the absolute kernel is at
most 1/12 in every case. Bilinearity gives the asserted coefficient bound,
and U_N is an average of these three-trajectory kernels. Exact enumeration
of all 729 pairs of three-value weak-order representatives gives support
{-1/12,-1/24,0,1/24,1/12} for one pair. This classical order structure must
be available to the comparator; using the loose fixed-focal width would
unnecessarily weaken it.

This is the classical distinct-reference rank product, evaluated after
the learned transformations. Hoeffding or the variance-aware block bound
applies to independent U_N blocks, not to their overlapping constituent
triples treated as if independent. The degree-three center can also be
computed across the complete pooled set of IID trajectories, with appropriate
U-statistic calibration and its larger compute charge.

For J>=2, at the exact same observed draw cost as (10), give this comparator all M
evaluation blocks and all n validation triples: each validation triple is
itself one unbiased U_3 block. Averaging those M+n independent block values
remains unbiased for theta_H; their distributions may differ, but their
common enclosing width W_U and independent non-IID empirical Bernstein bound
still apply. No law oracle or omitted validation charge is allowed.

Three trajectories are the minimal generic order for a fixed-kernel unbiased
estimator without extra law information. To see necessity, choose f=g=Y with
three ordered atoms having probabilities (t,(1-t)/2,(1-t)/2). The target is
Var M(Y)=[1-t^3-(1-t)^3/4]/12, with nonzero cubic coefficient -1/16. The
expectation of any fixed statistic of at most two IID draws is polynomial
of degree at most two in t. Thus it cannot equal this target for every t.
Three suffice by (12). This is an unbiased-kernel order boundary, not a
minimax testing lower bound; binary support or known rank information can
reduce the required order. It supplies no claim that the validated estimator
beats the classical direct one in power or runtime.

## 7. Complete information and cost contract

The validation branch uses exactly

    M(J+1)+3n trajectory draws.                                 (13)

If all transformations need L distinct raw coordinate IDs per trajectory,
the corresponding raw-coordinate draw count is L times (13). For irregular
windows, count the union of required primitive IDs in each block rather than
the sum of overlapping window lengths. This is the minimum number of distinct
coordinates required to evaluate that frozen construction, not a lower bound
for all estimators. Cache hits, repeated archive IDs, unique accessed labels,
full archive ingestion/census, fit/feature computation, and inference compute
must be reported separately. Unknown law constants are never free metadata.

The aggregate validator costs O(ST) arithmetic after computing dp,dq (or
O(J(S+T)) via the factorization Omega=C D^T); it reads three trajectories per
replicate. The original score costs O(J(S+T)); full U_N costs O(N^2(S+T))
comparisons and can in principle stream rows using O(N+S+T) temporary storage
after frozen map values are available. The simple supplied block API instead
materializes two batch-by-N-by-N arrays; the separate pooled API uses rank
sorting and Fenwick counts as described in POOLED_U_PROOF.md. Models and source/peer allocation may be chosen
from external training, but all training data and search compute remain
charged to the pipeline.

For the Hoeffding-only correction at fixed delta, the two uncertainty
contributions are a/sqrt(M)+b/sqrt(n), with
a=W_S sqrt(log(1/(alpha-delta))/2) and
b=W_R sqrt(log(1/delta)/2). Under c_M M+c_R n=C and positive a,b, continuous
minimization gives n/M=(b c_M/(a c_R))^(2/3). The exact finite-budget integer
minimum is computable by enumerating feasible M and using the largest feasible
n for each M, because the objective decreases in each sample count. Choose
this allocation from the frozen widths/costs before outcomes. The optimum is
for this displayed bound only; the direct estimator has a different tradeoff.

## 8. Heterogeneous peers and grouped references

For conditional independent peer trajectories with laws P_j,H, the common-law
matrix K is replaced by pair-indexed conditional-mean products. If
m_sj(z)=E_{P_j,H}[a(f_s(z),f_s(Z_j))-1/2], then the target is exactly
E[(sum_sj C_sj m_sj(Z_0))(sum_tk D_tk n_tk(Z_0))|H], including any nonzero
means induced by the heterogeneous references. The correction remains
sum_jst C_sj D_tj Lambda_st,j. Two independent clones of each P_j,H and a
focal from its declared law give the explicit validator

    R_hetero=(1/2)sum_j [sum_s C_sj(p_sj'-p_sj'')]
                             [sum_t D_tj(q_tj'-q_tj'')].          (14)

Its mean is the full correction and an enclosing width is
sum_j ||C_:j||_1 ||D_:j||_1. The cost is one focal plus two copies of each
declared peer trajectory law, not the common-law three-draw shortcut. Such
clones must exist in the actual access contract; one historical trajectory
per named heterogeneous entity does not supply them automatically.

If exact strata with known masses are part of the target, run matched-law
blocks within each declared stratum and combine their identified targets
with the declared weights. A simultaneous/weighted block bound supplies the
inference. Learned grouping functions can be fixed by external training, but
conditioning on realized audit memberships changes the reference laws and
requires this explicit conditional contract. Grouping all references changes
a global-rank target unless its reference mixture is preserved. Unknown
stratum masses require estimation/error accounting; no empirical plug-in
exemption is implied.

## 9. Contribution boundary

External-training conditioning, conditional covariance, covariance validation,
Hoeffding/empirical Bernstein inequalities, and the complete distinct-reference
U-statistic are classical. See closest_primary.md for primary-source mapping.
The usable deliverable is the conditional design contract, an exact indexed
source/reference operator with a directly observable aggregate correction,
sharp universal preservation conditions, and honest minimal-order/cost and
reference-law boundaries. It does not promise a score increase, MSE gain,
novelty priority, or a universal power advantage.
