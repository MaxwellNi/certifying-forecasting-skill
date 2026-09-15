# Operational boundaries for conditional trajectory designs

This note describes the trajectory identity and its complete pooled U-statistic comparator. It adds implementable input checks and a precise design-choice result. It does not establish new priority, a generic power improvement, or a new concentration inequality.

## Core identity and what it certifies

Condition on information H fixing scalar maps phi_s, psi_t, coefficient matrices C,D, the reference law, sample sizes and all error allocations. The sampled **whole trajectories** W_i must be conditionally IID. Arbitrary dependence inside a trajectory is allowed. A chronological split does not by itself establish this condition.

For centered comparisons p_sj and q_tj against reference draw j,

    E[p_sj q_tk | W_0,H]
      = m_s(W_0) n_t(W_0) + 1{j=k} Cov(p_s1,q_t1 | W_0,H).

Thus, with c=C1, d=D1, Omega=CD^T,

    E[AB|H] = theta_H + <Omega,Lambda>,
    theta_H = E[(c^T m)(d^T n)|H].

On three independent trajectories, Q=.5*(p_.1-p_.2)^T Omega*(q_.1-q_.2) has conditional expectation <Omega,Lambda>. The same reference pair must enter both differences. Replacing it by independent pairs across the factors gives zero, not the interaction.

The known enclosing ranges follow directly:

    |A| <= ||C||_1/2, |B| <= ||D||_1/2,
    |AB| <= ||C||_1||D||_1/4 = W_S/2;
    |Q| <= ||Omega||_1/2 = W_Q/2.

These are upper bounds on range widths. Actual widths may be smaller through constant or cancelling maps. Evaluation averages use M>=1 independently sampled blocks. Correction averages use n>=1 fresh triples when Omega is nonzero. All expectation and failure statements condition on H, not on realized correction outcomes.

Preservation for **every** map and law is equivalent to Omega=0, with C,D fixed. Isolating one Bernoulli map pair proves necessity. For one fixed map/law, Omega may be nonzero while <Omega,Lambda>=0. Therefore nonzero Omega identifies a design's potential for interference; it does not diagnose nonzero realized population interference by itself.

## A named baseline makes the target distinction explicit

Let the forecast, outcome and baseline all equal Y~Bernoulli(1/2). Its centered population midrank is r(Y) in {-1/4,1/4}. If both fitted baseline coefficients are 1/2, the two fitted residuals equal r(Y)/2. Consequently theta_H=1/64 even with disjoint reference pools and zero interaction. Conditioning fully on the baseline makes both residuals zero. This example separates three objects:

1. The observed finite-reference residual product, which may include reference interaction.
2. The specified fitted population-reference residual association theta_H, recovered after removing that interaction.
3. Full conditional residual association, which additionally requires correct conditioning or a justified nuisance-error allowance.

The fact that an arbitrary measurable trajectory map has a valid statistical target does not make it a valid forecast. A forecast map must separately use only information available at its issue time. Future outcomes are legitimate outcome maps, but illegitimate forecast inputs unless already known at issue time.

## All maps can be nonconstant when scalar cancellation fails

For independent U,V~Bernoulli(1/2), take phi=psi=(U,V) and C=D=(1,-1)^T with one common reference role. Every map and coefficient row is nonzero. Then

    Omega = [[1,-1],[-1,1]], sum_st Omega_st = 0,
    Lambda = diag(1/16,1/16),
    theta_H = 1/8, interaction = 1/8, E[AB] = 1/4.

This refutes summing all matrix entries. Its active-overlap diagonal sum is not zero, so it must not be presented as a counterexample to the more restricted overlap-aware scalar criterion.

The stronger existing heterogeneous example does defeat active scalar cancellation: independent Y_1,Y_2~Bernoulli(1/2), Y_3~Bernoulli(1/10), phi=(Y_1,Y_3), psi=(Y_1,Y_2,Y_3),

    C = [[1,0],[-1,0]],
    D = [[1/2,0],[-1/2,-1/2],[1/2,0]].

Its active coefficients +1/2,-1/2 sum to zero, but the corresponding Lambda entries are 1/16 and 9/400. The interaction equals 1/50. All transformations are nonconstant. Both exact examples are retained in the executable rational checks.

## Target-preserving design construction

Changing coefficients preserves the intended target for all maps/laws only if the same maps and reference law are retained and the declared row sums c,d are preserved. A simple same-law construction uses two independent reference roles:

    C_new=[c,0], D_new=[0,d].

It preserves c,d and forces C_new D_new^T=0. If c and d are nonzero, one reference role cannot do this: with one column, C=c and D=d, so CD^T=cd^T is nonzero. Thus two roles are sufficient and one is insufficient for nontrivial universally preserving designs in this unconstrained model. This elementary feasibility boundary is not a new minimax or sample-efficiency result. Reassigning a role means obtaining the required independent draw; merely relabeling a reused observation is invalid.

`design_contract.py` compiles explicit weighted (map ID, reference draw ID) terms. Repeated terms with the same draw-role ID accumulate in one column. It checks dimensions, signatures, reference-law declarations, and row sums using exact rational arithmetic. The preservation check also compares the exact binary64 row sums of the generated executable matrices. It rejects a mathematically exact rational reassignment if floating conversion changes those sums. One-third weights supply a regression test of this distinction. The builder does not infer raw independence from different coordinate names or role IDs.

## A strict radius comparison under a common information budget

Suppose M,n,r>=1, 0<delta<alpha<1 and W_S>0. Give the complete pooled degree-three U-statistic all N=M(r+1)+3n original trajectory draws, under the same conditional IID law and same fixed maps and target. Let q=floor(N/3), W_U=||c||_1||d||_1/6. Its pure-Hoeffding radius and the split evaluation radius are

    r_pool = W_U sqrt(log(1/alpha)/(2q)),
    r_score = W_S sqrt(log(1/(alpha-delta))/(2M)).

Exactly,

    r_pool/r_score
      = [||c||_1||d||_1/(3||C||_1||D||_1)]
        sqrt[(M/q) log(1/alpha)/log(1/(alpha-delta))]
      < 1/sqrt(6).

Indeed, q >= floor(2M/3)+1 > 2M/3; row-sum L1 norms do not exceed entrywise norms; and the logarithm ratio is strictly below one. Zero c or d makes the pooled radius zero. If W_S=0, both radii are zero and the ratio is undefined. The split bound also pays a nonnegative correction radius.

This is a comparison of these two stated **range-only radii**. It does not compare empirical-Bernstein/hybrid radii, samplewise centers, rejection decisions, or power. It is not a universal optimality statement. It gives a principled recommendation: use the complete pooled comparator when all needed IID raw trajectories can be recombined, and treat split correction as a way to explain the interference in an existing construction. Any efficiency claim favoring the split correction needs an actual access or computation constraint, with its cost documented rather than invented.

The pooled center costs O((S+T+ST)N log N) and O((S+T)N) storage, excluding map evaluation. O(N log N) is only the fixed-map-count specialization. The implementation already has this full complexity contract and exact integer/tie sorting; these were verified rather than duplicated.

## Fixed-delta p inversion and numerical scope

For the displayed pure-Hoeffding bound, fix delta before all test outcomes. Let

    G = mean(AB) - mean(Q) - W_Q sqrt(log(1/delta)/(2n)).

For W_S>0 and G>0,

    p_delta = min{1, delta + exp(-2M G^2/W_S^2)}.

For G<=0 set p_delta=1. When Omega=0 omit correction and set effective delta=0. The generic zero-score-width branch sets p_delta=delta if G>0, otherwise 1; valid zero-width design inputs here force G=0. The formula is the infimum of alpha in (delta,1) at which the corresponding lower bound becomes positive. Under the null, the fixed correction event plus Hoeffding controls rejection at every alpha>delta; alpha<=delta cannot reject a positive-width p value. A Bonferroni threshold uses the predeclared whole family.

`fixed_delta_inference.py` implements this formula, not an unannounced inversion of the hybrid rule. Exact rational score means and coefficient widths feed directed Decimal calculations. Transcendental outputs are enclosed conservatively and the final binary64 p value is rounded upward. Tiny tails never silently become zero. Family thresholds round downward and reject underflow to zero. Signed interactions are retained, invalid kernel support is rejected, and declared nonnegative mean-error allowances can cover upstream numerical discrepancies.

The outward claim is explicitly limited to inversion of the supplied numeric values. The older floating kernel code is not interval-certified. A rigorous end-to-end numerical guarantee still requires an independently justified bound on its mean error, or exact/interval kernel evaluation. Passing metadata checks does not verify stochastic assumptions.

## External learning and cross-fitting

Learned maps, weights and reference assignments can already be fixed by H when trained independently of the test draws. Additional generic words such as “learned” or “nonlinear” do not extend this theorem. Conditioning on all fitted objects is valid only if the claimed conditional common-law product experiment still holds.

Ordinary cross-fitting does not automatically make fold scores independent. With independent Rademacher X_1,X_2, use model m_1=X_2 to evaluate T_1=X_1 m_1 and m_2=X_1 to evaluate T_2=X_2 m_2. Each model is independent of its own evaluation variable, but T_1=T_2 exactly. Combining fold-specific valid bounds by a declared union allocation can remain valid; treating the two outputs as independent observations needs another argument. This is a sampling issue, not a limitation cured by merely renaming H.

## Verification receipt

Unittest discovery runs 43 tests: the 17 existing trajectory/comparator tests and 26 new contract/inversion tests. `EXACT_VERIFICATION.json` contains the 14 existing finite-law exact checks plus 5 new checks (19 in total), including 20,000 exact integer-budget floor cases. The new tests cover future-feature leakage, delayed models, reused training IDs, changed reference laws and target weights, rational-versus-floating preservation, smallest-positive alpha/delta, tied and huge integer maps (retained tests), zero ranges, signed correction, declared numerical errors, and family-threshold underflow. These are targeted verification, not a substitute for the analytic validity proof or an empirical guarantee for arbitrary panels.

## Unit-coefficient target distinction

Let baseline U be uniform on {-1,0,1}, with forecast and outcome both U squared. With row sums c=d=(1,-1), centered population-rank contrasts are (1/2,-1/3,-1/6). Their mean square is 7/54. Both fully baseline-adjusted targets are zero since the forecast and outcome are determined by U. Disjoint references remove interaction but preserve the positive fitted target. This boundary applies with the unit coefficients used in the application, not only under half-weighted fits. The exact finite-law check is included in contract_exact_checks.py.
