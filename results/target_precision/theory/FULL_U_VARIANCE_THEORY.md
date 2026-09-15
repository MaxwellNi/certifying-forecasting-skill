# Finite-sample variance calibration for the complete residual-midrank U statistic

This is a **specialization of classical concentration theory**, with an efficient
all-delete-one calculation. It establishes a usable algorithm and a uniform
finite-sample guarantee, including ties and first-projection degeneracy. It does
not establish a new concentration theorem or empirical superiority. In particular,
the multicategory remainder below can dominate at the existing sample sizes.

## 1. Sampling assumptions and notation

Condition throughout on training information H. The n inference rows
X_i=(V_i,W_i,C_i) are IID from P_H, n is fixed before inspecting those rows,
and every positive category probability p_c is supplied exactly. The predictor,
category definition, target and any selection/error-allocation rule are fixed
before inference. We do not condition additionally on the observed category
counts. Repeated positions sampled with replacement remain separate positions.

Let a(v,z)=1{z<v}+(1/2)1{z=v}. Define population midranks
R_V(v)=E[a(v,V')] and R_W analogously, and

    theta = E[R_V(V) R_W(W)]
            - sum_c p_c E[R_V(V)|C=c] E[R_W(W)|C=c].

The frozen estimator is U_n=U3_n-U4_n, equivalently the complete order-four
U statistic with the permutation-symmetric kernel

    h = Sym_4[a(V1,V2)a(W1,W3)
              - 1{C1=C2} a(V1,V3)a(W2,W4)/p_C1].

It is unbiased for theta. All identities below concern population midranks and
known p, not estimated frequencies or panel-standardized ranks.

For n>=5 let U_{-i} use all positions except i and define

    V_J = (n-1)/n * sum_i (U_{-i}-U_n)^2.

The deletion-average identity is n^{-1} sum_i U_{-i}=U_n. Each r-subset
contributes to n-r deletions, and
[(n-r)/n]/binom(n-1,r)=1/binom(n,r); apply this separately for r=3,4.

## 2. Two published probability inequalities used as inputs

We use [Maurer and Pontil (2018), Theorem 2 and equation (9)](https://proceedings.mlr.press/v75/maurer18a/maurer18a.pdf).
For a function f of k independent arguments, define M(f) as its largest absolute
one-argument replacement difference and J(f) as k times its largest absolute
two-argument mixed replacement difference. Let E_k(f) be the expected sum of
the k conditional variances. Their inequalities give

    Pr{ f-Ef > sqrt(2 E_k(f) x) + (2M(f)/3+J(f))x } <= exp(-x),       (P1)

and, for IID observations and their k+1-row estimator v_f,

    Pr{ sqrt(E_k(f)) > sqrt(v_f)
         + sqrt((2M(f)^2+8J(f)^2)x) } <= exp(-x).                   (P2)

The one-sided form of their Theorem 2 is used in (P2). For symmetric f,

    v_f(X_1,...,X_{k+1})
      = 1/[2(k+1)] sum_{i!=j}(f(X_{-i})-f(X_{-j}))^2
      = sum_i(f(X_{-i})-mean_j f(X_{-j}))^2.

Deleting j and replacing retained i by j gives the same multiset as deleting i;
symmetry proves the first equality. Expanding squared pair differences proves
the second. These are imported concentration inputs, not newly proved or
claimed results. The remaining target-specific and full-center reductions are
proved below.

## 3. Full Hoeffding decomposition and the correct variance transfer

Define g_s(x_1,...,x_s)=E[h(x_1,...,x_s,X_{s+1},...,X_4)], g_0=theta,
and canonical projections

    h_s(x_1,...,x_s)
      = sum_{A subset {1,...,s}} (-1)^(s-|A|) g_|A|(x_A).

For s>=1, integrating any argument gives zero: terms with that argument and
terms without it cancel in pairs. Inclusion-exclusion then gives

    h(x_1,...,x_4)=theta+sum_{nonempty A subset [4]} h_|A|(x_A).

Put zeta_s=E[h_s(X_1,...,X_s)^2]. Complete averaging yields

    U_k-theta = sum_{s=1}^4 binom(4,s)/binom(k,s)
                         * sum_{|A|=s} h_s(X_A).                 (1)

Any distinct canonical subset terms have zero cross-product expectation: pick
an argument in their symmetric difference and integrate it. Thus

    Var(U_k) = sum_{s=1}^4 binom(4,s)^2 zeta_s/binom(k,s).        (2)

The expected sum of conditional variances has the closely related exact form

    E_k := E_k(U_k)
         = sum_{s=1}^4 s binom(4,s)^2 zeta_s/binom(k,s).          (3)

To prove (3), the i-th conditional variance removes terms not containing i.
After expectation the surviving terms are orthogonal, as above. Summing over i
counts every size-s term s times. This also proves Var(U_k)<=E_k without an
asymptotic argument.

Since binom(n-1,s)/binom(n,s)=(n-s)/n <= (n-1)/n,

    E_n <= rho E_{n-1},     rho=(n-1)/n.                         (4)

Apply (P2) to f=U_{n-1} using all n observed rows. Its empirical variance is
v_f=V_J/rho. Consequently, except on an event of probability delta_v,

    sqrt(E_n) <= sqrt(V_J)
        + sqrt(rho*(2M_{n-1}^2+8J_{n-1}^2)*log(1/delta_v)).      (5)

This transfer is what permits the **full U_n center**. Simply moving the center
of a disjoint-block bound to U_n would not prove (5).

The exact expectation of the empirical quantity is

    E[V_J] = sum_{s=1}^4 [s(n-1)/(n-s)]
                         * binom(4,s)^2 zeta_s/binom(n,s).       (6)

It follows either from unbiasedness of v_f for E_{n-1}, or directly from
the orthogonal expansion. Thus V_J estimates an upper variance proxy; it is not
an exactly unbiased estimator of Var(U_n). Higher projections are retained.

## 4. Explicit interaction constants, including ties

The symmetrized order-three kernel for U3 lies in [1/6,1/3], so its width is
r3=1/6. To see this without continuity assumptions, strict three-row orders give
only 1/6 and 1/3. Break V and W ties independently and uniformly. The expectation
of the product of the resulting comparison bits equals the product of original
midrank comparisons, so tied kernels are convex combinations of strict cases.

The symmetrized nonnegative kernel for U4 is at most r4=1/(4p_min). In the
unweighted disjoint-pair expression, swapping each comparison's two endpoints
creates four products whose sum is one, because a(x,y)+a(y,x)=1, including ties.
Its full permutation mean is therefore 1/4. Each category weight is between
zero and 1/p_min, giving the bound. If there is just one category, U4 is the
constant 1/4 and r4 can instead be set to zero.

For any order-r complete U with kernel width d, a one-row replacement touches
a fraction r/k of its summands, each changing by at most d. A two-row mixed
replacement touches only summands containing both replaced positions, a
fraction r(r-1)/[k(k-1)], each mixed difference having magnitude at most 2d.
Hence

    M <= rd/k,           J <= 2r(r-1)d/(k-1).

The seminorm triangle inequality applied separately to U3 and U4 now gives

    M_k = A/k,           J_k = B/(k-1),
    A = 1/2 + 1/p_min,   B = 2 + 6/p_min,                       (7)

as valid upper bounds for multiple categories. For a single category,

    A=1/2,              B=2.                                    (8)

These constants are upper bounds, not claimed sharp. They slightly improve
the constants obtained by treating all of U3-U4 as one generic order-four
kernel. The full-kernel range width is

    R = 1/6+1/(4p_min)  (multiple categories),
    R = 1/6            (one category).

## 5. Finite-sample lower confidence theorem

Choose positive delta_t,delta_v with delta_t+delta_v=alpha before observing
inference rows, and set x_t=log(1/delta_t), x_v=log(1/delta_v). For n>=5,
under the assumptions in Section 1, define

    rad = sqrt(2 V_J x_t)
          + sqrt(2 rho (2M_{n-1}^2+8J_{n-1}^2) x_t x_v)
          + (2M_n/3+J_n)x_t.                                   (9)

Then

    Pr{ theta >= U_n-rad | H } >= 1-alpha.                     (10)

Proof: (P1) for U_n fails with probability at most delta_t. Inequality (5)
fails with probability at most delta_v. On their intersection substitute (5)
into (P1), obtaining U_n-theta<=rad. A union bound gives (10). Independence
between the two events is unnecessary. All constants are deterministic upper
bounds, so substitution only enlarges the radius. This proves the claimed
finite-sample bound for all conditional laws covered by Section 1.

The implementation uses delta_t=delta_v=alpha/2 by default. Its optional split
parameter must also be fixed before inspecting the inference sample. For the
equal split, with x=log(2/alpha),

    rad = sqrt(2 V_J x)
          + x sqrt(rho*(4M_{n-1}^2+16J_{n-1}^2))
          + (2M_n/3+J_n)x.

A fixed family needs a valid allocation across candidates. Taking a maximum
with another confidence bound after observation also needs a joint allocation.
`full_u_joint_bound` allocates alpha/2 to this empirical bound and alpha/2 to
the frozen complete-U Hoeffding bound, then takes their maximum. This follows
from the same union argument; no independence is required.

## 6. What happens near degeneracy

No division by a projection variance and no normal approximation occur in (9).
If zeta_1=0, every term in (6) is O(n^-2), hence V_J=O_P(n^-2) by Markov's
inequality. The first term of (9) and its deterministic remainder are then
O_P(n^-1) for fixed alpha and fixed positive p. For nonzero zeta_1 the leading
variance term is 16 zeta_1/n. All higher terms stay present.

More explicitly, for n>=6, divide the coefficient of zeta_s in (6) by
binom(4,s). For s=2 it is 24/[n(n-2)]; the s=3 and s=4 ratios to that
quantity are respectively 3/(n-3) and 4/[(n-3)(n-4)], both at most one.
Since Var(h)=sum_s binom(4,s)zeta_s<=R^2/4,

    E[V_J] <= 16 zeta_1/n + 6R^2/[n(n-2)].                     (11)

Thus the claim also covers a sequence of weak-projection laws with fixed
bounded R; no positive lower bound on zeta_1 is assumed. With probability at
least 1-eta, Markov gives V_J <= [16zeta_1/n+6R^2/(n(n-2))]/eta.
This last statement is only a rate diagnostic; the operational bound remains
the data-dependent (9).

A concrete tied, exactly first-projection-degenerate law is one category and
V=W~Bernoulli(p), p=1/2. Direct counting reduces the estimator to

    U_n = K(n-K)/[4n(n-1)],   theta=p(1-p)/4,

where K is the number of ones. Indeed the order-three kernel minus 1/4 is
zero for three equal bits and 1/12 otherwise. A mixed triple contains exactly
two unequal pairs; averaging reduces U_n to one eighth of the complete
unequal-pair statistic, proving the formula. Its order-two kernel is
q(x,y)=1{x!=y}/8. Its first projection is (1-2p)(x-p)/8 and its second
projection is -(x-p)(y-p)/4. Orthogonality therefore gives

    Var(U_n) = p(1-p)(1-2p)^2/(16n)
               + p^2(1-p)^2/[8n(n-1)].                        (12)

At p=1/2 the first term is exactly zero and the second equals
1/[128n(n-1)]. Choosing p close to 1/2 gives an explicit weak regime.
This is degeneracy of the final target statistic, not of a separate bias term.

There is also a direct obstruction to dropping a remainder when V_J=0. Take
one category, V~Bernoulli(p), and W=1-V. Then theta=-p(1-p)/4. On the event
that every V_i=0, all observed rows equal (0,1), so U_n=V_J=0, but the event
has probability (1-p)^n. A zero-radius lower bound therefore fails with
probability greater than alpha for every 0<p<1-alpha^(1/n). More generally,
suppose a deterministic proposed lower bound is -r_n on this constant dataset.
Uniform coverage forces

    r_n >= t_n(1-t_n)/4,
    t_n = min(1/2, 1-alpha^(1/n)).                               (13)

If r_n were smaller, continuity would give a p<t_n with p(1-p)/4>r_n and
(1-p)^n>alpha, contradicting coverage. When t_n=1/2 and alpha<2^-n,
p=1/2 itself also gives the contradiction; equality is handled by continuity
from below. For fixed alpha and growing n, the right-hand side is asymptotic
to log(1/alpha)/(4n). Thus the order of an n^-1 remainder is necessary on at
least this zero-observed-variance sample. This elementary lower bound does not
show that the constants in (9) are optimal.

## 7. All-delete-one calculation in quadratic time

Introduce formal row weights t_i. Let N3(t),N4(t) be the ordered distinct-tuple
numerators, multiplying each summand by the product of weights of its positions.
They are homogeneous multiaffine polynomials of degrees 3 and 4. Therefore

    N_r(with position i deleted) = N_r(1)-partial_i N_r(1).      (14)

This holds because each monomial either contains t_i exactly once or not at
all. In particular sum_i partial_i N_r(1)=rN_r(1), an arithmetic diagnostic.
Division by (n-1)_r gives the leave-one estimator components.

The collision-correct formula can be differentiated without expanding tuples.
Write A_i(t)=sum_{k!=i}t_k a_ik and B_i(t)=sum_{k!=i}t_k b_ik, and
g_ij=1{C_i=C_j}/p_Ci. Then

    N3(t)=sum_i t_i[A_i(t)B_i(t)-sum_{k!=i}t_k^2 a_ik b_ik],

    P(t)=sum_{i!=j}t_i t_j g_ij
             [A_i(t)-t_j a_ij][B_j(t)-t_i b_ji],

    Q(t)=sum_k t_k^2 sum_c (1/p_c)
          [(sum_{i!=k,C_i=c}t_i a_ik)(sum_{j!=k,C_j=c}t_j b_jk)
            -sum_{i!=k,C_i=c}t_i^2 a_ik b_ik],

    N4(t)=P(t)-Q(t).

P permits the two reference positions to coincide; Q removes exactly that
collision. The diagonal subtraction in Q excludes identical focus positions.
Although P and Q separately contain squared weights, they cancel in N4.

`full_u_all_delete_one` computes these values and derivatives with two passes
over comparison rows. For P it first accumulates derivatives with respect to
A_i and B_i; a second comparison pass propagates these derivatives to all t_k.
For Q, a shared reference weight has derivative two, and the two class-aggregate
products give the focus derivatives. N3's derivatives are the three position
roles in its distinct triple. Every comparison pass does O(n) vector operations
and class aggregation over at most n observed categories. Sorting exclusive
rank sums costs O(n log n). Validating complete support costs O(C). Total time
is O(n^2+C), extra memory O(n+C). No n-by-n comparison matrix or n separate
quadratic U computations are required.

Equal observed values use half-comparisons and are never deleted as duplicates.
The implementation reuses the frozen input validation, including exact integer
order checks. It uses floating point accumulation, so exact here means the full
mathematical tuple sum, not interval-certified numerical arithmetic.

## 8. Practical limitation and scope

For alpha=.05, the large-n deterministic remainder is approximately

    log(40)/n * [sqrt(4A^2+16B^2)+2A/3+B].                     (15)

This is an expansion of the explicit expression, not an extra confidence
formula. With one category it is about 38.35/n. With p_min=.1 it is about
1172.65/n. At n=32,000 the latter remainder alone is about .0367, already
comparable with the complete-U Hoeffding radius .0365; adding the observed
variance term generally makes it worse. At n=100,000 it is about .0117 versus
a Hoeffding radius of .0206, so there is room for an improvement when observed
variance is small. The frozen archive must be treated as previously inspected.
No untouched-task performance claim follows from this theorem or the checks.

The theorem supplies a valid full-U variance-sensitive comparator with a
degeneracy-safe remainder. It does not remove inverse-p dependence from the
original IID kernel, prove optimal constants, validate a real panel's sampling
design, justify adaptive stopping, or establish a novel result on concentration.
An alternative stratified sampling estimator has its own design and cost
analysis; it should not silently replace this IID-access comparator.

## 9. Checkable files

- `full_u_variance.py`: all-delete-one computation and bound.
- `check_full_u_variance.py`: independent tuple/deletion identities,
  finite-support projection and variance identities, degenerate/weak examples,
  bound-formula checks, and optional timing checks.
- `validation.json`: the recorded check output, with fixed seeds and commands.

Run from the module directory:

    python theory/check_full_u_variance.py
