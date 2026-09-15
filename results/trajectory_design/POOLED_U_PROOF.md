# Complete pooled rank U-statistic: exact fast center and finite bound

This comparator is classical. It uses every available trajectory in every distinct
role. A comparison restricted to the originally formed blocks can omit information
available for recombination; the complete pooled statistic must be acknowledged
when computation permits it. The separate implementation preserves the bytes of
the originally frozen block kernel and does not relabel an after-run extension as
part of that initial protocol.

## Exact O(N log N) center for a fixed finite map count

Let `p_sij = .5 sign(f_s(Z_i)-f_s(Z_j))`, with zero sign at a tie, and define `q`
analogously. Write `c=C1`, `d=D1`. The complete order-three center is

    U_N = sum_{i,j,k distinct} (sum_s c_s p_sij)(sum_t d_t q_tik)
                                      / [N(N-1)(N-2)].

Let `A_is=sum_j p_sij`, `B_it=sum_j q_tij`. These are the centered ordinary
midranks, namely midrank minus `(N+1)/2` with one-based ranks. Ties give exact
half-integer values. Let `K_st` be concordant minus discordant **unordered** pair
count between map columns `f_s,g_t`; a tie in either coordinate contributes zero.
Antisymmetry gives `sum_ij p_sij q_tij = K_st/2`. Therefore exactly

    U_N = [sum_i (c^T A_i)(d^T B_i) - .5 sum_st c_s d_t K_st]
                                      / [N(N-1)(N-2)].

Sort each map column to obtain its midranks. For each column pair, sort the first
coordinate and query a Fenwick count tree on ranks of the second coordinate.
Query all members of a tied-first-coordinate group before inserting that group;
subtract counts strictly above the second coordinate from counts strictly below.
This counts `K_st` without either type of tie. The total complexity is
`O((S+T+ST) N log N)` and storage `O((S+T) N)`, besides the learned map evaluation.
Signed coefficients enter only after exact ordering/counts. The supplied numerical
API uses floating coefficient contractions and is not an interval certificate.

## Complete-U support and permutation bound

The companion derivation proves that the symmetric scalar map-pair kernel on
three trajectories has absolute value at most `1/12`, including all tie patterns.
The fitted symmetric kernel `h` therefore lies in the known common interval
`[-W/2,W/2]`, where `W=||c||_1 ||d||_1/6`. `U_N` is its complete U-statistic.

Conditional on the frozen information, suppose **all pooled trajectories are
IID**. Set `q=floor(N/3)`. For a permutation of the N inputs, average h over the
first q disjoint triples. Averaging these incomplete means over all permutations
equals `U_N`. Jensen's inequality transfers each exponential-moment bound for the
independent triple mean to the complete statistic. Thus

    P(U_N-theta > W sqrt[x/(2q)]) <= exp(-x).

This is the classical bounded U-statistic argument in [Hoeffding (1963),
Section 5a](https://doi.org/10.1080/01621459.1963.10500830), with three observations
per kernel. `N choose 3` is not an independent sample size.

If `sigma²=Var h(Z1,Z2,Z3)`, the same argument transfers the standard Bernstein
moment bound to

    log E exp{lambda(U_N-theta)}
       <= sigma² lambda² / [2q(1-W lambda/(3q))],  0<=lambda<3q/W,

and hence

    P(U_N-theta > sigma sqrt(2x/q)+Wx/(3q)) <= exp(-x).

The Hoeffding and true-variance radii are deterministic given the frozen
information. Their minimum is one of these two deterministic valid bounds, so
the tail probability at that minimum is also at most `exp(-x)`, without an
additional union allocation between them.

## Variance-aware observable radius

For `q>=2`, calculate sample variance `s²` from the q **prespecified disjoint**
triple kernels on the same input ordering. [Maurer and Pontil (2009), Theorem 10,
Eq. (3)](https://arxiv.org/pdf/0907.3740) gives

    P(sigma > s+W sqrt[2x/(q-1)]) <= exp(-x).

This event and the pooled center can depend on the same raw draws; a union bound
does not require their independence. Set `x=log(2/alpha)`. Combining the events
gives one-sided failure at most alpha for the observable radius

    r = min{ W sqrt[x/(2q)],
             s sqrt[2x/q] + [2W/sqrt(q(q-1))+W/(3q)] x }.

The reported lower bound is `U_N-r`. For q=1 use pure Hoeffding at
`x=log(1/alpha)`; for W=0 the target and statistic are exactly zero. The proof is
an application of classical permutation and variance-concentration machinery.
It is not an invocation of the different variance-estimator constants in Peel,
Anthoine and Ralaivola's related empirical Bernstein U-statistic work.

The initial disjoint U_N/U3 block mixture allowed independent nonidentical block
distributions with a common expectation and range. That argument does **not**
authorize pooling nonidentical raw trajectories into this complete U-statistic.
All pooled inputs must share the same conditional law here. A fixed archive with
independent index draws supplies this conditional sampling law; it does not imply
the physical historical records are IID or establish future prediction utility.
