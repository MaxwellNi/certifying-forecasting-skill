This construction was selected analytically after the original six-law study was inspected and before any simulation of this addendum. Exact finite searches found no example among uniform three-atom weak orders and supplied four-atom examples. The threshold example below was chosen for its simple rational explanation. No Monte Carlo result or seed search informed that choice.

Let `B` be uniform on `{0,1,2,3}` and define `h=1{B=3}` and `Y=1{B>=2}`. The fixed candidate family is `(B,h,-h,0)`. Each candidate contrasts its centered population midrank with that of `B`, and the right contrast is `(Y,B)`.

| B | h | Y | Centered midrank of B | Centered midrank of h | Centered midrank of Y | Left contrast for h | Right contrast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | -3/8 | -1/8 | -1/4 | 1/4 | 1/8 |
| 1 | 0 | 0 | -1/8 | -1/8 | -1/4 | 0 | -1/8 |
| 2 | 0 | 1 | 1/8 | -1/8 | 1/4 | -1/4 | 1/8 |
| 3 | 1 | 1 | 3/8 | 3/8 | 1/4 | 0 | -1/8 |

The target for `h` is the average of the last two columns' products, namely `(1/32 + 0 - 1/32 + 0)/4 = 0`. Its left contrast is not identically zero: this is cancellation in the population target, not semantic identity with the baseline.

For two independent baseline draws, define the centered comparison `p_x(i,j)=sign(x_i-x_j)/2`. The shared score is

`S = [p_h(0,1)-p_B(0,1)] [p_Y(0,1)-p_B(0,1)]`.

Both threshold maps are nondecreasing coarsenings of B. A contrast is nonzero only when that threshold map ties on a pair with distinct baseline values. Both maps tie only on the distinct baseline states 0 and 1. Therefore S is 1/4 for the two ordered pairs `(0,1)` and `(1,0)`, and zero for the other 14 ordered pairs. It follows that `E[S]=(2/16)(1/4)=1/32`.

With the usual independent-reference covariance correction

`Q = [p_h(0,1)-p_B(0,1)-p_h(0,2)+p_B(0,2)] [p_Y(0,1)-p_B(0,1)-p_Y(0,2)+p_B(0,2)] / 2`,

the independent-copy covariance identity gives `E[Q]=E[S]-theta=1/32`. The independent rational oracle enumerates all 64 ordered state triples for each of the four candidates and checks this identity directly, as well as `E[U]=theta` for the six-role symmetric U kernel.

| Candidate | Exact target | True null `theta<=0` |
|---|---:|---|
| Baseline identity B | 0 | Yes |
| h | 0 | Yes |
| -h | 1/32 | No |
| Constant zero | 1/64 | No |

Thus `h` is a nonidentity null candidate with strictly positive shared-reference interaction `Gamma=1/32`. The family is not an all-null family. All maps are functions of B, so this construction provides no information beyond B; its null labels concern the stated rank-contrast target and must not be presented as conditional-independence labels.

The separate simulation uses the unchanged final common-baseline widths and rules at the original two budgets. It addresses this particular biased-null gap in the finite-law calibration. It is not new application evidence, an exhaustive null calibration, or a simulation proof of a coverage theorem.
