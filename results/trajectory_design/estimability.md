# Why the correction uses three draws

For an unrestricted unknown distribution, the interaction cannot always be
estimated without bias using at most two independent observations. This is
an application of the classical polynomial-degree argument for unbiased
estimability, not a new confidence-bound or forecasting-cost optimum.

Let an issue-time covariate X take values 0, 1, 2 with probabilities
(t, t, 1−2t), where 0<t<1/2. Take outcome Y=X, forecast h=X and a constant
baseline. The forecast uses the covariate. Under midrank comparison,

- expected shared score = t−3t²/2;
- target association = (t−2t²+t³)/2;
- reference interaction = (t−t²−t³)/2.

The expectation of a fixed statistic of two independent draws is quadratic
in these probabilities and therefore has degree at most two in t. It
cannot equal the cubic interaction for every t. The three-draw correction
does have that expectation. The companion gives the derivation.

The restriction counts **all observations that reveal the unknown law**.
Maps and auxiliary randomness are fixed independently of it, and the rule
uses at most two observations almost surely. Extra training information,
known probabilities, an oracle or occasional third observations can change
the conclusion. Restricted distributions can have lower order: for binary
X, the pair statistic 1{X1 differs from X2}/8 is already unbiased. A declared
identity forecast has zero interaction without sampling.

This explains the correction's three roles. It does not establish optimal
confidence intervals, testing power or practical acquisition cost.
