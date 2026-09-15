# Dependence-valid comparison of fixed forecasts using predictable loss ranges

This is a classical Hoeffding martingale argument with a finite union over
predeclared tuning constants. It is not a novel concentration result. The
bounded-variable exponential inequality originates in
[Hoeffding (1963)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830).
A self-contained proof of every inequality used here follows.

## Fixed contract and exact target

Condition on the completed training, calibration and selection information H.
Candidates A and B, their transformations, the eligibility rule, the finite
calendar of T=1416 publication hours, and all inference tuning are fixed before
confirmation outcomes are accessed. Delivered predictions a,b and delivered
targets y lie in [0,1]. In this protocol y is the training-scaled, clipped
log1p(TS144) target. Positive loss gain means A is better:

    gain(y) = (y-b)^2-(y-a)^2 = (b-a)(b+a-2y).

Let e_t be the end of nominal publication hour t. Its n_t eligible articles
and all their delivered predictions are fixed and available by e_t+1 hour.
No target-based exclusion can change this eligibility denominator. The target
for an article is its 48-hour value. Fix the hour's settlement time at

    tau_t = e_t + 48 hours.

The label availability decision and the gain are evaluated under the protocol's
fixed settlement rule by tau_t. For an unavailable TS144 target, set its paired
gain to zero. This does not assign a zero outcome or zero individual loss.
The rule must not use subsequent availability repairs. For n_t>0 let G_t be
the sum of available-target gains divided by **all n_t eligible articles**.
For n_t=0 set G_t=0. Every one of the 1416 calendar hours remains in the final
denominator, including hours with no eligible articles or no available targets.

Let F_t contain H and the information available by tau_t, including settled
hourly gains through t. The nested information sets must follow the stated
historical availability assumptions. Set F_0 at tau_0=e_1+47 hours. Since

    e_t+1 hour <= tau_{t-1}=e_t+47 hours,

the current hour's eligibility counts and forecasts are F_{t-1}-measurable.
The filtration can also contain more recent forecasts and partial outcomes
already available at that time. This causes no problem: define

    mu_t = E[G_t | F_{t-1}],      target = (1/T) sum_t mu_t.       (1)

The target is an average of conditional expected settled gains, given previous
hour settlements and all information already available then. It can be random.
It is **not** an expected-gain claim conditional only on the original forecast
issuance information, nor a claim about a further future deployment period.
The observed average gain remains a descriptive realized-performance quantity.
No independence across articles, hours, stories or overlapping horizons is
assumed. No stationarity or missing-at-random assumption is needed.

## Predictable interval for each hour

For article i in hour t define the two endpoint gains

    q0_ti = (b_ti-a_ti)(b_ti+a_ti),
    q1_ti = (b_ti-a_ti)(b_ti+a_ti-2).

Since a+b lies in [0,2], zero lies between q0 and q1. Every observed-target
gain is between these endpoints because gain(y) is affine in y in [0,1].
The missing-target gain zero is in the same interval, irrespective of why
the label is missing. Thus the available/missing indicator needs no conditional
independence or MAR property.

For n_t>0 put

    L_t = mean_i min(q0_ti,q1_ti),
    U_t = mean_i max(q0_ti,q1_ti),
    W_t = U_t-L_t = 2 mean_i |a_ti-b_ti|.                        (2)

For an empty hour set L_t=U_t=W_t=0. Summing the article intervals proves
G_t in [L_t,U_t]. The endpoints and W_t use only eligibility and predictions,
so are F_{t-1}-measurable. Their values may be random and serially dependent.
Also 0<=W_t<=2. In particular, dropping missing rows from the denominator,
computing a width on available rows alone, or discarding empty hours would
change this contract and invalidate this direct application.

## Conditional exponential bound, proved

For any bounded random Z in an interval of width w, define

    f(lambda)=log E exp(lambda Z).

Differentiation is valid because Z is bounded. Its second derivative is the
variance of Z under exponential reweighting. Every probability distribution
supported on an interval [l,u] has variance at most (u-l)^2/4: its variance
is the least expected squared distance to a constant and hence is no larger
than E[(Z-(l+u)/2)^2]<=(u-l)^2/4. Consequently f''(lambda)<=w^2/4.
Integrating twice from zero for lambda>=0 yields

    log E exp(lambda(Z-EZ)) <= lambda^2 w^2/8.                  (3)

Apply this to the conditional distribution of G_t given F_{t-1}; the known
conditional interval has width W_t. Thus for every fixed lambda>0,

    E[exp(lambda(G_t-mu_t)-lambda^2 W_t^2/8) | F_{t-1}] <= 1.   (4)

The predictable endpoints are finite, and mu_t is well defined because
|G_t|<=1. Multiplying (4) over time and using the tower property proves that

    M_T(lambda) = exp(lambda sum_t(G_t-mu_t)
                     -lambda^2 sum_t W_t^2/8)
    E[M_T(lambda) | H] <= 1.                                   (5)

More explicitly, the exponential expression through time t-1 is nonnegative
and F_{t-1}-measurable, so its product with the next factor has conditional
expectation at most its current value. Iteration from M_0=1 proves (5).
This argument handles random predictable widths directly.

## Frozen finite grid and one-sided confidence theorem

Fix before inference K=32 positive values

    Lambda = geomspace(0.01, 100000, 32),

and set alpha=.05/3 for each of the three primary comparisons in the final
preaccess protocol. Write
V_T=sum_t W_t^2 and c=log(K/alpha). For any lambda in this grid, Markov's
inequality applied to (5) gives

    Pr{ sum_t(G_t-mu_t) > c/lambda + lambda V_T/8 | H }
       <= exp(-c) = alpha/K.                                   (6)

Although V_T is random, the event in (6) is exactly the event
M_T(lambda)>exp(c), so no substitution of a random threshold into a
deterministic-variance theorem has occurred. A union bound over the K fixed
values now proves simultaneous validity. Therefore

    lower = mean_t G_t
             - (1/T) min_{lambda in Lambda}
                         [log(K/alpha)/lambda + lambda V_T/8], (7)

satisfies

    Pr{ lower <= (1/T) sum_t mu_t | H } >= 1-alpha.             (8)

The data may choose the minimizing member of this **predeclared grid** because
all its bounds hold simultaneously. They may not choose an unrestricted lambda
and discard the grid penalty. In particular, plugging random V_T directly into
the continuously optimized square-root formula with log(1/alpha) has not been
justified by (6). Grid, alpha, hours, candidate pair and missingness rule must
not be retuned after seeing confirmation outcomes.

Using alpha=.05/3 separately for three frozen primary comparisons and unioning
their failures gives simultaneous coverage at least .95. This requires no
independence between comparisons. A positive lower bound for all three supports
positive average conditional settled gain under (1) for all three comparisons, at
the stated joint error level. If any lower bound is nonpositive, the fixed
success criterion is not established. No result about unseen future-period
gain or the issuance-conditioned target follows from this theorem.

The T-hour target gives each calendar hour equal weight. It is distinct from
an article-weighted gain or an available-label mean loss; those should be
reported with their own descriptive definitions. Keeping all predetermined
hours avoids selecting a random set of nonempty or labeled hours after access.

## Implementation and limitations

`predictable_loss.py` loads no data. Its API is:

    hourly_gain_bounds(prediction_a, prediction_b, targets)
    predictable_loss_bound(hourly_gains, predictable_widths,
                           alpha=0.025, lambdas=None,
                           expected_hours=1416)

The API retains its generic alpha=.025 default. The frozen study explicitly
passes alpha=.05/3 from its configuration, for the stratified hybrid rule
against stratified profiled betting, pooled reference betting, and ungated
selection. The numerical default does not determine the study's error budget.

The first helper returns the hour's realized `gain`, `lower_endpoint`,
`upper_endpoint`, `predictable_width`, and eligibility/availability counts.
Nonfinite delivered targets denote unavailable labels; finite targets outside
[0,1] are rejected. The second function requires all frozen hours and returns
the lower bound, radius, realized average, width-square sum, complete grid,
selected lambda, and every grid radius. Predictable widths should be saved
before labels are settled; passing arrays cannot itself prove predictability.

The endpoint computation takes linear time in eligible articles. The bound
takes O(T+K) time and memory. No fitting, resampling or variance estimation is
performed. Arithmetic uses float64 and stable summation, not interval-certified
arithmetic. Checks establish formula and range identities, not the archive's
historical availability or filtration assumptions. The theorem can be much
tighter than using width two for every hour when predictions are close, but a
positive result is not guaranteed. The fixed-grid multiplicity also has a cost.

Run the independent mathematical checks from the module directory:

    python theory/check_predictable_loss.py
