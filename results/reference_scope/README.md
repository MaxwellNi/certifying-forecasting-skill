# Reference cost and sequential scope

The technical details prove two operational consequences of the reference design.
Deleting a column that is zero in both coefficient matrices preserves each score,
row sum, overlap matrix and declared range width. It removes an unused draw role.
For fixed range widths and levels, the allocation ratio is
M/n = (3a/(g b))^(2/3); integer enumeration respects the total budget gM+3n <= N.
This is an outcome-free allocation for the declared range radius, not a minimax
or hybrid-power optimality claim. See [the actual same-budget comparison](../trajectory_budget/README.md).

Marginally unbiased correction does not justify sequential stopping. Five latent
IID Bernoulli pairs repeated periodically give a strictly stationary sequence.
Within each aligned round, the score and validation correction use independent
roles, yet rounds are identical. For R = S - Q, the fitted target and E[R] are
zero, Var(R)=5/256, E[(1+R)^2]=261/256 and the eventual crossing probability at
any threshold above one is 7/32. The generic sequential tools remain valid for
their own predictable conditional-mean targets; identifying those targets with
the fitted rank target requires a separate conditional-centering argument.
Fresh conditional IID focal and reference trajectories under the same declared
law suffice. Independence of only the reference roles does not suffice.

Run `python exact_checks.py` to reproduce the finite rational enumeration.
It checks the explicit counterexample and 6,561 pruning cases. These checks
support the witnesses; the technical details give the general algebraic proof.
The three-role variant in the output is an additional witness; the paper's
preferred five-role example also separates evaluation and validation within a round.
