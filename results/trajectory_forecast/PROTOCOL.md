# Learned forecast trajectories: fixed study protocol

This local protocol is recorded before downloading or examining the new source
observations. It is not a public preregistration or proof that the data have
never been used by anyone. Results from every declared candidate and design
will be retained, including failure to establish additional utility.

## Source and forecasting task

Use Song Chen's UCI Beijing PM2.5 dataset (dataset 381), DOI
10.24432/C5JS49, under CC BY 4.0. The provider's schema/license metadata was read
before this protocol; the observation CSV has not been downloaded or read in
this task. Primary source URL:
https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip

Predict the next hourly PM2.5 measurement using observations available at the
previous hour. Build timestamp-aligned lags 1,2,3,6,12,24 of PM2.5; use previous
hour temperature, pressure, dewpoint, wind speed, snow, rain and wind direction,
plus known hour/day-of-week/month. Missing targets or any missing PM2.5 lag are
excluded with counts reported; meteorological missing values are imputed using
training medians, categorical missing values get an explicit category. No
future weather or outcome is used as a forecasting feature.

Training uses 2010–2012. Selection uses 2013; final chronological confirmation
uses 2014. No confirmation outcomes are used to select models or thresholds.
Persistence is the baseline. Fix eight candidates: persistence; Ridge with
alpha=1 and alpha=100; histogram gradient boosting on persistence residuals
with max_iter=100, max_leaf_nodes=7 and 31, learning_rate=.05, l2=1,
early_stopping=False, random_state=20260914; and the equal blend of persistence
with each of the three models Ridge1, HGB7 and HGB31. Numeric input scaling for
Ridge uses training means/standard deviations. Training includes all eligible
hours; selection and confirmation use the hour 14 origin each eligible day.
All predictions are clipped below at 0, a physical-domain constraint fixed
before outcomes. No upper clipping is used for the primary utility endpoint.

## Conditional archive experiment and target

Each eligible day's prediction-origin record, its complete lag/meteorological
history, learned forecast and target form one trajectory. The 2013 archive
is the finite conditional population. Sampling independent indices with
replacement creates iid complete trajectories conditional on this archive
and the trained models. This does NOT assert physical independence of successive
days, and does NOT establish population inference for the original dependent
time series.

For each candidate let forecast maps be (candidate,persistence), and outcome
maps be (realized target,persistence). The row-sum weights on each side are
(1,-1). The target is the archive expectation of the product of differences
of population midranks. It is a specified fitted-rank increment target, not
full conditional covariance, causal effect or a theorem of future MSE gain.
The full archive permits exact marginal midranks and therefore an exact census
target; its cost is reported as a strong available alternative.

Use three fixed reference designs with two peers: shared (both channels use
peer 1); separate (forecast uses peer 1, outcome uses peer 2); and crossed
(forecast candidate/persistence usepeer 1, outcome target uses peer 2 and
outcome persistence uses peer 1). Coefficient row sums and hence the target
are identical. The exact matrix of peer overlaps, not a scalar sum, determines
the additional interaction. Estimate its full signed contraction using
independent triples of trajectories.

For coverage/identity diagnostics use 300 paired seeds 2026091400+r, r=0,...,299,
evaluation blocks M=1024 (three trajectories each), validation triples n=512,
nominal alpha=.05 with delta=.025. One-sided bounds use the classical empirical
Bernstein inequality with the deterministic support widths; optionally take
the minimum with Hoeffding only after a union-budget allocation specified in
code before execution. Always preserve the same bound choice in comparison.
Report mean score, exact target, oracle interaction, estimated interaction,
coverage/noncoverage counts and exact binomial intervals for every candidate
and design. The same raw draws are offered to the full order-three distinct
U-statistic comparator, including all validation triples. Its bound needs no
reference-interaction validation. No unique efficiency claim is assumed.

## Selection and chronological utility

Use one additional fixed seed 2026091499, M=16384 evaluation triples, n=8192
validation triples, and simultaneous Bonferroni level .05/8 for each candidate
(delta half the marginal level for corrected bounds). The shared-score signed
correction and the all-role distinct U comparator are two separate gates.
Each gate selects the smallest 2013 raw-MSE candidate among positive lower
bounds, falling back to persistence if none pass. Baseline remains in every
family. Ungated selection minimizes the same 2013 loss over all candidates;
tie break uses the declared candidate order. The precise
draw budget is M*3+n*3=73728 whole-trajectory draws per candidate, identical
for the corrected and distinct-U procedures. Count the underlying distinct
archive records and coordinate reads as well; no label acquisition saving is
claimed from resampling a fully accessible archive.

Freeze selected candidate identities before examining 2014 losses. Report
every candidate's 2014 raw MSE and MAE, each rule's selected identity, absolute
and relative gains over persistence, and gains over ungated selection.
Distinguish useful augmentation from additional audit-gating benefit.
For descriptive uncertainty report weekly mean paired loss differences and
moving-block bootstrap intervals (block 4 weeks, 2000 draws, seed 2026091455),
explicitly without a claimed finite guarantee for the physical time series.

The success criteria are: correct identity and valid conditional-archive bound
coverage, followed separately by positive chronological loss improvement and
additional gating improvement. A positive mathematical result does not count
as a positive decision result. A null or adverse utility result will not be
removed, relabeled as irrelevant, or repaired by choosing another endpoint.
