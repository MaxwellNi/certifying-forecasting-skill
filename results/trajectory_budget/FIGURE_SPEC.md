# Reader figure from verified full-precision coordinates

Preferred main-paper design: a horizontal all-eight-candidate dot plot, with
candidate labels shared between two side-by-side panels. Preserve original
candidate order (persistence, ridge_1, ridge_100, boosting_7, boosting_31,
blend_ridge_1, blend_boosting_7, blend_boosting_31).

Left panel: **What the shared score contains**. Per candidate, show its original
raw score, original corrected center, and exact census fitted target in units
of 10^-3. Connect raw score to corrected center with a thin line to visualize
the estimated signed interaction. Exact target is a small black diamond.
The gap from score to corrected center is removal of an estimated interaction,
not a causal decomposition of forecast loss.

Right panel: **Which lower bounds cross zero at 73,728 draws?** Show original
shared hybrid, active-pruned/reallocated shared hybrid, triple U hybrid, and
complete pooled U variance lower bounds. Add a clear vertical zero line and
use the same distinct method colors/markers for all eight rows. Show exact
census target as a black diamond. Because the shared negatives are much farther
from zero than the U bounds, either use separate aligned narrow near-zero and
full-range insets clearly marked as different scales, or choose the simpler
full-range plot with direct “3/8 pass” labels beside the U legends. Do not use
an unlabeled broken axis. A two-column-width figure is preferable to illegible
single-column labels.

All values come from `results/certificate_all_eight.csv`, which has raw center,
interaction estimate, corrected center, both radii and lower bounds. The pooled
method's `bound` is named `variance`, rather than `hybrid`, because it uses the
complete-U variance formula. The original and reallocated shared methods both
cost 73,728 draws. Do not substitute `shared_pruned_same_counts`, which costs
57,344 and has the original lower bound exactly.

Suggested caption: “The learned-trajectory diagnostic on the corrected, already
inspected Beijing archive. Each candidate has the same 342-record fitted target
and 73,728 sampled draw positions. Deleting the zero reference column and
choosing M=13,086, n=15,852 by declared range widths tightens the shared bound
but retains no candidate. Triple U and pooled U retain three candidates using
their hybrid and variance bounds; all pure-Hoeffding lower bounds are negative.
Archive inference does not certify future MSE improvement.”

Compact alternative: use the boosting_31 four-method decomposition (recorded
shared, pruned/reallocated shared, triple U, pooled U), plotting raw center →
subtract signed interaction → subtract evaluation radius → subtract validation
radius. Use distinct encodings for interaction removal and uncertainty. Exact
target is 0.852133 in units of 10^-3. Coordinates are
`results/figure_boosting_certificate_coordinates.csv`; its columns ending in
`_times_1000` are already scaled. A small aligned panel lists charged/used draw
positions for original and pruned designs. Keep census visually distinct from
sampling lower bounds.

Future-loss values and intervals belong in the associated table/text or a
supplement: `results/all_eight_future_losses.csv` and
`results/future_rule_comparisons.csv`. A plot of only future loss would miss
the reference-interaction mechanism and allocation comparison addressed here.
