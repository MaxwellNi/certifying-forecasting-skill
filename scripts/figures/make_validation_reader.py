"""Draw learning allowances and sampling comparisons at publication size."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("rebuilt_reader_figures"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

source_a = PACKAGE / "results/certificate_factorial/summary.csv"
source_b = PACKAGE / "results/aggregate_bias/simulation/summary.csv"
factorial = pd.read_csv(source_a, float_precision="round_trip")
left = factorial[(factorial.design == "eight_rare") &
                 (factorial.signal == .25) & (factorial.fit == "estimated") &
                 (factorial.validation_pairs == 8192)].copy()
assert len(left) == 16
budgets = [26176, 30976, 50176, 126976]
assert sorted(left.total_observations.unique()) == budgets
assert left.replications.eq(1000).all()
assert np.allclose(left.power, left.rejections / left.replications, atol=0, rtol=0)
expected = {"absolute_range": [0., 0., 0., .559],
            "signed_range": [0., 0., 0., .739],
            "absolute_variance": [0., .274, 1., 1.],
            "signed_variance": [0., .435, 1., 1.]}
for name, values in expected.items():
    assert left[left.method == name].sort_values("total_observations").power.tolist() == values

summary = pd.read_csv(source_b, float_precision="round_trip")
keys = ["categories", "validation_pairs", "mass", "fit_error"]
assert len(summary) == 144 and summary.repetitions.eq(2000).all()
assert not summary.duplicated(keys + ["method"]).any()
points = summary.pivot(index=keys, columns="method", values="median_slack").reset_index()
points.columns.name = None
assert len(points) == 72
assert int((points["aggregate"] < points["rectangle"]).sum()) == 62
assert int((points["aggregate"] > points["rectangle"]).sum()) == 10
assert (points[["aggregate", "rectangle"]] > 0).all().all()

plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "rank-validation-figure"
})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0),
                         gridspec_kw={"width_ratios": [1.05, 1]})
fig.subplots_adjust(left=.115, right=.985, bottom=.24, top=.72, wspace=.45)
blue, orange = "#176b9d", "#b35a00"
styles = [
    ("absolute_range", "Absolute + range", orange, "x", "none", -.24),
    ("signed_range", "Signed + range", blue, "D", "white", -.08),
    ("absolute_variance", "Absolute + variance", orange, "s", "white", .08),
    ("signed_variance", "Signed + variance", blue, "o", blue, .24),
]
ax = axes[0]
for index in [0.5, 1.5, 2.5]:
    ax.axhline(index, color=".89", linewidth=.6, zorder=0)
handles = []
for method, label, color, marker, face, offset in styles:
    rows = left[left.method == method].sort_values("total_observations")
    y = np.arange(4) + offset
    ax.errorbar(rows.power, y,
                xerr=np.vstack([rows.power - rows.power_ci_lower,
                                rows.power_ci_upper - rows.power]),
                fmt=marker, color=color, markerfacecolor=face,
                markersize=2.8 if marker != "x" else 3.2, linewidth=.8,
                capsize=1.0, capthick=.7, linestyle="none", clip_on=False)
    handles.append(Line2D([], [], marker=marker, color=color,
                          markerfacecolor=face, markersize=4, linestyle="none",
                          label=label))
ax.set_xlim(-.035, 1.035)
ax.set_ylim(3.48, -.48)
ax.set_yticks(range(4), [f"{n:,}" for n in budgets])
ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
ax.set_ylabel("Total observations", labelpad=3)
ax.set_xlabel("Rejection rate", labelpad=3)
ax.set_title(r"(a) Eight categories, $\vartheta=0.01366$", pad=6)
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.46, 1.20),
          ncol=2, frameon=False, columnspacing=.85, handletextpad=.35,
          handlelength=.8, labelspacing=.35, borderpad=0)

ax = axes[1]
markers = {2: "o", 8: "s", 32: "D", 128: "^"}
for category, marker in markers.items():
    rows = points[points.categories == category]
    ax.scatter(rows["rectangle"], rows["aggregate"], s=12, marker=marker,
               facecolors="white", edgecolors=".20", linewidths=.65,
               zorder=3, clip_on=False)
extent = [.0005, .5]
ax.plot(extent, extent, color=".4", linewidth=.8, linestyle=(0, (3, 2)), zorder=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(extent)
ax.set_ylim(extent)
ax.set_xticks([.001, .01, .1], ["0.001", "0.01", "0.1"])
ax.set_yticks([.001, .01, .1], ["0.001", "0.01", "0.1"])
ax.minorticks_off()
ax.set_xlabel("Rectangle median slack", labelpad=3)
ax.set_ylabel("Aggregate median slack", labelpad=3)
ax.set_title("(b) Absolute validation slack", pad=6)
ax.text(.04, .92, "Equal slack", transform=ax.transAxes, ha="left", va="top",
        color=".30")
ax.text(.96, .06, "Below line:\naggregate tighter", transform=ax.transAxes,
        ha="right", va="bottom", color=".20", linespacing=1.1)
ax.legend(handles=[Line2D([], [], marker=marker, color=".2",
                          markerfacecolor="white", linestyle="none",
                          markersize=4, label=str(category))
                   for category, marker in markers.items()],
          title="Number of categories", title_fontsize=8.5,
          loc="lower center", bbox_to_anchor=(.5, 1.20), ncol=4,
          frameon=False, handlelength=.8, handletextpad=.3,
          columnspacing=.65, labelspacing=.35, borderpad=0)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.6, width=.55)
    ax.grid(axis="x" if ax is axes[0] else "both", color=".92", linewidth=.45)
    ax.set_axisbelow(True)

stem = args.output / "fig_validation_reader"
for extension in ["pdf", "svg", "png"]:
    metadata = {"CreationDate": None, "ModDate": None, "Creator": None,
                "Producer": None} if extension == "pdf" else (
                {"Date": None, "Creator": None} if extension == "svg" else None)
    fig.savefig(stem.with_suffix("." + extension), dpi=300, metadata=metadata)
plt.close(fig)
left.to_csv(args.output / "fig_validation_reader_power.csv", index=False,
            float_format="%.17g")
points.to_csv(args.output / "fig_validation_reader_slack.csv", index=False,
              float_format="%.17g")
record = {
    "native_size_inches": [7.0, 2.0], "minimum_text_points": 8.5,
    "input_sha256": {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [source_a, source_b]},
    "left": "All four recorded budgets for the previously shown fixed ablation; 1,000 repetitions, recorded pointwise exact binomial (Clopper-Pearson) 95% intervals.",
    "right": "All 72 recorded matched settings, 2,000 repetitions per method per setting; no jitter or result selection.",
    "counts": {"ablation_points": 16, "validation_settings": 72,
               "aggregate_tighter": 62, "rectangle_tighter": 10},
    "scope": "Absolute slack has the target's covariance units. These validation simulations have no matched evaluation statistic or decision threshold; the figure does not invent one.",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record["counts"]))
