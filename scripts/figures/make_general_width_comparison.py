"""Render all eight trajectory candidates from verified recorded coordinates.

No simulations, fitting, allocation choice or numerical-result edits. Portable
defaults resolve from this script's package root. Requires matplotlib, pandas,
numpy, PyMuPDF, pdflatex, IEEEtran and PGF. Output is a 7.1 x 2.6 inch vector PDF,
with all text at 9 TeX points using IEEEtran's Nimbus Roman and CM math fonts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

NAME = "fig_trajectory_comparison"
SIZE = (7.1, 2.6)
PREAMBLE = r"\usepackage{amsmath,amssymb,amsfonts}\renewcommand{\rmdefault}{ptm}"
NAMES = ["persistence", "ridge_1", "ridge_100", "boosting_7", "boosting_31",
         "blend_ridge_1", "blend_boosting_7", "blend_boosting_31"]
LABELS = ["Persistence", "Ridge 1", "Ridge 100", "Boosting 7", "Boosting 31",
          "Blend Ridge 1", "Blend Boosting 7", "Blend Boosting 31"]
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(package, output):
    # The artifact carries IEEEtran.cls with its manuscript sources. Respect
    # additional caller TEXINPUTS and retain TeX's default search path.
    tex_support = package / "results/trajectory_budget_figures/tex"
    os.environ["TEXINPUTS"] = str(tex_support) + ":" + str(package / "manuscript") + ":" + os.environ.get("TEXINPUTS", "") + ":"
    source = package / "results/trajectory_budget/results/certificate_all_eight.csv"
    data = pd.read_csv(source).set_index(["candidate", "method", "bound"])
    output.mkdir(parents=True, exist_ok=True)
    original = data.loc[(NAMES, "shared_original", "hybrid"), :].droplevel([1, 2]).loc[NAMES]
    pruned = data.loc[(NAMES, "shared_pruned_allocated", "hybrid"), :].droplevel([1, 2]).loc[NAMES]
    pooled = data.loc[(NAMES, "complete_pooled_u", "variance"), :].droplevel([1, 2]).loc[NAMES]
    assert list(original.index) == list(pruned.index) == list(pooled.index) == NAMES
    np.testing.assert_array_equal(original.exact_target, pruned.exact_target)
    np.testing.assert_array_equal(original.exact_target, pooled.exact_target)
    np.testing.assert_allclose(original.raw_center - original.interaction_estimate,
                               original.corrected_center, atol=2e-16, rtol=0)
    for frame in [original, pruned, pooled]:
        np.testing.assert_allclose(frame.corrected_center - frame.evaluation_radius - frame.validation_radius,
                                   frame.lower, atol=3e-16, rtol=0)
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "text.usetex": True, "pgf.texsystem": "pdflatex",
                         "pgf.rcfonts": False, "pgf.preamble": PREAMBLE,
                         "axes.unicode_minus": False, "axes.linewidth": .55,
                         "xtick.major.width": .5, "xtick.major.size": 2.5})
    fig = plt.figure(figsize=SIZE)
    # Inches specify a stable layout; no auto rescaling or broken-axis transform.
    def axes(left, width):
        return fig.add_axes([left / SIZE[0], .56 / SIZE[1], width / SIZE[0], 1.53 / SIZE[1]])
    left, full, zoom = axes(1.11, 2.12), axes(3.64, 1.56), axes(5.72, 1.26)
    y = np.arange(8)
    for ax in [left, full, zoom]:
        ax.set_ylim(7.5, -.5)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", pad=3)
        for yy in y:
            ax.axhline(yy, color=".92", lw=.45, zorder=0)
        ax.axvline(0, color=".50", lw=.65, zorder=1)
    left.set_yticklabels(LABELS)
    left.set_xlim(-1.05, 15.7)
    left.set_xticks([0, 5, 10, 15])
    full.set_xlim(-16.2, 1.1)
    full.set_xticks([-15, -10, -5, 0])
    zoom.set_xlim(-.46, 1.05)
    zoom.set_xticks([-.4, 0, .5, 1.0])
    zoom.set_xticklabels([r"$-0.4$", "0", "0.5", "1.0"])
    left.set_xlabel(r"Shared score and target $(10^{-3})$", labelpad=5)
    full.set_xlabel(r"Lower bound $(10^{-3})$" + "\nFull scale", labelpad=5)
    zoom.set_xlabel(r"Lower bound $(10^{-3})$" + "\nNear zero: separate scale", labelpad=5)
    fig.text(2.12 / SIZE[0], 2.52 / SIZE[1], "(a) Removing the reference interaction", ha="center", va="center")
    fig.text(5.30 / SIZE[0], 2.52 / SIZE[1], "(b) Same draw budget, different bounds", ha="center", va="center")
    records = []
    def plot_points(ax, axis_name, values, series, color, marker, size, face=None, z=3):
        x = np.asarray(values) * 1000
        line, = ax.plot(x, y, marker=marker, color=color,
                        markerfacecolor=color if face is None else face,
                        markersize=size, markeredgewidth=.7,
                        linestyle="none", zorder=z)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        for k, candidate in enumerate(NAMES):
            records.append({"axis": axis_name, "series": series, "candidate": candidate,
                            "y": int(y[k]), "unscaled_value": float(values.iloc[k]),
                            "plotted_x_times_1000": float(x[k])})
    for yy, raw, corrected in zip(y, original.raw_center, original.corrected_center):
        left.plot([raw * 1000, corrected * 1000], [yy, yy], color=".60", lw=.75, zorder=2)
    plot_points(left, "interaction", original.raw_center, "Raw shared score", ".2", "o", 4.7, "white", 4)
    plot_points(left, "interaction", original.corrected_center, "Corrected center", BLUE, "s", 3.8, z=5)
    plot_points(left, "interaction", original.exact_target, "Exact census target", ".1", "D", 2.4, z=6)
    for ax, axis_name in [(full, "lower_full"), (zoom, "lower_zoom")]:
        if ax is full:
            plot_points(ax, axis_name, original.lower, "Shared hybrid lower", BLUE, "o", 4.0, "white", 4)
            plot_points(ax, axis_name, pruned.lower, "Pruned hybrid lower", ORANGE, "s", 3.7, z=4)
        plot_points(ax, axis_name, pooled.lower, "Pooled variance lower", GREEN, "^", 4.3, z=5)
        plot_points(ax, axis_name, original.exact_target, "Exact census target", ".1", "D", 2.4, z=6)
    legend_left = [Line2D([], [], ls="none", marker="o", color=".2", mfc="white", ms=4.7, label="Raw"),
                   Line2D([], [], ls="none", marker="s", color=BLUE, ms=3.8, label="Corrected"),
                   Line2D([], [], ls="none", marker="D", color=".1", ms=2.4, label="Census")]
    legend_right = [Line2D([], [], ls="none", marker="o", color=BLUE, mfc="white", ms=4.0, label="Shared"),
                    Line2D([], [], ls="none", marker="s", color=ORANGE, ms=3.7, label="Reallocated"),
                    Line2D([], [], ls="none", marker="^", color=GREEN, ms=4.3, label="Pooled U"),
                    Line2D([], [], ls="none", marker="D", color=".1", ms=2.4, label="Census")]
    fig.legend(handles=legend_left, loc="center", bbox_to_anchor=(2.12 / SIZE[0], 2.27 / SIZE[1]),
               ncol=3, frameon=False, handletextpad=.35, columnspacing=.8, borderaxespad=0)
    fig.legend(handles=legend_right, loc="center", bbox_to_anchor=(5.30 / SIZE[0], 2.27 / SIZE[1]),
               ncol=4, frameon=False, handletextpad=.35, columnspacing=.65, borderaxespad=0)
    stem = output / NAME
    fig.savefig(stem.with_suffix(".pgf"), backend="pgf")
    wrapper = (r"\documentclass[conference]{IEEEtran}" + "\n" +
               r"\usepackage{amsmath,amssymb,amsfonts,pgf}\providecommand{\mathdefault}[1]{#1}" + "\n" +
               r"\usepackage[paperwidth=7.1in,paperheight=2.6in,margin=0pt]{geometry}" + "\n" +
               r"\pagestyle{empty}\setlength{\parindent}{0pt}" + "\n" +
               r"\pdfinfoomitdate=1\pdftrailerid{}\pdfsuppressptexinfo=15" + "\n" +
               r"\begin{document}\onecolumn\thispagestyle{empty}" + "\n" +
               rf"\noindent\input{{{NAME}.pgf}}" + "\n" + r"\end{document}" + "\n")
    stem.with_suffix(".tex").write_text(wrapper)
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", NAME + ".tex"],
                         cwd=output, text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError("PGF compilation failed:\n" + run.stdout + run.stderr)
    doc = fitz.open(stem.with_suffix(".pdf"))
    assert len(doc) == 1
    page = doc[0]
    assert not page.get_images(full=True), "Unexpected raster content"
    font_records = []
    for font in page.get_fonts():
        assert font[2] == "Type1" and doc.extract_font(font[0])[3], font
        font_records.append({"name": font[3], "type": font[2], "embedded": True})
    outside = [s["text"] for b in page.get_text("dict")["blocks"] if "lines" in b
               for line in b["lines"] for s in line["spans"]
               if not (page.rect + (-.1, -.1, .1, .1)).contains(fitz.Rect(s["bbox"]))]
    assert not outside, outside
    page.get_pixmap(matrix=fitz.Matrix(3, 3)).save(stem.with_suffix(".png"))
    stem.with_suffix(".svg").write_text(page.get_svg_image(text_as_path=True))
    coord = pd.DataFrame(records)
    assert len(coord) == 72
    coord.to_csv(output / "trajectory_comparison_coordinates.csv", index=False)
    source_copy = output / "trajectory_comparison_source.csv"
    source_copy.write_bytes(source.read_bytes())
    receipt = {"status": "PASS", "figure": NAME, "source": str(source.relative_to(package)),
               "source_sha256": sha(source), "generator_sha256": sha(Path(__file__)),
               "pdf_sha256": sha(stem.with_suffix(".pdf")), "size_inches": list(SIZE),
               "tex_support": {str(p.relative_to(package)): sha(p) for p in sorted(tex_support.glob("*.sty"))},
               "ordinary_text_tex_points": 9, "fonts": font_records, "raster_images": 0,
               "outside_page_text": outside, "coordinate_count": len(coord),
               "coordinate_validation": "Every plotted x equals 1000 times its source value exactly; every y is the declared candidate row, without jitter. Shared centers and all lower-bound decompositions also checked numerically.",
               "axis_limits": {"interaction": list(left.get_xlim()), "lower_full": list(full.get_xlim()), "lower_zoom": list(zoom.get_xlim())},
               "near_zero_scope": "Separate labeled linear scale displays pooled lower bounds and exact census targets only. Shared/pruned bounds are outside its range and remain visible on the full scale.",
               "passing_counts": {"shared_hybrid": int((original.lower > 0).sum()),
                                   "pruned_hybrid": int((pruned.lower > 0).sum()),
                                   "pooled_variance": int((pooled.lower > 0).sum())},
               "analysis_status": "Post-exposure diagnostic on corrected previously inspected cohorts; not independent confirmation."}
    (output / "TRAJECTORY_FIGURE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # Successful public renders retain portable source/receipts, without local
    # TeX filesystem paths or transient compiler logs.
    for suffix in [".aux", ".log", ".build.txt", ".tex"]:
        stem.with_suffix(suffix).unlink(missing_ok=True)
    plt.close(fig)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package_root.resolve()
    main(package, (args.output or package / "results/trajectory_budget_figures").resolve())
