"""Render the reference-design diagram with the IEEEtran template fonts.

Requires pdflatex, IEEEtran, TikZ and AMS packages. The --svg option also
requires PyMuPDF. LaTeX files are temporary build outputs, not paper sources.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile
import shutil

FIGURE1_TIKZ = r"""% Native vector diagram at publication size: IEEEtran default Times and CM math.
\begin{tikzpicture}[
 x=1mm,y=1mm,font=\fontsize{9}{10.2}\selectfont,>=Stealth,
 box/.style={draw=black!65,line width=.45pt,align=left,inner sep=1.1mm,anchor=north west},
 arrow/.style={->,line width=.55pt,draw=black!80},
 note/.style={align=left,inner sep=0pt,anchor=north west},
 ref/.style={draw=black!60,line width=.45pt,minimum width=8mm,minimum height=5mm,inner sep=0pt},
 val/.style={inner sep=0pt}]
\path[use as bounding box] (0,0) rectangle (174,-50);
\node[note] at (0,0) {\textbf{Training fixes learned maps and coefficients before validation and evaluation.}};
\draw[black!35,line width=.4pt] (87,-5) -- (87,-50);
\node[note] at (0,-5) {\textbf{(a) Category certificate}};
\node[note] at (91,-5) {\textbf{(b) Whole-trajectory design}};
\node[box,text width=36mm] (validate) at (0,-10)
 {\textbf{Validation $A,B$}\\Independent pairs\\Known category masses};
\node[box,text width=37mm] (evaluate) at (44,-10)
 {\textbf{Evaluation $E$}\\Independent groups};
\node[box,text width=36mm] (allowance) at (0,-27)
 {Learning allowance\\$\widehat B_{\rm agg}$: category errors};
\node[box,text width=37mm] (sampling) at (44,-27)
 {Corrected score $\bar D$\\Sampling radius $r_U$};
\draw[arrow] (validate.south) -- (validate.south |- allowance.north);
\draw[arrow] (evaluate.south) -- (evaluate.south |- sampling.north);
\node[box,text width=81mm] (cat) at (0,-41)
 {$L_\alpha=\bar D-\widehat B_{\rm agg}-r_U(\alpha)>0$\\Positive association beyond the declared categories};
\draw[arrow] (allowance.south) -- (allowance.south |- cat.north);
\draw[arrow] (sampling.south) -- (sampling.south |- cat.north);
\node[note,text width=83mm] at (91,-10)
 {$\mathsf W_i$ independent under one law given $\mathcal H$;\\dependence within each trajectory is unrestricted.};
\node[note] at (94,-19) {Shared reference};
\node[note] at (140,-19) {Distinct references};
\node[ref] (shared) at (111,-27) {$\mathsf W_1$};
\node[val,align=center] (sa) at (99,-35) {$A$: forecast\\contrast};
\node[val,align=center] (sb) at (123,-35) {$B$: outcome\\contrast};
\draw[arrow] (shared.south west) -- (sa.north);
\draw[arrow] (shared.south east) -- (sb.north);
\node[ref] (ra) at (147,-27) {$\mathsf W_1$};
\node[ref] (rb) at (167,-27) {$\mathsf W_2$};
\node[val,align=center] (da) at (147,-35) {$A$: forecast\\contrast};
\node[val,align=center] (db) at (167,-35) {$B$: outcome\\contrast};
\draw[arrow] (ra.south) -- (da.north);
\draw[arrow] (rb.south) -- (db.north);
\node[note,text width=83mm] at (91,-41)
 {Same focal $\mathsf W_0$ and row sums $c,d$: same target $\theta_{\mathcal H}$.\\Shared-source term $\langle \mathbf C\mathbf D^T,\Lambda\rangle$; disjoint pools set $\mathbf C\mathbf D^T=0$.};
\end{tikzpicture}
"""
WRAPPER = r"""\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning,calc}
\usepackage[paperwidth=178mm,paperheight=64mm,margin=2mm]{geometry}
\pagestyle{empty}\setlength{\parindent}{0pt}
\pdfinfoomitdate=1\pdftrailerid{}\pdfsuppressptexinfo=15
\begin{document}\onecolumn\thispagestyle{empty}
\noindent\input{figure1_dual_tikz.tex}
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reference_design.pdf"))
    parser.add_argument("--svg", action="store_true", help="Also write a vector SVG with outlined font glyphs")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reference-design-") as folder:
        work = Path(folder)
        (work / "figure1_dual_tikz.tex").write_text(FIGURE1_TIKZ)
        (work / "figure1_dual.tex").write_text(WRAPPER)
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure1_dual.tex"],
                                cwd=work, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        shutil.copyfile(work / "figure1_dual.pdf", args.output)
    if args.svg:
        import fitz
        with fitz.open(args.output) as document:
            if len(document) != 1:
                raise RuntimeError("Expected a one-page diagram")
            args.output.with_suffix(".svg").write_text(document[0].get_svg_image(text_as_path=True))
    print(args.output)


if __name__ == "__main__":
    main()
