"""Render supplementary public plots and the full model table with IEEE fonts.

Uses only recorded results. Requires the same dependencies as
make_companion_figures.py, which must be in the same directory.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import fitz

spec = importlib.util.spec_from_file_location('figure_renderer', Path(__file__).with_name('make_companion_figures.py'))
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)
PACKAGE = render.PACKAGE_ROOT
OUTPUT = render.BUILD_ROOT
render.SIZES.update({'peer_sampling':(6.9,2.7),'fig_certificate_efficiency':(6.9,2.95),
                     'source_time_overlap':(6.9,3.35)})
base_export = render.export


def export(fig, name):
    if name == 'peer_sampling':
        fig.axes[1].set_title('(b) Fixed peers,\naligned effects', pad=6)
        fig.axes[2].set_title('(c) Fixed peers,\nopposed effects', pad=6)
    elif name == 'fig_certificate_efficiency':
        fig.axes[0].set_title('(a) Two balanced categories\n'+r'$\vartheta=0.010$',loc='left',pad=5)
        fig.axes[1].set_title('(b) Eight categories with a rare cell\n'+r'$\vartheta=0.0137$',loc='left',pad=5)
        fig.subplots_adjust(top=.61,bottom=.19)
        legend=fig.legends[0]
        # Original five methods retained; use two columns to fit 9-point labels.
        handles,labels=fig.axes[0].get_legend_handles_labels()
        legend.remove()
        fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.53,1.01),
                   frameon=False,columnspacing=1.3,handlelength=2.1)
    elif name == 'source_time_overlap':
        for ax in fig.axes:
            for text in ax.texts:
                if text.get_text()=='c × d': text.set_text(r'$c\times d$')
        for text in fig.texts:
            if text.get_text().startswith('Fixed comparison coefficients'):
                text.set_text(r'Fixed comparison coefficients; independent uniform $\{-2,0,5\}$ source values.'+'\n'+
                              'Each population-reference product has mean zero.')
                text.set_position((.5,.018))
        fig.subplots_adjust(bottom=.15,top=.86)
    base_export(fig,name)


def escape(text):
    mapping={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(mapping.get(c,c)for c in str(text))


def complete_table():
    display=json.loads((PACKAGE/'results/publication_figure_assets/public_41_complete.display.json').read_text())
    assert len(display)==41
    titles={'electricity':'Electricity (MAE: original UCI load units)', 'ratings':'Ratings (MAE: rating points)',
            'retail':'Retail (MAE: weekly sales units)', 'portfolios':'Portfolios (MAE: percentage points of return)'}
    descriptions=[
      'Score: mean within-cluster Spearman correlation over clusters where it is defined; descriptive only. MAE: original-scale mean absolute error, with domain-specific units. The CSV retains full stored precision.',
      'RETAIN is the operational diagnostic screen at 0.05; real-panel calibration is not established. NOT_RETAINED means non-rejection; original CSV null codes denote the same decision. ABSTAIN denotes a guard or an undefined statistic. An asterisk marks the five finite copy or weak-order guards.',
      'Global mean and User mean have undefined descriptive scores and statistics. SVD interaction is an interaction-only score and has no rating-scale MAE. Training series mean has an undefined statistic scale. All three undefined statistic pairs remain in their model families.']
    source=[r'\documentclass[conference]{IEEEtran}',r'\usepackage{amsmath,amssymb,amsfonts}',
            r'\usepackage[paperwidth=11.7in,paperheight=8.3in,margin=.45in]{geometry}',
            r'\pagestyle{empty}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}',
            r'\pdfinfoomitdate=1\pdftrailerid{}\pdfsuppressptexinfo=15',
            r'\begin{document}\onecolumn\fontsize{9}{11}\selectfont']
    for page, domains in enumerate([('electricity','ratings'),('retail','portfolios')],1):
        if page>1:source.append(r'\newpage')
        source += [r'\textbf{Complete public-model comparison}\hfill '+str(page)+' / 2'+r'\par',
                   'All 41 scored models; complementary-fold fits and guarded within-domain '+r'$\beta=2$ BY decisions.\par',
                   r'\renewcommand{\arraystretch}{1.3}\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrl@{}}\hline',
                   r'Model & Score & MAE & $T$ ($\beta=2$) & $T$ (spline) & Guarded BY $p$ & Decision\\\hline']
        for domain in domains:
            source.append(r'\multicolumn{7}{l}{\textbf{'+escape(titles[domain])+r'}}\\')
            for row in display:
                if row['domain']!=domain:continue
                values=list(row['values']);values[0]+=' *'if row['finite_guard']else''
                source.append(' & '.join(escape(v)for v in values)+r'\\')
            source.append(r'\hline')
        source += [r'\end{tabular*}\par']+[escape(p)+r'\par'for p in descriptions]
    source.append(r'\end{document}')
    path=OUTPUT/'public_41_complete.tex';path.write_text('\n'.join(source)+'\n')
    proc=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',path.name],cwd=OUTPUT,text=True,capture_output=True)
    (OUTPUT/'public_41_complete.build.txt').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError('Complete table compilation failed')
    doc=fitz.open(OUTPUT/'public_41_complete.pdf');assert len(doc)==2
    for i,page in enumerate(doc,1):
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUTPUT/f'public_41_complete_page{i}.png')
        (OUTPUT/f'public_41_complete_page{i}.svg').write_text(page.get_svg_image(text_as_path=True))
    (OUTPUT/'public_41_complete_coordinates.json').write_text(json.dumps(display,indent=2)+'\n')


def main():
    render.export=export
    render.Figure.savefig=render.intercept_save
    sys.argv=['plot','--source',str(PACKAGE/'results/design_validation/peer_sampling.csv'),'--output-dir',str(OUTPUT/'peer_records')]
    m=render.module('scripts/figures/make_peer_sampling_figure.py');m['main']()
    sys.argv=['plot','--summary',str(PACKAGE/'results/reference_certificate_efficiency/summary.csv'),'--output-dir',str(OUTPUT/'efficiency_records')]
    m=render.module('scripts/figures/make_efficiency_figure.py');m['main']()
    m=render.module('results/reference_time_composition/make_overlap_guide.py');m['make'](OUTPUT/'overlap_records')
    complete_table()
    (OUTPUT/'DISPLAY_REBUILD_RECEIPT.json').write_text(json.dumps({'figures':render.RECORDS,
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'experiments_run':0},indent=2)+'\n')

if __name__=='__main__':main()
