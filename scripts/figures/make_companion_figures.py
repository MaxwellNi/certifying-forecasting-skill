"""Render companion figures from recorded results with IEEEtran text and math.

Requires NumPy, pandas, Matplotlib, PyMuPDF, pdflatex, IEEEtran, PGF and the
LaTeX underscore package. This runs plotting code only, never simulations or
training. PDF and SVG outputs are vector graphics; all ordinary labels use
9 TeX points at the companion's 6.9-inch placement width.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np

PARSER=argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[2])
PARSER.add_argument('--output',type=Path,required=True)
ARGS=PARSER.parse_args()
PACKAGE_ROOT=ARGS.package_root.resolve()
BUILD_ROOT=ARGS.output.resolve();BUILD_ROOT.mkdir(parents=True,exist_ok=True)
PREAMBLE=r'\usepackage{amsmath,amssymb,amsfonts}\renewcommand{\rmdefault}{ptm}'
SIZES={'fig_smooth_bin_bias':(6.9,2.4),'fig_canonical_comparison':(6.9,3.25),
       'fig_public_named_comparison':(6.9,5.0),'fig_public_directional':(6.9,4.6),
       'fig_oracle_rank':(6.9,3.9),'fig_historical_sensitivity':(6.9,5.9),
       'fig_gaussian_reference':(6.9,2.9),'fig_certificate_factorial':(6.9,2.55)}
OLD_SAVE=Figure.savefig
RECORDS={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def serial(value):
    array=np.asarray(value)
    if array.dtype.kind=='O':return str(value)
    return array.tolist()

def coordinates(fig):
    rows=[]
    for ax in fig.axes:
        row={'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
             'xscale':ax.get_xscale(),'yscale':ax.get_yscale(),
             'lines':[{'xy':serial(line.get_xydata())}for line in ax.lines], 'collections':[]}
        for collection in ax.collections:
            entry={'type':type(collection).__name__,'offsets':serial(collection.get_offsets())}
            if hasattr(collection,'get_segments'):entry['segments']=[serial(s)for s in collection.get_segments()]
            if hasattr(collection,'get_coordinates'):entry['coordinates']=serial(collection.get_coordinates())
            if collection.get_array() is not None:entry['array']=serial(collection.get_array())
            row['collections'].append(entry)
        rows.append(row)
    return rows

def tex_text(text):
    text=text.replace('\u2212',r'$-$')
    text=re.sub(r'(?<!\\)%',r'\\%',text)
    return text


def export(fig,name):
    if name in RECORDS:return
    before=coordinates(fig)
    plt.rcParams.update({'text.usetex':True,'font.family':'serif','font.size':9,
                         'pgf.texsystem':'pdflatex','pgf.rcfonts':False,'pgf.preamble':PREAMBLE,
                         'axes.unicode_minus':False})
    fig.set_size_inches(*SIZES[name],forward=True)
    for ax in fig.axes:
        ax.get_xticklabels();ax.get_yticklabels()
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily('serif');artist.set_fontsize(9);artist.set_usetex(True)
        artist.set_text(tex_text(artist.get_text()))
    if name=='fig_public_named_comparison':
        fig.subplots_adjust(left=.26,right=.985,wspace=1.6)
    if name=='fig_oracle_rank':
        fig.subplots_adjust(left=.275,right=.97,bottom=.13,top=.9,wspace=.38,hspace=.55)
    if name=='fig_certificate_factorial':
        fig.subplots_adjust(bottom=.205,top=.685)
        fig.axes[0].set_title(r'(a) Eight categories'+'\n'+r'$\vartheta=0.01366$',fontsize=9,pad=5)
        fig.axes[1].set_title('(b) Mean components',fontsize=9,pad=5)
        fig.axes[1].get_legend().set_bbox_to_anchor((1.05,1.47))
        fig.axes[2].set_title('(c) Opposed fits'+'\n'+r'$\vartheta=0.02$',fontsize=9,pad=5)
    for ax in fig.axes:
        for label in ax.get_xticklabels()+ax.get_yticklabels():
            label.set_fontfamily('serif');label.set_fontsize(9);label.set_usetex(True)
            label.set_text(tex_text(label.get_text()))
    after=coordinates(fig)
    assert json.dumps(before,sort_keys=True,allow_nan=True)==json.dumps(after,sort_keys=True,allow_nan=True),name
    stem=BUILD_ROOT/name
    OLD_SAVE(fig,stem.with_suffix('.pgf'),backend='pgf')
    w,h=SIZES[name]
    wrapper=(r'\documentclass[conference]{IEEEtran}'+'\n'+r'\usepackage{amsmath,amssymb,amsfonts,pgf}\providecommand{\mathdefault}[1]{#1}'+'\n'+
             rf'\usepackage[paperwidth={w}in,paperheight={h}in,margin=0pt]{{geometry}}'+'\n'+
             r'\pagestyle{empty}\setlength{\parindent}{0pt}'+'\n'+
             r'\pdfinfoomitdate=1\pdftrailerid{}\pdfsuppressptexinfo=15'+'\n'+
             r'\begin{document}\onecolumn\thispagestyle{empty}'+'\n'+
             rf'\noindent\input{{{name}.pgf}}'+'\n'+r'\end{document}'+'\n')
    stem.with_suffix('.tex').write_text(wrapper)
    run=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=BUILD_ROOT,text=True,capture_output=True)
    stem.with_suffix('.build.txt').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(f'LaTeX failed: {stem}')
    doc=fitz.open(stem.with_suffix('.pdf'));assert len(doc)==1,(name,len(doc))
    page=doc[0]
    stem.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True))
    page.get_pixmap(matrix=fitz.Matrix(2,2)).save(stem.with_suffix('.png'))
    outside=[s['text']for b in page.get_text('dict')['blocks']if 'lines'in b for line in b['lines']for s in line['spans'] if not(page.rect+(-.1,-.1,.1,.1)).contains(fitz.Rect(s['bbox']))]
    fonts=sorted({f[3].split('+')[-1]for f in page.get_fonts()})
    assert not page.get_images(full=True)
    for f in page.get_fonts():assert f[2]=='Type1' and doc.extract_font(f[0])[3]
    RECORDS[name]={'source_plot_coordinates':before,'data_coordinates_and_axis_scales_exactly_unchanged':True,
                   'size_inches':[w,h],'ordinary_font_tex_points':9,'font_names':fonts,'outside_text':outside,
                   'pdf_sha256':sha(stem.with_suffix('.pdf')),'svg_sha256':sha(stem.with_suffix('.svg'))}
    (BUILD_ROOT/'partial_receipt.json').write_text(json.dumps(RECORDS,indent=2,allow_nan=True)+'\n')
    print(name,fonts,'outside',outside,flush=True)
    plt.rcParams['text.usetex']=False


def intercept_save(fig,filename,*args,**kwargs):
    stem=Path(filename).stem
    stem={'gaussian_reference':'fig_gaussian_reference','historical_sensitivity':'fig_historical_sensitivity'}.get(stem,stem)
    if stem in SIZES:
        export(fig,stem)
        # Historical plotting routines may hash their local generated paths.
        source=(BUILD_ROOT/stem).with_suffix(Path(filename).suffix)
        target=Path(filename)
        if source.is_file() and source.resolve()!=target.resolve():
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        return
    return OLD_SAVE(fig,filename,*args,**kwargs)


def module(relative,changes=()):
    path=PACKAGE_ROOT/relative
    source=path.read_text().replace('exist_ok=False','exist_ok=True')
    for old,new in changes:
        assert old in source,(relative,old)
        source=source.replace(old,new)
    namespace={'__name__':'companion_plot_source','__file__':str(path),
               'PACKAGE_ROOT':PACKAGE_ROOT,'BUILD_ROOT':BUILD_ROOT}
    exec(compile(source,str(path),'exec'),namespace)
    return namespace


def save_record(fig,out,name,record):
    export(fig,name);plt.close(fig);return record


def directional():
    rows=list(csv.DictReader((PACKAGE_ROOT/'results/focused_directional/public_directional_models.csv').open()))
    assert len(rows)==82
    key={(r['domain'],r['model'],r['configuration']):r for r in rows}
    fig,axes=plt.subplots(2,2,figsize=SIZES['fig_public_directional'])
    fig.subplots_adjust(left=.265,right=.985,bottom=.10,top=.91,hspace=.42,wspace=1.6)
    panels=[('electricity','Electricity: 438 days',(-4,12),[0,5,10]),
            ('ratings','MovieLens: 168 user clusters',(-.5,14),[0,5,10]),
            ('retail','M5 retail: 17 weeks',(-21,15),[-20,-10,0,10]),
            ('portfolios','OSAP forecasts: 79 months',(-3,2.1),[-2,-1,0,1,2])]
    points=[]
    for ax,(domain,title,limits,ticks)in zip(axes.flat,panels):
        models=sorted({r['model']for r in rows if r['domain']==domain})
        ax.set(xlim=limits,ylim=(len(models)-.5,-.75),xticks=ticks,title=title)
        ax.spines[['left','right','top']].set_visible(False)
        ax.axvline(0,color='.8',linewidth=.5);ax.axvline(1.6448536269514722,color='.45',linewidth=.55,linestyle=':')
        labels=[]
        for y,model in enumerate(models):
            a,b=(key[domain,model,c]for c in ['complementary_middle','directional_middle'])
            x1=float(a['raw_statistic'])if a['raw_statistic']else None;x2=float(b['raw_statistic'])if b['raw_statistic']else None
            assert (x1 is None)==(x2 is None)
            labels.append(a['display_name']+(r'$^{\dagger}$'if a['original_policy_abstain']=='True'else''))
            if x1 is not None:
                ax.plot([x1,x2],[y,y],color='.67',lw=.6)
                ax.plot(x1,y,'o',color='.2',mfc='white',ms=3.25,mew=.65)
                ax.plot(x2,y,'s',color='#0072B2',ms=3.25,mew=.65)
            else:ax.text(.045,y,'undefined',transform=ax.get_yaxis_transform(),va='center',color='.4',fontstyle='italic')
            points.append({'domain':domain,'model':model,'complementary':x1,'directional':x2,'guarded':a['original_policy_abstain']=='True'})
        ax.set_yticks(range(len(models)),labels);ax.tick_params(axis='y',length=0,pad=4)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color='.2',marker='o',mfc='white',linestyle='none',label='Complementary fitting'),Line2D([],[],color='#0072B2',marker='s',linestyle='none',label='Directional fitting')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,1),ncol=2,frameon=False)
    fig.text(.53,.02,r'Raw residual statistic $T$; dotted line: nominal one-sided 5\% normal threshold.',ha='center')
    export(fig,'fig_public_directional');plt.close(fig)
    (BUILD_ROOT/'directional_coordinates.json').write_text(json.dumps(points,indent=2)+'\n')


def main():
    Figure.savefig=intercept_save
    m=module('scripts/figures/make_figures.py');m['save']=save_record;m['smooth'](BUILD_ROOT)
    m=module('scripts/figures/make_publication_figures.py');m['save_figure']=save_record
    rows,public,receipts=m['sources_and_data'](PACKAGE_ROOT,m['SOURCES'])
    old={};old['canonical']=m['canonical_plot'](rows,BUILD_ROOT);old['public_named']=m['public_plot'](public,BUILD_ROOT)
    (BUILD_ROOT/'original_plot_coordinates.json').write_text(json.dumps(old,indent=2)+'\n')
    directional()
    module('scripts/figures/make_oracle_rank_figure.py',[("DATA = ROOT/'rerun_oracle_figure'","DATA = BUILD_ROOT/'oracle_records'")])
    module('scripts/figures/make_historical_sensitivity.py',[("ROOT = PACKAGE/'rerun_historical_figure'","ROOT = BUILD_ROOT/'historical_records'")])
    sys.argv=['plot','--source',str(PACKAGE_ROOT/'results/gaussian_reference/summary.json'),'--output-dir',str(BUILD_ROOT/'gaussian_records')]
    m=module('scripts/figures/make_gaussian_reference.py');m['main']()
    sys.argv=['plot','--output',str(BUILD_ROOT/'factorial_records')]
    module('results/certificate_factorial/make_figure.py', [("Rendered fixed 7 × 2.05 inch figure and all12 predeclared focus table settings.", "Rendered the companion figure at 6.9 × 2.55 inches and all 12 predeclared focus table settings.")])
    # The plotting sources retain historical metadata; replace only fields that
    # describe this rendering. Recorded coordinates and uncertainty stay intact.
    for relative, name in [('gaussian_records/provenance.json', 'fig_gaussian_reference'),
                           ('factorial_records/fig_certificate_factorial.json', 'fig_certificate_factorial')]:
        path = BUILD_ROOT / relative
        record = json.loads(path.read_text())
        record['size_inches'] = list(SIZES[name])
        record['minimum_font_points'] = 9
        record['renderer_sha256'] = sha(Path(__file__))
        path.write_text(json.dumps(record, indent=2) + '\n')
    assert set(RECORDS)==set(SIZES)
    (BUILD_ROOT/'REBUILD_RECEIPT.json').write_text(json.dumps({'status':'RENDERED','generator_sha256':sha(Path(__file__)),'figures':RECORDS},indent=2,allow_nan=True)+'\n')

if __name__=='__main__':main()
