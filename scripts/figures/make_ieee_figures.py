"""Rebuild Figures 2--4 with the actual paper's LaTeX fonts, vector only.

No analysis, candidate selection, point jitter, interval change or new data.
Matplotlib creates PGF paths/text; an IEEEtran wrapper uses the same font
packages as main.tex: amsmath, amssymb, amsfonts, default Times text (no newtxmath).
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from statistics import NormalDist

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT
PREAMBLE=r"\usepackage{amsmath,amssymb,amsfonts}\renewcommand{\rmdefault}{ptm}"
BLUE="#0072B2"
ORANGE="#B35A00"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    plt.rcParams.update({"font.family":"serif","font.size":9,
        "axes.titlesize":9,"axes.labelsize":9,"xtick.labelsize":9,
        "ytick.labelsize":9,"legend.fontsize":9,"text.usetex":True,
        "pgf.texsystem":"pdflatex","pgf.rcfonts":False,"pgf.preamble":PREAMBLE,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6,
        "xtick.major.size":3,"ytick.major.size":3,
        "xtick.major.width":.55,"ytick.major.width":.55,
        "savefig.transparent":False})


def export(fig,stem):
    fig.savefig(stem.with_suffix(".pgf"))
    w,h=fig.get_size_inches()
    # IEEEtran supplies the exact template font context. All text inside the
    # PGF explicitly requests its declared 9pt size; math retains CM families.
    tex=(r"\documentclass[conference]{IEEEtran}"+"\n"+PREAMBLE+"\n"+
         r"\usepackage{pgf}"+"\n"+
         rf"\usepackage[paperwidth={w:.8f}in,paperheight={h:.8f}in,margin=0pt]{{geometry}}"+"\n"+
         r"\pagestyle{empty}\setlength{\parindent}{0pt}"+"\n"+
         r"\pdfinfoomitdate=1\pdftrailerid{}\pdfsuppressptexinfo=15"+"\n"+
         r"\begin{document}\onecolumn\thispagestyle{empty}"+"\n"+
         rf"\noindent\input{{{stem.name}.pgf}}"+"\n"+r"\end{document}"+"\n")
    stem.with_suffix(".tex").write_text(tex)
    proc=subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error",stem.name+".tex"],
                        cwd=stem.parent,text=True,capture_output=True,env=os.environ)
    stem.with_suffix(".build.txt").write_text(proc.stdout+proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"LaTeX failed for {stem}: see .build.txt")
    plt.close(fig)


def fig2(output):
    peer=PACKAGE/"results/design_validation/peer_sampling.csv"
    trained=PACKAGE/"results/learned_references/summary.csv"
    peers=pd.read_csv(peer,float_precision="round_trip")
    learned=pd.read_csv(trained,float_precision="round_trip")
    fig,axes=plt.subplots(1,4,figsize=(7,2.04),sharey=True)
    fig.subplots_adjust(left=.071,right=.991,bottom=(.245*2.18-.10)/2.04,top=(.685*2.18-.10)/2.04,wspace=.23)
    panels=[("resampled_peers","(a) Resampled peers\nExact means"),
            ("learned","(b) Resampled peers\nFitted means"),
            ("fixed_peers_aligned_effects","(c) Fixed peers\nAligned effects"),
            ("fixed_peers_opposed_effects","(d) Fixed peers\nOpposed effects")]
    coordinates=[]
    for ax,(regime,title) in zip(axes,panels):
        for short,long,label,color,marker,fill in [
            ("shared","shared_references","Shared references","#222222","o","white"),
            ("distinct","distinct_references","Distinct references",BLUE,"s",BLUE)]:
            if regime=="learned":
                data=learned[(learned.entities==64)&(learned.groups==400)&
                             (learned.training_rows>0)&(learned.method==short)].sort_values("training_rows")
                x=data.training_rows.to_numpy()
                assert list(x)==[32,512,8192]
            else:
                data=peers[(peers.peer_regime==regime)&(peers.method==long)].sort_values("periods")
                x=data.periods.to_numpy()
                assert list(x)==[25,100,400]
            y=data.rejections.to_numpy()/data.replications.to_numpy()
            # Use the recorded Wilson endpoints; independently verify formula.
            lo=data.wilson_low.to_numpy();hi=data.wilson_high.to_numpy()
            z=NormalDist().inv_cdf(.975);n=data.replications.to_numpy();den=1+z*z/n
            center=(y+z*z/(2*n))/den
            half=z*np.sqrt(y*(1-y)/n+z*z/(4*n*n))/den
            np.testing.assert_allclose(lo,np.maximum(0,center-half),rtol=0,atol=1e-12)
            np.testing.assert_allclose(hi,np.minimum(1,center+half),rtol=0,atol=1e-12)
            ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),
                        marker=marker,markerfacecolor=fill,markeredgecolor=color,
                        markersize=3.9,color=color,linewidth=.9,elinewidth=.7,
                        capsize=2,capthick=.7,label=label)
            coordinates.extend({"panel":regime,"method":short,"x":int(xx),"rate":float(yy),
                                "rejections":int(rr),"replications":int(nn),
                                "wilson_low":float(ll),"wilson_high":float(hh)}
                               for xx,yy,rr,nn,ll,hh in zip(x,y,data.rejections,n,lo,hi))
        if regime=="learned":
            ax.set_xscale("log",base=16);ax.set_xlim(20,13100)
            ax.set_xticks([32,512,8192],["32","512","8,192"])
            ax.set_xlabel(r"Training observations, $m$",labelpad=3)
        else:
            ax.set_xscale("log",base=4);ax.set_xlim(17,588)
            ax.set_xticks([25,100,400],["25","100","400"])
            ax.set_xlabel(r"Independent groups, $M$",labelpad=3)
        ax.minorticks_off();ax.set_ylim(-.025,1.025)
        ax.set_title(title,pad=7,linespacing=1.05)
        ax.axhline(.05,color=".4",linestyle=":",linewidth=.8,zorder=0)
        ax.grid(axis="y",color=".90",linewidth=.5);ax.set_axisbelow(True)
    axes[0].set_ylabel("Positive rejection rate",labelpad=3)
    axes[0].set_yticks([0,.25,.5,.75,1],["0","0.25","0.5","0.75","1"])
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,1.01),
               ncol=2,frameon=False,handlelength=2,columnspacing=2.5)
    stem=output/"fig_reference_training"
    export(fig,stem)
    pd.DataFrame(coordinates).to_csv(output/"figure2_coordinates.csv",index=False,float_format="%.17g")
    return {"figure":2,"stem":stem.name,"size_inches":[7,2.04],"points":coordinates,
            "inputs":{str(p.relative_to(PACKAGE)):sha(p) for p in [peer,trained]},
            "semantics":"All four reference designs, recorded rates and Wilson intervals unchanged; no point offsets.",
            "style":"Paper LaTeX fonts; explicit leading-zero decimal ticks; same logarithmic x axes."}


def fig3(output,height):
    sa=PACKAGE/"results/certificate_factorial/summary.csv"
    sb=PACKAGE/"results/aggregate_bias/simulation/summary.csv"
    factorial=pd.read_csv(sa,float_precision="round_trip")
    left=factorial[(factorial.design=="eight_rare")&(factorial.signal==.25)&
                   (factorial.fit=="estimated")&(factorial.validation_pairs==8192)].copy()
    budgets=[26176,30976,50176,126976]
    assert len(left)==16 and sorted(left.total_observations.unique())==budgets
    summary=pd.read_csv(sb,float_precision="round_trip")
    keys=["categories","validation_pairs","mass","fit_error"]
    points=summary.pivot(index=keys,columns="method",values="median_slack").reset_index()
    points.columns.name=None
    assert len(points)==72 and int((points["aggregate"]<points.rectangle).sum())==62
    assert not points.duplicated(["aggregate","rectangle"]).any()
    fig,axes=plt.subplots(1,2,figsize=(7,height),gridspec_kw={"width_ratios":[1.05,1]})
    fig.subplots_adjust(left=.116,right=.986,bottom=(.235*2.2-.07)/height,top=(.735*2.2-.07)/height,wspace=.47)
    styles=[("absolute_range","Absolute + range",ORANGE,"x","none",-.24),
            ("signed_range","Signed + range",BLUE,"D","white",-.08),
            ("absolute_variance","Absolute + variance",ORANGE,"s","white",.08),
            ("signed_variance","Signed + variance",BLUE,"o",BLUE,.24)]
    ax=axes[0];handles=[]
    for y in [.5,1.5,2.5]:ax.axhline(y,color=".89",linewidth=.6,zorder=0)
    for method,label,color,marker,face,offset in styles:
        rows=left[left.method==method].sort_values("total_observations")
        y=np.arange(4)+offset
        ax.errorbar(rows.power,y,xerr=np.vstack([rows.power-rows.power_ci_lower,
                                                rows.power_ci_upper-rows.power]),
                    fmt=marker,color=color,markerfacecolor=face,markersize=3.3,
                    linewidth=.8,capsize=1.2,capthick=.7,linestyle="none",clip_on=False)
        handles.append(Line2D([],[],marker=marker,color=color,markerfacecolor=face,
                              markersize=4,linestyle="none",label=label))
    ax.set_xlim(-.035,1.035);ax.set_ylim(3.48,-.48)
    ax.set_yticks(range(4),[f"{n:,}" for n in budgets]);ax.set_xticks([0,.5,1],["0","0.5","1"])
    ax.set_ylabel("Total observations",labelpad=3);ax.set_xlabel("Rejection rate",labelpad=3)
    ax.set_title(r"(a) Eight categories, $\vartheta\approx0.0137$",pad=7)
    ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(.47,1.16),ncol=2,
              frameon=False,columnspacing=.75,handletextpad=.35,handlelength=.8,
              labelspacing=.32,borderpad=0)
    ax=axes[1]
    # Transparent interiors expose coincident/nearby outlines. Different sizes
    # and shapes do not alter either numeric coordinate; there is no jitter.
    categories=[(2,"o",BLUE,5.0),(8,"s",ORANGE,4.5),
                (32,"D","#008571",4.1),(128,"^","#7B3294",4.1)]
    for category,marker,color,size in categories:
        rows=points[points.categories==category]
        ax.scatter(rows.rectangle,rows["aggregate"],s=size**2,marker=marker,
                   facecolors="none",edgecolors=color,linewidths=.75,zorder=3,clip_on=False)
    extent=[.0005,.5]
    ax.plot(extent,extent,color=".45",linewidth=.8,linestyle=(0,(3,2)),zorder=1)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(extent);ax.set_ylim(extent)
    ax.set_xticks([.001,.01,.1],["0.001","0.01","0.1"])
    ax.set_yticks([.001,.01,.1],["0.001","0.01","0.1"])
    ax.minorticks_off()
    ax.set_xlabel("Rectangle median slack",labelpad=3);ax.set_ylabel("Aggregate median slack",labelpad=3)
    ax.set_title("(b) Absolute validation slack",pad=7)
    ax.text(.035,.95,"Equal slack",transform=ax.transAxes,ha="left",va="top",color=".3")
    ax.text(.97,.045,"Below line:\naggregate tighter",transform=ax.transAxes,
            ha="right",va="bottom",color=".3",linespacing=1.05)
    ax.legend(handles=[Line2D([],[],marker=m,color=c,markerfacecolor="none",
                              linestyle="none",markersize=s,label=str(k)) for k,m,c,s in categories],
              title="Number of categories",title_fontsize=9,loc="lower center",
              bbox_to_anchor=(.5,1.16),ncol=4,frameon=False,handlelength=.8,
              handletextpad=.30,columnspacing=.75,labelspacing=.32,borderpad=0)
    for ax in axes:
        ax.grid(axis="x" if ax is axes[0] else "both",color=".92",linewidth=.45)
        ax.set_axisbelow(True)
    stem=output/"fig_aggregate_validation"
    export(fig,stem)
    left.to_csv(output/"figure3_power_coordinates.csv",index=False,float_format="%.17g")
    points.to_csv(output/"figure3_slack_coordinates.csv",index=False,float_format="%.17g")
    return {"figure":3,"stem":stem.name,"size_inches":[7,height],
            "inputs":{str(p.relative_to(PACKAGE)):sha(p) for p in [sa,sb]},
            "counts":{"ablation_points":16,"validation_settings":72,"aggregate_tighter":62,"rectangle_tighter":10},
            "semantics":"Every recorded power/Clopper-Pearson interval and absolute slack coordinate is preserved. Panel a retains existing categorical method offsets [-.24,-.08,.08,.24]; panel b has no coordinate offsets.",
            "style":"Paper fonts; four transparent color/shape category markers expose nearby outlines; legend encodes categories only; no jitter/size-based quantitative claim."}


def fig4(output):
    path=PACKAGE/"results/focused_directional/public_directional_models.csv"
    rows=list(csv.DictReader(path.open(newline="")))
    assert len(rows)==82
    keyed={(r["domain"],r["model"],r["configuration"]):r for r in rows}
    configs=("complementary_middle","directional_middle")
    fig,axes=plt.subplots(1,2,figsize=(7,2.52))
    fig.subplots_adjust(left=.235,right=.991,top=(.775*2.65-.08)/2.52,bottom=(.185*2.65-.08)/2.52,wspace=1.5)
    panels=[("retail","(a) M5 retail: 17 weeks",(-21.,15.),[-20,-10,0,10]),
            ("ratings","(b) MovieLens: 168 user clusters\nCluster-order sensitivity",(-.5,14.),[0,5,10])]
    coordinates=[]
    for ax,(domain,title,limits,ticks) in zip(axes,panels):
        models=sorted({r["model"] for r in rows if r["domain"]==domain})
        assert len(models)==10
        ax.set_xlim(limits)
        # Use the template's CM minus glyph; digits stay in the text family.
        # A TS1 textminus falls back to a bitmap font in this installed TeX.
        ax.set_xticks(ticks,[str(x) if x>=0 else r"$-$"+str(abs(x)) for x in ticks])
        ax.set_ylim(len(models)-.5,-.75);ax.spines["left"].set_visible(False)
        ax.axvline(0,color=".80",linewidth=.5,zorder=0)
        ax.axvline(NormalDist().inv_cdf(.95),color=".45",linewidth=.6,linestyle=":",zorder=0)
        labels=[]
        for y,model in enumerate(models):
            a,b=(keyed[(domain,model,config)] for config in configs)
            x1=float(a["raw_statistic"]) if a["raw_statistic"] else None
            x2=float(b["raw_statistic"]) if b["raw_statistic"] else None
            assert (x1 is None)==(x2 is None)
            guarded=a["original_policy_abstain"]=="True"
            labels.append(a["display_name"]+(r"$^{\dagger}$" if guarded else ""))
            if x1 is not None:
                ax.plot([x1,x2],[y,y],color=".65",linewidth=.6,zorder=1)
                for x,color,marker,fill in [(x1,".20","o","white"),(x2,BLUE,"s",BLUE)]:
                    ax.plot(x,y,marker=marker,markersize=3.5,color=color,
                            markerfacecolor=fill,markeredgewidth=.65,zorder=2)
            else:
                ax.text(.04,y,"undefined",transform=ax.get_yaxis_transform(),va="center",
                        color=".4",fontstyle="italic")
            coordinates.append({"domain":domain,"model":model,"complementary":x1,
                                "directional":x2,"guarded":guarded,"plot_y":y})
        ax.set_yticks(range(len(models)),labels);ax.tick_params(axis="y",length=0,pad=4)
        ax.set_title(title,pad=6,linespacing=1.05)
        if domain=="ratings":
            # The fixed labels consume the inter-panel gap. Center this long
            # title slightly left within its panel so its 9pt text stays on-page.
            ax.title.set_position((.32,1.))
    handles=[Line2D([],[],color=".20",marker="o",markerfacecolor="white",linestyle="none",
                    markersize=4,label="Complementary fitting"),
             Line2D([],[],color=BLUE,marker="s",linestyle="none",markersize=4,label="Directional fitting")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,1.012),ncol=2,
               frameon=False,handletextpad=.4,columnspacing=1.8)
    fig.text(.53,.035,r"Raw residual statistic $T$; dotted line: nominal one-sided 5\% normal threshold.",
             ha="center",fontsize=9)
    stem=output/"fig_public_directional_focus"
    export(fig,stem)
    pd.DataFrame(coordinates).to_csv(output/"figure4_coordinates.csv",index=False,float_format="%.17g")
    assert len(coordinates)==20 and sum(r["complementary"] is None for r in coordinates)==2
    return {"figure":4,"stem":stem.name,"size_inches":[7,2.52],"points":coordinates,
            "inputs":{str(path.relative_to(PACKAGE)):sha(path)},
            "counts":{"models":20,"finite_pairs":18,"undefined_pairs":2},
            "semantics":"All plotted paired statistics, fixed model order, guards and undefined rows unchanged. MovieLens is explicitly cluster-order sensitivity, not temporal causality; connectors remain raw comparisons, not confidence intervals.",
            "style":"Paper LaTeX fonts including CM dagger/T and IEEEtran default Times italic undefined labels."}


def main():
    global PACKAGE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"rebuilt")
    parser.add_argument("--package-root",type=Path,default=PACKAGE,
                        help="Public package containing the recorded results directory")
    parser.add_argument("--figure3-height",type=float,default=2.07)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    PACKAGE=args.package_root.resolve()
    setup()
    records=[fig2(args.output),fig3(args.output,args.figure3_height),fig4(args.output)]
    receipt={"font_setup":PREAMBLE,"text_size_points":9,"vector_only":True,
             "generator_sha256":sha(Path(__file__)),"figures":records}
    (args.output/"rebuild_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"generated":[x["stem"] for x in records],"output":str(args.output)}))


if __name__=="__main__":main()
