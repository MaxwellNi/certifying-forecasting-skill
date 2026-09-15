"""Preexecution-sealed, same-raw-draw binary synthetic method assessment.

No field data are read. Use --output with a new directory to repeat the complete experiment. See BINARY_BENCHMARK.md.
"""

import argparse
import csv
import datetime
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from binary_counts import BLOCK_VALUES_EXACT,full_u_binary_counts,full_u_binary_bound


HERE=Path(__file__).resolve().parent
OUT=HERE
BASE=HERE.parent
sys.path.insert(0,str(BASE/"betting"))
from finite_support_betting import FiniteSupportBetting

SEED=2026091331
REPETITIONS=500
BUDGETS=(512,2048,8192,32768)
LAWS=(
    ("balanced_independent",("1/4","1/4","1/4","1/4"),"0"),
    ("balanced_anticorrelated",("0","1/2","1/2","0"),"-1/16"),
    ("balanced_perfectly_correlated",("1/2","0","0","1/2"),"1/16"),
    ("rare_perfectly_correlated",("99/100","0","0","1/100"),"99/40000"),
)
METHODS=(
    "full_u_hoeffding","full_u_eb","full_u_joint",
    "binary_full_u_eb","binary_full_u_joint",
    "stratified_hoeffding","stratified_hybrid",
    "binary_stratified_hoeffding","binary_stratified_hybrid",
    "stratified_profiled_betting64","binary_stratified_betting64",
    "direct_four_row_betting64","binary_all_raw_pair_betting64",
)
ALPHA=.05
BET_FRACTIONS=tuple(float(x) for x in np.geomspace(1e-4,.99,64))
PAIR_SUPPORT=(F(-1,8),F(0),F(1,8))
BLOCK_SUPPORT=tuple(sorted(set(BLOCK_VALUES_EXACT)))
BLOCK_CATEGORY=np.asarray([BLOCK_SUPPORT.index(x) for x in BLOCK_VALUES_EXACT],dtype=np.int8)
CODE_PATHS=(Path(__file__),HERE/"binary_counts.py",HERE/"binary_checks.py",
            HERE/"BINARY_BENCHMARK.md",BASE/"betting/finite_support_betting.py",
            BASE/"betting/profiled_betting.py",
            HERE/"FULL_U_VARIANCE_THEORY.md")


def configuration():
    return {"seed":SEED,"repetitions":REPETITIONS,"budgets":list(BUDGETS),
            "laws":[{"name":name,"probabilities":list(p),"theta":theta} for name,p,theta in LAWS],
            "methods":list(METHODS),"raw_alpha":ALPHA,
            "betting_fractions":list(BET_FRACTIONS),"betting_precision":40,
            "replication_stream":"SeedSequence([seed,law_index,budget_index,repetition])",
            "paired_monte_carlo_interval":"pointwise 95% two-sided empirical Bernstein, range2, log80",
            "no_field_data":True,"no_posthoc_method_selection_claim":True}


def source_hashes():
    return {str(path.relative_to(BASE)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in CODE_PATHS}


def seal():
    path=OUT/"binary_benchmark_seal.json"
    if path.exists():
        raise FileExistsError("preserve the existing preexecution seal")
    record={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status":"SEALED_BEFORE_BENCHMARK_EXECUTION",
            "configuration":configuration(),"sha256":source_hashes()}
    path.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"seal":str(path),"created_utc":record["created_utc"],
                      "methods":len(METHODS),"total_repetitions":len(LAWS)*len(BUDGETS)*REPETITIONS}))


def _pair_counts(states):
    rows=states[:2*(len(states)//2)].reshape(-1,2)
    dv=(rows[:,0]//2).astype(int)-(rows[:,1]//2).astype(int)
    dw=(rows[:,0]%2).astype(int)-(rows[:,1]%2).astype(int)
    return np.bincount(dv*dw+1,minlength=3)


def _block_counts(states):
    rows=states.reshape(-1,4)
    code=rows[:,0]*64+rows[:,1]*16+rows[:,2]*4+rows[:,3]
    return np.bincount(BLOCK_CATEGORY[code],minlength=len(BLOCK_SUPPORT))


def _stratified_interval(counts,width=1.0,hybrid=False):
    q=int(sum(counts))
    center=(int(counts[2])-int(counts[0]))/(8*q)
    variance=max(0.,((int(counts[0])+int(counts[2]))/64-q*center*center)/(q-1))
    alpha=ALPHA/2 if hybrid else ALPHA
    h=width*math.sqrt(-math.log(alpha)/(2*q))
    x=math.log(2)-math.log(alpha)
    eb=math.sqrt(2*variance*x/q)+7*width*x/(3*(q-1))
    radius=min(h,eb) if hybrid else h
    return center,max(-.25,min(.25,center-radius))


def run():
    seal_path=OUT/"binary_benchmark_seal.json"
    record=json.loads(seal_path.read_text())
    if record["configuration"]!=configuration() or record["sha256"]!=source_hashes():
        raise RuntimeError("configuration or source differs from the preexecution seal")
    if (OUT/"binary_results.json").exists() or (OUT/"binary_replicates.npz").exists():
        raise FileExistsError("preserve the completed benchmark outputs")
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    shape=(len(LAWS),len(BUDGETS),REPETITIONS,len(METHODS))
    reject=np.zeros(shape,dtype=np.uint8)
    lower=np.full(shape,np.nan)
    estimate=np.full(shape,np.nan)
    pvalues=np.full(shape,np.nan)
    seconds=np.zeros(shape)
    wall_start=time.perf_counter()
    pair_generic=FiniteSupportBetting(PAIR_SUPPORT,F(-1,2),fractions=BET_FRACTIONS,precision=40)
    pair_tight=FiniteSupportBetting(PAIR_SUPPORT,F(-1,8),fractions=BET_FRACTIONS,precision=40)
    blocks=FiniteSupportBetting(BLOCK_SUPPORT,F(-1,12),fractions=BET_FRACTIONS,precision=40)
    betting_precomputation_seconds={"generic_pair":pair_generic.precompute_seconds,
                                    "binary_pair":pair_tight.precompute_seconds,
                                    "four_row_block":blocks.precompute_seconds}
    generation_seconds=0.0
    for law_index,(law,probability_strings,theta_string) in enumerate(LAWS):
        probabilities=np.asarray([float(F(x)) for x in probability_strings])
        for budget_index,n in enumerate(BUDGETS):
            for repetition in range(REPETITIONS):
                generation_start=time.perf_counter()
                rng=np.random.default_rng(np.random.SeedSequence([SEED,law_index,budget_index,repetition]))
                states=rng.choice(4,size=n,p=probabilities).astype(np.int16)
                generation_seconds+=time.perf_counter()-generation_start
                position=(law_index,budget_index,repetition)
                preparation_start=time.perf_counter()
                summary=full_u_binary_counts(np.bincount(states,minlength=4))
                full_preparation_seconds=time.perf_counter()-preparation_start
                for method_index,(rule,binary) in enumerate((("hoeffding",False),("eb",False),("joint",False),
                                                             ("eb",True),("joint",True))):
                    method_start=time.perf_counter()
                    output=full_u_binary_bound(summary,alpha=ALPHA,rule=rule,binary_interactions=binary)
                    estimate[position+(method_index,)]=output["estimate"]
                    lower[position+(method_index,)]=output["lower"]
                    reject[position+(method_index,)]=output["lower"]>0
                    seconds[position+(method_index,)]=full_preparation_seconds+time.perf_counter()-method_start
                preparation_start=time.perf_counter()
                strat_counts=_pair_counts(states[:n//2])
                strat_preparation_seconds=time.perf_counter()-preparation_start
                for method_index,width,hybrid in ((5,1.,False),(6,1.,True),(7,.25,False),(8,.25,True)):
                    method_start=time.perf_counter()
                    center,bound=_stratified_interval(strat_counts,width=width,hybrid=hybrid)
                    estimate[position+(method_index,)]=center
                    lower[position+(method_index,)]=bound
                    reject[position+(method_index,)]=bound>0
                    seconds[position+(method_index,)]=strat_preparation_seconds+time.perf_counter()-method_start
                for method_index,bet in ((9,pair_generic),(10,pair_tight)):
                    method_start=time.perf_counter()
                    p=bet.pvalue(strat_counts)
                    estimate[position+(method_index,)]=(int(strat_counts[2])-int(strat_counts[0]))/(8*int(sum(strat_counts)))
                    pvalues[position+(method_index,)]=p
                    reject[position+(method_index,)]=p<=ALPHA
                    seconds[position+(method_index,)]=strat_preparation_seconds+time.perf_counter()-method_start
                preparation_start=time.perf_counter()
                block_counts=_block_counts(states)
                block_preparation_seconds=time.perf_counter()-preparation_start
                method_start=time.perf_counter()
                p=blocks.pvalue(block_counts)
                method_index=11
                estimate[position+(method_index,)]=sum(float(x)*int(k) for x,k in zip(BLOCK_SUPPORT,block_counts))/int(sum(block_counts))
                pvalues[position+(method_index,)]=p
                reject[position+(method_index,)]=p<=ALPHA
                seconds[position+(method_index,)]=block_preparation_seconds+time.perf_counter()-method_start
                preparation_start=time.perf_counter()
                raw_pair_counts=_pair_counts(states)
                pair_preparation_seconds=time.perf_counter()-preparation_start
                method_start=time.perf_counter()
                p=pair_tight.pvalue(raw_pair_counts)
                method_index=12
                estimate[position+(method_index,)]=(int(raw_pair_counts[2])-int(raw_pair_counts[0]))/(8*int(sum(raw_pair_counts)))
                pvalues[position+(method_index,)]=p
                reject[position+(method_index,)]=p<=ALPHA
                seconds[position+(method_index,)]=pair_preparation_seconds+time.perf_counter()-method_start
            print(json.dumps({"completed_law":law,"raw_budget":n,"repetitions":REPETITIONS,
                              "elapsed_seconds":time.perf_counter()-wall_start}),flush=True)
    np.savez_compressed(OUT/"binary_replicates.npz",reject=reject,lower=lower,estimate=estimate,
                        pvalue=pvalues,method_seconds=seconds)
    summaries=[]
    paired=[]
    for li,(law,p,theta_string) in enumerate(LAWS):
        theta=float(F(theta_string))
        for bi,n in enumerate(BUDGETS):
            for mi,method in enumerate(METHODS):
                sample_lower=lower[li,bi,:,mi]
                present=np.isfinite(sample_lower)
                summaries.append({"law":law,"raw_budget":n,"theta":theta,"theta_exact":theta_string,
                                  "method":method,"raw_alpha":ALPHA,"repetitions":REPETITIONS,
                                  "rejections":int(reject[li,bi,:,mi].sum()),
                                  "rejection_rate":float(reject[li,bi,:,mi].mean()),
                                  "lower_bound_failures":int((sample_lower[present]>theta).sum()) if present.any() else None,
                                  "mean_estimate":float(estimate[li,bi,:,mi].mean()),
                                  "mean_lower_bound":float(sample_lower[present].mean()) if present.any() else None,
                                  "mean_method_seconds":float(seconds[li,bi,:,mi].mean()),
                                  "charged_raw_rows_per_repetition":n})
            for a,b in itertools.combinations(range(len(METHODS)),2):
                ra=reject[li,bi,:,a].astype(int)
                rb=reject[li,bi,:,b].astype(int)
                difference=ra-rb
                mean=float(difference.mean())
                variance=float(difference.var(ddof=1))
                radius=math.sqrt(2*variance*math.log(80)/REPETITIONS)+14*math.log(80)/(3*(REPETITIONS-1))
                paired.append({"law":law,"raw_budget":n,"method_a":METHODS[a],"method_b":METHODS[b],
                               "paired_rejection_rate_difference_a_minus_b":mean,
                               "pointwise_95_lower":max(-1.,mean-radius),"pointwise_95_upper":min(1.,mean+radius),
                               "a_only_rejections":int(((ra==1)&(rb==0)).sum()),
                               "b_only_rejections":int(((ra==0)&(rb==1)).sum()),
                               "paired_repetitions":REPETITIONS})
    for filename,rows in (("binary_summary.csv",summaries),("binary_paired_intervals.csv",paired)):
        with (OUT/filename).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result={"status":"COMPLETED_SYNTHETIC_ASSESSMENT","started_utc":started,
            "finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds":time.perf_counter()-wall_start,"raw_generation_seconds":generation_seconds,
            "betting_precomputation_seconds":betting_precomputation_seconds,
            "configuration":configuration(),"preexecution_seal_sha256":hashlib.sha256(seal_path.read_bytes()).hexdigest(),
            "executed_source_sha256":source_hashes(),"summary":summaries,
            "replicate_array_axes":["law","budget","repetition","method"],
            "artifacts":["binary_replicates.npz","binary_summary.csv","binary_paired_intervals.csv"],
            "limits":"Synthetic supplied binary-support laws only. Raw alpha=.05 per method; paired Monte Carlo intervals pointwise. No new field dataset, universal dominance or novelty claim."}
    (OUT/"binary_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"],
                      "summary_rows":len(summaries),"paired_interval_rows":len(paired)}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    OUT=args.output.resolve()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Output must be new and empty")
    OUT.mkdir(parents=True,exist_ok=True)
    seal()
    run()
