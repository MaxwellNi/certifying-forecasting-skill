"""Check whole-trajectory reference designs and freeze forecast selections."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'trajectory_design'))
from trajectory_kernel import shared_scores, validation_scores, full_u_scores, design_widths, mean_bound as _mean_bound, corrected_lower as _corrected_lower

def mean_bound(*args,**kwargs): return _mean_bound(*args,**kwargs)['bound']
def corrected_lower(*args,**kwargs): return _corrected_lower(*args,**kwargs)['lower']

C=np.array([[1.,0.],[-1.,0.]])
DESIGNS={'shared':np.array([[1.,0.],[-1.,0.]]),
         'separate':np.array([[0.,1.],[0.,-1.]]),
         'crossed':np.array([[0.,1.],[-1.,0.]])}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def cp(k,n):return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
                    1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def oracle(f,g):
    n=len(f);mf=(rankdata(f,axis=0,method='average')-.5)/n-.5
    mg=(rankdata(g,axis=0,method='average')-.5)/n-.5
    pf=(f[:,None,:]>f[None,:,:]).astype(float)+.5*(f[:,None,:]==f[None,:,:])-.5
    qg=(g[:,None,:]>g[None,:,:]).astype(float)+.5*(g[:,None,:]==g[None,:,:])-.5
    K=mf.T@mg/n
    same=np.einsum('ijs,ijt->st',pf,qg)/(n*n)
    return float(C.sum(1)@K@C.sum(1)),K,same-K

def run(data,output):
    output.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter()
    archive=np.load(data/'selection_forecasts.npz')
    forecasts=archive['predictions'];target=archive['target'];base=archive['baseline'];names=archive['names']
    g=np.column_stack([target,base]);maps=[np.column_stack([forecasts[:,j],base]) for j in range(8)]
    truth=[oracle(f,g) for f in maps]
    emit(output/'EXECUTION_PLAN.json',{'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
        'experiment_source_sha256':sha(Path(__file__)),
        'kernel_source_sha256':sha(ROOT.parent/'trajectory_design/trajectory_kernel.py'),
        'selection_input_sha256':sha(data/'selection_forecasts.npz'),
        'rule':'hybrid','replications':300,'evaluation_blocks':1024,'validation_triples':512,
        'alpha':.05,'delta':.025,'fresh_confirmation_losses_read':False,
        'bounds_condition_on':'Fixed2013 archive and models fitted on2010-2012; fresh independent index draws only.',
        'zero_overlap_budget':'Same prespecified1024/512 allocation in design diagnostics; full U can use every triple. No optimality claim for this allocation.'})
    oracle_rows=[]
    for k,name in enumerate(names):
        theta,K,Lambda=truth[k]
        for design,D in DESIGNS.items():
            omega=C@D.T
            oracle_rows.append({'candidate':str(name),'design':design,'target':theta,
                'reference_interaction':float(np.sum(omega*Lambda)),
                'expected_shared_score':theta+float(np.sum(omega*Lambda)),
                'overlap_matrix':json.dumps(omega.tolist()),'overlap_matrix_sum':float(omega.sum()),
                'population_covariance':json.dumps(K.tolist()),'reference_covariance':json.dumps(Lambda.tolist())})
    pd.DataFrame(oracle_rows).to_csv(output/'exact_archive_targets.csv',index=False)
    records=[]
    for rep in range(300):
        rng=np.random.default_rng(2026091400+rep)
        ev=rng.integers(len(target),size=(1024,3));va=rng.integers(len(target),size=(512,3))
        allix=np.concatenate([ev,va])
        for k,name in enumerate(names):
            f=maps[k];theta=truth[k][0]
            uv=full_u_scores(f[allix],g[allix],C,C)
            wu=design_widths(C,C)['full_u_width']
            lu=mean_bound(uv,wu,alpha=.05,side='lower',rule='hybrid')
            records.append({'replication':rep,'candidate':str(name),'design':'full_u',
                'mean_score':float(uv.mean()),'mean_interaction':0.,'lower':float(lu),
                'target':theta,'noncoverage':bool(lu>theta),'positive':bool(lu>0),
                'raw_lower':float(lu),'raw_noncoverage':bool(lu>theta)})
            for design,D in DESIGNS.items():
                sv=shared_scores(f[ev],g[ev],C,D);rv=validation_scores(f[va],g[va],C,D)
                lower=corrected_lower(sv,rv,C,D,alpha=.05,delta=.025,rule='hybrid')
                raw=mean_bound(sv,design_widths(C,D)['score_width'],alpha=.05,side='lower',rule='hybrid')
                records.append({'replication':rep,'candidate':str(name),'design':design,
                    'mean_score':float(sv.mean()),'mean_interaction':float(rv.mean()),
                    'lower':float(lower),'target':theta,'noncoverage':bool(lower>theta),
                    'positive':bool(lower>0),'raw_lower':float(raw),'raw_noncoverage':bool(raw>theta)})
    df=pd.DataFrame(records);df.to_csv(output/'coverage_replications.csv',index=False)
    summary=[]
    for (candidate,design),a in df.groupby(['candidate','design'],sort=False):
        row={'candidate':candidate,'design':design,'repetitions':len(a),'target':float(a.target.iloc[0])}
        for key in ['noncoverage','positive','raw_noncoverage']:
            count=int(a[key].sum());lo,hi=cp(count,len(a))
            row.update({key:count,key+'_ci_low':lo,key+'_ci_high':hi})
        row.update(mean_score=float(a.mean_score.mean()),mean_interaction=float(a.mean_interaction.mean()),
                   mean_lower=float(a.lower.mean()))
        summary.append(row)
    pd.DataFrame(summary).to_csv(output/'coverage_summary.csv',index=False)
    # Freeze gates and selected model identities before loading the confirmation archive.
    rng=np.random.default_rng(2026091499)
    ev=rng.integers(len(target),size=(16384,3));va=rng.integers(len(target),size=(8192,3));allix=np.concatenate([ev,va])
    np.savez_compressed(output/'selection_indices.npz',evaluation=ev,validation=va)
    mse=np.mean((forecasts-target[:,None])**2,axis=0);gates=[]
    alpha=.05/8;delta=alpha/2
    for k,name in enumerate(names):
        f=maps[k];sv=shared_scores(f[ev],g[ev],C,C);rv=validation_scores(f[va],g[va],C,C)
        lo=corrected_lower(sv,rv,C,C,alpha=alpha,delta=delta,rule='hybrid')
        uv=full_u_scores(f[allix],g[allix],C,C)
        ul=mean_bound(uv,design_widths(C,C)['full_u_width'],alpha=alpha,side='lower',rule='hybrid')
        gates.append({'candidate':str(name),'selection_mse':float(mse[k]),'target':truth[k][0],
            'corrected_lower':float(lo),'full_u_lower':float(ul),
            'corrected_pass':bool(lo>0),'full_u_pass':bool(ul>0)})
    gd=pd.DataFrame(gates);gd.to_csv(output/'selection_gates.csv',index=False)
    choices={'persistence':0,'ungated':int(np.argmin(mse))}
    for rule in ['corrected','full_u']:
        allowed=np.flatnonzero(gd[rule+'_pass'].to_numpy())
        choices[rule]=int(allowed[np.argmin(mse[allowed])]) if len(allowed) else 0
    emit(output/'FROZEN_SELECTIONS.json',{'candidates':{k:str(names[v]) for k,v in choices.items()},
        'candidate_indices':choices,'selection_gates_sha256':sha(output/'selection_gates.csv'),
        'frozen_before_confirmation_losses':True,'freeze_unix':time.time(),
        'family_size':8,'family_fwer':.05,'trajectory_draws_per_method_per_candidate':len(allix)*3,
        'distinct_selection_archive_records':int(len(np.unique(allix))),
        'input_coordinate_reads_per_draw':17,'distinct_extra_outcome_acquisitions':0,
        'primary_loss':'Unclipped raw-unit MSE; all forecasts clipped below zero before selection.',
        'rule_scope':'Positive fitted-rank increment in conditional archive, not a future-MSE guarantee.'})
    evaluate_confirmation(data,output,choices,names)
    emit(output/'EXECUTION.json',{'elapsed_seconds':time.perf_counter()-tick,
        'coverage_rows':len(records),'coverage_cells':len(summary),'source_array_records':len(target),
        'status':'Executed all declared cells and endpoints; substantive results recorded without a pass-by-performance filter.'})

def evaluate_confirmation(data,output,choices,names):
    assert (output/'FROZEN_SELECTIONS.json').exists()
    x=np.load(data/'confirmation_forecasts.npz');y=x['target'];pred=x['predictions']
    loss=(pred-y[:,None])**2;mae=np.abs(pred-y[:,None]);mse=loss.mean(0)
    pd.DataFrame({'candidate':names,'mse':mse,'mae':mae.mean(0),
        'relative_mse_gain_vs_persistence':1-mse/mse[0]}).to_csv(output/'confirmation_candidates.csv',index=False)
    times=pd.to_datetime(x['timestamps']);weeks=times.to_period('W-SUN').astype(str)
    weekly=pd.DataFrame(loss,columns=names).assign(week=weeks).groupby('week',sort=True).mean()
    weekly.to_csv(output/'confirmation_weekly_losses.csv')
    rng=np.random.default_rng(2026091455);n=len(weekly);b=4
    starts=rng.integers(n-b+1,size=(2000,int(np.ceil(n/b))))
    indices=(starts[:,:,None]+np.arange(b)).reshape(2000,-1)[:,:n]
    rows=[]
    for rule,k in choices.items():
        for comparator,j in [('persistence',0),('ungated',choices['ungated'])]:
            differences=weekly.iloc[:,j].to_numpy()-weekly.iloc[:,k].to_numpy()
            boot=differences[indices].mean(1)
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append({'rule':rule,'candidate':str(names[k]),'comparator':comparator,
                'confirmation_records':len(y),'confirmation_weeks':n,'raw_mse':float(mse[k]),
                'raw_mse_gain':float(mse[j]-mse[k]),'relative_mse_gain':float(1-mse[k]/mse[j]),
                'equal_week_mean_gain':float(differences.mean()),'bootstrap_low':float(lo),'bootstrap_high':float(hi),
                'identical_predictions':bool(np.array_equal(pred[:,j],pred[:,k])),
                'interval_scope':'Descriptive four-week moving-block bootstrap; no finite physical-time-series guarantee.'})
    pd.DataFrame(rows).to_csv(output/'confirmation_rules.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.data,args.output)
