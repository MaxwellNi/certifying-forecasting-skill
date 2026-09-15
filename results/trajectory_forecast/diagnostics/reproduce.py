"""Independent arithmetic replay of the frozen trajectory study; no gate tuning.

Only numpy/pandas/scipy are used; no author kernel or experiment module is imported.
The additional target-weighted bootstrap is a post hoc descriptive reporting check.
"""
from pathlib import Path
import argparse, hashlib, io, json, math, zipfile
import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent
REPO = PUBLIC.parents[1]
ROOT = None
DATA = PUBLIC / 'data/corrected'
REC = PUBLIC / 'recorded/corrected'
C = np.array([[1., 0.], [-1., 0.]])
DESIGNS = {'shared': C, 'separate': np.array([[0.,1.],[0.,-1.]]),
           'crossed': np.array([[0.,1.],[-1.,0.]])}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(name, x): (ROOT/name).write_text(json.dumps(x,indent=2)+'\n')
def cmp(a,b): return .5*((a>b).astype(float)-(a<b).astype(float))
def block(f,g,ix,D):
    ff,gg=f[ix],g[ix]
    pp=cmp(ff[:,0,None,:],ff[:,1:,:]);qq=cmp(gg[:,0,None,:],gg[:,1:,:])
    a=np.sum(pp*C.T,axis=(1,2)); b=np.sum(qq*D.T,axis=(1,2))
    return a*b
def correction(f,g,ix,D):
    ff,gg=f[ix],g[ix]
    p=cmp(ff[:,0],ff[:,1])-cmp(ff[:,0],ff[:,2])
    q=cmp(gg[:,0],gg[:,1])-cmp(gg[:,0],gg[:,2])
    return .5*np.einsum('bs,st,bt->b',p,C@D.T,q)
def triple(f,g,ix):
    out=np.zeros(len(ix))
    for a,b,c in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        p=cmp(f[ix[:,a]],f[ix[:,b]]) @ np.array([1.,-1.])
        q=cmp(g[ix[:,a]],g[ix[:,c]]) @ np.array([1.,-1.])
        out+=p*q/6
    return out
def radius(v,width,level):
    n=len(v); var=float(np.var(v,ddof=1)) if n>1 else 0.
    h=width*math.sqrt((math.log(2)-math.log(level))/(2*n))
    x=math.log(4)-math.log(level)
    eb=math.sqrt(2*var*x/n)+7*width*x/(3*(n-1))
    return {'radius':min(h,eb),'hoeffding_radius':h,'bernstein_radius':eb,'variance':var}
def identity(f,g):
    p=cmp(f[:,None,:],f[None,:,:]);q=cmp(g[:,None,:],g[None,:,:])
    mf=p.mean(1);mg=q.mean(1)
    K=mf.T@mg/len(f); same=np.einsum('ijs,ijt->st',p,q)/len(f)**2
    theta=float(np.mean((mf@np.array([1.,-1.]))*(mg@np.array([1.,-1.]))))
    return theta,same-K,p,q
def main():
    a=np.load(DATA/'selection_forecasts.npz'); z=np.load(DATA/'confirmation_forecasts.npz')
    g=np.column_stack([a['target'],a['baseline']]); names=list(a['names'])
    maps=[np.column_stack([a['predictions'][:,j],a['baseline']]) for j in range(8)]
    truths=[identity(f,g) for f in maps]
    ix=np.load(REC/'selection_indices.npz');ev,va=ix['evaluation'],ix['validation'];allix=np.r_[ev,va]
    alpha=.05/8;delta=alpha/2;rows=[];strong=pd.read_csv(PUBLIC/'recorded/strong/strong_gates.csv').set_index('candidate')
    gates=pd.read_csv(REC/'selection_gates.csv').set_index('candidate');errors=[]
    for k,name in enumerate(names):
        f=maps[k];theta,lam,p,q=truths[k]
        sv=block(f,g,ev,C);rv=correction(f,g,va,C)
        rs=radius(sv,2.,alpha-delta);rr=radius(rv,4.,delta)
        lower=float(sv.mean()-rv.mean()-rs['radius']-rr['radius'])
        uv=triple(f,g,allix);ru=radius(uv,2/3,alpha)
        mult=np.bincount(allix.reshape(-1),minlength=len(f));N=int(mult.sum())
        pc=p@np.array([1.,-1.]);qc=q@np.array([1.,-1.])
        center=float(np.dot(mult,(pc@mult)*(qc@mult)-(pc*qc)@mult)/(N*(N-1)*(N-2)))
        count=N//3;x=math.log(2)-math.log(alpha)
        pooled_radius=min((2/3)*math.sqrt(x/(2*count)),math.sqrt(np.var(uv,ddof=1))*math.sqrt(2*x/count)+(4/3/math.sqrt(count*(count-1))+(2/3)/(3*count))*x)
        row={'candidate':str(name),'exact_target':theta,'exact_shared_interaction':float(np.sum((C@C.T)*lam)),
          'raw_mean':float(sv.mean()),'interaction_mean':float(rv.mean()),'corrected_center':float(sv.mean()-rv.mean()),
          'evaluation_radius':rs['radius'],'validation_radius':rr['radius'],'split_lower':lower,
          'triple_center':float(uv.mean()),'triple_radius':ru['radius'],'triple_lower':float(uv.mean()-ru['radius']),
          'pooled_center':center,'pooled_radius':pooled_radius,'pooled_lower':center-pooled_radius,
          'hybrid_hoeffding_component_split_lower':float(sv.mean()-rv.mean()-rs['hoeffding_radius']-rr['hoeffding_radius']),
          'range_only_split_lower':float(sv.mean()-rv.mean()-2*math.sqrt(-math.log(alpha-delta)/(2*len(ev)))-4*math.sqrt(-math.log(delta)/(2*len(va)))),
          'M':len(ev),'J_reference':2,'n':len(va),'draws':N,'distinct_archive_rows':int((mult>0).sum()),
          'alpha':alpha,'delta':delta,'evaluation_variance':rs['variance'],'validation_variance':rr['variance'],
          'census_seconds_recorded':float(strong.loc[name,'census_seconds']),'pooled_seconds_recorded':float(strong.loc[name,'pooled_seconds'])}
        rows.append(row)
        for got,want in [(lower,gates.loc[name,'corrected_lower']),(row['triple_lower'],gates.loc[name,'full_u_lower']),
                         (center,strong.loc[name,'pooled_score']),(center-pooled_radius,strong.loc[name,'pooled_lower'])]:errors.append(abs(got-want))
    assert max(errors)<1e-12,max(errors)
    pd.DataFrame(rows).to_csv(ROOT/'certificate_decomposition.csv',index=False)
    # Complete original grid, all candidates and all four designs; no selected successes.
    stored=pd.read_csv(REC/'coverage_replications.csv').set_index(['replication','candidate','design'])
    coverage=[];griderr=[]
    for rep in range(300):
        rng=np.random.default_rng(2026091400+rep);ei=rng.integers(len(g),size=(1024,3));vi=rng.integers(len(g),size=(512,3));both=np.r_[ei,vi]
        for k,name in enumerate(names):
            f=maps[k];theta=truths[k][0]
            for design in ['shared','separate','crossed','full_u']:
                if design=='full_u':
                    scores=triple(f,g,both);raw=float(scores.mean());interaction=0.;rrad=0.;srad=radius(scores,2/3,.05)['radius'];W=2/3;WR=0.
                else:
                    D=DESIGNS[design];scores=block(f,g,ei,D);val=correction(f,g,vi,D)
                    raw=float(scores.mean());interaction=float(val.mean());WR=float(np.abs(C@D.T).sum());W=2.
                    srad=radius(scores,W,.05 if WR==0 else .025)['radius'];rrad=0. if WR==0 else radius(val,WR,.025)['radius']
                low=raw-interaction-srad-rrad
                rec=stored.loc[(rep,name,design)]
                griderr += [abs(low-rec.lower),abs(raw-rec.mean_score),abs(interaction-rec.mean_interaction)]
                coverage.append({'replication':rep,'candidate':str(name),'design':design,'target':theta,'mean_score':raw,'interaction':interaction,
                    'corrected_center':raw-interaction,'evaluation_radius':srad,'validation_radius':rrad,'lower':low,
                    'score_range_width':W,'interaction_range_width':WR,'noncoverage':int(low>theta),'detection':int(low>0)})
    assert max(griderr)<1e-12,max(griderr)
    grid=pd.DataFrame(coverage);grid.to_csv(ROOT/'coverage_decomposition_all_9600.csv',index=False)
    summaries=[]
    for (name,design),part in grid.groupby(['candidate','design'],sort=False):
        k=names.index(name);theta,lam,_,_=truths[k];gamma=0. if design=='full_u' else float(np.sum((C@DESIGNS[design].T)*lam))
        summaries.append({'candidate':name,'design':design,'target':theta,'true_interaction':gamma,'expected_raw_mean':theta+gamma,
          'score_range_width':float(part.score_range_width.iloc[0]),'interaction_range_width':float(part.interaction_range_width.iloc[0]),
          'mean_raw':float(part.mean_score.mean()),'mean_interaction':float(part.interaction.mean()),
          'mean_corrected_center':float(part.corrected_center.mean()),'mean_evaluation_radius':float(part.evaluation_radius.mean()),
          'mean_validation_radius':float(part.validation_radius.mean()),'mean_lower':float(part.lower.mean()),
          'lower_min':float(part.lower.min()),'lower_max':float(part.lower.max()),'replications':len(part),
          'noncoverage':int(part.noncoverage.sum()),'detection':int(part.detection.sum()),
          'zero_count_two_sided_95_upper':float(beta.ppf(.975,1,300))})
    pd.DataFrame(summaries).to_csv(ROOT/'coverage_all_32_cells.csv',index=False)
    # Same fixed candidates, paired temporal blocks. New weighted ratio is descriptive.
    loss=(z['predictions']-z['target'][:,None])**2; week=pd.to_datetime(z['timestamps']).to_period('W-SUN').astype(str)
    frame=pd.DataFrame(loss,columns=names).assign(week=week)
    sums=frame.groupby('week',sort=True).sum();means=frame.groupby('week',sort=True).mean();sizes=frame.groupby('week',sort=True).size().to_numpy()
    nweek=len(sizes);rng=np.random.default_rng(2026091455)
    starts=rng.integers(nweek-4+1,size=(2000,int(np.ceil(nweek/4))))
    indices=(starts[:,:,None]+np.arange(4)).reshape(2000,-1)[:,:nweek]
    utility=[]
    for candidate,comparator,label in [('persistence','boosting_31','gate_vs_ungated'),('persistence','persistence','gate_vs_baseline'),('boosting_31','persistence','augmentation_vs_baseline')]:
        k=names.index(candidate);j=names.index(comparator)
        totaldiff=(sums.iloc[:,j]-sums.iloc[:,k]).to_numpy();weeklydiff=(means.iloc[:,j]-means.iloc[:,k]).to_numpy()
        boot_weighted=totaldiff[indices].sum(1)/sizes[indices].sum(1)
        boot_relative=totaldiff[indices].sum(1)/sums.iloc[:,j].to_numpy()[indices].sum(1)
        boot_week=weeklydiff[indices].mean(1)
        utility.append({'comparison':label,'candidate':candidate,'comparator':comparator,'records':len(loss),'weeks':nweek,
         'candidate_mse':float(loss[:,k].mean()),'comparator_mse':float(loss[:,j].mean()),
         'target_weighted_absolute_gain':float((loss[:,j]-loss[:,k]).mean()),'target_weighted_absolute_ci':np.quantile(boot_weighted,[.025,.975]).tolist(),
         'target_weighted_relative_gain':float(1-loss[:,k].mean()/loss[:,j].mean()),'target_weighted_relative_ci':np.quantile(boot_relative,[.025,.975]).tolist(),
         'equal_week_absolute_gain':float(weeklydiff.mean()),'equal_week_absolute_ci':np.quantile(boot_week,[.025,.975]).tolist(),
         'interval_scope':'Descriptive paired four-week noncircular moving-block percentile bootstrap, 2000 draws, seed 2026091455. No physical-series finite guarantee.',
         'analysis_status':'Corrected already-viewed evaluation. Target-weighted interval is a subsequent reporting check; equal-week interval is the original specified endpoint.'})
    emit('utility_weighted_and_equal_week.json',{'comparisons':utility,'week_counts':dict(zip(sums.index,sizes.tolist()))})
    # Source timing and numerical feature audit. Calendar target metadata is known at issue.
    with zipfile.ZipFile(PUBLIC/'data/beijing_pm25.zip') as zz:
        raw=pd.read_csv(io.BytesIO(zz.read([x for x in zz.namelist() if x.endswith('.csv')][0])))
    raw.index=pd.to_datetime(raw[['year','month','day','hour']]);lags=[1,2,3,6,12,24]
    checks={};uniquehours=set()
    for role,arr in [('selection',a),('confirmation',z)]:
        stamps=pd.to_datetime(arr['timestamps']);assert np.all(stamps.hour==15)
        expected=np.column_stack([raw['pm2.5'].reindex(stamps-pd.Timedelta(hours=h)).to_numpy() for h in lags]+[raw[col].reindex(stamps-pd.Timedelta(hours=1)).to_numpy() for col in ['TEMP','PRES','DEWP','Iws','Is','Ir']]+[stamps.hour,stamps.dayofweek,stamps.month])
        assert np.array_equal(expected,arr['features'],equal_nan=True)
        assert np.array_equal(raw['pm2.5'].reindex(stamps).to_numpy(),arr['target'])
        assert np.array_equal(expected[:,0],arr['baseline'])
        checks[role]={'rows':len(stamps),'first_target':str(stamps.min()),'last_target':str(stamps.max()),'feature_match':True,'targets_match':True,'baseline_match':True}
        if role=='selection':
            for stamp in stamps:
                uniquehours.add(str(stamp))
                for h in lags:uniquehours.add(str(stamp-pd.Timedelta(hours=h)))
    emit('INDEPENDENT_CHECK.json',{'status':'PASS','selection_gate_max_absolute_error':max(errors),'all_9600_grid_max_absolute_error':max(griderr),
        'source_timing':checks,'logical_features':16,'persisted_numeric_features':15,'wind_direction_in_raw_source':True,
        'referenced_raw_hour_offsets_from_target':[0,-1,-2,-3,-6,-12,-24],
        'distinct_raw_hour_rows_across_selection':len(uniquehours),'maximum_history_window_hours':24,
        'timing_assumption':'Previous-hour measurements are assumed available at issue; the source does not establish actual publication latency.',
        'original_learning_rows':23006,'actual_new_labels_from_archive_resampling':0,
        'files':{str(p.relative_to(REPO)):sha(p) for p in [DATA/'selection_forecasts.npz',DATA/'confirmation_forecasts.npz',REC/'selection_indices.npz',REC/'coverage_replications.csv']}})
    print(json.dumps({'status':'PASS','gate_error':max(errors),'grid_error':max(griderr),'boosting31':rows[4],'utility':utility[0]}))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ROOT=args.output
    ROOT.mkdir(parents=True,exist_ok=False)
    main()
