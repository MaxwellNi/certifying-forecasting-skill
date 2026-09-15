"""Independent raw-block, target, p-value inversion and BY verification.

Does not import author estimator code or repeat the quadratic full-U pass.
The saved full-U center is used only for independent bound inversion.
"""
from pathlib import Path
from itertools import permutations
import argparse, hashlib, json, math, datetime
import numpy as np
import pandas as pd


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def residual_midrank(values, categories):
    ordered=np.sort(values)
    doubled=np.searchsorted(ordered,values,'left')+np.searchsorted(ordered,values,'right')
    counts=np.bincount(categories,minlength=32)
    sums=np.zeros(32,dtype=np.int64);np.add.at(sums,categories,doubled)
    numerator=counts[categories]*doubled-sums[categories]
    return numerator/(2.*len(values)*counts[categories])


def blocks(v,w,c,p):
    v,w,c=[a.reshape(-1,4) for a in (v,w,c)]
    av=(v[:,:,None]>v[:,None,:]).astype(float)+.5*(v[:,:,None]==v[:,None,:])
    bw=(w[:,:,None]>w[:,None,:]).astype(float)+.5*(w[:,:,None]==w[:,None,:])
    total=np.zeros(len(v))
    for i,j,k,l in permutations(range(4)):
        total+=av[:,i,j]*bw[:,i,k]-(c[:,i]==c[:,j])*av[:,i,k]*bw[:,j,l]/p[c[:,i]]
    return total/24


def eb_logp(mean,variance,q,width):
    if mean<=0:return 0.
    def lower(logp):
        x=math.log(2)-logp
        return mean-math.sqrt(2*variance*x/q)-7*width*x/(3*(q-1))
    if lower(0)<=0:return 0.
    left=-1.
    while lower(left)>0:left*=2
    right=0.
    for _ in range(100):
        mid=(left+right)/2
        if lower(mid)>0:right=mid
        else:left=mid
    return (left+right)/2


def independent_by(values):
    n=len(values);H=math.fsum(1/i for i in range(1,n+1))
    ordered=sorted(enumerate(values),key=lambda x:x[1]);answer=[None]*n
    for rank,(original,p) in enumerate(ordered,1):
        answer[original]=min(1.,min(n*H*value/j for j,(_,value) in enumerate(ordered,1) if j>=rank))
    return answer


def main():
    ap=argparse.ArgumentParser()
    here=Path(__file__).resolve().parent
    ap.add_argument('--results',type=Path,default=here/'archive')
    ap.add_argument('--source',type=Path,default=here.parent/'forecast_confirmation')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();frame=pd.read_csv(args.results/'all_candidates.csv')
    assert len(frame)==64 and frame.groupby(['task','baseline','method']).size().eq(8).all()
    src=args.source;protocol=json.loads((src/'protocol.json').read_text())
    cases=[];source_hashes={};maximum={};cache={}
    def compare(name,actual,expected,tol=1e-11):
        err=abs(float(actual)-float(expected));maximum[name]=max(maximum.get(name,0.),err)
        assert err<=tol,(name,actual,expected,err)
    for ti,task in enumerate(['appliances','metro']):
        archive=np.load(src/task/'forecast_archive.npz',allow_pickle=False)
        indices=np.load(src/task/'sampling_indices.npz',allow_pickle=False)
        select=archive['selection'];population=int(select.sum())
        generators=[np.random.default_rng(seed) for seed in np.random.SeedSequence(protocol['seed']+1000+ti).spawn(3)]
        expected=[generators[0].integers(population,size=4096),generators[1].integers(population,size=(8192,2)),generators[2].integers(population,size=12288)]
        for key,values in zip(['training','validation','evaluation'],expected):assert np.array_equal(values,indices[key])
        pooled=np.concatenate([x.ravel() for x in expected]);assert np.array_equal(pooled,indices['all_rows']) and len(pooled)==32768
        cache[task]=(archive,select,population,pooled)
        for name in ['forecast_archive.npz','sampling_indices.npz']:source_hashes[task+'/'+name]=sha(src/task/name)
    for (task,baseline,candidate),group in frame.groupby(['task','baseline','candidate']):
        archive,select,population,pooled=cache[task]
        c=archive[baseline+'__category'][select].astype(int);v=archive[baseline+'__'+candidate][select];w=archive['y'][select]
        counts=np.bincount(c,minlength=32);p=counts/population;positive=p[p>0]
        assert np.all(p[c]>0)
        target=float(np.mean(residual_midrank(v,c)*residual_midrank(w,c)))
        y=blocks(v[pooled],w[pooled],c[pooled],p)
        q=len(y);center=float(np.mean(y));variance=float(np.sum((y-center)**2)/(q-1))
        low=-1/12 if len(positive)==1 else 1/6-1/(4*min(positive));high=1/12 if len(positive)==1 else 1/3;R=high-low
        a_first=.05/(8*math.fsum(1/j for j in range(1,9)))
        assert np.all(y>=low-1e-12) and np.all(y<=high+1e-12)
        for row in group.itertuples():
            compare('exact_target',row.exact_archive_target,target,2e-13)
            compare('range',row.kernel_range,R)
            compare('p_min',row.p_min,min(positive),1e-15)
            assert row.raw_draw_calls==32768 and row.independent_blocks==8192
            assert row.distinct_sampled_rows==len(np.unique(pooled))
            if row.method=='direct_block_empirical_bernstein':
                compare('block_mean',row.estimate,center,2e-13)
                compare('block_variance',row.sample_variance,variance,2e-12)
                lp=eb_logp(center,variance,q,R)
                radius=math.sqrt(2*variance*(math.log(2)-math.log(.05))/q)+7*R*(math.log(2)-math.log(.05))/(3*(q-1))
                r_first=math.sqrt(2*variance*(math.log(2)-math.log(a_first))/q)+7*R*(math.log(2)-math.log(a_first))/(3*(q-1))
                compare('block_radius',row.radius,radius)
                compare('block_first_lower',row.first_threshold_lower,center-r_first)
            else:
                assert row.method=='direct_full_u_hoeffding'
                center_full=float(row.first_term)-float(row.second_term)
                compare('full_term_subtraction',row.estimate,center_full)
                lp=-2*q*(max(center_full,0.)/R)**2
                radius=R*math.sqrt(-math.log(.05)/(2*q))
                compare('full_radius',row.radius,radius)
                compare('full_first_lower',row.first_threshold_lower,center_full-R*math.sqrt(-math.log(a_first)/(2*q)))
            compare('log_p',row.log_p,lp)
            compare('p',row.p,math.exp(lp))
            compare('lower',row.lower,row.estimate-radius)
            assert bool(row.lower_covers_archive_target)==(row.lower<=target+1e-13)
        cases.append({'task':task,'baseline':baseline,'candidate':candidate,'block_mean':center,'block_variance':variance,'exact_target':target,'p_min':float(min(positive)),'independent_blocks':q})
    for keys,group in frame.groupby(['task','baseline','method']):
        adjusted=independent_by(group.p.tolist())
        for row,pby in zip(group.itertuples(),adjusted):
            compare('BY',row.p_BY,pby)
            assert bool(row.retained)==(pby<=.05)
    execution=json.loads((args.results/'execution.json').read_text())
    original_unchanged=all(sha(src/name)==digest for name,digest in execution['original_confirmation_hashes'].items());assert original_unchanged
    receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'candidate_count':32,'result_rows':64,'block_values_independently_recomputed':32*8192,'families':8,'family_size':8,'maximum_absolute_errors':maximum,'cases':cases,'results_sha256':sha(args.results/'all_candidates.csv'),'source_inputs_sha256':source_hashes,'auditor_source_sha256':sha(__file__),'original_confirmation_unchanged':original_unchanged,'author_estimator_imported':False,'quadratic_full_u_pass_repeated':False,'full_u_center_check':'saved first minus saved second; the complete algorithm is separately checked against rational sums by verify.py','scope':'All outcomes were previously viewed. Verification of fixed archive target, draw replay, block computation and family arithmetic, not unseen confirmation or superiority.','retentions':frame.groupby('method').retained.sum().astype(int).to_dict()}
    args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['passed','candidate_count','result_rows','maximum_absolute_errors','retentions']}))
if __name__=='__main__':main()
