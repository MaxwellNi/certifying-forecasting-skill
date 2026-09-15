"""Fixed retrospective same-pair comparison; never a new forecast confirmation."""
from pathlib import Path
import argparse, datetime, gzip, hashlib, importlib.util, itertools, json, sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parent.parent
spec=importlib.util.spec_from_file_location('original_sim', PUBLIC/'results/aggregate_bias/simulate.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
OUTCOMES=list(itertools.product([0,1],repeat=4))
HX=np.array([float(a>c)+.5*(a==c) for a,b,c,d in OUTCOMES])
HY=np.array([float(b>d)+.5*(b==d) for a,b,c,d in OUTCOMES])

def joint_counts(n,p,q,reference,rng):
    out=np.zeros((*n.shape,16),dtype=np.int64)
    for c in range(len(p)):
        weights=np.array([(q[c] if a else 1-q[c])*(q[c] if b else 1-q[c])*reference[u,v] for a,b,u,v in OUTCOMES]); weights/=weights.sum()
        left=n[:,c].copy();remaining=1.
        for j,w in enumerate(weights):
            draw=left if j==15 else rng.binomial(left,float(np.clip(w/remaining,0,1)))
            out[:,c,j]=draw;left=left-draw;remaining-=w
    return out

def stats(counts,f,g):
    n=counts.sum(axis=2);sx=counts@HX;sy=counts@HY
    return n,sx-n*f,sy-n*g,counts@(HX*HX)-2*f*sx+n*f*f,counts@(HY*HY)-2*g*sy+n*g*g,counts@(HX*HY)-f*sy-g*sx+n*f*g

def subset_counts(counts,rng):
    """Uniform within-category half-subset, independent of outcome values."""
    selected=np.zeros_like(counts);left_total=counts.sum(axis=2);need=left_total//2
    for j in range(15):
        good=counts[:,:,j];bad=left_total-good
        # Degenerate empty urns have no selected observations.
        draw=np.zeros_like(need);valid=left_total>0
        draw[valid]=rng.hypergeometric(good[valid],bad[valid],need[valid])
        selected[:,:,j]=draw;need-=draw;left_total-=good
    selected[:,:,-1]=need
    assert np.all(selected<=counts)
    return selected

def split_bound(p,nA,a,nB,b,corner,delta):
    valid=(nA>0)&(nB>0);pp=p*valid;x=np.log(3/delta)
    d=pp/(4*np.sqrt(np.maximum(nA,1)*np.maximum(nB,1)))
    center=np.sum(pp*a*b,axis=1)
    return (center+np.sqrt(x/2*np.sum(pp**2*b*b/np.maximum(nA,1),axis=1))+
            np.sqrt(x/2*np.sum(pp**2*a*a/np.maximum(nB,1),axis=1))+
            np.sqrt(2*x*np.sum(d*d,axis=1))+x*d.max(axis=1)+np.sum(p*(~valid)*corner,axis=1))

def pooled(p,S,corner,delta):
    n,sa,sb,saa,sbb,sab=S;valid=n>=2;pp=p*valid
    safe=np.maximum(n,2);k=safe//2;l=safe-k
    a=sa/safe;b=sb/safe
    va=np.maximum(saa-sa*a,0)/(safe-1);vb=np.maximum(sbb-sb*b,0)/(safe-1)
    U=np.sum(pp*(sa*sb-sab)/(safe*(safe-1)),axis=1)
    VA=np.sum(pp**2/(4*k)*(b*b+k*vb/(safe*l)),axis=1)
    VB=np.sum(pp**2/(4*l)*(a*a+l*va/(safe*k)),axis=1)
    d=pp/(4*np.sqrt(k*l));D=np.sum(d*d,axis=1);x=np.log(3/delta)
    lam=np.sqrt(2*x/np.maximum(D,1e-300));base=U+np.sqrt(2*x*D)+x*d.max(axis=1)+np.sum(p*(~valid)*corner,axis=1)
    fixed=base+2*x/lam+lam*(VA+VB)/2
    grid=lam[:,None]*2.**np.arange(-8,9);xg=np.log(3*17/delta)
    tuned=base+(xg/grid+grid*VA[:,None]/2).min(axis=1)+(xg/grid+grid*VB[:,None]/2).min(axis=1)
    rho=np.maximum(D,1e-300)/(2*x)
    mixed=base+np.sqrt((VA+rho)*(2*x+np.log1p(VA/rho)))+np.sqrt((VB+rho)*(2*x+np.log1p(VB/rho)))
    return fixed,tuned,mixed,U

def bounds(p,counts,left_original,right_original,left_balanced,f,g,delta):
    S=stats(counts,f,g);n,sa,sb,*_=S
    corner=np.maximum(f*g,(1-f)*(1-g))
    mx=np.divide(sa,n,out=np.zeros_like(sa),where=n>0)+f
    my=np.divide(sb,n,out=np.zeros_like(sb),where=n>0)+g
    rad=np.sqrt(np.log(4*len(p)/delta)/(2*np.maximum(n,1)))
    lx=np.where(n>0,np.maximum(0,mx-rad),0);ux=np.where(n>0,np.minimum(1,mx+rad),1)
    ly=np.where(n>0,np.maximum(0,my-rad),0);uy=np.where(n>0,np.minimum(1,my+rad),1)
    rect=np.maximum.reduce([(lx-f)*(ly-g),(lx-f)*(uy-g),(ux-f)*(ly-g),(ux-f)*(uy-g)])@p
    result={'rectangle':rect}
    for name,left,right in [('split_original',left_original,right_original),('split_balanced',left_balanced,counts-left_balanced)]:
        nl,al,*_=stats(left,f,g);nr,_,br,*_=stats(right,f,g)
        ma=np.divide(al,nl,out=np.zeros_like(al),where=nl>0);mb=np.divide(br,nr,out=np.zeros_like(br),where=nr>0)
        result[name]=split_bound(p,nl,ma,nr,mb,corner,delta)
    fixed,grid,mix,U=pooled(p,S,corner,delta)
    result.update(pooled_fixed=fixed,pooled_grid=grid,pooled_mixture=mix,pooled_center=U)
    return result,S

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists();args.output.mkdir()
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert protocol['repetitions']==old.REPS and protocol['seed']==old.SEED
    seal={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()}
    (args.output/'execution_seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    records=[];audit=[];example_stats=[]
    previous=pd.read_csv(PUBLIC/'results/aggregate_bias/simulation/replications.csv.gz')
    for C,m,mass,fit in itertools.product(protocol['categories'],protocol['validation_pairs'],protocol['mass_laws'],protocol['fit_errors']):
        fi=protocol['fit_errors'].index(fit);mi=int(mass=='one_heavy')
        rng=np.random.default_rng(np.random.SeedSequence([old.SEED,C,m,mi,fi]))
        p=np.full(C,1/C) if mass=='balanced' else np.r_[.7,np.full(C-1,.3/(C-1))]
        t=np.linspace(-1,1,C);t=t-p@t;q=.5+.35*t/np.max(np.abs(t));mu=.25+.5*q
        f=mu.copy();g=mu.copy()
        if fit=='same_direction':f+=.05;g+=.05
        elif fit=='opposite_direction':f+=.15;g-=.15
        truth=float(p@((mu-f)*(mu-g)))
        reference=np.array([[p@((1-q)**2),p@((1-q)*q)],[p@(q*(1-q)),p@(q*q)]])
        nA=old.categorical_counts(m//2,p,rng,old.REPS);nB=old.categorical_counts(m//2,p,rng,old.REPS)
        A=joint_counts(nA,p,q,reference,rng);B=joint_counts(nB,p,q,reference,rng);counts=A+B
        balanced=subset_counts(counts,np.random.default_rng(np.random.SeedSequence([old.SEED,C,m,mi,fi,991])))
        source=previous[(previous.categories==C)&(previous.validation_pairs==m)&(previous.mass==mass)&(previous.fit_error==fit)]
        assert len(source)==old.REPS
        for delta in protocol['deltas']:
            data,S=bounds(p,counts,A,B,balanced,f,g,delta)
            if delta==.05:
                err=max(np.max(np.abs(data['rectangle']-source.rectangle_upper)),np.max(np.abs(data['split_original']-source.aggregate_upper)))
                assert err<1e-12,(C,m,mass,fit,err)
                audit.append(dict(categories=C,validation_pairs=m,mass=mass,fit_error=fit,max_original_replay_difference=float(err)))
            fixed=dict(categories=C,validation_pairs=m,mass=mass,fit_error=fit,delta=delta,true_bias=truth)
            records.append(pd.DataFrame(dict(fixed,replicate=np.arange(old.REPS),**data)))
            example_stats.append(dict(fixed,masses=p.tolist(),corner=np.maximum(f*g,(1-f)*(1-g)).tolist(),stats=[s[0].tolist() for s in S],expected={k:float(v[0]) for k,v in data.items()}))
        print(C,m,mass,fit,flush=True)
    frame=pd.concat(records,ignore_index=True)
    with (args.output/'rows.csv.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:stream.write(frame.to_csv(index=False).encode())
    summaries=[]
    keys=['categories','validation_pairs','mass','fit_error','delta']
    for ident,cell in frame.groupby(keys,sort=False):
        for method in protocol['methods']:
            slack=cell[method]-cell.true_bias
            summaries.append(dict(zip(keys,ident),method=method,repetitions=len(cell),noncoverage=int((slack < -1e-14).sum()),median_slack=float(slack.median()),mean_slack=float(slack.mean()),median_upper=float(cell[method].median())))
    pd.DataFrame(summaries).to_csv(args.output/'summary.csv',index=False)
    (args.output/'example_sufficient_statistics.json').write_text(json.dumps(example_stats,indent=2)+'\n')
    (args.output/'original_replay.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('FINISHED',len(frame),len(summaries),flush=True)

if __name__=='__main__':main()
