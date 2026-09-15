"""Fixed same-cost audit methods for the new public-data confirmation.

No exact outcome-rank census is computed here. Category counts are supplied by
the charged metadata/availability census. The original rows remain dependent;
the inference experiment is conditional IID replacement sampling of the archive.
"""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE/'theory'), str(HERE/'betting'), str(HERE/'helpers')]
from stratified_reference import prepare, lower_bound, p_value
from full_u_variance import full_u_all_delete_one, interaction_bounds, baseline as direct
from aggregate_bias_bound import aggregate_upper_bound
from pooled_bias_bound import pooled_upper_bound
from reference_certificate_efficiency import one_sided_certificate, triple_scores, family_validation_delta
from peer_rank_products import peer_rank_products
from classical_reference_comparators import mixture_betting_pvalue

METHODS = ['full_u_joint', 'stratified_hybrid', 'stratified_hoeffding',
           'stratified_profiled_betting', 'aggregate_reference_u', 'aggregate_reference_betting',
           'pooled_reference_u', 'pooled_reference_betting', 'direct_loss']
TOTAL = 32768
PROFILED_FRACTIONS = tuple(repr(float(x)) for x in np.geomspace(1e-4, .99, 64))


def by_adjust(values):
    p=np.asarray(values,float); n=len(p); order=np.argsort(p,kind='stable')
    scaled=p[order]*n*math.fsum(1/j for j in range(1,n+1))/np.arange(1,n+1)
    out=np.empty(n);out[order]=np.minimum(1,np.minimum.accumulate(scaled[::-1])[::-1])
    return out


def draw_indices(n, seed):
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    train=streams[0].integers(n,size=4096)
    validation=streams[1].integers(n,size=(8192,2))
    evaluation=streams[2].integers(n,size=12288)
    return train,validation,evaluation,np.concatenate([train,validation.ravel(),evaluation])


def full_joint_from_summary(summary, probabilities, alpha=.05):
    n=summary['n'];center=summary['estimate'];q=n//4
    m,j=interaction_bounds(n,probabilities);mp,jp=interaction_bounds(n-1,probabilities)
    a=math.sqrt(2*summary['jackknife_variance'])
    b=math.sqrt((n-1)/n*(4*mp*mp+16*jp*jp))+2*m/3+j
    d=max(center,0.)
    root=2*d/(a+math.sqrt(a*a+4*b*d)) if d else 0.
    lo,hi=direct._kernel_bounds(probabilities);width=hi-lo
    pe=min(1.,4*math.exp(-root*root))
    ph=min(1.,2*math.exp(-2*q*(d/width)**2))
    radius=min(a*math.sqrt(math.log(4)-math.log(alpha))+b*(math.log(4)-math.log(alpha)),
               width*math.sqrt((math.log(2)-math.log(alpha))/(2*q)))
    return dict(p=min(pe,ph),estimate=center,lower=center-radius,radius=radius,
                jackknife_variance=summary['jackknife_variance'],eb_linear_remainder=b,
                full_u_eb_union_p=pe,full_u_hoeffding_union_p=ph)


def audit_candidates(preds, y, categories, baseline, blends, seed, family=6):
    from profiled_betting import profiled_betting
    x=np.asarray(preds,float); y=np.asarray(y,float);c=np.asarray(categories,int)
    base=np.asarray(baseline,float);blend=np.asarray(blends,float)
    if x.shape!=(len(y),family) or blend.shape!=x.shape or base.shape!=y.shape or c.shape!=y.shape:
        raise ValueError('candidate columns must match the complete frozen family')
    if not np.isfinite(y).all() or not np.isfinite(base).all() or (c<0).any():
        raise ValueError('observed archive outcomes, baseline and categories must be valid')
    if ((y<0)|(y>1)|(base<0)|(base>1)).any():
        raise ValueError('primary outcome and baseline must lie in [0,1]')
    finite_blends=blend[np.isfinite(blend)]
    if ((finite_blends<0)|(finite_blends>1)).any():
        raise ValueError('finite delivered augmentations must lie in [0,1]')
    train,val,evaluation,all_rows=draw_indices(len(y),seed)
    calls_hash=hashlib.sha256(all_rows.astype('<i8').tobytes()).hexdigest()
    nc=max(4,int(c.max())+1);counts=np.bincount(c,minlength=nc);p=counts/len(c)
    positive={i:float(v) for i,v in enumerate(p) if v>0}
    def means(z,v):
        counts=np.bincount(z,minlength=nc)
        return np.divide(np.bincount(z,weights=v,minlength=nc),counts,out=np.full(nc,.5),where=counts>0)
    def comp(v):return (v[val[:,0]]>v[val[:,1]]).astype(float)+.5*(v[val[:,0]]==v[val[:,1]])
    zt,zv,ze=c[train],c[val[:,0]],c[evaluation]
    g=means(zt,(rankdata(y[train])-1)/(len(train)-1));cy=comp(y)
    delta=family_validation_delta(family)
    rows=[]
    for candidate in range(family):
        start=time.perf_counter();v=x[:,candidate]
        meta=dict(candidate=candidate,raw_draw_calls=TOTAL,distinct_sampled_labels=int(len(np.unique(all_rows))),
                  sampled_index_sha256=calls_hash,archive_rows=len(y),positive_categories=len(positive),p_min=min(positive.values()))
        if not np.isfinite(v).all() or not np.isfinite(blend[:,candidate]).all():
            rows.extend(dict(meta,method=m,p=1.,failed=True) for m in METHODS);continue
        sx,sy,sz=v[all_rows],y[all_rows],c[all_rows]
        prepared=prepare(sx,sy,sz,positive)
        for rule in ['hybrid','hoeffding']:
            rows.append(dict(meta,method='stratified_'+rule,p=p_value(prepared,rule),
                             estimate=prepared['estimate'],lower=lower_bound(prepared,.05,rule),
                             missing_mass=prepared['missing_mass'],pairs=prepared['pairs'],allocation_variance_factor=prepared['allocation_variance_factor']))
        bet=profiled_betting(prepared, fractions=PROFILED_FRACTIONS)
        rows.append({**meta,**bet,'method':'stratified_profiled_betting'})
        summary=full_u_all_delete_one(sx,sy,sz,positive)
        rows.append(dict(meta,method='full_u_joint',**full_joint_from_summary(summary,positive)))
        f=means(zt,(rankdata(v[train])-1)/(len(train)-1));cx=comp(v)
        aa,bb=cx-f[zv],cy-g[zv];a_half=np.arange(len(val))<len(val)//2;b_half=~a_half
        ca=np.bincount(zv[a_half],minlength=nc);cb=np.bincount(zv[b_half],minlength=nc)
        ma=np.divide(np.bincount(zv[a_half],weights=aa[a_half],minlength=nc),ca,out=np.zeros(nc),where=ca>0)
        mb=np.divide(np.bincount(zv[b_half],weights=bb[b_half],minlength=nc),cb,out=np.zeros(nc),where=cb>0)
        corners=np.maximum(f*g,(1-f)*(1-g))
        agg=aggregate_upper_bound(p,ma,mb,ca,cb,delta,absent_upper=corners)
        stats=[np.bincount(zv,minlength=nc)]+[np.bincount(zv,weights=w,minlength=nc) for w in [aa,bb,aa*aa,bb*bb,aa*bb]]
        pooled=pooled_upper_bound(p,*stats,delta,absent_upper=corners,lambda_mode='fixed')
        h=triple_scores(*[z[None,:] for z in (v[evaluation],y[evaluation],f[ze],g[ze])])
        center=float(peer_rank_products(v[evaluation],y[evaluation],f[ze],g[ze])['corrected_residual_product'].mean())
        for label,budget in [('aggregate',agg),('pooled',pooled)]:
            certificate=one_sided_certificate(center,h,len(h),f,g,budget['upper'],delta=delta)
            betting=mixture_betting_pvalue(h,certificate['kernel_lower'],certificate['kernel_upper'],bias_upper=budget['upper'],delta=delta)
            rows.extend([dict(meta,method=label+'_reference_u',p=certificate['p'],lower=certificate['lower_bound'],estimate=center,bias=budget['upper']),
                         dict(meta,method=label+'_reference_betting',p=betting['p'],estimate=float(h.mean()),bias=budget['upper'])])
        gain=((y-base)**2-(y-blend[:,candidate])**2)[all_rows]
        lossbet=mixture_betting_pvalue(gain,-1.,1.,bias_upper=0.,delta=0.)
        rows.append(dict(meta,method='direct_loss',p=lossbet['p'],estimate=float(gain.mean())))
        for r in rows[-len(METHODS):]:r['candidate_all_methods_seconds']=time.perf_counter()-start
    assert len(rows)==family*len(METHODS)
    for row in rows:
        # Preserve positive finite mathematical tails when float exp underflows.
        row['p']=max(float(np.nextafter(0.,1.)),float(row['p']))
    assert all(0<=r['p']<=1 and math.isfinite(r['p']) for r in rows)
    return rows
