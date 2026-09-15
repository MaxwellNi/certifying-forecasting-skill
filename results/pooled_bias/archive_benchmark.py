"""Retrospective comparison on all32 previously inspected forecast candidates."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parent.parent
SOURCE=PUBLIC/'results/forecast_confirmation'
sys.path[:0]=[str(SOURCE),str(PUBLIC/'results/aggregate_bias'),str(HERE)]
from gates import means,by_values
from helpers.reference_certificate_efficiency import one_sided_certificate,triple_scores,family_validation_delta
from helpers.peer_rank_products import peer_rank_products
from helpers.classical_reference_comparators import mixture_betting_pvalue
from pooled_bias_bound import pooled_upper_bound

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    assert not out.exists();out.mkdir()
    (out/'execution_seal.json').write_text(json.dumps({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'comparator_sha256':hashlib.sha256((HERE/'pooled_bias_bound.py').read_bytes()).hexdigest(),'scope':'Prior32 candidates; no fresh forecast confirmation; modes evaluated separately, never unpenalized minimum'},indent=2)+'\n')
    old=pd.read_csv(PUBLIC/'results/aggregate_bias/archive/all_candidates.csv')
    rows=[];suff=[]
    for task in ['appliances','metro']:
        archive=np.load(SOURCE/task/'forecast_archive.npz');idx=np.load(SOURCE/task/'sampling_indices.npz')
        train,val,evaluation=[idx[k] for k in ['training','validation','evaluation']]
        mask=archive['selection'];y=archive['y'][mask]
        for baseline in ['ridge','seasonal_day']:
            z=archive[baseline+'__category'][mask].astype(int);p=np.bincount(z,minlength=32)/len(z)
            zt,zv,ze=z[train],z[val[:,0]],z[evaluation]
            g=means(zt,(rankdata(y[train])-1)/(len(train)-1));delta=family_validation_delta(8)
            comp=lambda a:(a[val[:,0]]>a[val[:,1]]).astype(float)+.5*(a[val[:,0]]==a[val[:,1]])
            yc=comp(y)
            for candidate in ['persistence','seasonal_day','seasonal_week','ridge_full','hist_gradient_boosting','extra_trees','category_copy','independent_noise']:
                x=archive[baseline+'__'+candidate][mask];f=means(zt,(rankdata(x[train])-1)/(len(train)-1));xc=comp(x)
                a=xc-f[zv];b=yc-g[zv];counts=np.bincount(zv,minlength=32)
                stats=[counts]+[np.bincount(zv,weights=w,minlength=32) for w in [a,b,a*a,b*b,a*b]]
                corner=np.maximum(f*g,(1-f)*(1-g))
                prior=old[(old.task==task)&(old.baseline==baseline)&(old.candidate==candidate)]
                assert len(prior)==4
                h=triple_scores(*[v[None,:] for v in (x[evaluation],y[evaluation],f[ze],g[ze])])
                mean=float(peer_rank_products(x[evaluation],y[evaluation],f[ze],g[ze])['corrected_residual_product'].mean())
                assert abs(mean-prior['mean'].iloc[0])<1e-12
                for mode in ['fixed','grid','mixture']:
                    bound=pooled_upper_bound(p,*stats,delta,absent_upper=corner,lambda_mode=mode)
                    B=bound['upper'];u=one_sided_certificate(mean,h,len(h),f,g,B,delta=delta)
                    bet=mixture_betting_pvalue(h,u['kernel_lower'],u['kernel_upper'],bias_upper=B,delta=delta)
                    rows.append(dict(task=task,baseline=baseline,candidate=candidate,budget_method='pooled_'+mode,bias=B,true_bias=prior.true_bias.iloc[0],mean=mean,U_lower=u['lower_bound'],U_p=u['p'],betting_p=bet['p'],bias_covers=B+1e-14>=prior.true_bias.iloc[0],queries=32768))
                suff.append(dict(task=task,baseline=baseline,candidate=candidate,masses=p.tolist(),stats=[s.tolist() for s in stats],corner=corner.tolist(),delta=delta))
    new=pd.DataFrame(rows)
    for name in ['U','betting']:
        new[name+'_BY']=new.groupby(['task','baseline','budget_method'])[name+'_p'].transform(by_values)
        new[name+'_retained']=new[name+'_BY']<=.05
    combined=pd.concat([old,new],ignore_index=True)
    combined.to_csv(out/'all_candidates.csv',index=False)
    (out/'sufficient_statistics.json').write_text(json.dumps(suff,indent=2)+'\n')
    summary=combined.groupby('budget_method')[['U_retained','betting_retained','bias_covers']].sum()
    summary.to_csv(out/'summary.csv');print(summary.to_string())
    (out/'receipt.json').write_text(json.dumps({'rows':len(combined),'new_rows':len(new),'candidates':len(suff),'scope':'Retrospective bias-bound substitution on the same fitted models and previously inspected evaluation scores; no re-selection or confirmation success claim','all_new_bias_bounds_cover_archive_bias':bool(new.bias_covers.all()),'output_sha256':hashlib.sha256((out/'all_candidates.csv').read_bytes()).hexdigest()},indent=2)+'\n')

if __name__=='__main__':main()
