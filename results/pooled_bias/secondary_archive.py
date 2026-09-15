"""Secondary square-MGF comparison, developed after the first benchmark."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import archive_benchmark as base

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists();args.output.mkdir()
    scope='Secondary retrospective extension developed after first comparator outcomes; no new labels, no joint claim from selecting methods'
    (args.output/'execution_seal.json').write_text(json.dumps({'scope':scope,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'comparator_sha256':hashlib.sha256((base.HERE/'pooled_bias_bound.py').read_bytes()).hexdigest()},indent=2)+'\n')
    stats=json.loads((args.primary/'sufficient_statistics.json').read_text());primary=pd.read_csv(args.primary/'all_candidates.csv');rows=[]
    for item in stats:
        task,baseline,candidate=[item[k] for k in ['task','baseline','candidate']]
        archive=np.load(base.SOURCE/task/'forecast_archive.npz');idx=np.load(base.SOURCE/task/'sampling_indices.npz')
        train,evaluation=idx['training'],idx['evaluation'];mask=archive['selection']
        x=archive[baseline+'__'+candidate][mask];y=archive['y'][mask];z=archive[baseline+'__category'][mask].astype(int)
        f=base.means(z[train],(rankdata(x[train])-1)/(len(train)-1));g=base.means(z[train],(rankdata(y[train])-1)/(len(train)-1))
        ze=z[evaluation];h=base.triple_scores(*[v[None,:] for v in (x[evaluation],y[evaluation],f[ze],g[ze])])
        prior=primary[(primary.task==task)&(primary.baseline==baseline)&(primary.candidate==candidate)&(primary.budget_method=='known_mass_aggregate')].iloc[0]
        for product in ['subgamma','exact_mgf']:
            b=base.pooled_upper_bound(item['masses'],*item['stats'],item['delta'],absent_upper=item['corner'],lambda_mode='square_mgf',product_mode=product)
            u=base.one_sided_certificate(prior['mean'],h,len(h),f,g,b['upper'],delta=item['delta'])
            bet=base.mixture_betting_pvalue(h,u['kernel_lower'],u['kernel_upper'],bias_upper=b['upper'],delta=item['delta'])
            rows.append(dict(task=task,baseline=baseline,candidate=candidate,budget_method='pooled_square_'+product,bias=b['upper'],true_bias=prior.true_bias,mean=prior['mean'],U_lower=u['lower_bound'],U_p=u['p'],betting_p=bet['p'],bias_covers=b['upper']+1e-14>=prior.true_bias,queries=32768))
    frame=pd.DataFrame(rows)
    for name in ['U','betting']:
        frame[name+'_BY']=frame.groupby(['task','baseline','budget_method'])[name+'_p'].transform(base.by_values)
        frame[name+'_retained']=frame[name+'_BY']<=.05
    frame.to_csv(args.output/'all_candidates.csv',index=False)
    summary=frame.groupby('budget_method')[['U_retained','betting_retained','bias_covers']].sum();summary.to_csv(args.output/'summary.csv');print(summary.to_string())
    (args.output/'receipt.json').write_text(json.dumps({'scope':scope,'rows':len(frame),'all_bounds_cover':bool(frame.bias_covers.all()),'output_sha256':hashlib.sha256((args.output/'all_candidates.csv').read_bytes()).hexdigest()},indent=2)+'\n')

if __name__=='__main__':main()
