"""Regenerate the fixed-design centering demonstration, including all outcomes."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from reference_time_audit import audit, panel_scores

PROTOCOL = {
    'seed':73190527,'repetitions':400,'groups':8192,'validation_triples':2048,
    'alpha':.05,'delta':.01,
    'laws':['Bernoulli(0.1)','Bernoulli(0.5)','uniform ternary','continuous uniform'],
    'designs':['positive overlap','negative overlap','zero overlap'],
    'purpose':'Check a developed exact interaction mechanism, not independent forecasting utility.',
    'comparators':'Uncorrected classical bound targets the wrong mean; exact-law correction is an oracle diagnostic; validated correction estimates the interaction from fresh triples.',
    'target':'Zero population-reference temporal product in each declared law and design.',
    'frozen_before_execution':True,
}


def draw(rng, law, shape):
    if law == 0:return rng.binomial(1,.1,size=shape)
    if law == 1:return rng.binomial(1,.5,size=shape)
    if law == 2:return np.array([-2.,0.,5.])[rng.integers(0,3,size=shape)]
    return rng.uniform(size=shape)


def interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    protocol=dict(PROTOCOL)
    protocol['producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol['audit_source_sha256']=hashlib.sha256(Path(__file__).with_name('reference_time_audit.py').read_bytes()).hexdigest()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    c=np.array([[1.,0.],[0.,0.],[-1.,0.]])
    ds=[np.array([[0.,-.5],[1.,0.],[-.5,0.]]),
        np.array([[-.5,0.],[1.,0.],[0.,-.5]]),
        np.array([[-.5,0.],[1.,0.],[-.5,0.]])]
    true_omegas=[.1*.9/4,.25/4,5/54,1/6]
    rows=[]
    for law in range(4):
        for rep in range(PROTOCOL['repetitions']):
            rng=np.random.default_rng(np.random.SeedSequence([PROTOCOL['seed'],law,rep]))
            panels=draw(rng,law,(PROTOCOL['groups'],3,3))
            val=draw(rng,law,(PROTOCOL['validation_triples'],3))
            for design,d in enumerate(ds):
                result=audit(panels,c,d,val,PROTOCOL['alpha'],PROTOCOL['delta'])
                radius=2*np.sqrt(np.log(1/PROTOCOL['alpha'])/(2*PROTOCOL['groups']))
                naive=result['score_mean']-radius
                known=naive-result['overlap_coefficient']*true_omegas[law]
                rows.append({'law':PROTOCOL['laws'][law],'design':PROTOCOL['designs'][design],
                    'replication':rep,'overlap':result['overlap_coefficient'],
                    'true_empirical_mean':result['overlap_coefficient']*true_omegas[law],
                    'true_population_reference_mean':0.,'score_mean':result['score_mean'],
                    'omega_hat':result['omega_hat'],'omega_lower':result['omega_interval'][0],
                    'omega_upper':result['omega_interval'][1],
                    'interaction_allowance':result['upper_interaction_allowance'],
                    'validated_radius':result['sampling_radius'],
                    'naive_lower':float(naive),'oracle_lower':float(known),'validated_lower':result['lower_bound'],
                    'naive_positive':int(naive>0),'oracle_positive':int(known>0),
                    'validated_positive':int(result['positive_lower_bound']),
                    'total_observations':result['total_observation_count']})
    with gzip.open(out/'replications.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for law in PROTOCOL['laws']:
        for design in PROTOCOL['designs']:
            selected=[r for r in rows if r['law']==law and r['design']==design]
            for method in ['naive','oracle','validated']:
                n=len(selected);count=sum(r[method+'_positive']for r in selected)
                lo,hi=interval(count,n)
                summary.append({'law':law,'design':design,'method':method,'repetitions':n,
                    'positives':count,'rate':count/n,'pointwise_95_lower':lo,'pointwise_95_upper':hi,
                    'population_target':0.,'empirical_mean':selected[0]['true_empirical_mean'],
                    'total_observations':selected[0]['total_observations']})
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    result={'protocol':protocol,'replication_rows':len(rows),'method_cells':len(summary),
            'status':'executed','zero_observed_failures_is_not_zero_error_probability':True}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output)
