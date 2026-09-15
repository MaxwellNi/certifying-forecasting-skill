"""Subsequent complete-U and census comparisons on all available trajectories."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'trajectory_design'))
from pooled_trajectory_kernel import pooled_u_joint_lower
from experiment import oracle,C,evaluate_confirmation

def main(data,recorded,output):
 output.mkdir(parents=True,exist_ok=False)
 x=np.load(data/'selection_forecasts.npz');pred=x['predictions'];base=x['baseline'];y=x['target'];names=x['names']
 ix=np.load(recorded/'selection_indices.npz');indices=np.concatenate([ix['evaluation'],ix['validation']]).reshape(-1)
 g=np.column_stack([y,base]);rows=[]
 for k,name in enumerate(names):
  f=np.column_stack([pred[:,k],base]);tick=time.perf_counter();theta=oracle(f,g)[0];census_seconds=time.perf_counter()-tick
  tick=time.perf_counter();u=pooled_u_joint_lower(f[indices],g[indices],C,C,alpha=.05/8);pooled_seconds=time.perf_counter()-tick
  rows.append({'candidate':str(name),'target':theta,'selection_mse':float(np.mean((pred[:,k]-y)**2)),
   'census_seconds':census_seconds,'census_records':len(y),'pooled_seconds':pooled_seconds,
   'pooled_score':u['score'],'pooled_lower':u['lower'],'pooled_radius':u['radius'],
   'trajectory_draws':len(indices),'independent_triples':u['independent_triples'],
   'pooled_pass':u['lower']>0,'census_positive':theta>1e-14})
 d=pd.DataFrame(rows);d.to_csv(output/'strong_gates.csv',index=False)
 choices={'persistence':0,'ungated':int(np.argmin(d.selection_mse))}
 for rule,mask in [('pooled_u',d.pooled_pass),('census',d.census_positive)]:
  allowed=np.flatnonzero(mask);choices[rule]=int(allowed[np.argmin(d.selection_mse.to_numpy()[allowed])])if len(allowed)else 0
 (output/'FROZEN_SELECTIONS.json').write_text(json.dumps({'candidate_indices':choices,'candidates':{k:str(names[v])for k,v in choices.items()},
  'analysis_status':'Subsequent strengthening after the first study outcomes; this is not an independent confirmation.',
  'selection_information':'Same corrected selection archive and all previously sampled raw trajectory indices.',
  'confirmation_used_for_choices':False,'freeze_unix':time.time()},indent=2)+'\n')
 evaluate_confirmation(data,output,choices,names)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--recorded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();main(a.data,a.recorded,a.output)
