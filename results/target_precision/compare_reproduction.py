"""Compare a source-retrained run with the saved reference, without changing either."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def array_record(a,b):
    same_shape=a.shape==b.shape
    numeric=a.dtype.kind in 'fciu' and b.dtype.kind in 'fciu'
    exact=bool(np.array_equal(a,b,equal_nan=True)) if numeric else bool(np.array_equal(a,b))
    max_abs=None
    if same_shape and numeric:
        finite=np.isfinite(a)&np.isfinite(b)
        if np.any(finite):max_abs=float(np.max(np.abs(a[finite].astype(float)-b[finite].astype(float))))
    return {'shape_equal':same_shape,'dtype_equal':a.dtype==b.dtype,'elementwise_equal':exact,'max_absolute_difference':max_abs}

def compare(run,reference):
    result={'scope':'Reproduction of an existing task, not new independent evidence. No tolerance is used to turn unequal predictions into equal arrays.','arrays':{},'numeric_gate_differences':{}}
    for stage in ('training','calibration','selection','confirmation'):
        for name in (stage+'_forecasts.npz','sealed/'+stage+'_labels.npz'):
            with np.load(run/name,allow_pickle=False) as a,np.load(reference/name,allow_pickle=False) as b:
                if set(a.files)!=set(b.files):raise ValueError('NPZ fields differ: '+name)
                result['arrays'][name]={key:array_record(a[key],b[key]) for key in a.files}
    a=read(run/'selectors.json');b=read(reference/'selectors.json')
    result['selectors_equal']=a['selectors']==b['selectors']
    result['calibration_weights_equal']=a['weights']==b['weights']
    ga={(x['method'],x['candidate']):x for x in read(run/'gate_decisions.json')}
    gb={(x['method'],x['candidate']):x for x in read(reference/'gate_decisions.json')}
    result['gate_keys_equal']=ga.keys()==gb.keys()
    result['gate_rows']=len(ga)
    result['all_gate_retention_equal']=result['gate_keys_equal'] and all(ga[k]['retained']==gb[k]['retained'] for k in ga)
    for field in ('p','estimate','lower','radius'):
        diffs=[abs(float(ga[k][field])-float(gb[k][field])) for k in ga.keys()&gb.keys() if field in ga[k] and field in gb[k] and ga[k][field] is not None and gb[k][field] is not None]
        result['numeric_gate_differences'][field]={'compared_rows':len(diffs),'max_absolute_difference':max(diffs,default=0.0)}
    ba=read(run/'confirmation_conditional_bounds.json');bb=read(reference/'confirmation_conditional_bounds.json')
    result['primary_bounds']={key:{'estimate_equal':ba[key]['estimate']==bb[key]['estimate'],'lower_difference':ba[key]['lower']-bb[key]['lower'],'gain_sum':ba[key]['gain_sum']} for key in ba}
    result['predefined_success']=read(run/'confirmation_adjudication.json')['observed_primary_success_under_stated_conditional_assumptions']
    result['source_sha256']=hashlib.sha256((run/'source.zip').read_bytes()).hexdigest()
    result['all_arrays_exactly_equal']=all(v['elementwise_equal'] for file in result['arrays'].values() for v in file.values())
    result['all_discrete_decisions_equal']=result['selectors_equal'] and result['calibration_weights_equal'] and result['all_gate_retention_equal']
    result['reference_primary_gains_reproduced']=all(x['estimate_equal'] and x['gain_sum']==0 for x in result['primary_bounds'].values())
    result['passed']=result['all_arrays_exactly_equal'] and result['all_discrete_decisions_equal'] and result['reference_primary_gains_reproduced']
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--reference',type=Path,default=HERE/'results/news432')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=compare(args.run.resolve(),args.reference.resolve())
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as f:f.write(text)
    print(text)
    raise SystemExit(0 if result['passed'] else 1)
