"""Check algebraic inversion against independently implemented bound calls."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE),str(BASE/'theory')]
from study_gates import full_joint_from_summary,by_adjust
from full_u_variance import full_u_all_delete_one,full_u_joint_bound
rng=np.random.default_rng(130920269)
errors=[];cases=0
for n in [6,8,12,19]:
 for k in [1,3]:
  for rep in range(3):
   v=rng.integers(0,4,n);w=rng.integers(0,5,n);c=rng.integers(k,size=n);p=dict(enumerate([1/k]*k))
   summary=full_u_all_delete_one(v,w,c,p)
   for alpha in [.001,.05,.4,.9]:
    fast=full_joint_from_summary(summary,p,alpha)
    full=full_u_joint_bound(v,w,c,p,alpha)
    errors.append(abs(fast['lower']-full['lower']))
    assert abs(fast['lower']-full['lower'])<1e-12
    assert (fast['p']<alpha)==(fast['lower']>0)
    cases+=1
for p in [np.array([.001,.012,.7,.001,.9,1]),np.array([1,1,1]),np.array([0,1e-8,.001,.006])]:
 out=by_adjust(p);n=len(p);h=math.fsum(1/i for i in range(1,n+1));order=np.argsort(p)
 for rank,i in enumerate(order,1):
  literal=min(1.,min(p[order[j-1]]*n*h/j for j in range(rank,n+1)))
  assert abs(out[i]-literal)<1e-15
rec={'bound_cases':cases,'max_bound_error':max(errors),'BY_literal_families':3,'scope':'Independent formula implementation agreement, not scientific performance evidence','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gate_sha256':hashlib.sha256((BASE/'study_gates.py').read_bytes()).hexdigest()}
(BASE/'checks/gate_composition.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec,indent=2))
