"""Independent finite-law checks and declared-design null simulations."""
from fractions import Fraction as F
from itertools import product
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import beta
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import stratified_reference as method

def a(x,y):return F(int(y<x))+F(int(y==x),2)
def laws():
 for dependence in [-1,0,1]:
  states=[]
  for c,v,w in product(range(2),repeat=3):
   pc=F(1,5) if c==0 else F(4,5)
   # No empty cells at independence; perfect signs on alternatives.
   prob=pc*(F(1,4) if dependence==0 else F(int((v==w)==(dependence>0)),2))
   if prob:states.append(((c,v,w),prob))
  yield dependence,states

exact=[]
for dep,law in laws():
 rv={s:sum(prob*a(s[1],t[1]) for t,prob in law) for s,_ in law}
 rw={s:sum(prob*a(s[2],t[2]) for t,prob in law) for s,_ in law}
 for c in [0,1]:
  pc=sum(p for s,p in law if s[0]==c)
  cond=[(s,p/pc) for s,p in law if s[0]==c]
  meanv=sum(p*rv[s] for s,p in cond);meanw=sum(p*rw[s] for s,p in cond)
  truth=sum(p*(rv[s]-meanv)*(rw[s]-meanw) for s,p in cond)
  value=F(0)
  for (s,ps),(t,pt),(r,pr),(u,pu) in product(cond,cond,law,law):
   kernel=(a(s[1],r[1])-a(t[1],r[1]))*(a(s[2],u[2])-a(t[2],u[2]))/2
   assert -F(1,2)<=kernel<=F(1,2)
   value+=ps*pt*pr*pu*kernel
  assert value==truth
  exact.append({'dependence':dep,'category':c,'kernel_expectation':str(value),'covariance':str(truth)})

# Ties, safely represented large integers and missing observed support.
v=np.array([2**60,2**60+1,2**60+3,2**60+2]*20,dtype=np.int64)
w=np.array([1,2,1,3]*20)
c=np.array([0,0,1,1]*20)
p={0:.49,1:.49,2:.02}
base=method.prepare(v,w,c,p)
small=method.prepare(v-v.min(),w,c,p)
assert base['estimate']==small['estimate']
assert base['missing_mass']==.02
assert base['total_raw_rows_charged']==80
assert base['reference_rows_used']==40
assert base['pairs']==20
for rule in ['hoeffding','cell_bernstein','pooled_bernstein','hybrid']:
 grid=[method.lower_bound(base,a,rule) for a in [.001,.01,.05,.1,.5,.99]]
 assert np.all(np.diff(grid)>=-1e-14)
 pv=method.p_value(base,rule)
 assert 0<=pv<=1
 if pv<1 and pv>1e-12:
  assert method.lower_bound(base,pv*.99,rule)<=0
  assert method.lower_bound(base,min(.999,pv*1.01),rule)>=0
try:
 method.prepare([2**63+1,-2**63]*4,range(8),[0]*8,{0:1.})
except ValueError: pass
else:raise AssertionError('unsafe conversion was not rejected')

# Entire stratified method under fresh draws; these are simulation checks, not a new task.
rng=np.random.default_rng(91320091)
results=[]
for name,pvec in [('balanced',np.ones(4)/4),('rare_missing',np.array([.899,.1,.001])),('many_categories',np.ones(32)/32)]:
 for sign in [0,-1]:
  count={rule:0 for rule in ['hoeffding','pooled_bernstein','hybrid']}
  lower_max={rule:-1 for rule in count}
  reps=1200;n=512
  for rep in range(reps):
   cats=rng.choice(len(pvec),size=n,p=pvec)
   vv=rng.integers(0,2,n);ww=rng.integers(0,2,n) if sign==0 else 1-vv
   prepared=method.prepare(vv,ww,cats,dict(enumerate(pvec)))
   for rule in count:
    lower=method.lower_bound(prepared,.05,rule)
    count[rule]+=int(lower>0)
    lower_max[rule]=max(lower_max[rule],lower)
  for rule,number in count.items():
   upper=float(beta.ppf(.975,number+1,reps-number)) if number<reps else 1.
   results.append({'law':name,'target_sign':sign,'method':rule,'replications':reps,'n':n,'nominal_alpha':.05,'null_rejections':number,'pointwise_cp95_upper':upper,'max_lower':lower_max[rule]})
output={'scope':'Exact conditional covariance identities plus prespecified fresh null simulation draws; no claim of optimality, exhaustive validity, independent field application or unique power.','exact_identity_checks':exact,'structural_cases':['ties','large safe integer translation','unsafe mixed integer rejection','missing category','cost','monotone inversion'],'simulation':results,'source_sha256':hashlib.sha256((BASE/'stratified_reference.py').read_bytes()).hexdigest()}
(BASE/'checks/stratified_results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'identities':len(exact),'simulation_cells':len(results),'replications':7200,'max_null_rejections':max(r['null_rejections'] for r in results)},indent=2))
