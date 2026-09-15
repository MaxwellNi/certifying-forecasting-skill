"""Exact oracle and separate full recomputation; imports no simulation implementation."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
from itertools import permutations, product
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, rankdata

HERE = Path(__file__).resolve().parent
H = (0, 0, 0, 1)
Y = (0, 0, 1, 1)
B = (0, 1, 2, 3)
NAMES = ("baseline_identity", "candidate", "reverse_candidate", "constant")
MAPS = (B, H, tuple(-h for h in H), (0, 0, 0, 0))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def compare(a, b):
    return F((a>b)-(a<b), 2)


def literal(h, y, b, ix):
    def p(i,j):return compare(h[i],h[j])-compare(b[i],b[j])
    def q(i,j):return compare(y[i],y[j])-compare(b[i],b[j])
    i,j,k = ix
    return (p(i,j)*q(i,j), (p(i,j)-p(i,k))*(q(i,j)-q(i,k))/2,
            sum((p(a,c)*q(a,d) for a,c,d in permutations(ix)),F())/6)


def exact_oracle():
    def ranks(values):
        return [sum((compare(v,w) for w in values),F())/4 for v in values]
    rb,ry = ranks(B),ranks(Y)
    out=[]
    for name,h in zip(NAMES,MAPS):
        rh=ranks(h)
        left,right = [a-c for a,c in zip(rh,rb)],[a-c for a,c in zip(ry,rb)]
        theta = sum((a*c for a,c in zip(left,right)),F())/4
        ks=[literal(h,Y,B,ix) for ix in product(range(4),repeat=3)]
        expectations=[sum((k[j] for k in ks),F())/64 for j in range(3)]
        assert expectations[0]-expectations[1]==expectations[2]==theta
        for k in ks:
            assert F(-1,4)<=k[0]<=1 and F(-1,2)<=k[1]<=2 and F(-1,6)<=k[2]<=F(1,3)
        out.append({"candidate":name,"theta_exact":str(theta),"shared_exact":str(expectations[0]),
                    "interaction_exact":str(expectations[1]),"null":theta<=0,
                    "left_rank_contrast":[str(v) for v in left],"right_rank_contrast":[str(v) for v in right],
                    "kernel_minima":[str(min(k[j] for k in ks)) for j in range(3)],
                    "kernel_maxima":[str(max(k[j] for k in ks)) for j in range(3)]})
    assert [F(r["theta_exact"]) for r in out]==[F(0),F(0),F(1,32),F(1,64)]
    assert F(out[1]["shared_exact"])==F(out[1]["interaction_exact"])==F(1,32)
    assert H!=B and out[1]["kernel_minima"][2]!=out[1]["kernel_maxima"][2]
    # Independent pair description: S is positive only for baseline states {0,1}.
    pair_values=[]
    for i,j in product(range(4),repeat=2):
        s=(compare(H[i],H[j])-compare(B[i],B[j]))*(compare(Y[i],Y[j])-compare(B[i],B[j]))
        assert s==(F(1,4) if {i,j}=={0,1} else F(0))
        pair_values.append(str(s))
    return {"status":"PASS","law":"biased_zero_rank_contrast",
            "definition":"B uniform on {0,1,2,3}; h=1{B=3}; Y=1{B>=2}",
            "weighted_triple_cases":256,"nonidentity_zero_target_has_positive_interaction":True,
            "targets":out,"candidate_shared_by_ordered_pair":pair_values,
            "analysis_scope":"Rank-contrast null, not a conditional-independence test; all maps here are functions of B."}


def radius(values,width,alpha,rule,complete=False):
    n=len(values)
    if width==0:return 0.
    if rule=="range" or n==1:return width*math.sqrt(math.log(1/alpha)/(2*n))
    s2=float(np.sum((values-values.mean())**2)/(n-1))
    h=width*math.sqrt(math.log(2/alpha)/(2*n))
    if complete:
        x=math.log(2/alpha)
        v=math.sqrt(2*s2*x/n)+(2*width/math.sqrt(n*(n-1))+width/(3*n))*x
    else:
        x=math.log(4/alpha)
        v=math.sqrt(2*s2*x/n)+7*width*x/(3*(n-1))
    return min(h,v)


def triples(h,y,b):
    p,q={},{}
    for i,j in permutations(range(3),2):
        base=np.sign(b[:,i]-b[:,j])/2
        p[i,j]=np.sign(h[:,i]-h[:,j])/2-base
        q[i,j]=np.sign(y[:,i]-y[:,j])/2-base
    return ((p[0,1]-p[0,2])*(q[0,1]-q[0,2])/2,
            sum(p[i,j]*q[i,k] for i,j,k in permutations(range(3)))/6)


def complete(h,y,b):
    n=len(h)
    a=rankdata(h,method="average")-rankdata(b,method="average")
    c=rankdata(y,method="average")-rankdata(b,method="average")
    states,counts=np.unique(np.column_stack((h,y,b)),axis=0,return_counts=True)
    forbidden=0.
    for v,ni in zip(states,counts):
        for w,nj in zip(states,counts):
            base=np.sign(v[2]-w[2])/2
            forbidden+=ni*nj*(np.sign(v[0]-w[0])/2-base)*(np.sign(v[1]-w[1])/2-base)
    return float((a@c-forbidden)/(n*(n-1)*(n-2)))


def verify_results(directory):
    protocol=json.loads((HERE/"PROTOCOL.json").read_text())
    freeze=json.loads((HERE/"FROZEN_BEFORE_SIMULATION.json").read_text())
    for name,digest in freeze["inputs"].items():assert sha(HERE/name)==digest
    oracle=exact_oracle()
    assert json.loads((HERE/"EXACT_ORACLE.json").read_text())["oracle"]==oracle
    targets={r["candidate"]:F(r["theta_exact"]) for r in oracle["targets"]}
    rows=list(csv.DictReader((directory/"replications.csv").open()))
    family_rows=list(csv.DictReader((directory/"family_replications.csv").open()))
    summaries=list(csv.DictReader((directory/"summary.csv").open()))
    lookup={(int(r["budget"]),int(r["replication"]),r["candidate"],r["method"],r["rule"]):r for r in rows}
    assert len(lookup)==len(rows)==14400
    aggregate,family=defaultdict(list),defaultdict(list)
    errors={x:0. for x in ("center","radius","lower","theta")}
    raw_means,correction_means=defaultdict(list),defaultdict(list)
    for budget in protocol["budgets"]:
        plan=protocol["allocations"][str(budget)]
        m,n=plan["M"],plan["n"]
        assert 2*m+3*n==budget
        a,d=protocol["alpha_candidate"],protocol["delta"]
        for rep in range(protocol["replications"]):
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([protocol["seed"],0,budget,rep])))
            b=rng.choice(4,size=budget,p=[.25]*4)
            hv=(b==3).astype(int);y=(b>=2).astype(int)
            for name,h in zip(NAMES,(b,hv,-hv,np.zeros(budget,dtype=int))):
                hh,yy,bb=(x[:2*m].reshape(m,2) for x in (h,y,b))
                base=np.sign(bb[:,0]-bb[:,1])/2
                sv=(np.sign(hh[:,0]-hh[:,1])/2-base)*(np.sign(yy[:,0]-yy[:,1])/2-base)
                qv,_=triples(*(x[2*m:].reshape(n,3) for x in (h,y,b)))
                _,uv=triples(*(x.reshape(budget//3,3) for x in (h,y,b)))
                if name=="candidate":
                    raw_means[budget].append(float(sv.mean()))
                    correction_means[budget].append(float(qv.mean()))
                centers={"shared_correction":float(sv.mean()-qv.mean()),
                         "disjoint_triple_u":float(uv.mean()),"complete_u":complete(h,y,b)}
                ws,wq,wu=(0.,0.,0.) if name=="baseline_identity" else (1.25,2.5,.5)
                for method,center in centers.items():
                    for rule in protocol["rules"]:
                        rad=radius(sv,ws,a-d,rule)+radius(qv,wq,d,rule) if method=="shared_correction" else radius(uv,wu,a,rule,method=="complete_u")
                        lower=center-rad;theta=float(targets[name])
                        row=lookup[(budget,rep,name,method,rule)]
                        assert row["law"]==oracle["law"]
                        for key,value in (("center",center),("radius",rad),("lower",lower),("theta",theta)):
                            errors[key]=max(errors[key],abs(float(row[key])-value))
                            assert errors[key]<1e-13
                        decision,fail=lower>0,lower>theta+1e-14
                        assert decision==(row["certifies"]=="True") and fail==(row["noncoverage"]=="True")
                        aggregate[(budget,name,method,rule)].append((decision,fail,lower,rad))
                        family[(budget,rep,method,rule)].append((fail,decision and theta<=0))
        print(f"independently checked biased-null budget {budget}: 300 replications",flush=True)
    assert len(family_rows)==len(family)==3600
    for row in family_rows:
        values=family[(int(row["budget"]),int(row["replication"]),row["method"],row["rule"])]
        assert len(values)==4
        assert (row["family_noncoverage"]=="True")==any(x[0] for x in values)
        assert (row["false_certification"]=="True")==any(x[1] for x in values)
    counts=[]
    assert len(summaries)==len(aggregate)==48
    for row in summaries:
        key=(int(row["budget"]),row["candidate"],row["method"],row["rule"])
        values=aggregate[key]
        assert len(values)==int(row["replications"])==300
        k,f=sum(x[0] for x in values),sum(x[1] for x in values)
        families=[family[(key[0],rep,key[2],key[3])] for rep in range(300)]
        ff=sum(any(x[0] for x in v) for v in families)
        fw=sum(any(x[1] for x in v) for v in families)
        assert (k,f,ff,fw)==tuple(int(row[x]) for x in ("certifications","noncoverage","family_noncoverage","family_false_certification"))
        assert abs(float(row["certification_rate"])-k/300)<1e-13
        assert float(row["theta"])==float(targets[key[1]])
        assert abs(float(row["mean_lower"])-np.mean([x[2] for x in values]))<1e-13
        assert abs(float(row["mean_radius"])-np.mean([x[3] for x in values]))<1e-13
        for label,count in (("certification",k),("noncoverage",f),("family_noncoverage",ff),("family_false",fw)):
            lo=0. if count==0 else beta.ppf(.025,count,301-count)
            hi=1. if count==300 else beta.ppf(.975,count+1,300-count)
            assert abs(float(row[label+"_ci_low"])-lo)<1e-13
            assert abs(float(row[label+"_ci_high"])-hi)<1e-13
        counts.append({"budget":key[0],"candidate":key[1],"method":key[2],"rule":key[3],
                       "certifications":k,"noncoverage":f,"family_noncoverage":ff,"family_false_certification":fw})
    return {"status":"PASS","checked_at_utc":datetime.now(timezone.utc).isoformat(),
            "checker_sha256":sha(Path(__file__)),"exact_oracle":oracle,
            "independent_replications":600,"draw_positions":4608000,"candidate_rows":14400,"family_rows":3600,"summary_rows":48,
            "max_absolute_differences":errors,"counts":counts,
            "candidate_raw_shared_means":{str(k):float(np.mean(v)) for k,v in raw_means.items()},
            "candidate_interaction_means":{str(k):float(np.mean(v)) for k,v in correction_means.items()},
            "zero_of_300_pointwise_two_sided_95pct_upper":float(1-.025**(1/300)),
            "result_hashes":{p.name:sha(p) for p in directory.glob("*.csv")}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle",action="store_true")
    parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--output",type=Path,default=HERE/"INDEPENDENT_CHECK.json")
    args=parser.parse_args()
    if args.oracle:
        output=HERE/"EXACT_ORACLE.json"
        if output.exists():raise FileExistsError(output)
        emit(output,{"computed_at_utc":datetime.now(timezone.utc).isoformat(),
                     "source_sha256":sha(Path(__file__)),"oracle":exact_oracle(),"simulation_results_exist":(HERE/"results").exists()})
        print("Exact oracle passed; no simulation performed.")
    else:
        emit(args.output,verify_results(args.results))
        print("Independent full result check PASS.")
