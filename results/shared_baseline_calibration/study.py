"""Frozen finite-law checks of sharp common-baseline lower bounds.

This is an implementation/calibration study on declared synthetic IID laws.
It is not independent application validation or a new statistical guarantee.
"""
import argparse
import csv
from fractions import Fraction as F
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
FAMILY = ("baseline_identity", "candidate", "reverse_candidate", "constant")
METHODS = ("shared_correction", "disjoint_triple_u", "complete_u")
RULES = ("range", "hybrid")
BUDGETS = (3072, 12288)
REPETITIONS = 300
ALPHA_FAMILY = .05
ALPHA = ALPHA_FAMILY / len(FAMILY)
DELTA = ALPHA / 2
SEED = 20260915
STATUS = "Prespecified finite-law implementation/calibration study; not independent application validation."


def emit(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def laws():
    uniform = {-1: F(1, 3), 0: F(1, 3), 1: F(1, 3)}
    rare = {-1: F(1, 100), 0: F(49, 50), 1: F(1, 100)}
    out = []
    for name, marginal, rho in [
        ("tied_null", uniform, F(0)),
        ("tied_weak", uniform, F(1, 10)),
        ("tied_positive", uniform, F(1)),
        ("near_degenerate_null", rare, F(0)),
        ("near_degenerate_positive", rare, F(1)),
    ]:
        cells = []
        for h, y in product(marginal, repeat=2):
            weight = (1-rho)*marginal[h]*marginal[y] + (rho*marginal[h] if h == y else 0)
            if weight:
                cells.append((h, y, 0, weight))
        out.append((name, cells))
    out.append(("informative_nonlinear_baseline", [(b*b, b*b, b, F(1, 3)) for b in (-1, 0, 1)]))
    return out


def compare(x, y):
    return .5 * ((np.asarray(x) > y).astype(float) - (np.asarray(x) < y).astype(float))


def candidate_maps(cells):
    h, y, b = np.asarray([r[:3] for r in cells], dtype=int).T
    return np.column_stack([b, h, -h, np.zeros(len(h), dtype=int)]), y, b


def exact_law(cells):
    """Compute exact targets directly from marginal midranks, without kernels."""
    hm, y, b = candidate_maps(cells)
    weights = [r[3] for r in cells]
    def midrank(values):
        return [sum((wj * (F(int(v > z)) + F(int(v == z), 2) - F(1,2))
                     for z, wj in zip(values, weights)), F()) for v in values]
    rb, ry = midrank(b), midrank(y)
    result = []
    for k, name in enumerate(FAMILY):
        rh = midrank(hm[:, k])
        theta = sum((w*(a-c)*(d-c) for w,a,c,d in zip(weights,rh,rb,ry)), F())
        shared = F()
        for i, wi in enumerate(weights):
            for j, wj in enumerate(weights):
                p = F(int(hm[i,k] > hm[j,k])-int(hm[i,k] < hm[j,k]),2) - F(int(b[i] > b[j])-int(b[i] < b[j]),2)
                q = F(int(y[i] > y[j])-int(y[i] < y[j]),2) - F(int(b[i] > b[j])-int(b[i] < b[j]),2)
                shared += wi*wj*p*q
        result.append({"candidate":name,"theta_exact":str(theta),"theta":float(theta),
                       "shared_expectation_exact":str(shared),"interaction_exact":str(shared-theta),
                       "null":theta <= 0})
    return result


def allocation(budget):
    # Range-only minimizer, independent of candidate outcomes or variances.
    a = 1.25 * math.sqrt(-math.log(ALPHA-DELTA)/2)
    b = 2.5 * math.sqrt(-math.log(DELTA)/2)
    radius, n, m = min((a/math.sqrt((budget-3*n)//2)+b/math.sqrt(n), n, (budget-3*n)//2)
                       for n in range(1,(budget-2)//3+1))
    return {"M":m,"n":n,"draws_used":2*m+3*n,"unused_budget":budget-2*m-3*n,"range_radius":radius}


def mean_radius(n, variance, width, alpha, rule):
    if width == 0: return 0.
    if rule == "range" or n == 1:
        return width * math.sqrt(-math.log(alpha)/(2*n))
    x = math.log(4/alpha)
    return min(width*math.sqrt(math.log(2/alpha)/(2*n)),
               math.sqrt(2*variance*x/n)+7*width*x/(3*(n-1)))


def pooled_radius(n, variance, width, alpha, rule):
    if width == 0: return 0.
    if rule == "range" or n == 1:
        return width*math.sqrt(-math.log(alpha)/(2*n))
    x = math.log(2/alpha)
    return min(width*math.sqrt(x/(2*n)), math.sqrt(2*variance*x/n)+
               (2*width/math.sqrt(n*(n-1))+width/(3*n))*x)


def kernels(cells):
    maps, y, b = candidate_maps(cells)
    base = compare(b[:,None], b[None,:])
    q = compare(y[:,None], y[None,:])-base
    return [(compare(maps[:,k,None],maps[None,:,k])-base,q) for k in range(len(FAMILY))]


def triple_values(p,q,stream):
    ix = stream[:3*(len(stream)//3)].reshape(-1,3)
    out = np.zeros(len(ix))
    for i,j,k in ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)):
        out += p[ix[:,i],ix[:,j]]*q[ix[:,i],ix[:,k]]/6
    return out


def complete_center(p,q,stream):
    # Exact distinct-index contraction over the empirical multiplicities.
    # Repeated state values remain separate IID draws, including their ties.
    counts = np.bincount(stream,minlength=len(p))
    n = len(stream)
    return float(counts @ ((p@counts)*(q@counts)-(p*q)@counts)/(n*(n-1)*(n-2)))


def evaluate(cells,stream,plan):
    out = []
    m,n = plan["M"],plan["n"]
    e = stream[:2*m].reshape(m,2)
    v = stream[2*m:2*m+3*n].reshape(n,3)
    for name,(p,q) in zip(FAMILY,kernels(cells)):
        identity = name == "baseline_identity"  # Declared functional map identity.
        ws,wq,wu = (0.,0.,0.) if identity else (1.25,2.5,.5)
        sv = p[e[:,0],e[:,1]]*q[e[:,0],e[:,1]]
        qv = .5*(p[v[:,0],v[:,1]]-p[v[:,0],v[:,2]])*(q[v[:,0],v[:,1]]-q[v[:,0],v[:,2]])
        uv = triple_values(p,q,stream)
        v_s,v_q,v_u = (float(np.var(x,ddof=1)) for x in (sv,qv,uv))
        centers = (float(sv.mean()-qv.mean()),float(uv.mean()),complete_center(p,q,stream))
        for method,center in zip(METHODS,centers):
            for rule in RULES:
                if method == "shared_correction":
                    radius = mean_radius(m,v_s,ws,ALPHA-DELTA,rule)+mean_radius(n,v_q,wq,DELTA,rule)
                elif method == "disjoint_triple_u":
                    radius = mean_radius(len(uv),v_u,wu,ALPHA,rule)
                else:
                    radius = pooled_radius(len(uv),v_u,wu,ALPHA,rule)
                out.append({"candidate":name,"method":method,"rule":rule,"center":center,"radius":radius,"lower":center-radius})
    return out


def interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def protocol():
    return {"status":STATUS,"seed":SEED,"bit_generator":"numpy.random.PCG64",
            "seed_sequence":["seed","law_index","budget","replication_index"],
            "family":list(FAMILY),"alpha_family":ALPHA_FAMILY,"alpha_candidate":ALPHA,"delta":DELTA,
            "methods":list(METHODS),"rules":list(RULES),"budgets":list(BUDGETS),"replications":REPETITIONS,
            "laws":[{"name":name,"cells":[{"h":h,"Y":y,"baseline":b,"probability":str(w)} for h,y,b,w in cells],
                     "exact_targets":exact_law(cells)} for name,cells in laws()],
            "allocations":{str(n):allocation(n) for n in BUDGETS},
            "widths":{"score":"5/4","correction":"5/2","triple_u":"1/2","declared_identity":"0"},
            "cost":{"common_iid_draw_stream":True,"all_observed_draws_available_to_complete_u":True,
                    "training_draws":0,"law_known_for_target_check_only":True,"candidate_family_size":4,
                    "target_or_law_not_used_in_bounds":True,"additional_support_specific_widths_not_used":True,"comparison_scope":"Verify the predeclared final common-baseline rule, not globally optimal widths for each known law","shared_allocation":"range-only integer optimum"},
            "endpoints":{"family_noncoverage":"any candidate lower bound exceeds its exact target by more than numerical tolerance 1e-14",
                         "false_certification":"any null candidate has lower > 0",
                         "power":"each positive candidate has lower > 0",
                         "precision":"pointwise two-sided 95% Clopper-Pearson intervals; not simultaneous over cells"},
            "decision_policy":"All 12 cells and every candidate, method and bound are reported. No stopping, seed search, rule tuning or extra success-driven runs.",
            "interpretation":"Monte Carlo estimates probe the frozen implementation; no finite simulation establishes the theorem or a physical-panel guarantee.",
            "source_sha256":sha(HERE/"study.py")}


def run():
    frozen = json.loads((HERE/"PROTOCOL.json").read_text())
    if frozen != protocol():
        raise ValueError("Source or protocol changed after freezing; run is prohibited")
    out = HERE/"results"
    if out.exists():
        raise FileExistsError("Do not overwrite a run; use --verify for deterministic replay into a separate path")
    return execute(out,frozen)


def execute(out,frozen):
    out.mkdir(parents=True)
    start=time.time()
    raw=[]
    families=[]
    for li,(law,cells) in enumerate(laws()):
        probabilities=np.asarray([float(r[3]) for r in cells])
        targets={r["candidate"]:r for r in exact_law(cells)}
        for budget in BUDGETS:
            plan=allocation(budget)
            for rep in range(REPETITIONS):
                stream=np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED,li,budget,rep]))).choice(len(cells),size=budget,p=probabilities)
                values=evaluate(cells,stream,plan)
                for r in values:
                    theta=targets[r["candidate"]]["theta"]
                    raw.append({"law":law,"budget":budget,"replication":rep,**r,"theta":theta,
                                "noncoverage":r["lower"]>theta+1e-14,"certifies":r["lower"]>0})
                for method in METHODS:
                    for rule in RULES:
                        group=[r for r in raw[-24:] if r["method"]==method and r["rule"]==rule]
                        families.append({"law":law,"budget":budget,"replication":rep,"method":method,"rule":rule,
                                         "family_noncoverage":any(r["noncoverage"] for r in group),
                                         "false_certification":any(r["certifies"] and r["theta"]<=0 for r in group)})
            print(f"completed {law}, budget {budget}, {REPETITIONS} repetitions",flush=True)
    def csvwrite(path,rows):
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvwrite(out/"replications.csv",raw)
    csvwrite(out/"family_replications.csv",families)
    cells=[]
    for law,_ in laws():
        for budget in BUDGETS:
            for method in METHODS:
                for rule in RULES:
                    fg=[r for r in families if (r["law"],r["budget"],r["method"],r["rule"])==(law,budget,method,rule)]
                    for name in FAMILY:
                        group=[r for r in raw if (r["law"],r["budget"],r["candidate"],r["method"],r["rule"])==(law,budget,name,method,rule)]
                        k=sum(r["certifies"] for r in group)
                        failures=sum(r["noncoverage"] for r in group)
                        fam_fail=sum(r["family_noncoverage"] for r in fg)
                        false=sum(r["false_certification"] for r in fg)
                        clo,chi=interval(k,len(group));flo,fhi=interval(failures,len(group))
                        falo,fahi=interval(fam_fail,len(fg));fwlo,fwhi=interval(false,len(fg))
                        cells.append({"law":law,"budget":budget,"candidate":name,"method":method,"rule":rule,"theta":group[0]["theta"],
                                      "replications":len(group),"certifications":k,"certification_rate":k/len(group),"certification_ci_low":clo,"certification_ci_high":chi,
                                      "noncoverage":failures,"noncoverage_ci_low":flo,"noncoverage_ci_high":fhi,
                                      "family_noncoverage":fam_fail,"family_noncoverage_ci_low":falo,"family_noncoverage_ci_high":fahi,
                                      "family_false_certification":false,"family_false_ci_low":fwlo,"family_false_ci_high":fwhi,
                                      "mean_lower":float(np.mean([r["lower"] for r in group])),"mean_radius":float(np.mean([r["radius"] for r in group]))})
    csvwrite(out/"summary.csv",cells)
    emit(out/"MANIFEST.json",{"status":STATUS,"protocol_sha256":sha(HERE/"PROTOCOL.json"),"source_sha256":sha(HERE/"study.py"),
                               "python":platform.python_version(),"numpy":np.__version__,"seconds":time.time()-start,
                               "law_budget_cells":len(laws())*len(BUDGETS),"independent_replications":len(laws())*len(BUDGETS)*REPETITIONS,
                               "raw_draw_positions":len(laws())*REPETITIONS*sum(BUDGETS),"candidate_method_rule_rows":len(raw),"family_rows":len(families),
                               "files":{p.name:sha(p) for p in out.glob("*.csv")},"zero_of_300_two_sided_95pct_upper":interval(0,REPETITIONS)[1]})
    return cells


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--verify",type=Path)
    args=parser.parse_args()
    if args.freeze:
        if (HERE/"PROTOCOL.json").exists():raise FileExistsError("A frozen protocol already exists")
        emit(HERE/"PROTOCOL.json",protocol())
        print("Frozen protocol:",sha(HERE/"PROTOCOL.json"))
    elif args.verify:
        if args.verify.exists():raise FileExistsError(args.verify)
        if json.loads((HERE/"PROTOCOL.json").read_text()) != protocol():raise ValueError("Frozen source mismatch")
        execute(args.verify,protocol())
        expected=json.loads((HERE/"results/MANIFEST.json").read_text())["files"]
        actual={name:sha(args.verify/name) for name in expected}
        emit(args.verify/"REPLAY_CHECK.json",{"matches":actual==expected,"expected":expected,"actual":actual})
        if actual!=expected:raise AssertionError("Replay CSV mismatch")
    else:run()
