"""Independently recount saved coverage and certification decisions."""
import argparse
import csv
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def verify(result_dir):
    protocol=json.loads((HERE/"PROTOCOL.json").read_text())
    target={(law["name"],row["candidate"]):float(Fraction(row["theta_exact"]))
            for law in protocol["laws"] for row in law["exact_targets"]}
    groups=defaultdict(list)
    families=defaultdict(list)
    raw_path=result_dir/"replications.csv"
    rows=0
    for row in csv.DictReader(raw_path.open()):
        rows+=1
        key=(row["law"],int(row["budget"]),row["candidate"],row["method"],row["rule"])
        theta=target[(row["law"],row["candidate"])]
        center,radius,lower=map(float,(row["center"],row["radius"],row["lower"]))
        assert radius>=0 and math.isfinite(lower)
        assert lower==center-radius
        assert float(row["theta"])==theta
        failure=lower>theta+1e-14
        reject=lower>0
        assert (row["noncoverage"]=="True")==failure
        assert (row["certifies"]=="True")==reject
        if row["candidate"]=="baseline_identity":assert center==radius==lower==theta==0
        groups[key].append((int(row["replication"]),failure,reject,lower,radius))
        families[(key[0],key[1],int(row["replication"]),key[3],key[4])].append((failure,reject and theta<=0))
    summaries=list(csv.DictReader((result_dir/"summary.csv").open()))
    assert rows==86400 and len(groups)==len(summaries)==288 and len(families)==21600
    for row in summaries:
        key=(row["law"],int(row["budget"]),row["candidate"],row["method"],row["rule"])
        group=groups[key]
        assert sorted(r[0] for r in group)==list(range(300))
        assert int(row["noncoverage"])==sum(r[1] for r in group)
        assert int(row["certifications"])==sum(r[2] for r in group)
        fam=[families[(key[0],key[1],rep,key[3],key[4])] for rep in range(300)]
        assert all(len(r)==4 for r in fam)
        assert int(row["family_noncoverage"])==sum(any(t[0] for t in r) for r in fam)
        assert int(row["family_false_certification"])==sum(any(t[1] for t in r) for r in fam)
        assert math.isclose(float(row["mean_radius"]),math.fsum(r[4] for r in group)/300,abs_tol=1e-14)
        assert math.isclose(float(row["mean_lower"]),math.fsum(r[3] for r in group)/300,abs_tol=1e-14)
    family_rows=list(csv.DictReader((result_dir/"family_replications.csv").open()))
    assert len(family_rows)==21600
    for row in family_rows:
        group=families[(row["law"],int(row["budget"]),int(row["replication"]),row["method"],row["rule"])]
        assert (row["family_noncoverage"]=="True")==any(r[0] for r in group)
        assert (row["false_certification"]=="True")==any(r[1] for r in group)
    manifest=json.loads((result_dir/"MANIFEST.json").read_text())
    for name,expected in manifest["files"].items():
        assert hashlib.sha256((result_dir/name).read_bytes()).hexdigest()==expected
    return {"status":"PASS","replication_rows":rows,"candidate_summary_rows":len(groups),
            "family_rows":len(family_rows),"targets":"exact rational protocol targets",
            "identity_rows_zero":True,"all_decisions_recounted":True,"all_summary_counts_reconstructed":True,
            "numeric_tolerance_for_noncoverage":1e-14,
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--output",type=Path,default=HERE/"CHECK_RESULTS.json")
    args=parser.parse_args()
    result=verify(args.results)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
