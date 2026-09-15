"""Separately frozen biased-null addendum using byte-identical final study methods."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
METHOD_SOURCE_SHA="754132bc8a1b067e2f151c390af97191822df01eda23bc5f94d47348a5ca950d"
INPUTS=("study.py","experiment.py","independent_verify.py","EXACT_ORACLE.json","DERIVATION.md")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def emit(path,obj):path.write_text(json.dumps(obj,indent=2)+"\n")


def configured_methods():
    # No method execution occurs before the independently derived exact oracle.
    oracle=json.loads((HERE/"EXACT_ORACLE.json").read_text())
    assert oracle["oracle"]["status"]=="PASS" and not oracle["simulation_results_exist"]
    assert oracle["source_sha256"]==sha(HERE/"independent_verify.py")
    assert sha(HERE/"study.py")==METHOD_SOURCE_SHA
    spec=importlib.util.spec_from_file_location("frozen_final_methods",HERE/"study.py")
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    s.SEED=2026091501
    s.STATUS="Separately prespecified biased-null finite-law addendum; follows inspection of the original six-law study; not independent application validation."
    cells=[(0,0,0,F(1,4)),(0,0,1,F(1,4)),(0,1,2,F(1,4)),(1,1,3,F(1,4))]
    s.laws=lambda:[("biased_zero_rank_contrast",cells)]
    for observed,expected in zip(s.exact_law(cells),oracle["oracle"]["targets"]):
        assert observed["candidate"]==expected["candidate"]
        assert observed["theta_exact"]==expected["theta_exact"]
        assert observed["shared_expectation_exact"]==expected["shared_exact"]
        assert observed["interaction_exact"]==expected["interaction_exact"]
        assert observed["null"]==expected["null"]
    return s


def protocol(s):
    out=s.protocol()
    out["decision_policy"]="All two budget cells, four candidates, three methods and both rules are reported. Exactly 300 repetitions per budget. No Monte Carlo seed search, stopping, retuning, or success-driven additional runs."
    out["addendum"]={
        "chronology":"Constructed after inspecting the original six-law calibration; this addendum is frozen separately before its own simulation.",
        "law_selection":"Exact deterministic finite search: no uniform three-atom weak-order assignment had theta=0 and positive shared bias; four-atom enumeration supplied examples; selected the simple threshold construction B~Unif{0,1,2,3}, h=1{B=3}, Y=1{B>=2}. No Monte Carlo outcome was used to choose the law.",
        "gap_addressed":"Nonidentity candidate h has exact theta=0 and Gamma=E[S]=1/32>0.",
        "scope":"One analytically selected synthetic IID law. Two of four candidates are null; reverse and constant maps have positive rank-contrast targets. No new application or conditional-independence test.",
        "separation":"12 procedure cells and 600 replications; not part of the original 72 cells or original freeze.",
        "parent_protocol_sha256":"f2415d41fe46d72dcdf1ddd0b8d1f5ea64a743aa4be289396b353d87cb37a727",
        "method_implementation_sha256":METHOD_SOURCE_SHA,
        "frozen_input_sha256":{name:sha(HERE/name) for name in INPUTS},
        "supplementary_diagnostics":"Independent verifier also reports raw shared-score and interaction sample means for h; these are diagnostics, not extra decision procedures."
    }
    return out


def validate_frozen(s):
    frozen=json.loads((HERE/"PROTOCOL.json").read_text())
    if frozen!=protocol(s):raise ValueError("Frozen protocol, methods, or independent oracle changed")
    receipt=json.loads((HERE/"FROZEN_BEFORE_SIMULATION.json").read_text())
    for name,digest in receipt["inputs"].items():
        if sha(HERE/name)!=digest:raise ValueError("Frozen source digest mismatch: "+name)
    return frozen


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--verify",type=Path)
    args=parser.parse_args()
    s=configured_methods()
    if args.freeze:
        for name in ("PROTOCOL.json","FROZEN_BEFORE_SIMULATION.json","results"):
            if (HERE/name).exists():raise FileExistsError(HERE/name)
        emit(HERE/"PROTOCOL.json",protocol(s))
        emit(HERE/"FROZEN_BEFORE_SIMULATION.json",{
            "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
            "simulation_results_exist":False,
            "oracle_before_simulation":"Independent rational midrank and 256 weighted triple checks passed before importing the reused method code; reused exact targets match the oracle.",
            "inputs":{name:sha(HERE/name) for name in (*INPUTS,"PROTOCOL.json")},
            "scope":"Separate post-original-study addendum, frozen before generating or inspecting any of its simulation outcomes. Local receipt, not externally timestamped registration."})
        print("Separately frozen biased-null protocol",sha(HERE/"PROTOCOL.json"))
    else:
        frozen=validate_frozen(s)
        output=args.verify if args.verify else HERE/"results"
        if output.exists():raise FileExistsError(output)
        s.execute(output,frozen)
        if args.verify:
            expected=json.loads((HERE/"results/MANIFEST.json").read_text())["files"]
            actual={name:sha(output/name) for name in expected}
            emit(output/"REPLAY_CHECK.json",{"matches":actual==expected,"expected":expected,"actual":actual})
            if actual!=expected:raise AssertionError("Replay result digest mismatch")
