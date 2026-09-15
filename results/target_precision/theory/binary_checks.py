"""Independent exact type-sum, row-estimator and paired-score checks."""

from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

from binary_counts import (
    STATES, exact_u_counts, full_u_binary_counts, ordered_quadruple_oracle,
    exact_symmetrized_block,binary_block_values,binary_pair_values,full_u_binary_bound,
)
from full_u_variance import baseline,full_u_all_delete_one,full_u_empirical_bernstein,full_u_joint_bound

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratified_reference import prepare,lower_bound


def run():
    rng=np.random.default_rng(3900913)
    cases=0
    max_error=0.0
    for n in (5,6,10,30,100):
        for _ in range(10):
            states=rng.integers(0,4,size=n)
            counts=np.bincount(states,minlength=4)
            summary=full_u_binary_counts(counts)
            assert F(summary["estimate_exact"]) == ordered_quadruple_oracle(counts)
            v,w=states//2,states%2
            row=full_u_all_delete_one(v,w,[0]*n,{0:1.0})
            max_error=max(max_error,abs(summary["estimate"]-row["estimate"]))
            assert abs(summary["estimate"]-row["estimate"])<1e-13
            assert abs(summary["jackknife_variance"]-row["jackknife_variance"])<1e-13
            for i,s in enumerate(states):
                assert abs(summary["delete_one_by_state"][s]-row["delete_one"][i])<1e-13
            for rule,fn in (("eb",full_u_empirical_bernstein),("joint",full_u_joint_bound)):
                direct=fn(v,w,[0]*n,{0:1.0})
                count=full_u_binary_bound(summary,rule=rule)
                assert abs(direct["radius"]-count["radius"])<1e-12
            prep=prepare(v,w,[0]*n,{0:1.0})
            pair=binary_pair_values(states[:n//2])
            assert np.array_equal(pair,prep["cells"][0]["values"])
            cases+=1
    exact_blocks=0
    support=set()
    for states in product(range(4),repeat=4):
        value=exact_symmetrized_block(states)
        counts=np.bincount(states,minlength=4)
        assert value == ordered_quadruple_oracle(counts) == exact_u_counts(counts)
        lookup=binary_block_values(states)[0]
        v=[STATES[s][0] for s in states]
        w=[STATES[s][1] for s in states]
        direct=baseline.symmetrized_four_row_kernel(v,w,[0]*4,{0:1.0})
        assert abs(lookup-direct)<1e-15
        assert F(-1,12)<=value<=F(1,12)
        support.add(value)
        exact_blocks+=1
    large=[32400,0,0,368]
    summary=full_u_binary_counts(large)
    assert F(summary["estimate_exact"])==ordered_quadruple_oracle(large)
    # Verify the benchmark's accelerated one-cell interval and block counts
    # against the independently implemented generic row preparation/calibration.
    from binary_benchmark import _stratified_interval,_pair_counts,_block_counts,BLOCK_SUPPORT
    runner_cases=0
    for n in (8,20,100,1000):
        for _ in range(10):
            s=rng.integers(0,4,size=n)
            pre=prepare(s//2,s%2,[0]*n,{0:1.})
            pc=_pair_counts(s[:n//2])
            for rule,hybrid in (("hoeffding",False),("hybrid",True)):
                center,bound=_stratified_interval(pc,hybrid=hybrid)
                assert abs(bound-lower_bound(pre,.05,rule))<1e-13
            bc=_block_counts(s)
            assert abs(sum(float(x)*int(c) for x,c in zip(BLOCK_SUPPORT,bc))/sum(bc)-binary_block_values(s).mean())<1e-15
            runner_cases+=1
    result={"all_checks_pass":True,"row_cases":cases,"all_256_ordered_type_blocks":exact_blocks,
            "maximum_row_center_error":max_error,"stratified_pair_identity_pass":True,
            "bound_reduction_checks_pass":True,"large_count_exact_check_pass":True,
            "independent_runner_equivalence_cases":runner_cases,
            "binary_block_support_exact":[str(x) for x in sorted(support)],
            "seed":3900913,"new_dataset_access":False}
    Path(__file__).with_name("binary_checks.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    run()
