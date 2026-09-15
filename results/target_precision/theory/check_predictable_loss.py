"""No-data mathematical checks for predictable_loss.py; no Monte Carlo coverage.

Checks direct squared losses, endpoint identities, missing/empty rules,
fixed-grid arithmetic, and a fully enumerated dependent finite probability tree.
"""

from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from predictable_loss import (
    DEFAULT_LAMBDA_GRID, hourly_gain_bounds, predictable_loss_bound,
)


def run():
    endpoint_cases = 0
    max_loss_identity_error = 0.0
    for a,b in product((0.0,.2,.5,.8,1.0),repeat=2):
        baseline_bounds = hourly_gain_bounds([a],[b],[float("nan")])
        assert baseline_bounds["gain"] == 0
        assert baseline_bounds["lower_endpoint"] <= 0 <= baseline_bounds["upper_endpoint"]
        for y in (0.0,.25,.5,.75,1.0):
            out = hourly_gain_bounds([a],[b],[y])
            direct = (y-b)**2-(y-a)**2
            error=abs(out["gain"]-direct)
            max_loss_identity_error=max(max_loss_identity_error,error)
            assert error < 5e-16
            assert out["lower_endpoint"]-1e-15 <= direct <= out["upper_endpoint"]+1e-15
            for field in ("lower_endpoint","upper_endpoint","predictable_width"):
                assert out[field] == baseline_bounds[field]
            assert abs(out["upper_endpoint"]-out["lower_endpoint"]-2*abs(a-b)) < 5e-16
            endpoint_cases += 1

    empty=hourly_gain_bounds([],[],[])
    assert all(value == 0 for value in empty.values())
    missing=hourly_gain_bounds([.1,.4,.7],[.9,.6,.3],[.2,float("nan"),float("inf")])
    direct_first=(.2-.9)**2-(.2-.1)**2
    assert abs(missing["gain"]-direct_first/3) < 1e-15
    assert missing["eligible_rows"] == 3
    assert missing["observed_targets"] == 1
    assert missing["missing_targets"] == 2
    complete=hourly_gain_bounds([.1,.4,.7],[.9,.6,.3],[.2,.2,.2])
    for field in ("lower_endpoint","upper_endpoint","predictable_width"):
        assert missing[field] == complete[field]

    # Independent scalar evaluation of the predeclared 32-point formula.
    widths=[.2]*1416
    gains=[.03]*1416
    out=predictable_loss_bound(gains,widths)
    scalar_grid=[.01*(1e7**(i/31)) for i in range(32)]
    scalar_radii=[(math.log(32/.025)/lam+lam*sum(w*w for w in widths)/8)/1416
                  for lam in scalar_grid]
    assert abs(out["radius"]-min(scalar_radii)) < 1e-14
    assert abs(out["lower"]-(.03-min(scalar_radii))) < 1e-14
    assert out["grid_size"] == 32
    assert out["hours"] == 1416
    assert out["lambda_grid"][0] == .01
    assert out["lambda_grid"][-1] == 100000.0

    zero=predictable_loss_bound([0.0]*1416,[0.0]*1416)
    assert zero["selected_lambda"] == 100000.0
    assert abs(zero["radius"]-math.log(32/.025)/(100000*1416)) < 1e-20

    # All 256 paths: current Bernoulli probability and interval width depend on
    # prior outcomes. Both are predictable; the sum of widths is random.
    hours=8
    audit_lambdas=(.01,.1,.8,2.0,10.0)
    mgf_expectations={str(lam):0.0 for lam in audit_lambdas}
    failure_probability=F()
    mass_sum=F()
    possible_width_sums=set()
    grid=(1.0,5.0,10.0)
    alpha=.2
    for path in product((0,1),repeat=hours):
        probability=F(1)
        past_ones=0
        realized=[]
        means=[]
        predictable=[]
        for bit in path:
            p=F(2+(past_ones%5),10)
            width=F(1+(past_ones%4),10)
            low=-width/4
            probability *= p if bit else 1-p
            realized.append(low+width*bit)
            means.append(low+width*p)
            predictable.append(width)
            past_ones += bit
        deviation=sum(realized,F())-sum(means,F())
        width_sum=sum((w*w for w in predictable),F())
        possible_width_sums.add(width_sum)
        for lam in audit_lambdas:
            exponent=lam*float(deviation)-lam*lam*float(width_sum)/8
            mgf_expectations[str(lam)] += float(probability)*math.exp(exponent)
        bound=predictable_loss_bound([float(x) for x in realized],
                                     [float(x) for x in predictable],
                                     alpha=alpha,lambdas=grid,expected_hours=hours)
        if bound["lower"] > float(sum(means,F())/hours):
            failure_probability += probability
        mass_sum += probability
    assert mass_sum == 1
    assert len(possible_width_sums) > 1
    assert all(value <= 1+1e-14 for value in mgf_expectations.values())
    assert failure_probability <= F(1,5)

    rejected=0
    invalid_calls=[
        lambda: hourly_gain_bounds([1.1],[.1],[.3]),
        lambda: hourly_gain_bounds([.1],[.2],[-.1]),
        lambda: hourly_gain_bounds([.1],[.2],[1.1]),
        lambda: hourly_gain_bounds([.1],[],[.2]),
        lambda: predictable_loss_bound([0.0],[.1]),
        lambda: predictable_loss_bound([0.0],[2.1],expected_hours=1),
        lambda: predictable_loss_bound([.2],[.1],expected_hours=1),
        lambda: predictable_loss_bound([0.0],[.1],expected_hours=1,lambdas=[0.0]),
        lambda: predictable_loss_bound([0.0],[.1],expected_hours=1,lambdas=[1.0,1.0]),
        lambda: predictable_loss_bound([0.0],[.1],expected_hours=1,alpha=0),
    ]
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            rejected += 1
    assert rejected == len(invalid_calls)
    report={
        "command":"python theory/check_predictable_loss.py",
        "all_checks_pass":True,
        "dataset_access":False,
        "endpoint_loss_cases":endpoint_cases,
        "maximum_direct_loss_identity_error":max_loss_identity_error,
        "empty_hour_check_pass":True,
        "missing_target_denominator_and_predictable_width_checks_pass":True,
        "scalar_grid_formula_check_pass":True,
        "fixed_grid":list(DEFAULT_LAMBDA_GRID),
        "dependent_tree":{
            "hours":hours,"paths_enumerated":2**hours,
            "distinct_predictable_width_square_sums":len(possible_width_sums),
            "fixed_lambda_exponential_expectations":mgf_expectations,
            "grid":list(grid),"alpha":alpha,
            "failure_probability_exact_rational":str(failure_probability),
            "failure_probability":float(failure_probability),
            "all_probability_mass_exact":str(mass_sum),
        },
        "rejected_invalid_inputs":rejected,
        "synthetic_formula_example":{key:out[key] for key in
                                     ("estimate","lower","radius","sum_predictable_width_squares","selected_lambda")},
        "limits":"Formula and finite-tree checks, not validation of historical availability, chronology or any real-data inferential assumptions. No Monte Carlo or new dataset access.",
    }
    output=Path(__file__).resolve().parent/"predictable_loss_validation.json"
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    run()
