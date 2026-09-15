"""Independent exact 512-array check; standard library, no project imports."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json


def rank(row, i):
    return sum(F(row[j] < row[i])+F(row[j] == row[i], 2)
               for j in range(3) if j != i)/2


def check():
    result = {key: F() for key in ("ranked_score", "population_score",
              "fit_product", "contemporaneous_rank_covariance", "rank_variance",
              "earlier_path", "later_path", "rank_cross_time_moment")}
    for flat in product((0, 1), repeat=9):
        y = [flat[3*t:3*(t+1)] for t in range(3)]
        x = [(0, 0, 0), y[0], y[1]]
        for i in range(3):
            q = [rank(row, i) for row in y]
            p = [rank(row, i) for row in x]
            f, g = (p[0]+p[2])/2, (q[0]+q[2])/2
            weight = F(1, 512*3)
            result["ranked_score"] += weight*(p[1]-f)*(q[1]-g)
            result["fit_product"] += weight*(F(1, 2)-f)*(F(1, 2)-g)
            result["contemporaneous_rank_covariance"] += weight*(p[1]-F(1, 2))*(q[1]-F(1, 2))
            r = [u-F(1, 2) for u in q]
            result["rank_variance"] += weight*r[1]**2
            result["earlier_path"] -= weight*r[0]**2/2
            result["later_path"] -= weight*r[1]**2/2
            result["rank_cross_time_moment"] += weight*r[0]*r[1]
            qq = [F(1, 4)+F(row[i], 2) for row in y]
            pp = [F(1, 2), qq[0], qq[1]]
            ff, gg = (pp[0]+pp[2])/2, (qq[0]+qq[2])/2
            result["population_score"] += weight*(pp[1]-ff)*(qq[1]-gg)
    assert result["ranked_score"] == -F(3, 32)
    assert result["population_score"] == -F(1, 16)
    assert result["rank_variance"] == F(3, 32)
    assert result["earlier_path"] == result["later_path"] == -F(3, 64)
    for key in ("fit_product", "contemporaneous_rank_covariance", "rank_cross_time_moment"):
        assert result[key] == 0
    # Direct focal/reference enumeration at middle time, separately from
    # within-panel score calculation. There are 16 binary ordered pairs.
    reuse = F()
    for v0, w0, v1, w1 in product((0, 1), repeat=4):
        av = F(v1 < v0)+F(v1 == v0, 2)
        bw = F(w1 < w0)+F(w1 == w0, 2)
        reuse += (av*bw-(F(1, 4)+F(v0, 2))*(F(1, 4)+F(w0, 2)))/16
    assert reuse == 0
    return {"status": "ALL_ASSERTIONS_PASSED", "arrays": 512,
            "score_averaging": "Equal average over 3 entities, then expectation over all 512 arrays",
            "moments_exact": {key: str(val) for key, val in result.items()},
            "contemporaneous_reuse_exact": str(reuse)}


if __name__ == "__main__":
    result = check()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        with args.output.open("x") as f:
            f.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
