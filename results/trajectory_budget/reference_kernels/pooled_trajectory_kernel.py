"""Complete degree-three rank U center using every available trajectory.

This separate module extends the comparison without changing the originally
frozen block-kernel module. Integer map values retain exact ordering and ties.
Cost O((S+T+ST)*N*log(N)) time and O((S+T)*N) storage. Computations are ordinary
floating point after exact rank ordering, not interval-certified arithmetic.
"""

import math
import numpy as np

from trajectory_kernel import _design, _map_array, full_u_scores


def _rank_codes_and_centered_sums(column):
    _, codes, counts = np.unique(column, return_inverse=True, return_counts=True)
    # Twice sum_j centered_midcompare(x_i,x_j), including its zero diagonal.
    twice_by_code = 2 * np.cumsum(counts, dtype=np.int64) - counts - column.size
    return codes, counts.size, twice_by_code[codes].astype(float) * 0.5


def _concordance_minus_discordance(x_codes, y_codes, y_cardinality):
    """Exact unordered-pair signed count; either-coordinate ties contribute zero."""
    n = len(x_codes)
    if n < 2 or y_cardinality < 2 or np.all(x_codes == x_codes[0]):
        return 0
    order = np.argsort(x_codes, kind="stable")
    ordered_x = x_codes[order]
    boundaries = np.r_[0, np.flatnonzero(ordered_x[1:] != ordered_x[:-1]) + 1, n]
    tree = [0] * (int(y_cardinality) + 1)

    def prefix(count):
        total = 0
        while count:
            total += tree[count]
            count -= count & -count
        return total

    seen, signed_count = 0, 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = order[start:end]
        # Query the whole tied-x group before inserting any of its observations.
        for index in group:
            y = int(y_codes[index])
            below, at_or_below = prefix(y), prefix(y + 1)
            signed_count += below - (seen - at_or_below)
        for index in group:
            node = int(y_codes[index]) + 1
            while node <= y_cardinality:
                tree[node] += 1
                node += node & -node
        seen += int(end - start)
    return signed_count


def pooled_u_details(f_maps, g_maps, C, D):
    """Input maps have shape (N,S)/(N,T), N>=3; return center and rank diagnostics."""
    f, g = np.asarray(f_maps), np.asarray(g_maps)
    if f.ndim != 2 or g.ndim != 2:
        raise ValueError("pooled maps require shape (trajectories, maps)")
    f, g = _map_array(f[None], "f_maps")[0], _map_array(g[None], "g_maps")[0]
    C, D, _, c, d, _, _, width_focal = _design(C,D)
    n = f.shape[0]
    if n < 3 or g.shape[0] != n or f.shape[1] != len(c) or g.shape[1] != len(d):
        raise ValueError("map rows must match at N>=3 and map columns match design rows")
    fs = [_rank_codes_and_centered_sums(f[:,s]) for s in range(len(c))]
    gs = [_rank_codes_and_centered_sums(g[:,t]) for t in range(len(d))]
    a = sum((float(c[s]) * fs[s][2] for s in range(len(c))), np.zeros(n))
    b = sum((float(d[t]) * gs[t][2] for t in range(len(d))), np.zeros(n))
    all_reference_product_sum = math.fsum(float(x) * float(y) for x,y in zip(a,b))
    concordances, same_reference_terms = [], []
    for s, weight_s in enumerate(c):
        row = []
        for t, weight_t in enumerate(d):
            signed = _concordance_minus_discordance(fs[s][0], gs[t][0], gs[t][1])
            row.append(signed)
            same_reference_terms.append(0.5 * float(weight_s) * float(weight_t) * signed)
        concordances.append(row)
    same_reference_product_sum = math.fsum(same_reference_terms)
    denominator = n * (n - 1) * (n - 2)
    score = (all_reference_product_sum - same_reference_product_sum) / denominator
    if not math.isfinite(score):
        raise ValueError("floating coefficient contractions overflow")
    return {
        "score": score, "n": n, "full_u_width": width_focal / 3.0,
        "all_reference_product_sum": all_reference_product_sum,
        "same_reference_product_sum": same_reference_product_sum,
        "concordance_minus_discordance": concordances,
        "ordered_distinct_triples": denominator,
        "f_unique_values": [int(x[1]) for x in fs],
        "g_unique_values": [int(x[1]) for x in gs],
    }


def pooled_u_score(f_maps, g_maps, C, D):
    """Scalar convenience API for the complete all-trajectory U-statistic center."""
    return pooled_u_details(f_maps,g_maps,C,D)["score"]


def pooled_u_hoeffding_lower(f_maps, g_maps, C, D, alpha=0.05):
    """Classical U-statistic Hoeffding bound, effective floor(N/3) observations.

    Symmetrize the exponential moment over disjoint triples in every permutation
    (Hoeffding 1963, Section 5a). Do not use N choose 3 as the independent count.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    out = pooled_u_details(f_maps,g_maps,C,D)
    independent_triples = out['n'] // 3
    radius = out['full_u_width'] * math.sqrt(-math.log(alpha) / (2 * independent_triples))
    out.update({"lower":out['score']-radius, "radius":radius, "alpha":alpha,
                "independent_triples":independent_triples, "rule":"pooled_u_hoeffding"})
    return out


def pooled_u_joint_lower(f_maps, g_maps, C, D, alpha=0.05):
    """Variance-aware complete-U bound using prespecified disjoint triple variance.

    Complete pooled center and variance estimates may use the same raw draws.
    The pooled trajectories must be conditionally IID, with maps/design fixed.
    Classical permutation/Jensen MGF plus Maurer--Pontil (2009), Theorem 10.
    See POOLED_U_PROOF.md. This is not a claim of a new concentration inequality.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    out = pooled_u_details(f_maps,g_maps,C,D)
    q, width = out['n'] // 3, out['full_u_width']
    if q < 2:
        radius = width * math.sqrt(-math.log(alpha)/(2*q))
        variance, h_radius, v_radius = 0.0, radius, None
    else:
        f, g = np.asarray(f_maps), np.asarray(g_maps)
        # This partition is index-only; it is not selected using observed maps.
        blocks = full_u_scores(f[:3*q].reshape(q,3,f.shape[1]),
                               g[:3*q].reshape(q,3,g.shape[1]),C,D)
        mean = math.fsum(float(x)/q for x in blocks)
        variance = math.fsum((float(x)-mean)**2/(q-1) for x in blocks)
        x = math.log(2.0)-math.log(alpha)
        h_radius = width*math.sqrt(x/(2*q))
        v_radius = math.sqrt(variance)*math.sqrt(2*x/q) + (
            2*width/math.sqrt(q*(q-1)) + width/(3*q)
        )*x
        radius = min(h_radius,v_radius)
    out.update({"lower":out['score']-radius, "radius":radius, "alpha":alpha,
                "independent_triples":q, "triple_sample_variance":variance,
                "hoeffding_radius":h_radius, "variance_radius":v_radius,
                "variance_draws":3*q, "rule":"pooled_u_joint"})
    return out
