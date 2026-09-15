"""Independent-reference, within-category pair estimator of rank covariance.

Classical stratified covariance identities and bounded-mean concentration.
No claim of priority or universal dominance. Exact category probabilities and
independent raw rows are required. See STRATIFIED_THEORY.md for the contract.
"""
import math
from typing import Mapping

import numpy as np


def prepare(v, w, category, probabilities: Mapping):
    original_v, original_w = np.asarray(v, dtype=object), np.asarray(w, dtype=object)
    v, w = np.asarray(v), np.asarray(w)
    c = list(category)
    n = len(c)
    if v.ndim != 1 or w.ndim != 1 or len(v) != n or len(w) != n or n < 4:
        raise ValueError('need at least four equally sized one-dimensional raw-row arrays')
    if v.dtype.kind not in 'biuf' or w.dtype.kind not in 'biuf':
        raise ValueError('use finite real arrays with a safe exact ordering representation')
    if not np.isfinite(v).all() or not np.isfinite(w).all():
        raise ValueError('values must be finite')
    # Refuse silent integer-to-float coercion when lists combine incompatible dtypes.
    for original, converted in [(original_v, v), (original_w, w)]:
        if converted.dtype.kind == 'f':
            for before, after in zip(original.flat, converted.flat):
                if isinstance(before, (int, np.integer)) and int(after) != int(before):
                    raise ValueError('integer input loses exact order during conversion')
    keys = list(probabilities)
    p = np.array([probabilities[k] for k in keys], dtype=float)
    if len(p) == 0 or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('complete positive category masses are required')
    if not math.isclose(math.fsum(p), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('category masses must sum to one')
    mapping = {k: j for j, k in enumerate(keys)}
    try:
        labels = np.array([mapping[k] for k in c], dtype=int)
    except KeyError as exc:
        raise ValueError('observed category absent from positive mass support') from exc
    focal_count = n // 2
    cursor = focal_count
    cells = []
    for j, key in enumerate(keys):
        positions = np.flatnonzero(labels[:focal_count] == j)
        count = len(positions) // 2
        positions = positions[:2*count]
        first, second = positions[::2], positions[1::2]
        rv = np.arange(cursor, cursor+2*count, 2)
        rw = rv+1
        cursor += 2*count
        assert cursor <= n
        av1 = (v[rv] < v[first]).astype(float) + .5*(v[rv] == v[first])
        av2 = (v[rv] < v[second]).astype(float) + .5*(v[rv] == v[second])
        bw1 = (w[rw] < w[first]).astype(float) + .5*(w[rw] == w[first])
        bw2 = (w[rw] < w[second]).astype(float) + .5*(w[rw] == w[second])
        z = .5*(av1-av2)*(bw1-bw2)
        cells.append({'category': key, 'mass': float(p[j]), 'focal_count': int((labels[:focal_count] == j).sum()),
                      'pairs': count, 'values': z, 'mean': float(z.mean()) if count else 0.,
                      'variance': float(z.var(ddof=1)) if count > 1 else 0.})
    used = [cell for cell in cells if cell['pairs']]
    missing_mass = math.fsum(cell['mass'] for cell in cells if not cell['pairs'])
    center = math.fsum(cell['mass']*cell['mean'] for cell in used)
    k = math.fsum(cell['mass']**2/cell['pairs'] for cell in used)
    return {'n': n, 'focal_rows': focal_count, 'available_reference_rows': n-focal_count,
            'reference_rows_used': cursor-focal_count, 'pairs': sum(cell['pairs'] for cell in cells),
            'total_raw_rows_charged': n, 'unused_focal_rows': focal_count-2*sum(cell['pairs'] for cell in cells),
            'unused_reference_rows': n-cursor, 'cells': cells, 'estimate': center,
            'missing_mass': missing_mass, 'allocation_variance_factor': k}


def lower_bound(prepared, alpha, rule='hoeffding'):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one')
    h = .25*prepared['missing_mass']
    if rule == 'hoeffding':
        radius = math.sqrt(.5*prepared['allocation_variance_factor']*(-math.log(alpha)))
        return max(-.25, min(.25, prepared['estimate']-h-radius))
    if rule == 'hybrid':
        return max(lower_bound(prepared, alpha/2, 'hoeffding'),
                   lower_bound(prepared, alpha/2, 'pooled_bernstein'))
    if rule == 'pooled_bernstein':
        used = [c for c in prepared['cells'] if c['pairs']]
        count = prepared['pairs']
        if count < 2:
            return -.25
        weighted = np.concatenate([c['mass']/c['pairs']*c['values'] for c in used])
        maximum_weight = max(c['mass']/c['pairs'] for c in used)
        log_term = math.log(2)-math.log(alpha)
        radius = math.sqrt(2*count*float(weighted.var(ddof=1))*log_term) + 7*count*maximum_weight*log_term/(3*(count-1))
        return max(-.25, min(.25, prepared['estimate']-h-radius))
    if rule != 'cell_bernstein':
        raise ValueError('unknown prespecified rule')
    observed_mass = 1-prepared['missing_mass']
    if observed_mass <= 0:
        return -.25
    lower = -h
    for cell in prepared['cells']:
        q = cell['pairs']
        if q == 0:
            continue
        log_inv_cell_alpha = -math.log(alpha)-math.log(cell['mass'])+math.log(observed_mass)
        # Both candidate bounds have half of the cell's conditional error allocation.
        if q > 1:
            log_h = math.log(2)+log_inv_cell_alpha
            log_b = math.log(4)+log_inv_cell_alpha
            rh = math.sqrt(log_h/(2*q))
            rb = math.sqrt(2*cell['variance']*log_b/q)+7*log_b/(3*(q-1))
            radius = min(rh, rb)
        else:
            radius = math.sqrt(log_inv_cell_alpha/2)
        cell_lower = max(-.25, min(.25, cell['mean']-radius))
        lower += cell['mass']*cell_lower
    return lower


def p_value(prepared, rule='hoeffding'):
    """Valid fixed-sample one-sided p value by inversion of nested lower bounds."""
    lo, hi = 0., 1.
    # alpha=1 excluded; a nonpositive limiting bound means p=1.
    if lower_bound(prepared, np.nextafter(1., 0.), rule) <= 0:
        return 1.
    for _ in range(64):
        mid = (lo+hi)/2
        if lower_bound(prepared, mid, rule) > 0:
            hi = mid
        else:
            lo = mid
    return hi


def power_requirement(prepared, alpha, eta, effect):
    if not 0 < alpha < 1 or not 0 < eta < 1 or effect <= 0:
        raise ValueError('require error probabilities in (0,1) and positive effect')
    threshold = .5*prepared['missing_mass'] + math.sqrt(.5*prepared['allocation_variance_factor'])*(math.sqrt(-math.log(alpha))+math.sqrt(-math.log(eta)))
    return {'conditional_power_at_least': 1-eta, 'effect_required_strictly_above': threshold,
            'specified_effect': effect, 'sufficient_condition_met': bool(effect > threshold)}
