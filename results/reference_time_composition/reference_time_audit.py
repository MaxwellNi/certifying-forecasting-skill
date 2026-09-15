"""A finite bound for fixed rank designs on independent source panels.

The target replaces each peer comparison by its population midrank before
applying the same coefficients. It is not a guarantee for dependent panels,
data-selected coefficients, or future forecasting loss.
"""
import numpy as np


def comparison(x, y):
    return (np.asarray(x) > np.asarray(y)).astype(float) + 0.5 * (np.asarray(x) == np.asarray(y))


def coefficients(c, d):
    c, d = np.asarray(c, dtype=float), np.asarray(d, dtype=float)
    if c.ndim != 2 or c.shape != d.shape or min(c.shape) < 1:
        raise ValueError('Coefficient matrices must have the same nonempty time-by-peer shape.')
    if not (np.isfinite(c).all() and np.isfinite(d).all()):
        raise ValueError('Coefficients must be finite and fixed independently of these data.')
    return c, d


def panel_scores(panels, c, d):
    c, d = coefficients(c, d)
    p = np.asarray(panels)
    if p.ndim != 3 or p.shape[0] < 1 or p.shape[1:] != (c.shape[0], c.shape[1]+1):
        raise ValueError('Panels must have shape (groups, source times, focal plus peers).')
    if p.dtype.kind not in 'biuf' or not np.isfinite(p).all():
        raise ValueError('Panel observations must be finite real numbers.')
    z = comparison(p[:, :, :1], p[:, :, 1:])-0.5
    a = np.einsum('gtj,tj->g', z, c)
    b = np.einsum('gtj,tj->g', z, d)
    return a*b


def audit(panels, c, d, validation_triples, alpha=0.05, delta=0.01):
    """Return a lower bound under the stated common-law independence design.

    All source observations across times/entities/groups are independent with
    the same law. Validation triples are fresh independent draws from that law,
    independent of the panels. These assumptions cannot be inferred from arrays.
    The coefficient matrices and alpha/delta must be fixed before evaluation.
    """
    c, d = coefficients(c, d)
    if not 0 < delta < alpha < 1:
        raise ValueError('Require 0 < delta < alpha < 1.')
    scores = panel_scores(panels, c, d)
    triples = np.asarray(validation_triples)
    if triples.ndim != 2 or triples.shape[1] != 3 or len(triples) < 1 or triples.dtype.kind not in 'biuf' or not np.isfinite(triples).all():
        raise ValueError('Validation requires a nonempty finite array with three columns.')
    t = 0.5*(comparison(triples[:, 0], triples[:, 1])-comparison(triples[:, 0], triples[:, 2]))**2
    omega_hat = float(t.mean())
    epsilon = 0.5*np.sqrt(np.log(2/delta)/(2*len(triples)))
    # [0, 1/2] uses the kernel range; no sharper distributional formula is needed.
    omega_lo, omega_hi = max(0.0, omega_hat-epsilon), min(0.5, omega_hat+epsilon)
    overlap = float(np.sum(c*d))
    upper_bias = overlap*(omega_hi if overlap >= 0 else omega_lo)
    width = float(0.5*np.abs(c).sum()*np.abs(d).sum())
    radius = float(width*np.sqrt(np.log(1/(alpha-delta))/(2*len(scores))))
    lower = float(scores.mean()-upper_bias-radius)
    return {
        'target':'Population-reference product after the same fixed linear coefficients',
        'guarantee_scope':'Independent common-law source observations, fixed coefficients, independent validation triples',
        'groups':len(scores), 'validation_triples':len(triples),
        'source_observation_count':int(np.asarray(panels).size),
        'validation_observation_count':int(triples.size),
        'total_observation_count':int(np.asarray(panels).size+triples.size),
        'alpha':alpha, 'validation_failure_budget':delta,
        'overlap_coefficient':overlap, 'omega_hat':omega_hat,
        'omega_interval':[float(omega_lo), float(omega_hi)],
        'upper_interaction_allowance':float(upper_bias),
        'score_mean':float(scores.mean()), 'score_range_width':width,
        'sampling_radius':radius, 'lower_bound':lower,
        'positive_lower_bound':bool(lower > 0),
    }
