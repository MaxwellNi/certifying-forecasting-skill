"""Known-mass split aggregate allowance, full-U evaluation, and complete-family BY.

This runnable example composes the existing public helpers. It does not infer
sampling independence or exact category probabilities from the input files.
See examples/aggregate_reference_family/README.md for the four CSV schemas.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PUBLIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC / 'scripts' / 'analysis'))
sys.path.insert(0, str(PUBLIC / 'results' / 'aggregate_bias'))
from aggregate_bias_bound import aggregate_upper_bound
from peer_rank_products import peer_rank_products
from reference_certificate_efficiency import family_validation_delta, one_sided_certificate, triple_scores


def _required(frame, columns, label):
    if frame.empty or not set(columns).issubset(frame.columns):
        raise ValueError(f'{label} requires nonempty columns {columns}')
    if frame[columns].isna().any().any():
        raise ValueError(f'{label} has missing required values')


def by_adjust(pvalues):
    """BY adjusted p-values in original family order, including every member."""
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('BY requires a nonempty finite p-value vector in [0,1]')
    k = len(p)
    order = np.argsort(p, kind='stable')
    adjusted = p[order] * k * math.fsum(1/j for j in range(1, k+1)) / np.arange(1, k+1)
    adjusted = np.minimum(1., np.minimum.accumulate(adjusted[::-1])[::-1])
    out = np.empty(k)
    out[order] = adjusted
    return out


def _exact_order_codes(values, label):
    """Encode a numeric channel without rounding away its order or ties.

    CSV strings are parsed as exact decimal values; Python/NumPy integer
    values stay exact. Existing binary64 values retain their supplied value.
    Codes are a common strictly increasing relabeling, used only to evaluate
    the same comparisons as the original observations.
    """
    parsed = []
    for value in values:
        try:
            if isinstance(value, (int, np.integer, bool, np.bool_)):
                number = Decimal(int(value))
            elif isinstance(value, (float, np.floating)):
                if isinstance(value, np.floating) and value.dtype.itemsize > 8:
                    raise ValueError('Supply wider-than-binary64 values as exact numeric strings')
                # Widening float32 is exact; Decimal.from_float also preserves
                # binary64 input exactly instead of rounding its printed form.
                number = Decimal.from_float(float(value))
            else:
                number = value if isinstance(value, Decimal) else Decimal(str(value).strip())
        except (InvalidOperation, ValueError, TypeError, OverflowError) as error:
            raise ValueError(f'{label} has an unsupported numeric value') from error
        if not number.is_finite():
            raise ValueError(f'{label} scores must be finite')
        parsed.append(number)
    levels = sorted(set(parsed))
    if len(levels) > 2**53:
        raise ValueError('Too many distinct values for exact comparison codes')
    lookup = {number: index for index, number in enumerate(levels)}
    return np.asarray([lookup[number] for number in parsed], dtype=np.int64)


def audit_family(masses, fits, validation, evaluation, *, training_calls,
                 mass_source, exact_masses_declared, family_level=.05, validation_fraction=.1,
                 sampling_bound='range', metadata_rows=0):
    """Audit the whole declared model family using one preselected construction.

    ``training_calls`` counts shared raw training draws once for this family.
    Every validation pair and every evaluation row is a separate draw/call;
    optional repeated population IDs are neither removed nor used as proof of
    dependence or independence. Streams and within-group input order must be
    fixed without looking at outcomes. Fits must be trained independently.
    ``exact_masses_declared`` must be the literal boolean True. This records
    the caller's exactness assertion, not a check of its truth; plug-in observed
    frequencies are insufficient without a separate mass-uncertainty argument.
    """
    if sampling_bound not in ('range', 'variance'):
        raise ValueError('sampling_bound must be range or variance')
    for value, name in ((training_calls, 'training_calls'), (metadata_rows, 'metadata_rows')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if exact_masses_declared is not True:
        raise ValueError('Explicitly declare exact_masses_declared=True; estimated or unknown category masses need a separate uncertainty budget')
    if not isinstance(mass_source, str) or not mass_source.strip():
        raise ValueError('Describe the source of the supplied exact category masses')
    masses, fits, validation, evaluation = [frame.copy() for frame in (masses, fits, validation, evaluation)]
    _required(masses, ['category', 'probability'], 'masses')
    _required(fits, ['model', 'category', 'forecast_mean', 'outcome_mean'], 'fits')
    for frame in (masses, fits):
        frame['category'] = frame.category.astype(str)
    fits['model'] = fits.model.astype(str)
    if masses.category.duplicated().any() or fits.duplicated(['model', 'category']).any():
        raise ValueError('Masses and each model require one entry per category')
    p = masses.probability.to_numpy(float)
    if not np.isfinite(p).all() or np.any(p < 0) or not math.isclose(float(p.sum()), 1., abs_tol=1e-12, rel_tol=0):
        raise ValueError('Supplied exact masses must be finite, nonnegative, and sum to one')
    models = list(fits.model.drop_duplicates())
    k = len(models)
    delta = family_validation_delta(k, q=family_level, rho=validation_fraction)
    harmonic = math.fsum(1/j for j in range(1, k+1))
    first_threshold = family_level/(k*harmonic)
    categories = {category: i for i, category in enumerate(masses.category)}
    columns = ['forecast__'+model for model in models]
    for frame, basic, name in (
        (validation, ['pair', 'role', 'stream', 'category', 'outcome'], 'validation'),
        (evaluation, ['group', 'category', 'outcome'], 'evaluation'),
    ):
        _required(frame, basic+columns, name)
        actual_forecasts = {c for c in frame.columns if c.startswith('forecast__')}
        if actual_forecasts != set(columns):
            raise ValueError(f'{name} forecast columns must match the complete declared family')
        frame['category'] = frame.category.astype(str)
        if not frame.category.isin(categories).all():
            raise ValueError(f'{name} contains an undeclared category')
        observed_codes = frame.category.map(categories).to_numpy(int)
        if np.any(p[observed_codes] == 0):
            raise ValueError(f'{name} observes a category declared to have exact probability zero')
    # One order encoding across both stages preserves every focal/reference
    # comparison and every evaluation comparison. It estimates no nuisance
    # and changes no score; it avoids binary64 creating ties in large integers
    # or finely separated decimal inputs before the existing rank helpers run.
    for column in ['outcome']+columns:
        encoded = _exact_order_codes(list(validation[column])+list(evaluation[column]), column)
        validation[column] = encoded[:len(validation)]
        evaluation[column] = encoded[len(validation):]
    if set(validation.role) != {'focal', 'reference'} or not set(validation.stream).issubset({'A', 'B'}):
        raise ValueError('Validation roles must be focal/reference and streams A/B')
    if validation.duplicated(['pair', 'role']).any() or not validation.groupby('pair', sort=False).size().eq(2).all():
        raise ValueError('Each validation pair must have exactly one focal and one reference row')
    if not validation.groupby('pair', sort=False).stream.nunique().eq(1).all():
        raise ValueError('The two members of a validation pair must have the same predeclared stream')
    focal = validation[validation.role == 'focal'].set_index('pair')
    reference = validation[validation.role == 'reference'].set_index('pair').reindex(focal.index)
    z = focal.category.map(categories).to_numpy(int)
    stream_a = focal.stream.to_numpy() == 'A'
    stream_b = ~stream_a
    n_a = np.bincount(z[stream_a], minlength=len(p))
    n_b = np.bincount(z[stream_b], minlength=len(p))
    sizes = evaluation.groupby('group', sort=False).size()
    if sizes.nunique() != 1 or sizes.iloc[0] < 3:
        raise ValueError('Evaluation groups need equal size N>=3')
    groups, peers = len(sizes), int(sizes.iloc[0])
    effective = groups*(peers//3)
    if sampling_bound == 'variance' and effective < 2:
        raise ValueError('The variance construction needs at least two disjoint triples')
    rows = []
    for model in models:
        model_fits = fits[fits.model == model].set_index('category')
        if set(model_fits.index) != set(categories):
            raise ValueError(f'Model {model} must declare both fits for every category')
        model_fits = model_fits.reindex(masses.category)
        f, g = (model_fits[c].to_numpy(float) for c in ('forecast_mean', 'outcome_mean'))
        if not np.isfinite(f).all() or not np.isfinite(g).all() or np.any((f < 0) | (f > 1) | (g < 0) | (g > 1)):
            raise ValueError('Independently supplied fits must lie in [0,1]')
        residuals = []
        for name, nuisance, mask, count in (
            ('forecast__'+model, f, stream_a, n_a), ('outcome', g, stream_b, n_b),
        ):
            left, right = focal[name].to_numpy(float), reference[name].to_numpy(float)
            comparison = (left > right).astype(float)+.5*(left == right)
            total = np.bincount(z[mask], weights=(comparison-nuisance[z])[mask], minlength=len(p))
            residuals.append(np.divide(total, count, out=np.zeros(len(p)), where=count > 0))
        corners = np.maximum(f*g, (1-f)*(1-g))
        bias = aggregate_upper_bound(p, *residuals, n_a, n_b, delta, absent_upper=corners)
        group_means, triples = [], []
        for _, group in evaluation.groupby('group', sort=False):
            c = group.category.map(categories).to_numpy(int)
            v, w = group['forecast__'+model].to_numpy(float), group.outcome.to_numpy(float)
            group_means.append(float(peer_rank_products(v, w, f[c], g[c])['corrected_residual_product'].mean()))
            if sampling_bound == 'variance':
                triples.append(triple_scores(v[None, :], w[None, :], f[c][None, :], g[c][None, :]))
        h = np.concatenate(triples) if triples else None
        certificate = one_sided_certificate(float(np.mean(group_means)), h, effective, f, g,
                                             bias['upper'], delta=delta, alpha=first_threshold,
                                             kind=sampling_bound)
        rows.append(dict(model=model, bias_center=bias['center'], bias_upper=bias['upper'],
                         bias_linear_a_radius=bias['linear_a_radius'], bias_linear_b_radius=bias['linear_b_radius'],
                         bias_product_radius=bias['product_radius'], absent_allowance=bias['absent_allowance'],
                         categories_used=bias['categories_used'], mean=certificate['mean'],
                         kernel_lower=certificate['kernel_lower'], kernel_upper=certificate['kernel_upper'],
                         sampling_radius_at_first_by=certificate['radius'],
                         lower_bound_at_first_by=certificate['lower_bound'], p=certificate['p']))
    adjusted = by_adjust([row['p'] for row in rows])
    for row, value in zip(rows, adjusted):
        row.update(by_adjusted_p=float(value), retained=bool(value <= family_level))
    receipt = dict(
        scope='Declared common-law marginal-midrank residual covariance; executable composition, not new forecasting evidence',
        models=models, family_size=k, family_level=family_level, family_harmonic=harmonic,
        first_by_threshold=first_threshold, validation_fraction=validation_fraction,
        per_candidate_validation_delta=delta, sampling_bound=sampling_bound,
        category_order=list(masses.category), exact_category_masses=p.tolist(), mass_source=mass_source,
        exact_masses_declared=exact_masses_declared,
        category_counts_a=n_a.tolist(), category_counts_b=n_b.tolist(),
        validation_pairs=len(focal), validation_raw_calls=len(validation),
        evaluation_groups=groups, peers_per_group=peers, effective_triples=effective,
        evaluation_raw_calls=len(evaluation), training_raw_calls=int(training_calls),
        total_raw_calls=int(training_calls)+len(validation)+len(evaluation),
        control_metadata_rows=int(metadata_rows), metadata_accounting='Reported separately; complete control metadata need not be new outcome calls',
        full_family_adjustment_applied=True, mean_uses_all_evaluation_rows=True,
        variance_partition='Consecutive disjoint triples in supplied within-group order' if sampling_bound == 'variance' else None,
        validation_partition='Supplied A/B stream labels fixed before values; A channel in A, B channel in B',
        sampling_assumptions_verified=False, exact_mass_provenance_verified=False,
        required_design='Fixed independent training; independent validation pairs; fresh iid rows in independent equal-size evaluation groups, all from one common law; reference categories remain unconditioned; declare family, splits, q, rho and sampling bound before evaluation',
        identifier_accounting='Counts raw draws/calls, not unique population IDs; repeated IDs are not deduplicated and do not establish or refute sampling independence',
        score_representation='Exact decimal/integer input order encoded jointly across validation and evaluation; all observed comparisons and ties are preserved before binary64 rank helpers',
        interpretation='BY uses all supplied models. Retention concerns the stated rank target; it does not establish raw-unit forecast gain. The first-threshold lower bound is not the final step-up decision.',
    )
    return {'results': rows, 'receipt': receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('masses', 'fits', 'validation', 'evaluation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--training-calls', type=int, required=True)
    parser.add_argument('--mass-source', required=True)
    parser.add_argument('--exact-category-masses', action='store_true', required=True,
                        help='Declare that supplied masses are exact under the target sampling law; this assertion is recorded, not verified')
    parser.add_argument('--metadata-rows', type=int, default=0)
    parser.add_argument('--family-level', type=float, default=.05)
    parser.add_argument('--validation-fraction', type=float, default=.1)
    parser.add_argument('--sampling-bound', choices=['range', 'variance'], default='range')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; existing results are not overwritten')
    paths = [args.masses, args.fits, args.validation, args.evaluation]
    # Parse raw numeric lexemes before pandas can round large/mixed integers.
    frames = [pd.read_csv(path, dtype=str) for path in paths]
    try:
        result = audit_family(*frames, training_calls=args.training_calls, mass_source=args.mass_source,
                              exact_masses_declared=args.exact_category_masses,
                              metadata_rows=args.metadata_rows, family_level=args.family_level,
                              validation_fraction=args.validation_fraction, sampling_bound=args.sampling_bound)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    pd.DataFrame(result['results']).to_csv(args.output/'results.csv', index=False)
    result['receipt']['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for name, path in zip(('masses', 'fits', 'validation', 'evaluation'), paths)}
    result['receipt']['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper_paths = ('results/aggregate_bias/aggregate_bias_bound.py',
                    'scripts/analysis/peer_rank_products.py',
                    'scripts/analysis/reference_certificate_efficiency.py',
                    'scripts/analysis/category_reference_certificate.py')
    result['receipt']['helper_sha256'] = {name: hashlib.sha256((PUBLIC/name).read_bytes()).hexdigest()
                                         for name in helper_paths}
    result['receipt']['results_sha256'] = hashlib.sha256((args.output/'results.csv').read_bytes()).hexdigest()
    (args.output/'receipt.json').write_text(json.dumps(result['receipt'], indent=2, allow_nan=False)+'\n')
    print(json.dumps({'models': len(result['results']), 'retained': sum(row['retained'] for row in result['results']),
                      'total_raw_calls': result['receipt']['total_raw_calls'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
