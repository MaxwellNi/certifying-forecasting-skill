"""Generate reproducible synthetic inputs; this is a wiring demonstration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ['coupled', 'independent', 'constant']
SEED = 1309202642


def demo_frames(seed=SEED, training=512, validation_pairs=4096, groups=2, peers=3072):
    if training < 2 or validation_pairs < 2 or groups < 1 or peers < 3:
        raise ValueError('Need training>=2, validation_pairs>=2, groups>=1, peers>=3')
    rng = np.random.default_rng(seed)

    def draw(n):
        category = rng.integers(0, 2, size=n)
        probability = .2+.6*category
        outcome = rng.binomial(1, probability)
        return pd.DataFrame(dict(category=category.astype(str), outcome=outcome,
                                 forecast__coupled=outcome,
                                 forecast__independent=rng.binomial(1, probability),
                                 forecast__constant=np.zeros(n, dtype=int)))

    training_rows = draw(training)
    if set(training_rows.category) != {'0', '1'}:
        raise ValueError('This synthetic training draw missed a category; choose a larger declared training budget')

    def midranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts)-.5*counts)[inverse]/len(values)

    outcome_rank = midranks(training_rows.outcome)
    fits = []
    for model in MODELS:
        forecast_rank = midranks(training_rows['forecast__'+model])
        for category in ('0', '1'):
            mask = training_rows.category == category
            fits.append(dict(model=model, category=category, forecast_mean=float(forecast_rank[mask].mean()),
                             outcome_mean=float(outcome_rank[mask].mean())))
    # Two new raw draws per pair, with reference categories drawn marginally.
    validation = draw(2*validation_pairs)
    validation.insert(0, 'pair', np.repeat(np.arange(validation_pairs), 2))
    validation.insert(1, 'role', np.tile(['focal', 'reference'], validation_pairs))
    validation.insert(2, 'stream', np.repeat(np.where(np.arange(validation_pairs) < validation_pairs//2, 'A', 'B'), 2))
    evaluation = draw(groups*peers)
    evaluation.insert(0, 'group', np.repeat(np.arange(groups), peers))
    masses = pd.DataFrame(dict(category=['0', '1'], probability=[.5, .5]))
    return masses, pd.DataFrame(fits), validation, evaluation, training_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    frames = demo_frames(seed=args.seed)
    args.output.mkdir(parents=True)
    for name, frame in zip(('masses', 'fits', 'validation', 'evaluation', 'training'), frames):
        frame.to_csv(args.output/(name+'.csv'), index=False)
    (args.output/'design.json').write_text(json.dumps(dict(
        scope='Synthetic executable demonstration, not a new scientific validation or forecasting comparison',
        seed=args.seed, training_raw_calls=len(frames[-1]), validation_pairs=len(frames[2])//2,
        evaluation_raw_calls=len(frames[3]), category_probabilities=[.5, .5],
        mass_source='Exact two-category Bernoulli sampling design', exact_masses_declared=True,
        fixed_family=MODELS, fits='Conditional means of empirical marginal midranks from separate training draws',
        truth='Coupled has positive residual midrank association; independent and constant have zero target under the generating law',
        stages='One RNG stream supplies successive independent training, validation and evaluation draws; A/B labels are fixed by pair index',
    ), indent=2)+'\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
