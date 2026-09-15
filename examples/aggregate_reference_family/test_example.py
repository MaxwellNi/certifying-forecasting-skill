"""Focused wiring checks against direct triple enumeration and existing bounds."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from aggregate_reference_family import audit_family, by_adjust
from generate_demo import demo_frames
import numpy as np
import pandas as pd


class AggregateFamilyExampleTests(unittest.TestCase):
    def setUp(self):
        self.frames = demo_frames(training=80, validation_pairs=24, groups=2, peers=7)[:4]
        self.options = dict(training_calls=80, mass_source='Exact synthetic p=(.5,.5)', exact_masses_declared=True)

    def test_full_u_matches_brute_ordered_triples_and_cost_counts_leftovers(self):
        masses, fits, validation, evaluation = self.frames
        result = audit_family(*self.frames, **self.options)
        for row in result['results']:
            model = row['model']
            fitted = fits[fits.model == model].set_index('category')
            means = []
            for _, group in evaluation.groupby('group', sort=False):
                values = group.reset_index(drop=True)
                products = []
                for i, j, k in itertools.permutations(range(len(values)), 3):
                    ci = values.iloc[i].category
                    v = values['forecast__'+model]
                    w = values.outcome
                    av = float(v.iloc[i] > v.iloc[j])+.5*(v.iloc[i] == v.iloc[j])
                    bw = float(w.iloc[i] > w.iloc[k])+.5*(w.iloc[i] == w.iloc[k])
                    products.append((av-fitted.loc[ci, 'forecast_mean'])*(bw-fitted.loc[ci, 'outcome_mean']))
                means.append(np.mean(products))
            self.assertAlmostEqual(row['mean'], float(np.mean(means)), places=13)
        receipt = result['receipt']
        self.assertEqual(receipt['effective_triples'], 4)
        self.assertEqual(receipt['evaluation_raw_calls'], 14)
        self.assertEqual(receipt['total_raw_calls'], 80+48+14)
        self.assertFalse(receipt['sampling_assumptions_verified'])

    def test_exactness_declaration_required_and_not_provenance_verification(self):
        options = dict(self.options)
        del options['exact_masses_declared']
        with self.assertRaisesRegex(TypeError, 'exact_masses_declared'):
            audit_family(*self.frames, **options)
        for declaration in (False, None, 'unknown', 'True', 1, np.bool_(True)):
            with self.subTest(declaration=repr(declaration)):
                with self.assertRaisesRegex(ValueError, 'Explicitly declare'):
                    audit_family(*self.frames, **dict(self.options, exact_masses_declared=declaration))
        for source in ('', '  ', None, 1):
            with self.subTest(source=repr(source)):
                with self.assertRaisesRegex(ValueError, 'source'):
                    audit_family(*self.frames, **dict(self.options, mass_source=source))
        with self.assertRaisesRegex(ValueError, 'Explicitly declare'):
            audit_family(*self.frames, **dict(self.options, exact_masses_declared=False,
                         mass_source='Estimated frequencies from the validation stream'))
        receipt = audit_family(*self.frames, **self.options)['receipt']
        self.assertIs(receipt['exact_masses_declared'], True)
        self.assertIs(receipt['exact_mass_provenance_verified'], False)

    def test_complete_family_by_and_repeat_ids(self):
        expected = [.055, .0825, .9*5.5/3]
        expected[-1] = 1.
        np.testing.assert_allclose(by_adjust([.01, .03, .9]), expected)
        pvalues = by_adjust([.9, .01, .03])
        np.testing.assert_allclose(pvalues, [1., .055, .0825])
        repeated = [frame.copy() for frame in self.frames]
        repeated[2]['population_id'] = 'same_ID_can_recur_under_sampling_with_replacement'
        repeated[3]['population_id'] = 'same_ID_can_recur_under_sampling_with_replacement'
        first = audit_family(*self.frames, **self.options)
        second = audit_family(*repeated, **self.options)
        self.assertEqual(first, second)
        self.assertEqual(len(first['results']), 3)
        self.assertEqual(first['results'][2]['p'], 1.)

    def test_reference_labels_do_not_define_validation_strata(self):
        changed = [frame.copy() for frame in self.frames]
        mask = changed[2].role == 'reference'
        changed[2].loc[mask, 'category'] = changed[2].loc[mask, 'category'].map({'0': '1', '1': '0'})
        self.assertEqual(audit_family(*self.frames, **self.options), audit_family(*changed, **self.options))

    def test_large_integer_rank_shift_and_extreme_decimal_parse(self):
        original = audit_family(*self.frames, **self.options)
        changed = [frame.copy() for frame in self.frames]
        for frame in changed[2:]:
            for column in ['outcome']+[c for c in frame if c.startswith('forecast__')]:
                frame[column] = [str(2**80+int(value)) for value in frame[column]]
        shifted = audit_family(*changed, **self.options)
        self.assertEqual(original, shifted)
        # A literal beyond Decimal's supported exponent range must be rejected
        # explicitly, rather than becoming infinity or a different tied value.
        changed[2].loc[0, 'outcome'] = '1e999999999999999999999999999'
        with self.assertRaisesRegex(ValueError, 'unsupported numeric'):
            audit_family(*changed, **self.options)

    def test_missing_cell_and_malformed_pair(self):
        masses, fits, validation, evaluation = [frame.copy() for frame in self.frames]
        validation.loc[(validation.role == 'focal') & (validation.stream == 'B'), 'category'] = '0'
        result = audit_family(masses, fits, validation, evaluation, **self.options)
        for row in result['results']:
            cell = fits[(fits.model == row['model']) & (fits.category == '1')].iloc[0]
            expected = .5*max(cell.forecast_mean*cell.outcome_mean, (1-cell.forecast_mean)*(1-cell.outcome_mean))
            self.assertAlmostEqual(row['absent_allowance'], expected)
        bad = validation.iloc[:-1]
        with self.assertRaisesRegex(ValueError, 'exactly one focal'):
            audit_family(masses, fits, bad, evaluation, **self.options)
        missing_model = evaluation.drop(columns=['forecast__constant'])
        with self.assertRaises(ValueError):
            audit_family(masses, fits, validation, missing_model, **self.options)

    def test_zero_mass_categories_must_be_unobserved(self):
        masses, fits, validation, evaluation = [frame.copy() for frame in self.frames]
        masses['probability'] = [1., 0.]
        with self.assertRaisesRegex(ValueError, 'exact probability zero'):
            audit_family(masses, fits, validation, evaluation, **self.options)
        masses = pd.concat([self.frames[0], pd.DataFrame([{'category': 'unobserved', 'probability': 0.}])], ignore_index=True)
        extra = [dict(model=model, category='unobserved', forecast_mean=.5, outcome_mean=.5) for model in fits.model.unique()]
        fits = pd.concat([fits, pd.DataFrame(extra)], ignore_index=True)
        result = audit_family(masses, fits, validation, evaluation, **self.options)
        self.assertEqual(result['receipt']['category_counts_a'][-1], 0)
        self.assertEqual(result['receipt']['category_counts_b'][-1], 0)

    def test_alternate_allocation_variance_and_cli(self):
        result = audit_family(*self.frames, **self.options, family_level=.1,
                              validation_fraction=.2, sampling_bound='variance')
        self.assertAlmostEqual(result['receipt']['per_candidate_validation_delta'], .2*.1/(3*(1+1/2+1/3)))
        self.assertTrue(all(math.isfinite(row['p']) for row in result['results']))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            command = [sys.executable, str(HERE.parent/'aggregate_reference_family.py')]
            for name, frame in zip(('masses', 'fits', 'validation', 'evaluation'), self.frames):
                path = base/(name+'.csv')
                frame.to_csv(path, index=False)
                command += ['--'+name, str(path)]
            output = base/'result'
            command += ['--training-calls', '80', '--mass-source', 'Exact test design', '--exact-category-masses', '--output', str(output)]
            missing_declaration = [part for part in command if part != '--exact-category-masses']
            rejected = subprocess.run(missing_declaration, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('--exact-category-masses', rejected.stderr)
            self.assertFalse(output.exists())
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads((output/'receipt.json').read_text())
            self.assertEqual(receipt['total_raw_calls'], 142)
            self.assertEqual(set(receipt['input_sha256']), {'masses', 'fits', 'validation', 'evaluation'})
            self.assertEqual(len(pd.read_csv(output/'results.csv')), 3)
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            # CSV inference must not collapse 2^53 and 2^53+1 into a tie.
            for name in ('validation', 'evaluation'):
                path = base/(name+'.csv')
                frame = pd.read_csv(path)
                for column in ['outcome']+[c for c in frame if c.startswith('forecast__')]:
                    frame[column] = [2**53+int(value) for value in frame[column]]
                frame.to_csv(path, index=False)
            shifted_command = command[:-1]+[str(base/'shifted')]
            shifted = subprocess.run(shifted_command, capture_output=True, text=True)
            self.assertEqual(shifted.returncode, 0, shifted.stderr)
            pd.testing.assert_frame_equal(pd.read_csv(output/'results.csv'), pd.read_csv(base/'shifted'/'results.csv'))


if __name__ == '__main__':
    unittest.main()
