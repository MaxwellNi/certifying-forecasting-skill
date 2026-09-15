"""Boundary, label-staging and independently computed utility checks; no source data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("news_confirmation", HERE / "experiment.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


class ProtocolChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = ex.read_config()

    def test_source_hash_checked_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/"wrong.zip"
            src.write_bytes(b"incorrect dataset")
            with self.assertRaises(ValueError):
                ex.copy_source(src,tmp,self.cfg)
            self.assertFalse((Path(tmp)/"source.zip").exists())

    def test_loader_boundaries_negatives_and_conflicting_ids(self):
        dates = ["2016-01-29 23:59:59", "2016-01-30 00:00:00", "2016-02-01 00:00:00",
                 "2016-04-29 00:00:00", "2016-05-01 00:00:00", "2016-06-29 00:00:00",
                 "2016-05-02 00:00:00", "2016-05-03 00:00:00"]
        news = pd.DataFrame([dict(IDLink=str(i), Title="t", Headline="h", Source="s", Topic="economy", PublishDate=d)
                             for i, d in enumerate(dates)])
        social = pd.DataFrame([dict(IDLink=str(i), TS1=-1, TS2=0, TS3=0, TS4=999, TS144=-1 if i == 4 else 10)
                               for i in range(len(dates))])
        social.loc[6, "TS3"] = -1
        conflict = social.loc[[7]].copy()
        conflict["TS1"] = 1  # Conflicting first-hour information excludes the complete ID.
        social = pd.concat([social, conflict, social.loc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            frame = ex.ingest_frames(news, social, tmp, self.cfg)
            self.assertEqual(frame.IDLink.tolist(), ["0", "2", "4"])
            self.assertTrue(frame.TS1.isna().all())
            self.assertTrue((frame.TS3 == 0).all())
            self.assertFalse(frame.loc[frame.IDLink == "4", "label_available"].item())
            with np.load(Path(tmp) / "sealed/confirmation_labels.npz") as z:
                self.assertTrue(np.isnan(z["raw"][0]))
            meta = ex.load(Path(tmp) / "preparation_metadata.json")
            self.assertEqual(meta["duplicates"]["social"]["conflicting_ids"], 1)
            self.assertEqual(meta["duplicates"]["social"]["identical_duplicates_removed"], 1)

    def test_future_duplicate_values_never_change_issuance_support(self):
        news = pd.DataFrame([dict(IDLink="a", Title="t", Headline="h", Source="s", Topic="economy",
                                  PublishDate="2016-05-01 00:10:00", Facebook=0, GooglePlus=0)])
        news = pd.concat([news, news], ignore_index=True)
        social = pd.DataFrame([dict(IDLink="a", TS1=0, TS2=1, TS3=2, TS4=3, TS144=4)] * 2)
        changed_news = news.copy()
        changed_news.loc[1, ["Facebook", "GooglePlus"]] = [10000, -1]
        changed_social = social.copy()
        changed_social.loc[1, ["TS4", "TS144"]] = [99999, 1000]
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / "left", Path(tmp) / "right"
            left.mkdir()
            right.mkdir()
            before = ex.ingest_frames(news, social, left, self.cfg)
            after = ex.ingest_frames(changed_news, changed_social, right, self.cfg)
            pd.testing.assert_frame_equal(before.drop(columns="label_available"), after.drop(columns="label_available"))
            self.assertEqual(len(after), 1)
            self.assertTrue(before.label_available.item())
            self.assertFalse(after.label_available.item())
            meta = ex.load(right / "preparation_metadata.json")
            self.assertEqual(meta["duplicates"]["social"]["target_conflicting_ids_retained_eligible_if_TS3_available"], 1)
            for frame, y in [(before, [0.0]), (after, [np.nan])]:
                _, count, gain, width, _ = ex.utility_hours(frame.PublishDate, np.array([.2]), np.array([.8]), np.array(y),
                                                           *self.cfg["chronology"]["confirmation"])
                self.assertEqual(count[0], 1)
                self.assertAlmostEqual(width[0], 1.2)

    def test_future_features_do_not_change_training_preprocessing(self):
        news, social = ex.synthetic_frames(self.cfg)
        with tempfile.TemporaryDirectory() as tmp:
            frame = ex.ingest_frames(news, social, tmp, self.cfg)
        train = (frame.stage == "training").to_numpy()
        before, recipe = ex.fit_transform_features(frame, train, rich=True)
        altered = frame.copy()
        altered.loc[~train, "TS1"] = 1e12
        altered.loc[~train, "Source"] = "future_unseen_source"
        after, changed_recipe = ex.fit_transform_features(altered, train, rich=True)
        np.testing.assert_array_equal(before[train], after[train])
        self.assertEqual(recipe["source_levels"], changed_recipe["source_levels"])

    def test_by_and_exact_category_copy(self):
        np.testing.assert_array_equal(ex.by_rejections([.003, .008, .4, .8, 1, 1]), [True, False, False, False, False, False])
        c = np.array([0, 0, 1, 1, 1, 2])
        y = np.array([0, 1, 0, .5, .5, 1])
        self.assertEqual(ex.exact_theta(c[:, None] / 3, y, c)[0], 0)
        self.assertEqual(ex.choose(np.array([.1, .1, .1, .1, .1, .1]), .1, np.ones(6, bool)), -1)

    def test_hourly_gain_denominator_and_predictability(self):
        a = np.array([.2, .2, .5])
        b = np.array([.8, .8, .5])
        y = np.array([0., np.nan, 1.])
        dates = ["2016-05-01 00:00:00", "2016-05-01 00:59:59", "2016-05-01 02:00:00"]
        grid, count, gain, width, rows = ex.utility_hours(dates, a, b, y, *self.cfg["chronology"]["confirmation"])
        self.assertEqual(len(grid), 1416)
        self.assertEqual(count[0], 2)
        self.assertAlmostEqual(gain[0], .3)
        self.assertAlmostEqual(width[0], 1.2)
        self.assertEqual(gain[1], 0)
        self.assertEqual(width[1], 0)
        changed = ex.utility_hours(dates, a, b, np.array([1., 0., np.nan]), *self.cfg["chronology"]["confirmation"])
        np.testing.assert_array_equal(width, changed[3])
        bound = ex.predictable_grid_bound(gain, width, .025)
        self.assertEqual(bound["hours"], 1416)
        self.assertLessEqual(bound["lower_bound"], gain.mean())

    def test_complete_synthetic_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            cfg = ex.start_run(out, HERE / "config.json", synthetic=True)
            news, social = ex.synthetic_frames(cfg)
            frame = ex.ingest_frames(news, social, out, cfg)
            original = ex.exact_theta
            def after_freeze(*args):
                self.assertTrue((out / "selector_freeze.json").exists())
                return original(*args)
            with patch.object(ex, "exact_theta", side_effect=after_freeze), patch.object(ex.urllib.request, "urlopen") as network:
                ex.run_phases(out, cfg, frame, ex.synthetic_audit)
                network.assert_not_called()
            events = [json.loads(line) for line in (out / "access_receipts.jsonl").read_text().splitlines()]
            access = [e["stage"] for e in events if e["event"] == "model_stage_label_access"]
            self.assertEqual(access, list(ex.STAGES))
            names = [e["event"] for e in events]
            self.assertLess(names.index("all_selectors_frozen"), names.index("selection_exact_rank_diagnostic"))
            self.assertLess(names.index("selection_exact_rank_diagnostic"), max(i for i, e in enumerate(events) if e["event"] == "model_stage_label_access"))
            self.assertEqual(len(ex.load(out / "gate_decisions.json")), 54)
            self.assertEqual(ex.load(out / "confirmation_adjudication.json")["independent_dataset_tasks"], 0)
            self.assertEqual(len(pd.read_csv(out / "confirmation_hour_gains.csv")), 1416 * 12)


if __name__ == "__main__":
    unittest.main()
