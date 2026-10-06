import gzip
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from itertools import product

import numpy as np

from grape_auto_labeling.reproduction.tables import (
    detector_records, exact_sign_flip, holm, write_csv,
    subset_records, generate_subset_tables, adabn_selection_statistics,
)


class NumericReproductionTests(unittest.TestCase):
    def test_exact_paired_test_and_holm_family(self):
        self.assertEqual(exact_sign_flip([1, 1, 1]), .25)
        self.assertEqual(exact_sign_flip([1, -1]), 1.0)
        self.assertEqual(exact_sign_flip([0, 0, 0]), 1.0)
        self.assertEqual(holm([.01, .04, .03]), [.03, .06, .06])

    def test_frame_presence_unions_tiles_and_uses_frame_ground_truth(self):
        # A false-positive tile in a positive frame must not create a background
        # frame alert; two predicted tiles in that frame count only once.
        run = dict(run_id="r1", arm="cold", real_positive_budget="", period="T1",
                   composition="deployment", train_seed="111", object_tp=1,
                   object_fp=2, object_fn=1, precision=1/3, recall=.5, f1=.4,
                   image_tp=1, image_fp=2, image_fn=1, image_tn=20, ap50=.6, ap50_95=.4)
        tiles = [dict(tile_id=f"t{i}", frame_id=f"f{i//8}", period="T1",
                      gt_boxes=int(i in (0, 8))) for i in range(24)]
        protocol = {"detector_runs":1, "deployment_runs":1, "seed_order":[111]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "detector_runs.csv", [run])
            write_csv(root / "test_tiles.csv", tiles)
            with gzip.open(root / "deployment_presence.jsonl.gz", "wt") as f:
                f.write(json.dumps({"run_id":"r1", "positive_tile_ids":["t0", "t1", "t16"]}) + "\n")
            _, records = detector_records(root, protocol)
            result = records[0]
            self.assertEqual(result["source_frame_tp"], 1)
            self.assertEqual(result["source_frame_recall"], .5)
            self.assertEqual(result["false_alert_frames"], 1)
            self.assertEqual(result["false_alert_frame_fraction"], 1.0)
            self.assertAlmostEqual(result["false_alert_tile_fraction"], 2/22)

            # Reject missing paired seeds before any statistical calculation.
            with self.assertRaisesRegex(ValueError, "Unpaired or duplicate seeds"):
                detector_records(root, {**protocol, "seed_order":[111, 222]})


class SubsetResamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = {"runs":180, "budgets":[10, 25, 50], "periods":["early", "harvest"],
                         "data_subset_seed_order":[409, 1824, 3657, 4012, 4506],
                         "training_seed_order":[1679, 2289, 8935], "bootstrap_seed":20260727,
                         "bootstrap_resamples":200, "permutation_unit":"data_subset_seed_mean",
                         "holm_family_size":6}
        self.protocol = {"subset_resampling":self.settings}
        self.rows = []
        self.differences = np.asarray([[.1 + .01 * d + .02 * s + .005 * d * s
                                       for s in range(3)] for d in range(5)])
        for arm, budget, period, d, s in product(("real", "finetune"), (10,25,50), ("early","harvest"), range(5), range(3)):
            ap = .1 + (self.differences[d,s] if arm == "finetune" else 0)
            self.rows.append(dict(run_id=f"{arm}.{budget}.{period}.{d}.{s}", arm=arm,
                real_positive_budget=budget, period=period, composition="curated",
                data_subset_seed=self.settings["data_subset_seed_order"][d],
                train_seed=self.settings["training_seed_order"][s], object_tp=5, object_fp=5, object_fn=5,
                precision=.5, recall=.5, f1=.5, ap50=ap, ap50_95=ap, applied_conf=.4))
        aggregate_rows = [dict(composition="curated", test_period=p, scenario=scenario, real_positive_budget=str(k))
                          for scenario, p, k in product(("real-only", "synthetic-pretrain + fine-tune"), ("early","harvest"), (10,25,50))]
        paired_rows = [dict(composition="curated", test_period=p, real_positive_budget=str(k), contrast="FT minus real")
                       for p, k in product(("early","harvest"), (10,25,50))]
        self.layouts = {"low_budget_subset_results.csv":{"rows":aggregate_rows},
                        "low_budget_subset_paired_effects.csv":{"rows":paired_rows}}
        write_csv(self.root/"subset_resampling_runs.csv", self.rows)

    def test_crossed_bootstrap_and_five_unit_exact_test(self):
        cells = subset_records(self.root, self.protocol)
        actual = generate_subset_tables(self.layouts, cells, self.protocol)
        rng = np.random.default_rng(20260727)
        subset_draws = rng.integers(0,5,size=(200,5))
        seed_draws = rng.integers(0,3,size=(200,3))
        # Explicit Cartesian resampling is an independent check of vectorized indexing.
        means = [self.differences[np.ix_(ds, ts)].mean() for ds,ts in zip(subset_draws,seed_draws)]
        lo, hi = np.quantile(means,[.025,.975])
        for row in actual["low_budget_subset_paired_effects.csv"]:
            self.assertEqual(row["two_way_bootstrap_ci_lower"], f"{lo:.8f}")
            self.assertEqual(row["two_way_bootstrap_ci_upper"], f"{hi:.8f}")
            self.assertEqual(row["raw_permutation_p"], "0.06250000")
            self.assertEqual(row["holm_adjusted_p"], "0.37500000")
            self.assertEqual(row["n_positive_cells"], "15")
            self.assertEqual(row["n_positive_subset_means"], "5")
        ft = next(r for r in actual["low_budget_subset_results.csv"] if r["scenario"] != "real-only")
        self.assertEqual(ft["ap50_95_sample_sd"], f"{self.differences.std(ddof=1):.6f}")
        # Pair by both seed identifiers, regardless of CSV order.
        write_csv(self.root/"subset_resampling_runs.csv", list(reversed(self.rows)))
        shuffled = generate_subset_tables(self.layouts, subset_records(self.root,self.protocol), self.protocol)
        self.assertEqual(actual, shuffled)

    def test_incomplete_or_duplicate_crossed_cells_are_rejected(self):
        for rows in (self.rows[:-1], self.rows[:-1]+[self.rows[0]],
                     [dict(self.rows[0],train_seed=111)]+self.rows[1:]):
            with self.subTest(length=len(rows)):
                write_csv(self.root/"subset_resampling_runs.csv", rows)
                with self.assertRaises(ValueError):
                    subset_records(self.root,self.protocol)

    def test_invalid_metrics_and_counts_are_rejected(self):
        for change in ({"ap50_95":float("nan")}, {"ap50_95":1.1}, {"precision":.9}, {"object_tp":1.5}):
            with self.subTest(change=change):
                rows = [dict(self.rows[0],**change)]+self.rows[1:]
                write_csv(self.root/"subset_resampling_runs.csv", rows)
                with self.assertRaises(ValueError):
                    subset_records(self.root,self.protocol)

    def test_wrong_permutation_unit_or_incomplete_holm_family_is_rejected(self):
        cells = subset_records(self.root,self.protocol)
        for change in ({"permutation_unit":"training_seed"}, {"holm_family_size":5}):
            protocol = deepcopy(self.protocol)
            protocol["subset_resampling"].update(change)
            with self.assertRaises(ValueError):
                generate_subset_tables(self.layouts,cells,protocol)


class AdaBNSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = {"seed_order":[111, 222], "adabn_setting_selection":{
            "background_draws":3, "bootstrap_seed":42, "bootstrap_resamples":200}}
        self.differences = np.asarray([[.01, .06, .03], [-.03, .02, -.01]])
        self.rows = [dict(training_seed=seed, draw=draw, adaptation_images=n,
                         ap50_95=.5 + (self.differences[i,draw] if n == 128 else 0))
                     for n, (i,seed), draw in product((128,512), enumerate((111,222)), range(3))]

    def test_crossed_resampling_preserves_shared_draw_pairing(self):
        write_csv(self.root/"adabn_selection_runs.csv", self.rows)
        actual = adabn_selection_statistics(self.root,self.protocol)
        rng = np.random.default_rng(42)
        seeds = rng.integers(0,2,size=(200,2))
        draws = rng.integers(0,3,size=(200,3))
        expected = [self.differences[np.ix_(s,d)].mean() for s,d in zip(seeds,draws)]
        np.testing.assert_allclose(actual["crossed_bootstrap_ci95"],np.quantile(expected,[.025,.975]))
        self.assertAlmostEqual(actual["ap50_95_paired_effect"],self.differences.mean())
        self.assertEqual(actual["n_paired_cells"],6)
        write_csv(self.root/"adabn_selection_runs.csv",list(reversed(self.rows)))
        self.assertEqual(actual,adabn_selection_statistics(self.root,self.protocol))

    def test_invalid_or_incomplete_cells_are_rejected(self):
        for rows in (self.rows[:-1], self.rows[:-1]+[self.rows[0]],
                     [dict(self.rows[0],ap50_95=float("nan"))]+self.rows[1:]):
            with self.subTest(rows=len(rows)):
                write_csv(self.root/"adabn_selection_runs.csv",rows)
                with self.assertRaises(ValueError):
                    adabn_selection_statistics(self.root,self.protocol)


if __name__ == "__main__":
    unittest.main()
