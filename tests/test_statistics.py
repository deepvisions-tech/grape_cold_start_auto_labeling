import csv
import json
import tempfile
import unittest
from pathlib import Path

from grape_auto_labeling.reproduction.statistics import (
    analyze_records,
    load_evaluation_rows,
    load_gate_run_rows,
    main,
    validate_reference_summary,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class LabelRecoveryStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = load_evaluation_rows(FIXTURE_DIR / "per_image_evaluation.jsonl")
        self.analysis_kwargs = {
            "comparisons": [("sam3_exemplar_dinov2", "sam3_exemplar")],
            "gate_comparison": (
                "sam3_exemplar_dinov2",
                "sam3_exemplar_dinov2_pre_gate",
            ),
            "bootstrap_iterations": 500,
            "permutation_iterations": 1_000,
            "seed": 17,
        }

    def test_point_metrics_match_deterministic_fixture(self) -> None:
        result = analyze_records(self.rows, **self.analysis_kwargs)
        expected = json.loads((FIXTURE_DIR / "expected_metrics.json").read_text(encoding="utf-8"))
        for method, values in expected.items():
            observed = result["methods"][method]
            self.assertEqual(
                observed["totals"],
                {name: values[name] for name in ("tp", "fp", "fn")},
            )
            for metric in ("precision", "recall", "f1"):
                self.assertAlmostEqual(observed["metrics"][metric], values[metric])

    def test_bootstrap_and_permutation_are_reproducible(self) -> None:
        first = analyze_records(self.rows, **self.analysis_kwargs)
        second = analyze_records(self.rows, **self.analysis_kwargs)
        self.assertEqual(first, second)
        permutation = first["paired_comparisons"][
            "sam3_exemplar_dinov2_minus_sam3_exemplar"
        ]["permutation_test"]
        self.assertEqual(permutation["alternative"], "two-sided")
        self.assertTrue(permutation["plus_one_correction"])
        self.assertAlmostEqual(
            permutation["p_value"],
            (permutation["extreme_count"] + 1) / (permutation["iterations"] + 1),
        )

    def test_explicit_order_preserves_statistics_after_anonymous_ids_are_changed(self):
        original_ids=sorted({r["image_id"] for r in self.rows})
        mapping={name:f"anon_{len(original_ids)-i:03d}" for i,name in enumerate(original_ids)}
        renamed=[{**r,"image_id":mapping[r["image_id"]]} for r in self.rows]
        first=analyze_records(self.rows,**self.analysis_kwargs)
        second=analyze_records(renamed, image_order=[mapping[n] for n in original_ids], **self.analysis_kwargs)
        self.assertEqual(first["methods"],second["methods"])
        self.assertEqual(first["paired_comparisons"],second["paired_comparisons"])
        with self.assertRaisesRegex(ValueError,"exactly once"):
            analyze_records(renamed,image_order=[mapping[original_ids[0]]]*len(original_ids),**self.analysis_kwargs)

    def test_gate_comparison_requires_matching_candidate_pool_ids(self) -> None:
        altered = [dict(row) for row in self.rows]
        for row in altered:
            if (
                row["image_id"] == "image_001"
                and row["method"] == "sam3_exemplar_dinov2_pre_gate"
            ):
                row["candidate_pool_id"] = "different_pool"
        with self.assertRaisesRegex(ValueError, "do not share a candidate pool"):
            analyze_records(altered, **self.analysis_kwargs)

    def test_method_image_sets_must_match(self) -> None:
        incomplete = [
            row
            for row in self.rows
            if not (row["method"] == "sam3_only" and row["image_id"] == "image_004")
        ]
        with self.assertRaisesRegex(ValueError, "Image ID set differs"):
            analyze_records(incomplete, **self.analysis_kwargs)

    def test_gate_run_file_derives_pre_gate_counts_from_the_same_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_text:
            temp_dir = Path(temp_dir_text)
            gt_path = temp_dir / "image.txt"
            gt_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            run_path = temp_dir / "per_image_results.json"
            run_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "relative_path": "image.jpg",
                                "width": 100,
                                "height": 100,
                                "sam3_detections_before_dinov2": [
                                    {"box_xyxy": [40, 40, 60, 60]},
                                    {"box_xyxy": [0, 0, 10, 10]},
                                ],
                                "performance": {
                                    "status": "ok",
                                    "tp": 1,
                                    "fp": 0,
                                    "fn": 0,
                                    "gt_label_path": str(gt_path),
                                    "target_class_id": 0,
                                    "iou_threshold": 0.5,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = load_gate_run_rows(
                run_path,
                after_method="post",
                before_method="pre",
            )
        indexed = {row["method"]: row for row in rows}
        self.assertEqual((indexed["pre"]["tp"], indexed["pre"]["fp"], indexed["pre"]["fn"]), (1, 1, 0))
        self.assertEqual(indexed["pre"]["candidate_pool_id"], indexed["post"]["candidate_pool_id"])

    def test_reference_csv_validates_rounded_point_metrics(self) -> None:
        result = analyze_records(self.rows, **self.analysis_kwargs)
        with tempfile.TemporaryDirectory() as temp_dir_text:
            reference_path = Path(temp_dir_text) / "reference.csv"
            with reference_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["method", "n_images", "tp", "fp", "fn", "precision", "recall", "f1"])
                writer.writerow(["sam3_only", 4, 3, 2, 3, "0.6000", "0.5000", "0.5455"])
            validation = validate_reference_summary(reference_path, result["methods"])
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["checked_methods"], ["sam3_only"])

    def test_cli_writes_json_and_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_text:
            output_dir = Path(temp_dir_text) / "analysis"
            exit_code = main(
                [
                    "--input",
                    str(FIXTURE_DIR / "per_image_evaluation.jsonl"),
                    "--compare",
                    "sam3_exemplar_dinov2",
                    "sam3_exemplar",
                    "--gate-comparison",
                    "sam3_exemplar_dinov2",
                    "sam3_exemplar_dinov2_pre_gate",
                    "--bootstrap-iterations",
                    "100",
                    "--permutation-iterations",
                    "100",
                    "--seed",
                    "17",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            result = json.loads((output_dir / "analysis_results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["settings"]["seed"], 17)
            with (output_dir / "analysis_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["record_type"] for row in rows},
                {"method", "paired_comparison", "gate_diagnostic"},
            )


if __name__ == "__main__":
    unittest.main()
