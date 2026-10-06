"""Image-cluster statistics for label-recovery evaluation results.

The public analysis operates on per-image TP/FP/FN counts. Metrics are always
micro-aggregated: counts are summed first, then precision, recall, and F1 are
computed. Image-cluster bootstrap resampling and paired comparisons therefore
preserve the image as the statistical unit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from grape_auto_labeling.labeling.detections import match_prediction_boxes_to_gt
from grape_auto_labeling.utils.yolo_labels import MISSING_FILE_ERROR, load_yolo_boxes


METRIC_NAMES = ("precision", "recall", "f1")
DEFAULT_BOOTSTRAP_ITERATIONS = 50_000
DEFAULT_PERMUTATION_ITERATIONS = 500_000
DEFAULT_SEED = 42


def metrics_from_totals(totals: np.ndarray | Sequence[int]) -> np.ndarray:
    """Return precision, recall, and F1 for arrays ending in TP/FP/FN.

    A metric with a zero denominator is defined as 0.0. This is the same rule
    used by the label-recovery evaluator.
    """

    values = np.asarray(totals, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(f"Expected TP/FP/FN on the last axis, got shape {values.shape}")
    tp, fp, fn = np.moveaxis(values, -1, 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return np.stack([precision, recall, f1], axis=-1)


def _metric_dict(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(METRIC_NAMES, values)}


def _ci_dict(bounds: np.ndarray | Sequence[Sequence[float]]) -> dict[str, list[float]]:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2, 3):
        raise ValueError(f"Expected 2x3 confidence-interval bounds, got {values.shape}")
    return {
        name: [float(values[0, index]), float(values[1, index])]
        for index, name in enumerate(METRIC_NAMES)
    }


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON row list or an object containing 'rows': {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Every input row must be a JSON object: {path}")
    return rows


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object on {path}:{line_number}")
        rows.append(row)
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_raw_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, JSON, or CSV evaluation rows."""

    if not path.is_file():
        raise FileNotFoundError(f"Evaluation input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl_rows(path)
    if suffix == ".json":
        return _read_json_rows(path)
    if suffix == ".csv":
        return _read_csv_rows(path)
    raise ValueError(f"Unsupported evaluation input format for {path}; use .jsonl, .json, or .csv")


def _nonnegative_int(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer for {context}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(f"{field} must be a non-negative integer for {context}: {value!r}")
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative for {context}: {parsed}")
    return parsed


def normalize_evaluation_row(
    row: dict[str, Any],
    *,
    method_override: str | None = None,
    use_basename_image_id: bool = False,
    context: str = "input row",
) -> dict[str, Any]:
    """Normalize a flat row or a pipeline ``per_image_results.json`` row."""

    image_id_value = row.get("image_id", row.get("relative_path"))
    if image_id_value is None or not str(image_id_value).strip():
        raise ValueError(f"Missing image_id or relative_path in {context}")
    image_id = str(image_id_value).strip().replace("\\", "/")
    if use_basename_image_id:
        image_id = Path(image_id).name

    method_value = method_override if method_override is not None else row.get("method")
    if method_value is None or not str(method_value).strip():
        raise ValueError(f"Missing method in {context}")
    method = str(method_value).strip()

    performance = row.get("performance")
    count_source = performance if isinstance(performance, dict) else row
    if isinstance(performance, dict):
        status = performance.get("status")
        if status is not None and status != "ok":
            raise ValueError(f"Performance status is not 'ok' for {image_id}: {status!r}")

    counts = {
        name: _nonnegative_int(count_source.get(name), field=name, context=f"{method}/{image_id}")
        for name in ("tp", "fp", "fn")
    }
    candidate_pool_id = row.get("candidate_pool_id")
    if candidate_pool_id is not None:
        candidate_pool_id = str(candidate_pool_id).strip()
        if not candidate_pool_id:
            candidate_pool_id = None
    return {
        "image_id": image_id,
        "method": method,
        **counts,
        "candidate_pool_id": candidate_pool_id,
    }


def load_evaluation_rows(
    path: Path,
    *,
    method_override: str | None = None,
    use_basename_image_id: bool = False,
) -> list[dict[str, Any]]:
    raw_rows = read_raw_rows(path)
    return [
        normalize_evaluation_row(
            row,
            method_override=method_override,
            use_basename_image_id=use_basename_image_id,
            context=f"{path} row {index}",
        )
        for index, row in enumerate(raw_rows, 1)
    ]


def _load_yolo_boxes(path: Path, *, width: int, height: int, class_id: int) -> list[list[float]]:
    """Read ground truth through the same parser the labeling pipeline uses.

    A gate comparison derives its before side here and its after side from the
    pipeline, so both must apply identical clipping and minimum-size rules.
    An analysis run names its label files explicitly, so a missing file is an
    error rather than an empty box list.
    """
    return load_yolo_boxes(
        path,
        width=width,
        height=height,
        target_class_id=class_id,
        require_exact_fields=True,
        on_missing_file=MISSING_FILE_ERROR,
    )


def load_gate_run_rows(
    path: Path,
    *,
    after_method: str,
    before_method: str,
) -> list[dict[str, Any]]:
    """Derive before/after counts from one DINOv2 pipeline result file.

    Both stages originate from ``sam3_detections_before_dinov2`` and the final
    performance record in the same row, so the shared candidate-pool condition
    is guaranteed by construction.
    """

    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(read_raw_rows(path), 1):
        context = f"{path} row {index}"
        after = normalize_evaluation_row(
            raw_row,
            method_override=after_method,
            use_basename_image_id=True,
            context=context,
        )
        performance = raw_row.get("performance")
        if not isinstance(performance, dict):
            raise ValueError(f"Gate-run row has no performance object: {context}")
        required = ("gt_label_path", "target_class_id", "iou_threshold")
        missing = [field for field in required if performance.get(field) is None]
        if missing:
            raise ValueError(f"Gate-run row is missing {', '.join(missing)}: {context}")
        if raw_row.get("width") is None or raw_row.get("height") is None:
            raise ValueError(f"Gate-run row is missing image dimensions: {context}")
        detections = raw_row.get("sam3_detections_before_dinov2")
        if not isinstance(detections, list):
            raise ValueError(f"Gate-run row has no pre-gate detections: {context}")

        gt_path = Path(str(performance["gt_label_path"]))
        if not gt_path.is_absolute():
            gt_path = path.parent / gt_path
        gt_boxes = _load_yolo_boxes(
            gt_path,
            width=int(raw_row["width"]),
            height=int(raw_row["height"]),
            class_id=int(performance["target_class_id"]),
        )
        prediction_boxes: list[list[float]] = []
        for detection_index, detection in enumerate(detections, 1):
            if not isinstance(detection, dict) or not isinstance(detection.get("box_xyxy"), list):
                raise ValueError(
                    f"Invalid pre-gate detection {detection_index} for {after['image_id']}"
                )
            box = [float(value) for value in detection["box_xyxy"]]
            if len(box) != 4:
                raise ValueError(f"Expected four box coordinates for {after['image_id']}")
            prediction_boxes.append(box)
        match_result = match_prediction_boxes_to_gt(
            prediction_boxes=prediction_boxes,
            gt_boxes=gt_boxes,
            iou_threshold=float(performance["iou_threshold"]),
        )
        pool_id = f"same_gate_run:{after['image_id']}"
        after["candidate_pool_id"] = pool_id
        before = {
            "image_id": after["image_id"],
            "method": before_method,
            "tp": match_result["tp"],
            "fp": match_result["fp"],
            "fn": match_result["fn"],
            "candidate_pool_id": pool_id,
        }
        rows.extend((after, before))
    return rows


def build_count_arrays(
    rows: Iterable[dict[str, Any]],
    *,
    image_order: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[str, str | None]]]:
    indexed: dict[str, dict[str, tuple[int, int, int]]] = {}
    pool_ids: dict[str, dict[str, str | None]] = {}
    for row in rows:
        method = str(row["method"])
        image_id = str(row["image_id"])
        method_rows = indexed.setdefault(method, {})
        if image_id in method_rows:
            raise ValueError(f"Duplicate method/image row: {method}/{image_id}")
        method_rows[image_id] = (int(row["tp"]), int(row["fp"]), int(row["fn"]))
        pool_ids.setdefault(method, {})[image_id] = row.get("candidate_pool_id")

    if not indexed:
        raise ValueError("No evaluation rows were provided")
    first_method = next(iter(indexed))
    image_ids = sorted(indexed[first_method])
    expected_set = set(image_ids)
    if image_order is not None:
        if len(image_order) != len(expected_set) or set(image_order) != expected_set:
            raise ValueError("Explicit analysis order must contain each image ID exactly once")
        image_ids = list(image_order)
    for method, method_rows in indexed.items():
        actual_set = set(method_rows)
        if actual_set != expected_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            raise ValueError(
                f"Image ID set differs for {method}; missing={missing[:5]}, extra={extra[:5]}"
            )

    for image_id in image_ids:
        gt_counts = {
            method: values[image_id][0] + values[image_id][2]
            for method, values in indexed.items()
        }
        if len(set(gt_counts.values())) != 1:
            raise ValueError(f"GT count differs across methods for {image_id}: {gt_counts}")

    counts = {
        method: np.asarray([method_rows[image_id] for image_id in image_ids], dtype=np.int64)
        for method, method_rows in indexed.items()
    }
    return image_ids, counts, pool_ids


def bootstrap_intervals(
    counts: dict[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    *,
    iterations: int,
    seed: int,
    chunk_size: int = 2_000,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute method and paired-difference percentile intervals."""

    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if chunk_size <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    names = list(counts)
    n_images = counts[names[0]].shape[0]
    samples: dict[str, list[np.ndarray]] = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, chunk_size):
        size = min(chunk_size, iterations - start)
        indices = rng.integers(0, n_images, size=(size, n_images))
        for name, values in counts.items():
            samples[name].append(metrics_from_totals(values[indices].sum(axis=1)))
    metric_samples = {name: np.concatenate(chunks, axis=0) for name, chunks in samples.items()}
    method_intervals = {
        name: np.quantile(values, [0.025, 0.975], axis=0)
        for name, values in metric_samples.items()
    }
    comparison_intervals = {
        f"{left}_minus_{right}": np.quantile(
            metric_samples[left] - metric_samples[right],
            [0.025, 0.975],
            axis=0,
        )
        for left, right in comparisons
    }
    return method_intervals, comparison_intervals


def paired_swap_permutation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int,
    seed: int,
    chunk_size: int = 2_000,
) -> dict[str, Any]:
    """Two-sided paired image-wise label-swap test for micro F1 difference."""

    if permutations <= 0:
        raise ValueError("permutation iterations must be positive")
    if left.shape != right.shape:
        raise ValueError(f"Paired arrays must have identical shape: {left.shape} vs {right.shape}")
    observed = float(
        metrics_from_totals(left.sum(axis=0))[2]
        - metrics_from_totals(right.sum(axis=0))[2]
    )
    rng = np.random.default_rng(seed)
    extreme_count = 0
    n_images = left.shape[0]
    for start in range(0, permutations, chunk_size):
        size = min(chunk_size, permutations - start)
        swap = rng.integers(0, 2, size=(size, n_images), dtype=np.int8).astype(bool)
        left_totals = np.where(swap[..., None], right, left).sum(axis=1)
        right_totals = np.where(swap[..., None], left, right).sum(axis=1)
        differences = (
            metrics_from_totals(left_totals)[:, 2]
            - metrics_from_totals(right_totals)[:, 2]
        )
        extreme_count += int(np.count_nonzero(np.abs(differences) >= abs(observed)))
    return {
        "metric": "micro_f1_difference",
        "alternative": "two-sided",
        "extremeness_rule": "abs(permuted_difference) >= abs(observed_difference)",
        "observed_difference": observed,
        "iterations": permutations,
        "extreme_count": extreme_count,
        "plus_one_correction": True,
        "p_value": float((extreme_count + 1) / (permutations + 1)),
    }


def _validate_comparisons(
    comparisons: Sequence[tuple[str, str]],
    counts: dict[str, np.ndarray],
) -> list[tuple[str, str]]:
    validated: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for left, right in comparisons:
        if left == right:
            raise ValueError(f"A comparison requires two different methods: {left}")
        missing = [method for method in (left, right) if method not in counts]
        if missing:
            raise ValueError(f"Comparison references unknown method(s): {', '.join(missing)}")
        pair = (left, right)
        if pair not in seen:
            validated.append(pair)
            seen.add(pair)
    return validated


def _verify_shared_candidate_pool(
    after_method: str,
    before_method: str,
    image_ids: Sequence[str],
    pool_ids: dict[str, dict[str, str | None]],
) -> None:
    for image_id in image_ids:
        after_pool = pool_ids[after_method].get(image_id)
        before_pool = pool_ids[before_method].get(image_id)
        if after_pool is None or before_pool is None:
            raise ValueError(
                "Gate comparison requires candidate_pool_id for both stages on every image; "
                f"missing for {image_id}"
            )
        if after_pool != before_pool:
            raise ValueError(
                f"Gate stages do not share a candidate pool for {image_id}: "
                f"{after_pool!r} vs {before_pool!r}"
            )


def analyze_records(
    rows: Iterable[dict[str, Any]],
    *,
    comparisons: Sequence[tuple[str, str]] = (),
    gate_comparison: tuple[str, str] | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    permutation_iterations: int = DEFAULT_PERMUTATION_ITERATIONS,
    seed: int = DEFAULT_SEED,
    image_order: Sequence[str] | None = None,
    permutation_seed: int | None = None,
) -> dict[str, Any]:
    image_ids, counts, pool_ids = build_count_arrays(rows, image_order=image_order)
    comparisons = _validate_comparisons(comparisons, counts)
    gate_pair: tuple[str, str] | None = None
    if gate_comparison is not None:
        gate_pair = _validate_comparisons([gate_comparison], counts)[0]
        _verify_shared_candidate_pool(gate_pair[0], gate_pair[1], image_ids, pool_ids)

    bootstrap_pairs = list(comparisons)
    if gate_pair is not None and gate_pair not in bootstrap_pairs:
        bootstrap_pairs.append(gate_pair)
    method_intervals, comparison_intervals = bootstrap_intervals(
        counts,
        bootstrap_pairs,
        iterations=bootstrap_iterations,
        seed=seed,
    )

    methods: dict[str, Any] = {}
    point_metrics: dict[str, np.ndarray] = {}
    for method, values in counts.items():
        totals = values.sum(axis=0)
        metrics = metrics_from_totals(totals)
        point_metrics[method] = metrics
        methods[method] = {
            "n_images": len(image_ids),
            "totals": {"tp": int(totals[0]), "fp": int(totals[1]), "fn": int(totals[2])},
            "metrics": _metric_dict(metrics),
            "ci_95": _ci_dict(method_intervals[method]),
        }

    paired_results: dict[str, Any] = {}
    for left, right in comparisons:
        key = f"{left}_minus_{right}"
        paired_results[key] = {
            "left_method": left,
            "right_method": right,
            "n_paired_images": len(image_ids),
            "metric_difference": _metric_dict(point_metrics[left] - point_metrics[right]),
            "ci_95": _ci_dict(comparison_intervals[key]),
            "permutation_test": paired_swap_permutation(
                counts[left],
                counts[right],
                permutations=permutation_iterations,
                seed=seed if permutation_seed is None else permutation_seed,
            ),
        }

    gate_diagnostic: dict[str, Any] | None = None
    if gate_pair is not None:
        after_method, before_method = gate_pair
        key = f"{after_method}_minus_{before_method}"
        gate_diagnostic = {
            "after_method": after_method,
            "before_method": before_method,
            "n_paired_images": len(image_ids),
            "same_candidate_pool_verified": True,
            "before": methods[before_method],
            "after": methods[after_method],
            "metric_difference": _metric_dict(
                point_metrics[after_method] - point_metrics[before_method]
            ),
            "ci_95": _ci_dict(comparison_intervals[key]),
        }

    return {
        "settings": {
            "bootstrap_iterations": bootstrap_iterations,
            "permutation_iterations": permutation_iterations,
            "seed": seed,
            "permutation_seed": seed if permutation_seed is None else permutation_seed,
            "image_order": "explicit" if image_order is not None else "sorted_image_id",
            "rng": "numpy.random.Generator(PCG64)",
            "metric_aggregation": "micro_from_summed_tp_fp_fn",
            "bootstrap_unit": "image_cluster",
            "bootstrap_interval": "percentile_95",
            "permutation_unit": "paired_image",
            "zero_denominator_rule": "return_0.0",
        },
        "dataset": {
            "n_images": len(image_ids),
            "methods": list(counts),
        },
        "methods": methods,
        "paired_comparisons": paired_results,
        "gate_diagnostic": gate_diagnostic,
    }


def validate_reference_summary(
    path: Path,
    methods: dict[str, Any],
    *,
    tolerance: float = 5e-5,
) -> dict[str, Any]:
    """Validate point estimates against an aggregate result CSV.

    Aggregate CSV rows cannot validate bootstrap or permutation calculations;
    they verify only image counts, TP/FP/FN totals, and point metrics.
    """

    rows = _read_csv_rows(path)
    required = {"method", "n_images", "tp", "fp", "fn", *METRIC_NAMES}
    if not rows:
        raise ValueError(f"Reference summary CSV is empty: {path}")
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(
            f"Reference summary CSV lacks columns: {', '.join(sorted(missing_columns))}"
        )
    checked: list[str] = []
    for row in rows:
        method = str(row["method"]).strip()
        if method not in methods:
            raise ValueError(f"Reference CSV method is absent from per-image input: {method}")
        actual = methods[method]
        if int(row["n_images"]) != actual["n_images"]:
            raise ValueError(f"n_images mismatch for {method}")
        for field in ("tp", "fp", "fn"):
            if int(row[field]) != actual["totals"][field]:
                raise ValueError(
                    f"{field} mismatch for {method}: {actual['totals'][field]} vs {row[field]}"
                )
        for metric in METRIC_NAMES:
            expected = float(row[metric])
            observed = float(actual["metrics"][metric])
            if abs(observed - expected) > tolerance:
                raise ValueError(
                    f"{metric} mismatch for {method}: {observed:.10f} vs {expected:.10f}"
                )
        checked.append(method)
    return {
        "status": "passed",
        "reference_file": path.name,
        "checked_methods": checked,
        "point_metric_tolerance": tolerance,
        "scope": "n_images, TP/FP/FN totals, and point precision/recall/F1",
    }


SUMMARY_COLUMNS = (
    "record_type",
    "name",
    "left_method",
    "right_method",
    "n_images",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "precision_ci_lower",
    "precision_ci_upper",
    "recall_ci_lower",
    "recall_ci_upper",
    "f1_ci_lower",
    "f1_ci_upper",
    "permutation_p_value",
    "permutation_extreme_count",
)


def _summary_row(
    *,
    record_type: str,
    name: str,
    n_images: int,
    metrics: dict[str, float],
    ci_95: dict[str, list[float]],
    left_method: str = "",
    right_method: str = "",
    totals: dict[str, int] | None = None,
    permutation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    totals = totals or {}
    permutation = permutation or {}
    return {
        "record_type": record_type,
        "name": name,
        "left_method": left_method,
        "right_method": right_method,
        "n_images": n_images,
        "tp": totals.get("tp", ""),
        "fp": totals.get("fp", ""),
        "fn": totals.get("fn", ""),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "precision_ci_lower": ci_95["precision"][0],
        "precision_ci_upper": ci_95["precision"][1],
        "recall_ci_lower": ci_95["recall"][0],
        "recall_ci_upper": ci_95["recall"][1],
        "f1_ci_lower": ci_95["f1"][0],
        "f1_ci_upper": ci_95["f1"][1],
        "permutation_p_value": permutation.get("p_value", ""),
        "permutation_extreme_count": permutation.get("extreme_count", ""),
    }


def write_analysis_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis_results.json"
    csv_path = output_dir / "analysis_summary.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for method, values in result["methods"].items():
        summary_rows.append(
            _summary_row(
                record_type="method",
                name=method,
                n_images=values["n_images"],
                totals=values["totals"],
                metrics=values["metrics"],
                ci_95=values["ci_95"],
            )
        )
    for name, values in result["paired_comparisons"].items():
        summary_rows.append(
            _summary_row(
                record_type="paired_comparison",
                name=name,
                left_method=values["left_method"],
                right_method=values["right_method"],
                n_images=values["n_paired_images"],
                metrics=values["metric_difference"],
                ci_95=values["ci_95"],
                permutation=values["permutation_test"],
            )
        )
    gate = result.get("gate_diagnostic")
    if gate is not None:
        summary_rows.append(
            _summary_row(
                record_type="gate_diagnostic",
                name=f"{gate['after_method']}_minus_{gate['before_method']}",
                left_method=gate["after_method"],
                right_method=gate["before_method"],
                n_images=gate["n_paired_images"],
                metrics=gate["metric_difference"],
                ci_95=gate["ci_95"],
            )
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return json_path, csv_path


def _parse_method_file(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("method file must use METHOD=PATH")
    method, path_text = spec.split("=", 1)
    if not method.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("method file must use non-empty METHOD=PATH")
    return method.strip(), Path(path_text)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze image-level TP/FP/FN label-recovery results.",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Combined .jsonl/.json/.csv rows containing image_id, method, tp, fp, fn.",
    )
    parser.add_argument(
        "--method-file",
        action="append",
        type=_parse_method_file,
        default=[],
        metavar="METHOD=PATH",
        help="Assign METHOD to every row in a pipeline per_image_results.json or flat input.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        nargs=2,
        default=[],
        metavar=("LEFT", "RIGHT"),
        help="Add LEFT-minus-RIGHT paired CIs and a two-sided F1 label-swap test.",
    )
    parser.add_argument(
        "--gate-comparison",
        nargs=2,
        metavar=("AFTER", "BEFORE"),
        help="Compare gate stages; matching candidate_pool_id values are required.",
    )
    parser.add_argument(
        "--gate-run-file",
        type=Path,
        help="DINOv2 per_image_results.json used to derive verified before/after gate counts.",
    )
    parser.add_argument("--gate-after-method", default="sam3_exemplar_dinov2")
    parser.add_argument("--gate-before-method", default="sam3_exemplar_dinov2_pre_gate")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=DEFAULT_PERMUTATION_ITERATIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-summary-csv",
        type=Path,
        help="Optionally verify aggregate point estimates against a result table.",
    )
    parser.add_argument("--reference-tolerance", type=float, default=5e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.input and not args.method_file and args.gate_run_file is None:
        parser.error("provide --input, --method-file, or --gate-run-file")
    if args.gate_run_file is not None and args.gate_comparison is not None:
        parser.error("use either --gate-run-file or --gate-comparison, not both")

    rows: list[dict[str, Any]] = []
    for path in args.input:
        rows.extend(load_evaluation_rows(path))
    for method, path in args.method_file:
        rows.extend(
            load_evaluation_rows(
                path,
                method_override=method,
                use_basename_image_id=True,
            )
        )

    gate_pair = tuple(args.gate_comparison) if args.gate_comparison is not None else None
    if args.gate_run_file is not None:
        rows.extend(
            load_gate_run_rows(
                args.gate_run_file,
                after_method=args.gate_after_method,
                before_method=args.gate_before_method,
            )
        )
        gate_pair = (args.gate_after_method, args.gate_before_method)

    result = analyze_records(
        rows,
        comparisons=[tuple(pair) for pair in args.compare],
        gate_comparison=gate_pair,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
    )
    if args.reference_summary_csv is not None:
        result["reference_validation"] = validate_reference_summary(
            args.reference_summary_csv,
            result["methods"],
            tolerance=args.reference_tolerance,
        )
    json_path, csv_path = write_analysis_outputs(result, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
