"""Detection records, box geometry, NMS, and ground-truth evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from grape_auto_labeling.utils.yolo_labels import load_yolo_boxes


@dataclass
class Detection:
    box_xyxy: List[float]
    score: float
    prompt: str
    class_id: int
    mask: Optional[np.ndarray] = None


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_prediction_boxes_to_gt(
    prediction_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float,
) -> Dict[str, Any]:
    candidates = []
    for pred_index, pred_box in enumerate(prediction_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            iou = box_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gt_index))

    candidates.sort(reverse=True)
    used_predictions = set()
    used_gt = set()
    matches = []
    for iou, pred_index, gt_index in candidates:
        if pred_index in used_predictions or gt_index in used_gt:
            continue
        used_predictions.add(pred_index)
        used_gt.add(gt_index)
        matches.append(
            {
                "prediction_index": int(pred_index),
                "gt_index": int(gt_index),
                "iou": float(iou),
            }
        )

    tp = len(matches)
    fp = len(prediction_boxes) - tp
    fn = len(gt_boxes) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_predictions": len(prediction_boxes),
        "num_gt_boxes": len(gt_boxes),
        "matches": matches,
    }


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def safe_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Return count/mean/std/p25/p50/p75 for a sequence of floats."""
    if not values:
        return {"count": 0, "mean": None, "std": None, "p25": None, "p50": None, "p75": None}
    arr = np.array(values, dtype=np.float32)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def evaluate_detections_against_gt(
    image_path: Path,
    detections: Sequence[Detection],
    width: int,
    height: int,
    gt_index: Dict[str, List[Path]],
    target_class_id: Optional[int],
    iou_threshold: float,
) -> Dict[str, Any]:
    gt_matches = gt_index.get(image_path.stem, [])
    if len(gt_matches) == 0:
        raise FileNotFoundError(f"Missing GT label file for image: {image_path}")
    if len(gt_matches) > 1:
        raise RuntimeError(
            f"Multiple GT label files matched image {image_path}: "
            + ", ".join(str(path) for path in gt_matches)
        )

    gt_label_path = gt_matches[0]
    gt_boxes = load_yolo_boxes(
        gt_label_path,
        width=width,
        height=height,
        target_class_id=target_class_id,
    )

    prediction_boxes = [detection.box_xyxy for detection in detections]
    result = match_prediction_boxes_to_gt(
        prediction_boxes=prediction_boxes,
        gt_boxes=gt_boxes,
        iou_threshold=iou_threshold,
    )
    result.update(
        {
            "status": "ok",
            "gt_label_path": str(gt_label_path),
            "iou_threshold": float(iou_threshold),
            "target_class_id": target_class_id,
        }
    )
    result.update(compute_prf(result["tp"], result["fp"], result["fn"]))
    return result


def accumulate_performance_result(totals: Dict[str, int], result: Dict[str, Any]) -> None:
    totals["tp"] += int(result.get("tp", 0))
    totals["fp"] += int(result.get("fp", 0))
    totals["fn"] += int(result.get("fn", 0))
    totals["num_predictions"] += int(result.get("num_predictions", 0))
    totals["num_gt_boxes"] += int(result.get("num_gt_boxes", 0))


def nms_detections(
    detections: Sequence[Detection],
    iou_threshold: Optional[float],
    max_detections: Optional[int],
) -> List[Detection]:
    ordered = sorted(detections, key=lambda detection: detection.score, reverse=True)
    if iou_threshold is None:
        return ordered[:max_detections] if max_detections else ordered

    kept: List[Detection] = []
    for detection in ordered:
        if max_detections is not None and len(kept) >= max_detections:
            break
        if all(box_iou(detection.box_xyxy, kept_detection.box_xyxy) <= iou_threshold for kept_detection in kept):
            kept.append(detection)
    return kept


def detection_to_record(detection: Detection) -> Dict[str, Any]:
    return {
        "box_xyxy": [float(value) for value in detection.box_xyxy],
        "score": float(detection.score),
        "prompt": detection.prompt,
        "class_id": int(detection.class_id),
        "has_mask": detection.mask is not None,
    }
