"""Candidate-adaptive DINOv2 filtering for paired-image differences."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from grape_auto_labeling.labeling.alignment import AlignmentResult
from grape_auto_labeling.config import (
    get_dinov2_require_patch_fully_inside_sam3_mask,
    resolve_dinov2_adaptive_threshold_offset,
    strict_config_bool,
)
from grape_auto_labeling.labeling.detections import Detection, safe_stats
from grape_auto_labeling.models.dinov2_runtime import DinoPatch
from grape_auto_labeling.labeling.visualize import (
    draw_dinov2_box_heatmap,
    draw_dinov2_diff_heatmap,
)


@dataclass
class DetectionDiagnostic:
    detection_index: int
    within_mask_mean_diff: Optional[float]
    within_mask_max_diff: Optional[float]
    within_mask_p75_diff: Optional[float]
    within_mask_patch_count: int
    adaptive_threshold: Optional[float]
    kept: bool
    kept_reason: Optional[str]
    candidate_score: Optional[float] = None
    candidate_cutoff: Optional[float] = None
    candidate_score_field: Optional[str] = None


@dataclass
class DinoFilterResult:
    detections: List[Detection]
    metadata: Dict[str, Any]


def box_contains_box(container: Sequence[float], inner: Sequence[float]) -> bool:
    cx1, cy1, cx2, cy2 = [float(value) for value in container]
    ix1, iy1, ix2, iy2 = [float(value) for value in inner]
    return ix1 >= cx1 and iy1 >= cy1 and ix2 <= cx2 and iy2 <= cy2


def boxes_have_positive_intersection(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def box_to_mask_slice(
    box_xyxy: Sequence[float],
    mask_width: int,
    mask_height: int,
) -> Optional[Tuple[slice, slice]]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    left = max(0, min(mask_width, int(math.floor(x1))))
    top = max(0, min(mask_height, int(math.floor(y1))))
    right = max(0, min(mask_width, int(math.ceil(x2))))
    bottom = max(0, min(mask_height, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        return None
    return slice(top, bottom), slice(left, right)


def patch_supported_by_detection_mask(
    detection: Detection,
    patch_box_xyxy: Sequence[float],
    require_patch_fully_inside_mask: bool,
) -> bool:
    if detection.mask is None:
        if require_patch_fully_inside_mask:
            return box_contains_box(detection.box_xyxy, patch_box_xyxy)
        return boxes_have_positive_intersection(detection.box_xyxy, patch_box_xyxy)

    mask = np.asarray(detection.mask).astype(bool, copy=False)
    if mask.ndim != 2:
        return False

    mask_h, mask_w = mask.shape
    patch_slice = box_to_mask_slice(patch_box_xyxy, mask_width=mask_w, mask_height=mask_h)
    if patch_slice is None:
        return False

    patch_region = mask[patch_slice]
    if patch_region.size == 0:
        return False
    if require_patch_fully_inside_mask:
        return bool(np.all(patch_region))
    return bool(np.any(patch_region))


def _otsu_threshold(scores: np.ndarray) -> float:
    """1-D Otsu threshold using numpy only (no skimage dependency).

    Finds the threshold that maximises between-class variance for a
    bimodal distribution of cosine-diff scores in [0, 1].
    Falls back to the 80th percentile when the distribution is degenerate
    (empty, all-zero, or perfectly uniform).
    """
    if scores.size == 0:
        return 0.5
    scores = np.clip(scores.astype(np.float32), 0.0, 1.0)
    hist, bin_edges = np.histogram(scores, bins=256, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = float(hist.sum())
    if total == 0.0:
        return float(np.percentile(scores, 80))

    best_thresh = float(bin_edges[len(bin_edges) // 2])
    best_var = 0.0
    weight1 = 0.0
    mean1_sum = 0.0
    total_mean = float(np.dot(hist.astype(np.float64), bin_centers))

    for i in range(len(hist) - 1):
        weight1 += float(hist[i])
        mean1_sum += float(hist[i]) * float(bin_centers[i])
        weight2 = total - weight1
        if weight1 == 0.0 or weight2 == 0.0:
            continue
        mean1 = mean1_sum / weight1
        mean2 = (total_mean - mean1_sum) / weight2
        var = (weight1 * weight2 / (total * total)) * ((mean1 - mean2) ** 2)
        if var > best_var:
            best_var = var
            # Use the upper edge of bin i as the threshold so that
            # class-0 = scores <= threshold, class-1 = scores > threshold.
            best_thresh = float(bin_edges[i + 1])

    if best_var == 0.0:
        return float(np.percentile(scores, 80))
    return best_thresh


def compute_adaptive_threshold(
    all_scores: np.ndarray,
    method: str = "otsu",
) -> Tuple[float, str]:
    """Compute a per-image adaptive diff threshold.

    Returns (threshold, method_actually_used).
    """
    if all_scores.size == 0:
        return 0.25, "fallback_empty"

    if method == "otsu":
        threshold = _otsu_threshold(all_scores)
        if threshold < 0.05:
            threshold = float(np.percentile(all_scores, 80))
            return threshold, "otsu_fallback_percentile_80"
        return threshold, "otsu"

    if method == "percentile_80":
        return float(np.percentile(all_scores, 80)), "percentile_80"

    return float(np.percentile(all_scores, 80)), f"unknown_method_fallback_{method}"


def compute_per_detection_stats(
    detection: "Detection",
    all_patches: Sequence["DinoPatch"],
    require_patch_fully_inside_mask: bool,
) -> Dict[str, Any]:
    """Compute diff statistics for patches inside a single detection's mask."""
    mask_scores = [
        patch.score
        for patch in all_patches
        if patch_supported_by_detection_mask(
            detection,
            patch.label_box_xyxy,
            require_patch_fully_inside_mask=require_patch_fully_inside_mask,
        )
    ]
    if not mask_scores:
        return {
            "within_mask_mean": None,
            "within_mask_max": None,
            "within_mask_p75": None,
            "within_mask_patch_count": 0,
        }
    arr = np.array(mask_scores, dtype=np.float32)
    return {
        "within_mask_mean": float(np.mean(arr)),
        "within_mask_max": float(np.max(arr)),
        "within_mask_p75": float(np.percentile(arr, 75)),
        "within_mask_patch_count": len(mask_scores),
    }


def _mask_integral(mask: np.ndarray) -> np.ndarray:
    mask_int = np.asarray(mask, dtype=np.int64)
    return np.pad(mask_int, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)


def _integral_region_sum(integral: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    return int(integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left])


def _per_detection_stats_from_scores(scores: np.ndarray) -> Dict[str, Any]:
    if scores.size == 0:
        return {
            "within_mask_mean": None,
            "within_mask_max": None,
            "within_mask_p75": None,
            "within_mask_patch_count": 0,
        }
    return {
        "within_mask_mean": float(np.mean(scores)),
        "within_mask_max": float(np.max(scores)),
        "within_mask_p75": float(np.percentile(scores, 75)),
        "within_mask_patch_count": int(scores.size),
    }


def build_per_detection_stats(
    all_patches: Sequence["DinoPatch"],
    detections: Sequence["Detection"],
    require_patch_fully_inside_mask: bool,
) -> List[Dict[str, Any]]:
    scores = np.array([patch.score for patch in all_patches], dtype=np.float32)
    num_patches = len(all_patches)
    num_detections = len(detections)
    filter_membership = np.zeros((num_patches, num_detections), dtype=bool)

    for det_index, detection in enumerate(detections):
        if detection.mask is None:
            for patch_index, patch in enumerate(all_patches):
                any_supported = boxes_have_positive_intersection(detection.box_xyxy, patch.label_box_xyxy)
                filter_membership[patch_index, det_index] = (
                    box_contains_box(detection.box_xyxy, patch.label_box_xyxy)
                    if require_patch_fully_inside_mask
                    else any_supported
                )
            continue

        mask = np.asarray(detection.mask).astype(bool, copy=False)
        if mask.ndim != 2:
            continue

        mask_h, mask_w = mask.shape
        integral = _mask_integral(mask)
        for patch_index, patch in enumerate(all_patches):
            patch_slice = box_to_mask_slice(patch.label_box_xyxy, mask_width=mask_w, mask_height=mask_h)
            if patch_slice is None:
                continue

            row_slice, col_slice = patch_slice
            top = int(row_slice.start)
            bottom = int(row_slice.stop)
            left = int(col_slice.start)
            right = int(col_slice.stop)
            patch_area = (bottom - top) * (right - left)
            if patch_area <= 0:
                continue

            mask_count = _integral_region_sum(integral, top=top, bottom=bottom, left=left, right=right)
            any_supported = mask_count > 0
            filter_membership[patch_index, det_index] = (
                mask_count == patch_area if require_patch_fully_inside_mask else any_supported
            )

    per_detection_stats = [
        _per_detection_stats_from_scores(scores[filter_membership[:, det_index]])
        for det_index in range(num_detections)
    ]
    return per_detection_stats


def compute_candidate_adaptive_cutoff(
    candidate_scores: Sequence[float],
    fallback_cutoff: float,
) -> Dict[str, Any]:
    valid_scores = [
        float(score)
        for score in candidate_scores
        if score is not None and np.isfinite(float(score))
    ]
    if not valid_scores:
        return {
            "cutoff": fallback_cutoff,
            "method": "fallback_no_valid_candidate_scores",
            "num_candidate_scores": 0,
        }

    if len(valid_scores) < 4:
        return {
            "cutoff": fallback_cutoff,
            "method": "fallback_too_few_candidate_scores",
            "num_candidate_scores": len(valid_scores),
        }

    scores = np.array(valid_scores, dtype=np.float32)
    cutoff, method = compute_adaptive_threshold(scores, method="otsu")
    return {
        "cutoff": float(cutoff),
        "method": f"candidate_{method}",
        "num_candidate_scores": len(valid_scores),
        "score_stats": safe_stats(valid_scores),
    }


def filter_detections_candidate_adaptive(
    detections: Sequence[Detection],
    all_patches: Sequence[DinoPatch],
    require_patch_fully_inside_mask: bool,
    fallback_cutoff: float,
    per_detection_stats: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Detection], List[DetectionDiagnostic], Dict[str, Any]]:
    """Filter SAM3 detections by the candidate-level DINOv2 diff distribution.

    Each SAM3 detection becomes one candidate. The candidate score is the
    p75 DINOv2 diff inside its mask, which is more tolerant than mean diff
    when only part of a leaf changed. A per-image Otsu cutoff over those
    candidate scores decides which candidates are kept.
    """
    score_field = "within_mask_p75_diff"

    per_detection_rows: List[Tuple[int, Detection, Dict[str, Any], Optional[float]]] = []
    candidate_scores: List[float] = []
    for i, detection in enumerate(detections):
        det_stats = (
            per_detection_stats[i]
            if per_detection_stats is not None and i < len(per_detection_stats)
            else compute_per_detection_stats(detection, all_patches, require_patch_fully_inside_mask)
        )
        candidate_score: Optional[float] = det_stats["within_mask_p75"]
        if candidate_score is not None:
            candidate_scores.append(float(candidate_score))
        per_detection_rows.append((i, detection, det_stats, candidate_score))

    cutoff_metadata = compute_candidate_adaptive_cutoff(
        candidate_scores=candidate_scores,
        fallback_cutoff=fallback_cutoff,
    )
    cutoff = float(cutoff_metadata["cutoff"])

    kept: List[Detection] = []
    diagnostics: List[DetectionDiagnostic] = []
    for i, detection, det_stats, candidate_score in per_detection_rows:
        patch_count = det_stats["within_mask_patch_count"]
        keep = (
            candidate_score is not None
            and float(candidate_score) >= float(cutoff)
        )
        diagnostics.append(
            DetectionDiagnostic(
                detection_index=i,
                within_mask_mean_diff=det_stats["within_mask_mean"],
                within_mask_max_diff=det_stats["within_mask_max"],
                within_mask_p75_diff=det_stats["within_mask_p75"],
                within_mask_patch_count=patch_count,
                adaptive_threshold=cutoff,
                kept=keep,
                kept_reason="candidate_adaptive" if keep else None,
                candidate_score=float(candidate_score) if candidate_score is not None else None,
                candidate_cutoff=cutoff,
                candidate_score_field=score_field,
            )
        )
        if keep:
            kept.append(detection)

    return kept, diagnostics, cutoff_metadata


def apply_dinov2_filter_for_image(
    cfg: Dict[str, Any],
    label_image: Image.Image,
    sam3_detections: Sequence[Detection],
    alignment: AlignmentResult,
    all_patches: Sequence[DinoPatch],
    patch_metadata: Dict[str, Any],
    artifact_paths: Optional[Dict[str, Path]],
    use_amp: bool = False,
) -> DinoFilterResult:
    require_fully_inside = get_dinov2_require_patch_fully_inside_sam3_mask(cfg)
    save_artifacts = strict_config_bool(cfg, "DINOV2_SAVE_ARTIFACTS", False) and artifact_paths is not None
    adaptive_threshold_method = str(cfg.get("DINOV2_ADAPTIVE_THRESHOLD_METHOD", "otsu")).strip().lower()
    adaptive_threshold_offset = resolve_dinov2_adaptive_threshold_offset(cfg)

    all_scores = np.array([patch.score for patch in all_patches], dtype=np.float32)
    adaptive_threshold_raw_value, adaptive_threshold_method_used = compute_adaptive_threshold(
        all_scores,
        method=adaptive_threshold_method,
    )
    adaptive_threshold_value = max(
        0.0,
        adaptive_threshold_raw_value - adaptive_threshold_offset,
    )
    per_detection_stats = build_per_detection_stats(
        all_patches,
        sam3_detections,
        require_patch_fully_inside_mask=require_fully_inside,
    )
    detections, diagnostics, candidate_cutoff_metadata = filter_detections_candidate_adaptive(
        detections=sam3_detections,
        all_patches=all_patches,
        require_patch_fully_inside_mask=require_fully_inside,
        fallback_cutoff=adaptive_threshold_value,
        per_detection_stats=per_detection_stats,
    )

    artifact_path_records: Dict[str, Optional[str]] = {}
    if save_artifacts and artifact_paths is not None:
        draw_dinov2_diff_heatmap(label_image, all_patches, artifact_paths["diff_heatmap"])
        draw_dinov2_box_heatmap(label_image, all_patches, artifact_paths["box_heatmap"])
        artifact_path_records = {key: str(path) for key, path in artifact_paths.items()}

    metadata: Dict[str, Any] = {
        "enabled": True,
        "status": "ok",
        "filter_strategy": "candidate_adaptive",
        "alignment": {
            "method": alignment.method,
            "status": alignment.status,
            "matches": alignment.matches,
            "inliers": alignment.inliers,
            "matrix_label_to_original": alignment.matrix_label_to_original,
            "matrix_original_to_label": alignment.matrix_original_to_label,
            "original_size": list(alignment.original_size),
            "label_size": list(alignment.label_size),
        },
        "patches": patch_metadata,
        "require_patch_fully_inside_sam3_mask": require_fully_inside,
        "sam3_detections_with_masks": sum(1 for detection in sam3_detections if detection.mask is not None),
        "sam3_boxes_before_filter": len(sam3_detections),
        "sam3_boxes_after_filter": len(detections),
        "sam3_boxes_filtered": len(sam3_detections) - len(detections),
        "amp": bool(use_amp),
        "amp_dtype": "float16" if use_amp else None,
        "artifact_paths": artifact_path_records,
    }

    metadata["adaptive_threshold"] = adaptive_threshold_value
    metadata["adaptive_threshold_raw"] = adaptive_threshold_raw_value
    metadata["adaptive_threshold_offset"] = adaptive_threshold_offset
    metadata["adaptive_threshold_method"] = adaptive_threshold_method_used
    metadata["all_scores_percentiles"] = {
        "p25": float(np.percentile(all_scores, 25)) if all_scores.size else None,
        "p50": float(np.percentile(all_scores, 50)) if all_scores.size else None,
        "p75": float(np.percentile(all_scores, 75)) if all_scores.size else None,
        "p95": float(np.percentile(all_scores, 95)) if all_scores.size else None,
    }
    metadata["detection_diagnostics"] = [dataclasses.asdict(d) for d in diagnostics]
    metadata["candidate_cutoff"] = candidate_cutoff_metadata

    return DinoFilterResult(
        detections=detections,
        metadata=metadata,
    )
