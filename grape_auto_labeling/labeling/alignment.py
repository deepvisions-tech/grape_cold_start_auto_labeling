"""Homography alignment of a generated image to its original."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from grape_auto_labeling.utils.tensor_utils import matrix_to_list
from grape_auto_labeling.utils.yolo_labels import clip_box


@dataclass
class AlignmentResult:
    method: str
    status: str
    matrix_label_to_original: List[List[float]]
    matrix_original_to_label: List[List[float]]
    matches: int
    inliers: int
    original_size: Tuple[int, int]
    label_size: Tuple[int, int]


def transform_boxes_with_matrix(
    boxes_xyxy: np.ndarray,
    matrix: np.ndarray,
    width: int,
    height: int,
) -> List[Optional[List[float]]]:
    import cv2

    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return []

    points = np.empty((boxes.shape[0], 4, 1, 2), dtype=np.float32)
    points[:, 0, 0, 0] = boxes[:, 0]
    points[:, 0, 0, 1] = boxes[:, 1]
    points[:, 1, 0, 0] = boxes[:, 2]
    points[:, 1, 0, 1] = boxes[:, 1]
    points[:, 2, 0, 0] = boxes[:, 2]
    points[:, 2, 0, 1] = boxes[:, 3]
    points[:, 3, 0, 0] = boxes[:, 0]
    points[:, 3, 0, 1] = boxes[:, 3]

    transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 4, 2)
    out_boxes = np.stack(
        [
            np.min(transformed[:, :, 0], axis=1),
            np.min(transformed[:, :, 1], axis=1),
            np.max(transformed[:, :, 0], axis=1),
            np.max(transformed[:, :, 1], axis=1),
        ],
        axis=1,
    )
    return [clip_box(box, width=width, height=height) for box in out_boxes]


def try_sift_homography(label_gray: np.ndarray, original_gray: np.ndarray):
    import cv2

    if not hasattr(cv2, "SIFT_create"):
        return None

    detector = cv2.SIFT_create(nfeatures=6000)
    label_keypoints, label_desc = detector.detectAndCompute(label_gray, None)
    original_keypoints, original_desc = detector.detectAndCompute(original_gray, None)
    if label_desc is None or original_desc is None:
        return None
    if len(label_desc) < 12 or len(original_desc) < 12:
        return None

    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=64),
    )
    raw_matches = matcher.knnMatch(label_desc, original_desc, k=2)
    good_matches = [
        first
        for pair in raw_matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    if len(good_matches) < 12:
        return None

    src = np.float32(
        [label_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    dst = np.float32(
        [original_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if matrix is None or inliers < 8:
        return None
    return matrix.astype(np.float64), len(good_matches), inliers


def try_orb_homography(label_gray: np.ndarray, original_gray: np.ndarray):
    import cv2

    detector = cv2.ORB_create(nfeatures=8000)
    label_keypoints, label_desc = detector.detectAndCompute(label_gray, None)
    original_keypoints, original_desc = detector.detectAndCompute(original_gray, None)
    if label_desc is None or original_desc is None:
        return None
    if len(label_desc) < 12 or len(original_desc) < 12:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(label_desc, original_desc, k=2)
    good_matches = [
        first
        for pair in raw_matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    if len(good_matches) < 12:
        return None

    src = np.float32(
        [label_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    dst = np.float32(
        [original_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if matrix is None or inliers < 8:
        return None
    return matrix.astype(np.float64), len(good_matches), inliers


def estimate_label_to_original_alignment(
    label_image: Image.Image,
    original_image: Image.Image,
) -> Tuple[np.ndarray, AlignmentResult]:
    import cv2

    label_rgb = np.asarray(label_image.convert("RGB"))
    original_rgb = np.asarray(original_image.convert("RGB"))
    label_h, label_w = label_rgb.shape[:2]
    original_h, original_w = original_rgb.shape[:2]

    label_gray = cv2.cvtColor(label_rgb, cv2.COLOR_RGB2GRAY)
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    method = "resize_only"
    status = "fallback_resize_only"
    matches = 0
    inliers = 0
    matrix = None

    sift_result = try_sift_homography(label_gray, original_gray)
    if sift_result is not None:
        matrix, matches, inliers = sift_result
        method = "sift_homography"
        status = "ok"
    else:
        orb_result = try_orb_homography(label_gray, original_gray)
        if orb_result is not None:
            matrix, matches, inliers = orb_result
            method = "orb_homography"
            status = "ok"

    if matrix is None:
        matrix = np.asarray(
            [
                [float(original_w) / float(label_w), 0.0, 0.0],
                [0.0, float(original_h) / float(label_h), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    try:
        inverse_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        matrix = np.asarray(
            [
                [float(original_w) / float(label_w), 0.0, 0.0],
                [0.0, float(original_h) / float(label_h), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        inverse_matrix = np.linalg.inv(matrix)
        method = "resize_only"
        status = "fallback_singular_matrix"
        matches = 0
        inliers = 0

    aligned_label = cv2.warpPerspective(
        label_rgb,
        matrix,
        (original_w, original_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    result = AlignmentResult(
        method=method,
        status=status,
        matrix_label_to_original=matrix_to_list(matrix),
        matrix_original_to_label=matrix_to_list(inverse_matrix),
        matches=matches,
        inliers=inliers,
        original_size=(original_w, original_h),
        label_size=(label_w, label_h),
    )
    return aligned_label, result
