"""Resolved configuration, runtime environment, and alignment provenance.

These records are what makes a run auditable after the fact, so they report the
effective values rather than the requested ones."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from grape_auto_labeling.config import (
    build_class_names,
    get_dinov2_require_patch_fully_inside_sam3_mask,
    normalize_optional_text,
    optional_int,
    resolve_dinov2_adaptive_threshold_offset,
    resolve_dinov2_edge_suppress_patches,
    resolve_dinov2_tile_batch_size,
    resolve_nms_iou_threshold,
    resolve_sam3_confidence_threshold,
    strict_config_bool,
)
from grape_auto_labeling.labeling.detections import safe_stats
from grape_auto_labeling.models.dinov2_runtime import build_dinov2_tiling_config
from grape_auto_labeling.utils.serialization import yaml_safe_value
from grape_auto_labeling.labeling.preflight import AutoLabelPreflight


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalized_optional_path(value: Any) -> Optional[str]:
    text = normalize_optional_text(value)
    if text is None:
        return None
    return str(Path(text).expanduser().resolve(strict=False))


def build_resolved_auto_labeling_config(
    cfg: Dict[str, Any],
    preflight: AutoLabelPreflight,
) -> Dict[str, Any]:
    """Merge user values with effective runtime defaults using canonical keys."""
    performance_test_enabled = preflight.performance_test_enabled
    save_visual_outputs = strict_config_bool(
        cfg,
        "SAVE_VISUAL_OUTPUTS",
        not performance_test_enabled,
    )
    use_amp = strict_config_bool(cfg, "AMP", True)
    sam3_checkpoint_path = _normalized_optional_path(cfg.get("SAM3_CHECKPOINT_PATH"))
    sam3_repo_dir = _normalized_optional_path(cfg.get("SAM3_REPO_DIR"))
    load_from_hf = strict_config_bool(
        cfg,
        "SAM3_LOAD_FROM_HF",
        sam3_checkpoint_path is None,
    )
    model_name = str(cfg.get("DINOV2_MODEL_NAME", "dinov2_vitb14")).strip()
    adaptive_method = str(cfg.get("DINOV2_ADAPTIVE_THRESHOLD_METHOD", "otsu")).strip().lower()

    resolved = yaml_safe_value(dict(cfg))
    resolved.update(
        {
            "DEVICE": cfg.get("DEVICE", "auto"),
            "AMP": bool(use_amp),
            "LABEL_IMAGE_PATH": str(preflight.image_path),
            "LABEL_OUTPUT_DIR": str(preflight.output_dir),
            "OUTPUT_EXISTING_POLICY": preflight.output_existing_policy,
            "SAM3_PROMPT_MODE": preflight.sam3_prompt_mode,
            "TEXT_PROMPTS": list(preflight.prompts),
            "SAM3_IMAGE_EXEMPLARS": [
                {
                    "image_path": str(exemplar.spec.image_path),
                    "label_path": str(exemplar.spec.label_path),
                    "class_id": int(exemplar.spec.class_id),
                }
                for exemplar in preflight.loaded_exemplars
            ],
            "LABEL_CLASS_ID": int(preflight.class_id),
            "CLASS_NAMES": build_class_names(cfg.get("CLASS_NAMES"), preflight.class_id),
            "SAM3_REPO_DIR": sam3_repo_dir,
            "SAM3_LOAD_FROM_HF": bool(load_from_hf),
            "SAM3_CHECKPOINT_PATH": sam3_checkpoint_path,
            "SAM3_CONFIDENCE_THRESHOLD": resolve_sam3_confidence_threshold(cfg),
            "NMS_IOU_THRESHOLD": resolve_nms_iou_threshold(cfg),
            "MAX_DETECTIONS_PER_IMAGE": optional_int(
                cfg.get("MAX_DETECTIONS_PER_IMAGE"),
                default=None,
            ),
            "MAX_IMAGES": optional_int(cfg.get("MAX_IMAGES"), default=None),
            "DISABLE_TQDM": strict_config_bool(cfg, "DISABLE_TQDM", False),
            "SAVE_VISUAL_OUTPUTS": bool(save_visual_outputs),
            "AUTO_LABELING_BATCH_SIZE": optional_int(
                cfg.get("AUTO_LABELING_BATCH_SIZE"),
                default=1,
            ),
            "AUTO_LABELING_NUM_WORKERS": optional_int(
                cfg.get("AUTO_LABELING_NUM_WORKERS"),
                default=1,
            ),
            "DINOV2_FILTER_ENABLED": bool(preflight.dinov2_enabled),
            "DINOV2_ORIGINAL_IMAGE_PATH": (
                str(preflight.original_path) if preflight.original_path is not None else None
            ),
            "DINOV2_MODEL_NAME": model_name,
            "DINOV2_HUB_REPO": str(
                cfg.get("DINOV2_HUB_REPO", "facebookresearch/dinov2")
            ).strip(),
            "DINOV2_HUB_REF": normalize_optional_text(cfg.get("DINOV2_HUB_REF")),
            "DINOV2_REPO_DIR": _normalized_optional_path(cfg.get("DINOV2_REPO_DIR")),
            "DINOV2_CHECKPOINT_PATH": _normalized_optional_path(
                cfg.get("DINOV2_CHECKPOINT_PATH")
            ),
            "DINOV2_CHECKPOINT_FILENAME": str(
                cfg.get("DINOV2_CHECKPOINT_FILENAME", f"{model_name}_pretrain.pth")
            ).strip(),
            "DINOV2_TILE_BATCH_SIZE": resolve_dinov2_tile_batch_size(cfg),
            "DINOV2_EDGE_SUPPRESS_PATCHES": resolve_dinov2_edge_suppress_patches(cfg),
            "DINOV2_REQUIRE_PATCH_FULLY_INSIDE_SAM3_MASK": (
                get_dinov2_require_patch_fully_inside_sam3_mask(cfg)
            ),
            "DINOV2_SAVE_ARTIFACTS": (
                strict_config_bool(cfg, "DINOV2_SAVE_ARTIFACTS", False)
                and not performance_test_enabled
            ),
            "DINOV2_ADAPTIVE_THRESHOLD_METHOD": adaptive_method,
            "DINOV2_ADAPTIVE_THRESHOLD_OFFSET": resolve_dinov2_adaptive_threshold_offset(cfg),
            "PERFORMANCE_TEST_ENABLED": bool(performance_test_enabled),
            "GT_LABEL_DIR": (
                str(preflight.gt_label_dir) if preflight.gt_label_dir is not None else None
            ),
            "PERFORMANCE_IOU_THRESHOLD": float(preflight.performance_iou_threshold),
        }
    )

    if preflight.dinov2_enabled:
        tiling = build_dinov2_tiling_config(cfg)
        resolved["DINOV2_INPUT_SIZE"] = int(tiling.input_size)
        resolved["DINOV2_TILE_WIDTH_DIVISOR"] = float(tiling.tile_width_divisor)
        resolved["DINOV2_TILE_HEIGHT_DIVISOR"] = float(tiling.tile_height_divisor)
        resolved["DINOV2_TILE_STRIDE_RATIO"] = float(tiling.tile_stride_ratio)

    return resolved


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def collect_runtime_environment() -> Dict[str, Any]:
    """Collect best-effort runtime versions without making provenance fatal."""
    environment: Dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {
            "numpy": str(np.__version__),
            "pillow": _distribution_version("Pillow"),
            "pyyaml": _distribution_version("PyYAML"),
            "opencv_python": _distribution_version("opencv-python"),
            "opencv_python_headless": _distribution_version("opencv-python-headless"),
            "torch": _distribution_version("torch"),
            "torchvision": _distribution_version("torchvision"),
            "tqdm": _distribution_version("tqdm"),
        },
    }

    try:
        import cv2

        environment["opencv_runtime_version"] = str(cv2.__version__)
    except Exception as exc:
        environment["opencv_runtime_version"] = None
        environment["opencv_collection_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import torch

        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        environment["pytorch"] = {
            "version": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
        }
    except Exception as exc:
        environment["pytorch"] = None
        environment["pytorch_collection_error"] = f"{type(exc).__name__}: {exc}"

    return environment


def summarize_alignment_results(
    per_image_results: Sequence[Dict[str, Any]],
    *,
    dinov2_enabled: bool,
) -> Dict[str, Any]:
    method_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    method_status_counts: Dict[str, int] = {}
    matches: List[float] = []
    inliers: List[float] = []
    images_with_alignment = 0

    for row in per_image_results:
        dinov2 = row.get("dinov2")
        alignment = dinov2.get("alignment") if isinstance(dinov2, dict) else None
        if not isinstance(alignment, dict):
            continue
        method = str(alignment.get("method") or "unknown")
        status = str(alignment.get("status") or "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        combination = f"{method}:{status}"
        method_status_counts[combination] = method_status_counts.get(combination, 0) + 1
        if alignment.get("matches") is not None:
            matches.append(float(alignment["matches"]))
        if alignment.get("inliers") is not None:
            inliers.append(float(alignment["inliers"]))
        images_with_alignment += 1

    expected_images = len(per_image_results) if dinov2_enabled else 0
    return {
        "enabled": bool(dinov2_enabled),
        "images_with_alignment": images_with_alignment,
        "images_without_alignment": max(0, expected_images - images_with_alignment),
        "method_counts": dict(sorted(method_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "method_status_counts": dict(sorted(method_status_counts.items())),
        "fallback_images": sum(
            count for status, count in status_counts.items() if status != "ok"
        ),
        "matches": safe_stats(matches),
        "inliers": safe_stats(inliers),
    }
