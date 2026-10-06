"""Model-free validation of configuration, inputs, pairing, and output safety."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from grape_auto_labeling.config import (
    normalize_optional_text,
    SUPPORTED_ADAPTIVE_THRESHOLD_METHODS,
    build_class_names,
    normalize_prompts,
    require_config_float,
    require_config_int,
    resolve_dinov2_adaptive_threshold_offset,
    resolve_dinov2_edge_suppress_patches,
    resolve_dinov2_tile_batch_size,
    resolve_nms_iou_threshold,
    resolve_optional_existing_path,
    resolve_output_existing_policy,
    resolve_path,
    resolve_sam3_confidence_threshold,
    strict_config_bool,
    validate_known_config_keys,
)
from grape_auto_labeling.utils.devices import validate_device_config
from grape_auto_labeling.models.dinov2_runtime import build_dinov2_tiling_config
from grape_auto_labeling.utils.io_paths import (
    index_images_by_name,
    index_yolo_labels_by_stem,
    list_input_images,
    require_unique_gt_labels,
    require_unique_original_matches,
    validate_safe_output_path,
    validate_unique_output_targets,
)
from grape_auto_labeling.models.model_artifacts import (
    build_torch_hub_repo_spec,
)
from grape_auto_labeling.models.sam3_exemplar import (
    load_sam3_image_exemplars,
    normalize_sam3_prompt_mode,
    sam3_prompt_mode_uses_exemplar,
)
from grape_auto_labeling.utils.yolo_labels import load_yolo_boxes


@dataclass
class AutoLabelPreflight:
    sam3_prompt_mode: str
    prompts: List[str]
    class_id: int
    image_path: Path
    output_dir: Path
    image_paths: List[Path]
    input_root: Path
    output_existing_policy: str
    output_has_existing_entries: bool
    performance_test_enabled: bool
    performance_iou_threshold: float
    gt_label_dir: Optional[Path]
    gt_index: Dict[str, List[Path]]
    dinov2_enabled: bool
    original_path: Optional[Path]
    original_index: Dict[str, List[Path]]
    original_matches: Dict[Path, Path]
    original_root: Optional[Path]
    loaded_exemplars: List[Any]


def preflight_auto_labeling_config(
    cfg: Dict[str, Any],
    *,
    overwrite: Optional[bool] = None,
) -> AutoLabelPreflight:
    """Validate config, inputs, and output safety without importing model runtimes."""
    validate_known_config_keys(cfg)
    validate_device_config(cfg)

    sam3_prompt_mode = normalize_sam3_prompt_mode(cfg.get("SAM3_PROMPT_MODE", "text"))
    prompts = normalize_prompts(cfg.get("TEXT_PROMPTS"))
    class_id = require_config_int(cfg, "LABEL_CLASS_ID", default=0, minimum=0)
    assert class_id is not None
    build_class_names(cfg.get("CLASS_NAMES"), class_id=class_id)

    for key, default in {
        "AMP": True,
        "DISABLE_TQDM": False,
        "PERFORMANCE_TEST_ENABLED": False,
        "DINOV2_FILTER_ENABLED": False,
        "DINOV2_SAVE_ARTIFACTS": False,
        "DINOV2_REQUIRE_PATCH_FULLY_INSIDE_SAM3_MASK": True,
    }.items():
        strict_config_bool(cfg, key, default)
    if "SAVE_VISUAL_OUTPUTS" in cfg:
        strict_config_bool(cfg, "SAVE_VISUAL_OUTPUTS", True)

    resolve_sam3_confidence_threshold(cfg)
    resolve_nms_iou_threshold(cfg)
    performance_iou_threshold = require_config_float(
        cfg,
        "PERFORMANCE_IOU_THRESHOLD",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
    )
    assert performance_iou_threshold is not None
    require_config_int(
        cfg,
        "MAX_DETECTIONS_PER_IMAGE",
        minimum=1,
        allow_none=True,
    )
    max_images = require_config_int(cfg, "MAX_IMAGES", minimum=1, allow_none=True)
    require_config_int(cfg, "AUTO_LABELING_BATCH_SIZE", default=1, minimum=1)
    require_config_int(cfg, "AUTO_LABELING_NUM_WORKERS", default=1, minimum=1)

    performance_test_enabled = strict_config_bool(
        cfg,
        "PERFORMANCE_TEST_ENABLED",
        False,
    )
    dinov2_enabled = strict_config_bool(cfg, "DINOV2_FILTER_ENABLED", False)
    adaptive_method = str(cfg.get("DINOV2_ADAPTIVE_THRESHOLD_METHOD", "otsu")).strip().lower()
    if adaptive_method not in SUPPORTED_ADAPTIVE_THRESHOLD_METHODS:
        raise ValueError(
            "DINOV2_ADAPTIVE_THRESHOLD_METHOD must be one of "
            f"{', '.join(sorted(SUPPORTED_ADAPTIVE_THRESHOLD_METHODS))}; "
            f"got {adaptive_method!r}."
        )

    resolve_dinov2_tile_batch_size(cfg)
    resolve_dinov2_edge_suppress_patches(cfg)
    resolve_dinov2_adaptive_threshold_offset(cfg)
    build_dinov2_tiling_config(cfg)

    image_path = resolve_path(cfg.get("LABEL_IMAGE_PATH"), "LABEL_IMAGE_PATH", must_exist=True)
    output_dir = resolve_path(cfg.get("LABEL_OUTPUT_DIR"), "LABEL_OUTPUT_DIR", must_exist=False)
    image_paths, input_root = list_input_images(image_path)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise RuntimeError(f"No images to process after MAX_IMAGES filtering: {image_path}")
    validate_unique_output_targets(image_paths, input_root)

    output_existing_policy = resolve_output_existing_policy(cfg, overwrite=overwrite)
    output_has_existing_entries = validate_safe_output_path(
        image_path=image_path,
        output_dir=output_dir,
        output_existing_policy=output_existing_policy,
    )

    gt_label_dir: Optional[Path] = None
    gt_index: Dict[str, List[Path]] = {}
    if performance_test_enabled:
        gt_label_dir = resolve_path(cfg.get("GT_LABEL_DIR"), "GT_LABEL_DIR", must_exist=True)
        gt_index = index_yolo_labels_by_stem(gt_label_dir)
        require_unique_gt_labels(image_paths, gt_index)

    original_path: Optional[Path] = None
    original_index: Dict[str, List[Path]] = {}
    original_matches: Dict[Path, Path] = {}
    original_root: Optional[Path] = None
    if dinov2_enabled:
        original_path = resolve_path(
            cfg.get("DINOV2_ORIGINAL_IMAGE_PATH"),
            "DINOV2_ORIGINAL_IMAGE_PATH",
            must_exist=True,
        )
        original_index, original_root = index_images_by_name(original_path)
        original_matches = require_unique_original_matches(image_paths, original_index)

    resolve_optional_existing_path(cfg.get("SAM3_REPO_DIR"), "SAM3_REPO_DIR", kind="directory")
    sam3_checkpoint_path = resolve_optional_existing_path(
        cfg.get("SAM3_CHECKPOINT_PATH"),
        "SAM3_CHECKPOINT_PATH",
        kind="file",
    )
    load_from_hf = strict_config_bool(
        cfg,
        "SAM3_LOAD_FROM_HF",
        sam3_checkpoint_path is None,
    )
    if sam3_checkpoint_path is None and not load_from_hf:
        raise ValueError("SAM3_CHECKPOINT_PATH must be set when SAM3_LOAD_FROM_HF is false.")

    if dinov2_enabled:
        dinov2_repo_dir = resolve_optional_existing_path(
            cfg.get("DINOV2_REPO_DIR"),
            "DINOV2_REPO_DIR",
            kind="directory",
        )
        hub_ref = normalize_optional_text(cfg.get("DINOV2_HUB_REF"))
        if dinov2_repo_dir is not None and hub_ref is not None:
            raise ValueError(
                "DINOV2_REPO_DIR and DINOV2_HUB_REF are mutually exclusive. "
                "Use a local repository or a Torch Hub source, not both."
            )
        resolve_optional_existing_path(
            cfg.get("DINOV2_CHECKPOINT_PATH"),
            "DINOV2_CHECKPOINT_PATH",
            kind="file",
        )
        hub_repo = str(cfg.get("DINOV2_HUB_REPO", "facebookresearch/dinov2")).strip()
        build_torch_hub_repo_spec(hub_repo, hub_ref)
        checkpoint_filename = str(
            cfg.get("DINOV2_CHECKPOINT_FILENAME", "dinov2_vitb14_pretrain.pth")
        ).strip()
        if not checkpoint_filename:
            raise ValueError("DINOV2_CHECKPOINT_FILENAME must not be empty.")

    loaded_exemplars: List[Any] = []
    if sam3_prompt_mode_uses_exemplar(sam3_prompt_mode):
        loaded_exemplars = load_sam3_image_exemplars(
            cfg.get("SAM3_IMAGE_EXEMPLARS"),
            default_class_id=class_id,
            load_yolo_boxes=load_yolo_boxes,
        )

    return AutoLabelPreflight(
        sam3_prompt_mode=sam3_prompt_mode,
        prompts=prompts,
        class_id=class_id,
        image_path=image_path,
        output_dir=output_dir,
        image_paths=image_paths,
        input_root=input_root,
        output_existing_policy=output_existing_policy,
        output_has_existing_entries=output_has_existing_entries,
        performance_test_enabled=performance_test_enabled,
        performance_iou_threshold=performance_iou_threshold,
        gt_label_dir=gt_label_dir,
        gt_index=gt_index,
        dinov2_enabled=dinov2_enabled,
        original_path=original_path,
        original_index=original_index,
        original_matches=original_matches,
        original_root=original_root,
        loaded_exemplars=loaded_exemplars,
    )


def check_auto_labeling_config(
    cfg: Dict[str, Any],
    *,
    overwrite: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the model-free preflight and return a JSON-serializable summary."""
    preflight = preflight_auto_labeling_config(cfg, overwrite=overwrite)
    result = {
        "status": "ok",
        "check_only": True,
        "input_path": str(preflight.image_path),
        "input_root": str(preflight.input_root),
        "num_images": len(preflight.image_paths),
        "output_dir": str(preflight.output_dir),
        "output_existing_policy": preflight.output_existing_policy,
        "output_has_existing_entries": preflight.output_has_existing_entries,
        "sam3_prompt_mode": preflight.sam3_prompt_mode,
        "num_text_prompts": len(preflight.prompts),
        "num_exemplars": len(preflight.loaded_exemplars),
        "dinov2_enabled": preflight.dinov2_enabled,
        "performance_test_enabled": preflight.performance_test_enabled,
    }
    print("[check-only] Configuration and input preflight passed")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
