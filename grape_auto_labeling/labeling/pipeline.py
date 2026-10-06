"""SAM 3 and DINOv2 label-recovery run loop.

The command-line entry point lives in grape_auto_labeling.cli, which applies the
single-GPU CUDA_VISIBLE_DEVICES mask before any model runtime is imported."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from grape_auto_labeling.labeling.alignment import estimate_label_to_original_alignment
from grape_auto_labeling.utils.concurrency import iter_batches, ordered_worker_map
from grape_auto_labeling.config import (
    config_int,
    build_class_names,
    get_dinov2_require_patch_fully_inside_sam3_mask,
    optional_int,
    resolve_dinov2_adaptive_threshold_offset,
    resolve_dinov2_edge_suppress_patches,
    resolve_dinov2_tile_batch_size,
    resolve_nms_iou_threshold,
    resolve_sam3_confidence_threshold,
    strict_config_bool,
)
from grape_auto_labeling.labeling.detections import (
    Detection,
    safe_stats,
    accumulate_performance_result,
    compute_prf,
    detection_to_record,
    evaluate_detections_against_gt,
    nms_detections,
)
from grape_auto_labeling.utils.devices import (
    amp_inference_context,
    configure_auto_label_runtime,
)
from grape_auto_labeling.labeling.dinov2_gate import apply_dinov2_filter_for_image
from grape_auto_labeling.models.dinov2_runtime import (
    DinoPatchDiffInput,
    build_dinov2_runtime,
    build_dinov2_tiling_config,
    compute_dinov2_patch_diffs_batch,
)
from grape_auto_labeling.utils.serialization import save_json, save_yaml
from grape_auto_labeling.utils.yolo_labels import save_classes_txt, save_yolo_label
from grape_auto_labeling.labeling.preflight import preflight_auto_labeling_config
from grape_auto_labeling.labeling.provenance import (
    build_resolved_auto_labeling_config,
    collect_runtime_environment,
    summarize_alignment_results,
    utc_now_iso,
)
from grape_auto_labeling.models.sam3_exemplar import (
    Sam3ImageExemplarPredictor,
    sam3_prompt_mode_uses_exemplar,
)
from grape_auto_labeling.models.sam3_runtime import (
    build_sam3_processor,
    run_sam3_prompt_mode_on_image,
)
from grape_auto_labeling.labeling.visualize import draw_visualization


@dataclass
class AutoLabelImageState:
    img_path: Path
    rel_path: Path
    label_path: Path
    vis_path: Optional[Path]
    image: Optional[Image.Image] = None
    original_image: Optional[Image.Image] = None
    original_match_path: Optional[Path] = None
    dinov2_metadata: Dict[str, Any] = field(default_factory=dict)
    detections_before_nms: List[Detection] = field(default_factory=list)
    sam3_detections: List[Detection] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    performance_result: Optional[Dict[str, Any]] = None
    width: Optional[int] = None
    height: Optional[int] = None


def build_auto_label_result_record(
    state: AutoLabelImageState,
    dinov2_enabled: bool,
) -> Dict[str, Any]:
    return {
        "image_path": str(state.img_path),
        "relative_path": str(state.rel_path),
        "output_label_path": str(state.label_path),
        "output_visualization_path": str(state.vis_path) if state.vis_path is not None else None,
        "width": state.width,
        "height": state.height,
        "num_boxes_before_nms": len(state.detections_before_nms),
        "num_boxes_before_dinov2": len(state.sam3_detections),
        "num_boxes": len(state.detections),
        "detections": [detection_to_record(detection) for detection in state.detections],
        "sam3_detections_before_dinov2": (
            [detection_to_record(detection) for detection in state.sam3_detections]
            if dinov2_enabled
            else None
        ),
        "dinov2": state.dinov2_metadata,
        "performance": state.performance_result,
    }


def run_auto_labeling(
    cfg: Dict[str, Any],
    *,
    overwrite: Optional[bool] = None,
) -> Dict[str, Any]:
    execution_started_at = utc_now_iso()
    execution_started_monotonic = time.perf_counter()
    preflight = preflight_auto_labeling_config(cfg, overwrite=overwrite)
    runtime_info = configure_auto_label_runtime(cfg)
    sam3_prompt_mode = preflight.sam3_prompt_mode
    class_id = preflight.class_id
    prompts = preflight.prompts
    image_path = preflight.image_path
    output_dir = preflight.output_dir
    image_paths = preflight.image_paths
    input_root = preflight.input_root
    labels_dir = output_dir / "labels"
    visualizations_dir = output_dir / "visualizations"
    performance_test_enabled = preflight.performance_test_enabled
    save_visual_outputs = strict_config_bool(
        cfg,
        "SAVE_VISUAL_OUTPUTS",
        not performance_test_enabled,
    )
    labels_dir.mkdir(parents=True, exist_ok=True)
    if save_visual_outputs:
        visualizations_dir.mkdir(parents=True, exist_ok=True)

    class_names = build_class_names(cfg.get("CLASS_NAMES"), class_id=class_id)
    confidence_threshold = resolve_sam3_confidence_threshold(cfg)
    use_amp = strict_config_bool(cfg, "AMP", True)
    nms_iou_threshold = resolve_nms_iou_threshold(cfg)
    max_detections = optional_int(cfg.get("MAX_DETECTIONS_PER_IMAGE"), default=None)
    max_images = optional_int(cfg.get("MAX_IMAGES"), default=None)
    disable_tqdm = strict_config_bool(cfg, "DISABLE_TQDM", False)
    auto_labeling_batch_size = config_int(cfg, "AUTO_LABELING_BATCH_SIZE", default=1, minimum=1)
    auto_labeling_num_workers = config_int(cfg, "AUTO_LABELING_NUM_WORKERS", default=1, minimum=1)
    dinov2_tile_batch_size = resolve_dinov2_tile_batch_size(cfg)
    dinov2_enabled = preflight.dinov2_enabled
    dinov2_tiling_config = build_dinov2_tiling_config(cfg) if dinov2_enabled else None
    dinov2_save_artifacts = (
        strict_config_bool(cfg, "DINOV2_SAVE_ARTIFACTS", False)
        and not performance_test_enabled
    )
    performance_iou_threshold = preflight.performance_iou_threshold
    gt_label_dir = preflight.gt_label_dir
    gt_index = preflight.gt_index
    original_path = preflight.original_path
    original_index = preflight.original_index
    original_matches = preflight.original_matches
    original_root = preflight.original_root

    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config = build_resolved_auto_labeling_config(cfg, preflight)
    save_yaml(resolved_config_path, resolved_config)

    processor, sam3_info = build_sam3_processor(
        cfg,
        confidence_threshold=confidence_threshold,
        use_amp=use_amp,
    )
    sam3_info["amp"] = bool(use_amp)
    sam3_info["amp_dtype"] = "float16" if use_amp else None
    runtime_info["effective_device"] = sam3_info["device"]
    runtime_info["amp"] = bool(use_amp)
    runtime_info["sam3_amp"] = bool(use_amp)
    runtime_info["sam3_amp_dtype"] = sam3_info["amp_dtype"]
    runtime_info["dinov2_amp_dtype"] = "float16" if use_amp and dinov2_enabled else None
    sam3_info["prompt_mode"] = sam3_prompt_mode
    exemplar_predictor: Optional[Sam3ImageExemplarPredictor] = None
    if sam3_prompt_mode_uses_exemplar(sam3_prompt_mode):
        exemplar_predictor = Sam3ImageExemplarPredictor(
            processor=processor,
            exemplars=preflight.loaded_exemplars,
        )
        with amp_inference_context(use_amp, dtype="float16"):
            exemplar_predictor.build_embeddings()
        sam3_info["image_exemplars"] = exemplar_predictor.metadata()
    else:
        sam3_info["image_exemplars"] = {
            "prompt_label": None,
            "num_exemplars": 0,
            "num_exemplar_boxes": 0,
            "exemplars": [],
        }
    dinov2_model = None
    dinov2_info: Dict[str, Any] = {
        "enabled": dinov2_enabled,
        "amp": bool(use_amp),
        "amp_dtype": "float16" if use_amp else None,
    }
    dinov2_artifact_dirs: Dict[str, Path] = {}
    if dinov2_enabled:
        dinov2_model, dinov2_runtime_info = build_dinov2_runtime(cfg)
        dinov2_info.update(dinov2_runtime_info)
        dinov2_info["original_image_path"] = str(original_path)
        dinov2_info["original_root"] = str(original_root)
        dinov2_info["num_original_images_indexed"] = sum(len(paths) for paths in original_index.values())
        dinov2_info["num_duplicate_original_names"] = sum(1 for paths in original_index.values() if len(paths) > 1)
        if dinov2_save_artifacts:
            dinov2_root = output_dir / "dinov2_visualizations"
            dinov2_artifact_dirs = {
                "diff_heatmap": dinov2_root / "diff_heatmap",
                "box_heatmap": dinov2_root / "box_heatmap",
            }
            for artifact_dir in dinov2_artifact_dirs.values():
                artifact_dir.mkdir(parents=True, exist_ok=True)
            dinov2_info["artifact_dirs"] = {key: str(path) for key, path in dinov2_artifact_dirs.items()}

    candidate_diagnostics_all_images: List[Dict[str, Any]] = []

    per_image_results: List[Dict[str, Any]] = []
    total_boxes_before_nms = 0
    total_boxes = 0
    detected_images = 0
    total_sam3_boxes_after_nms = 0
    total_boxes_filtered_by_dinov2 = 0
    dinov2_images_processed = 0
    performance_totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "num_predictions": 0,
        "num_gt_boxes": 0,
        "images_evaluated": 0,
    }

    progress = tqdm(total=len(image_paths), desc="Auto labeling", disable=disable_tqdm)

    def finalize_state(state: AutoLabelImageState) -> None:
        nonlocal detected_images, total_boxes, total_boxes_before_nms

        if state.image is None:
            raise RuntimeError(f"Image was not loaded before finalization: {state.img_path}")

        width, height = state.image.size
        state.width = width
        state.height = height
        save_yolo_label(state.label_path, detections=state.detections, width=width, height=height)
        if save_visual_outputs and state.vis_path is not None:
            draw_visualization(image=state.image, detections=state.detections, save_path=state.vis_path)

        if performance_test_enabled:
            state.performance_result = evaluate_detections_against_gt(
                image_path=state.img_path,
                detections=state.detections,
                width=width,
                height=height,
                gt_index=gt_index,
                target_class_id=class_id,
                iou_threshold=performance_iou_threshold,
            )
            accumulate_performance_result(performance_totals, state.performance_result)
            performance_totals["images_evaluated"] += 1

        total_boxes_before_nms += len(state.detections_before_nms)
        total_boxes += len(state.detections)
        if state.detections:
            detected_images += 1
        per_image_results.append(build_auto_label_result_record(state, dinov2_enabled=dinov2_enabled))
        progress.update(1)

    def prepare_state_for_image(img_path: Path) -> AutoLabelImageState:
        rel_path = img_path.relative_to(input_root)
        state = AutoLabelImageState(
            img_path=img_path,
            rel_path=rel_path,
            label_path=(labels_dir / rel_path).with_suffix(".txt"),
            vis_path=(visualizations_dir / rel_path).with_suffix(".jpg") if save_visual_outputs else None,
            dinov2_metadata={
                "enabled": dinov2_enabled,
                "amp": bool(use_amp),
                "amp_dtype": "float16" if use_amp else None,
            },
        )

        try:
            state.image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open generated image: {img_path}") from exc
        state.width, state.height = state.image.size

        if dinov2_enabled:
            state.original_match_path = original_matches[img_path]
            try:
                state.original_image = Image.open(state.original_match_path).convert("RGB")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to open original image matched to {img_path}: "
                    f"{state.original_match_path}"
                ) from exc

        return state

    cpu_executor = (
        ThreadPoolExecutor(max_workers=auto_labeling_num_workers)
        if auto_labeling_num_workers > 1
        else None
    )

    try:
        for batch_paths in iter_batches(image_paths, auto_labeling_batch_size):
            prepared_states = ordered_worker_map(cpu_executor, prepare_state_for_image, batch_paths)
            batch_states: List[AutoLabelImageState] = list(prepared_states)

            for state in batch_states:
                state.detections_before_nms = run_sam3_prompt_mode_on_image(
                    processor=processor,
                    exemplar_predictor=exemplar_predictor,
                    image=state.image,
                    prompts=prompts,
                    class_id=class_id,
                    prompt_mode=sam3_prompt_mode,
                    use_amp=use_amp,
                )
                state.sam3_detections = nms_detections(
                    detections=state.detections_before_nms,
                    iou_threshold=nms_iou_threshold,
                    max_detections=max_detections,
                )
                state.detections = list(state.sam3_detections)
                total_sam3_boxes_after_nms += len(state.sam3_detections)

            if dinov2_enabled:
                dino_states = [
                    state
                    for state in batch_states
                    if state.image is not None and state.original_image is not None
                ]
                if dino_states:
                    edge_suppress_patches = resolve_dinov2_edge_suppress_patches(cfg)

                    def build_dinov2_job(state: AutoLabelImageState):
                        aligned_label, alignment = estimate_label_to_original_alignment(
                            label_image=state.image,
                            original_image=state.original_image,
                        )
                        return (
                            state,
                            alignment,
                            DinoPatchDiffInput(
                                original_image=state.original_image,
                                aligned_label_array=aligned_label,
                                alignment=alignment,
                            ),
                        )

                    dino_jobs = ordered_worker_map(cpu_executor, build_dinov2_job, dino_states)
                    ordered_dino_states = [item[0] for item in dino_jobs]
                    alignments = [item[1] for item in dino_jobs]
                    jobs = [item[2] for item in dino_jobs]
                    patch_outputs = compute_dinov2_patch_diffs_batch(
                        model=dinov2_model,
                        device=dinov2_info["device"],
                        jobs=jobs,
                        tiling_config=dinov2_tiling_config,
                        edge_suppress_patches=edge_suppress_patches,
                        tile_batch_size=dinov2_tile_batch_size,
                        use_amp=use_amp,
                    )

                    def apply_dinov2_result(item):
                        state, alignment, patch_output = item
                        artifact_paths = None
                        if dinov2_save_artifacts:
                            artifact_paths = {
                                key: (artifact_dir / state.rel_path).with_suffix(".jpg")
                                for key, artifact_dir in dinov2_artifact_dirs.items()
                            }
                        dinov2_result = apply_dinov2_filter_for_image(
                            cfg=cfg,
                            label_image=state.image,
                            sam3_detections=state.sam3_detections,
                            alignment=alignment,
                            all_patches=patch_output.all_patches,
                            patch_metadata=patch_output.metadata,
                            artifact_paths=artifact_paths,
                            use_amp=use_amp,
                        )
                        return state, dinov2_result

                    dino_results = ordered_worker_map(
                        cpu_executor,
                        apply_dinov2_result,
                        list(zip(ordered_dino_states, alignments, patch_outputs)),
                    )

                    for state, dinov2_result in dino_results:
                        state.detections = dinov2_result.detections
                        state.dinov2_metadata = dinov2_result.metadata
                        state.dinov2_metadata["matched_original_path"] = str(state.original_match_path)
                        state.dinov2_metadata["num_original_matches"] = 1
                        dinov2_images_processed += 1
                        total_boxes_filtered_by_dinov2 += len(state.sam3_detections) - len(state.detections)
                        for diag in dinov2_result.metadata.get("detection_diagnostics", []):
                            candidate_diagnostics_all_images.append(
                                {
                                    "image_path": str(state.img_path),
                                    "relative_path": str(state.rel_path),
                                    **diag,
                                }
                            )

            for state in batch_states:
                finalize_state(state)
    finally:
        progress.close()
        if cpu_executor is not None:
            cpu_executor.shutdown(wait=True)

    classes_txt_path = save_classes_txt(labels_dir, class_names)
    performance_metrics = compute_prf(
        performance_totals["tp"],
        performance_totals["fp"],
        performance_totals["fn"],
    )
    alignment_summary = summarize_alignment_results(
        per_image_results,
        dinov2_enabled=dinov2_enabled,
    )
    environment_summary = collect_runtime_environment()
    execution_ended_at = utc_now_iso()
    execution_duration_seconds = time.perf_counter() - execution_started_monotonic
    summary = {
        "input_path": str(image_path),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "labels_dir": str(labels_dir),
        "visualizations_dir": str(visualizations_dir) if save_visual_outputs else None,
        "classes_txt_path": str(classes_txt_path),
        "resolved_config_path": str(resolved_config_path),
        "output_existing_policy": preflight.output_existing_policy,
        "output_had_existing_entries": preflight.output_has_existing_entries,
        "execution": {
            "started_at_utc": execution_started_at,
            "ended_at_utc": execution_ended_at,
            "duration_seconds": float(execution_duration_seconds),
        },
        "environment": environment_summary,
        "num_images": len(image_paths),
        "detected_images": detected_images,
        "images_without_boxes": len(image_paths) - detected_images,
        "total_boxes_before_nms": total_boxes_before_nms,
        "total_sam3_boxes_after_nms_before_dinov2": total_sam3_boxes_after_nms,
        "total_boxes": total_boxes,
        "total_boxes_filtered_by_dinov2": total_boxes_filtered_by_dinov2,
        "settings": {
            "device": runtime_info["requested_device"],
            "amp": bool(use_amp),
            "sam3_prompt_mode": sam3_prompt_mode,
            "text_prompts": prompts,
            "sam3_image_exemplar_count": sam3_info["image_exemplars"]["num_exemplars"],
            "sam3_image_exemplar_box_count": sam3_info["image_exemplars"]["num_exemplar_boxes"],
            "label_class_id": class_id,
            "class_names": class_names,
            "sam3_confidence_threshold": confidence_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "max_detections_per_image": max_detections,
            "max_images": max_images,
            "auto_labeling_batch_size": auto_labeling_batch_size,
            "auto_labeling_num_workers": auto_labeling_num_workers,
            "save_visual_outputs": save_visual_outputs,
            "performance_test_enabled": performance_test_enabled,
        },
        "runtime": runtime_info,
        "sam3": sam3_info,
        "alignment": alignment_summary,
        "dinov2": {
            **dinov2_info,
            "save_artifacts": dinov2_save_artifacts,
            "filter_strategy": "candidate_adaptive",
            "images_processed": dinov2_images_processed,
            "tiling_mode": "relative" if dinov2_tiling_config is not None else None,
            "input_size": dinov2_tiling_config.input_size if dinov2_tiling_config is not None else None,
            "tile_width_divisor": (
                dinov2_tiling_config.tile_width_divisor if dinov2_tiling_config is not None else None
            ),
            "tile_height_divisor": (
                dinov2_tiling_config.tile_height_divisor if dinov2_tiling_config is not None else None
            ),
            "tile_stride_ratio": (
                dinov2_tiling_config.tile_stride_ratio if dinov2_tiling_config is not None else None
            ),
            "tile_batch_size": dinov2_tile_batch_size,
            "edge_suppress_patches": resolve_dinov2_edge_suppress_patches(cfg),
            "require_patch_fully_inside_sam3_mask": get_dinov2_require_patch_fully_inside_sam3_mask(cfg),
            "adaptive_threshold_method": str(cfg.get("DINOV2_ADAPTIVE_THRESHOLD_METHOD", "otsu")),
            "adaptive_threshold_offset": resolve_dinov2_adaptive_threshold_offset(cfg),
            "candidate_score_field": "within_mask_p75_diff",
        },
        "performance": {
            "enabled": performance_test_enabled,
            "gt_label_dir": str(gt_label_dir) if gt_label_dir is not None else None,
            "iou_threshold": float(performance_iou_threshold),
            "target_class_id": class_id,
            **performance_totals,
            **performance_metrics,
        },
    }

    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "per_image_results.json", {"rows": per_image_results})

    if candidate_diagnostics_all_images:
        kept_diags = [d for d in candidate_diagnostics_all_images if d.get("kept")]
        rejected_diags = [d for d in candidate_diagnostics_all_images if not d.get("kept")]
        reason_counts: Dict[str, int] = {}
        for d in candidate_diagnostics_all_images:
            reason = d.get("kept_reason") or "rejected"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        diagnostic_summary: Dict[str, Any] = {
            "filter_strategy": "candidate_adaptive",
            "total_detections": len(candidate_diagnostics_all_images),
            "kept": len(kept_diags),
            "rejected": len(rejected_diags),
            "kept_reason_counts": reason_counts,
            "adaptive_threshold_stats": safe_stats(
                [
                    d["adaptive_threshold"]
                    for d in candidate_diagnostics_all_images
                    if d.get("adaptive_threshold") is not None
                ]
            ),
            "candidate_score_stats": safe_stats(
                [
                    d["candidate_score"]
                    for d in candidate_diagnostics_all_images
                    if d.get("candidate_score") is not None
                ]
            ),
            "candidate_cutoff_stats": safe_stats(
                [
                    d["candidate_cutoff"]
                    for d in candidate_diagnostics_all_images
                    if d.get("candidate_cutoff") is not None
                ]
            ),
            "rows": candidate_diagnostics_all_images,
        }
        save_json(output_dir / "diagnostic_summary.json", diagnostic_summary)
        print(f"- diagnostic summary : {output_dir / 'diagnostic_summary.json'}")

    print("[done] Auto labeling complete")
    print(f"- input images       : {len(image_paths)}")
    print(f"- labels dir         : {labels_dir}")
    print(f"- resolved config    : {resolved_config_path}")
    print(f"- visualizations dir : {visualizations_dir if save_visual_outputs else 'disabled'}")
    print(f"- total boxes        : {total_boxes}")
    print(f"- images without box : {len(image_paths) - detected_images}")
    if dinov2_enabled:
        print(f"- DINOv2 processed   : {dinov2_images_processed}")
        print(f"- filtered by DINOv2 : {total_boxes_filtered_by_dinov2}")
    if performance_test_enabled:
        print(
            "[performance] "
            f"P={performance_metrics['precision']:.4f} "
            f"R={performance_metrics['recall']:.4f} "
            f"F1={performance_metrics['f1']:.4f} "
            f"TP={performance_totals['tp']} "
            f"FP={performance_totals['fp']} "
            f"FN={performance_totals['fn']}"
        )
    return summary
