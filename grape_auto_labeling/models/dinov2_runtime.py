"""DINOv2 model construction, tiling geometry, and patch difference scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from grape_auto_labeling.labeling.alignment import AlignmentResult, transform_boxes_with_matrix
from grape_auto_labeling.config import (
    normalize_optional_text,
    DEFAULT_DINOV2_INPUT_SIZE,
    DEFAULT_DINOV2_TILE_HEIGHT_DIVISOR,
    DEFAULT_DINOV2_TILE_STRIDE_RATIO,
    DEFAULT_DINOV2_TILE_WIDTH_DIVISOR,
    DINOV2_PATCH_SIZE,
    config_float,
    config_int,
)
from grape_auto_labeling.utils.devices import (
    amp_inference_context,
    resolve_torch_device,
)
from grape_auto_labeling.models.model_artifacts import (
    build_torch_hub_repo_spec,
    torch_hub_checkpoint_path,
)
from grape_auto_labeling.utils.tensor_utils import tensor_to_numpy


@dataclass
class DinoPatch:
    original_box_xyxy: List[float]
    label_box_xyxy: List[float]
    score: float


@dataclass
class DinoPatchDiffInput:
    original_image: Image.Image
    aligned_label_array: np.ndarray
    alignment: AlignmentResult


@dataclass
class DinoTilingConfig:
    input_size: int
    tile_width_divisor: float
    tile_height_divisor: float
    tile_stride_ratio: float


@dataclass
class DinoPatchDiffAccumulator:
    all_patches: List[DinoPatch] = field(default_factory=list)
    num_patch_boxes: int = 0
    max_diff: float = 0.0
    min_diff: Optional[float] = None
    total_diff: float = 0.0
    num_tiles: int = 0


@dataclass
class DinoPatchDiffOutput:
    all_patches: List[DinoPatch]
    metadata: Dict[str, Any]


def build_dinov2_runtime(cfg: Dict[str, Any]):
    import torch

    model_name = str(cfg.get("DINOV2_MODEL_NAME", "dinov2_vitb14")).strip()
    device = resolve_torch_device(cfg.get("DEVICE", "auto"), "DEVICE")

    hub_repo = str(cfg.get("DINOV2_HUB_REPO", "facebookresearch/dinov2")).strip()
    hub_ref = normalize_optional_text(cfg.get("DINOV2_HUB_REF"))
    repo_dir_text = normalize_optional_text(cfg.get("DINOV2_REPO_DIR"))
    repo_dir = (
        Path(repo_dir_text).expanduser().resolve(strict=False)
        if repo_dir_text is not None
        else None
    )
    if repo_dir is not None and hub_ref is not None:
        raise ValueError(
            "DINOV2_REPO_DIR and DINOV2_HUB_REF are mutually exclusive. "
            "Use a local repository or a Torch Hub source, not both."
        )

    if repo_dir is not None:
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"DINOV2_REPO_DIR is not a directory: {repo_dir}")
        hub_source = "local"
        hub_repo_spec = str(repo_dir)
    else:
        hub_source = "github"
        hub_repo_spec = build_torch_hub_repo_spec(hub_repo, hub_ref)

    checkpoint_text = normalize_optional_text(
        cfg.get("DINOV2_CHECKPOINT_PATH")
    )
    checkpoint_path = (
        Path(checkpoint_text).expanduser().resolve(strict=False)
        if checkpoint_text is not None
        else None
    )
    checkpoint_filename = str(
        cfg.get("DINOV2_CHECKPOINT_FILENAME", f"{model_name}_pretrain.pth")
    ).strip()
    if checkpoint_path is None:
        checkpoint_path = torch_hub_checkpoint_path(torch, checkpoint_filename)
    elif not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DINOV2_CHECKPOINT_PATH is not a file: {checkpoint_path}"
        )

    model_kwargs: Dict[str, Any] = {}
    if checkpoint_text is not None:
        model_kwargs["weights"] = str(checkpoint_path)
    try:
        if hub_source == "local":
            model = torch.hub.load(
                hub_repo_spec,
                model_name,
                source="local",
                **model_kwargs,
            )
        else:
            hub_load_options: Dict[str, Any] = {"trust_repo": True}
            if hub_ref is not None:
                # Torch Hub's default validation only recognizes branch/tag
                # heads. Historical full commit SHAs require direct ref loading.
                hub_load_options["skip_validation"] = True
            model = torch.hub.load(
                hub_repo_spec,
                model_name,
                **hub_load_options,
                **model_kwargs,
            )
    except TypeError as exc:
        # Compatibility with older torch releases that predate these Hub
        # options. Do not hide TypeErrors raised by the model entry point.
        unsupported_hub_options = {
            key
            for key in ("trust_repo", "skip_validation")
            if key in str(exc)
        }
        if hub_source == "local" or not unsupported_hub_options:
            raise
        fallback_options = {
            key: value
            for key, value in hub_load_options.items()
            if key not in unsupported_hub_options
        }
        model = torch.hub.load(
            hub_repo_spec,
            model_name,
            **fallback_options,
            **model_kwargs,
        )

    model.to(device)
    model.eval()
    return model, {
        "model_name": model_name,
        "device": device,
        "hub_repo": hub_repo,
        "hub_ref": hub_ref,
        "hub_repo_spec": hub_repo_spec,
        "hub_source": hub_source,
        "repo_dir": str(repo_dir) if repo_dir is not None else None,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_filename": checkpoint_filename,
    }


def build_dinov2_tiling_config(cfg: Dict[str, Any]) -> DinoTilingConfig:
    input_size = config_int(
        cfg,
        "DINOV2_INPUT_SIZE",
        default=DEFAULT_DINOV2_INPUT_SIZE,
        minimum=1,
    )
    if input_size % DINOV2_PATCH_SIZE:
        raise ValueError(
            f"DINOV2_INPUT_SIZE must be a multiple of the DINOv2 patch size "
            f"{DINOV2_PATCH_SIZE}; got {input_size}. Otherwise the patch token grid "
            "covers only part of the resized crop and every patch box is shifted."
        )

    width_divisor = config_float(
        cfg,
        "DINOV2_TILE_WIDTH_DIVISOR",
        default=DEFAULT_DINOV2_TILE_WIDTH_DIVISOR,
        minimum=0.0,
    )
    height_divisor = config_float(
        cfg,
        "DINOV2_TILE_HEIGHT_DIVISOR",
        default=DEFAULT_DINOV2_TILE_HEIGHT_DIVISOR,
        minimum=0.0,
    )
    stride_ratio = config_float(
        cfg,
        "DINOV2_TILE_STRIDE_RATIO",
        default=DEFAULT_DINOV2_TILE_STRIDE_RATIO,
        minimum=0.0,
    )
    if width_divisor <= 0.0 or height_divisor <= 0.0:
        raise ValueError("DINOv2 tile divisors must be positive.")
    if stride_ratio <= 0.0:
        raise ValueError("DINOV2_TILE_STRIDE_RATIO must be positive.")
    return DinoTilingConfig(
        input_size=input_size,
        tile_width_divisor=width_divisor,
        tile_height_divisor=height_divisor,
        tile_stride_ratio=stride_ratio,
    )


def resolve_dinov2_tile_geometry(
    width: int,
    height: int,
    tiling_config: DinoTilingConfig,
) -> Dict[str, Any]:
    tile_width = max(1, min(width, int(round(float(width) / tiling_config.tile_width_divisor))))
    tile_height = max(1, min(height, int(round(float(height) / tiling_config.tile_height_divisor))))
    stride_x = max(1, int(round(float(tile_width) * tiling_config.tile_stride_ratio)))
    stride_y = max(1, int(round(float(tile_height) * tiling_config.tile_stride_ratio)))
    return {
        "mode": "relative",
        "input_size": tiling_config.input_size,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "tile_width_divisor": tiling_config.tile_width_divisor,
        "tile_height_divisor": tiling_config.tile_height_divisor,
        "tile_stride_ratio": tiling_config.tile_stride_ratio,
    }


def iter_tile_windows(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    stride_x: int,
    stride_y: int,
):
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("DINOv2 source tile width/height must be positive.")
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("DINOv2 source tile stride must be positive.")

    if width <= tile_width:
        xs = [0]
    else:
        xs = list(range(0, width - tile_width + 1, stride_x))
        if xs[-1] != width - tile_width:
            xs.append(width - tile_width)

    if height <= tile_height:
        ys = [0]
    else:
        ys = list(range(0, height - tile_height + 1, stride_y))
        if ys[-1] != height - tile_height:
            ys.append(height - tile_height)

    for y1 in ys:
        for x1 in xs:
            yield x1, y1, min(x1 + tile_width, width), min(y1 + tile_height, height)


def prepare_dinov2_batch_from_crops(
    crops: Sequence[np.ndarray],
    input_size: int,
    device: str,
):
    import torch

    tensors = []
    for crop in crops:
        pil = Image.fromarray(crop.astype(np.uint8), mode="RGB").resize(
            (input_size, input_size),
            Image.BICUBIC,
        )
        array = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor)

    if not tensors:
        raise ValueError("DINOv2 crop batch must contain at least one crop.")

    batch = torch.stack(tensors, dim=0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def extract_dinov2_patch_tokens(model: Any, batch: Any):
    if hasattr(model, "forward_features"):
        output = model.forward_features(batch)
    else:
        output = model(batch)
    if isinstance(output, dict):
        tokens = output.get("x_norm_patchtokens")
        if tokens is None:
            raise KeyError("DINOv2 forward_features output has no x_norm_patchtokens key.")
        return tokens
    return output


def infer_square_patch_grid(num_tokens: int) -> int:
    grid = int(round(math.sqrt(float(num_tokens))))
    if grid * grid != num_tokens:
        raise ValueError(f"Expected square DINOv2 patch token grid, got {num_tokens} tokens.")
    return grid


def append_dinov2_tile_patches(
    accumulator: DinoPatchDiffAccumulator,
    diffs_np: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
    original_to_label: np.ndarray,
    label_width: int,
    label_height: int,
    edge_suppress_patches: int,
) -> None:
    grid = int(diffs_np.shape[0])
    patch_w = float(x2 - x1) / float(grid)
    patch_h = float(y2 - y1) / float(grid)
    rows, cols = np.indices((grid, grid), dtype=np.float64)
    original_boxes = np.stack(
        [
            float(x1) + cols * patch_w,
            float(y1) + rows * patch_h,
            float(x1) + (cols + 1.0) * patch_w,
            float(y1) + (rows + 1.0) * patch_h,
        ],
        axis=-1,
    ).reshape(-1, 4)
    scores = diffs_np.reshape(-1).astype(np.float32, copy=True)

    if edge_suppress_patches > 0:
        margin_x = float(edge_suppress_patches) * patch_w
        margin_y = float(edge_suppress_patches) * patch_h
        center_x = (original_boxes[:, 0] + original_boxes[:, 2]) / 2.0
        center_y = (original_boxes[:, 1] + original_boxes[:, 3]) / 2.0
        suppress = (
            (center_x < margin_x)
            | (center_y < margin_y)
            | (center_x > float(image_width) - margin_x)
            | (center_y > float(image_height) - margin_y)
        )
        scores[suppress] = 0.0

    label_boxes = transform_boxes_with_matrix(
        original_boxes,
        original_to_label,
        width=label_width,
        height=label_height,
    )
    accumulator.num_tiles += 1
    for original_box, label_box, score_value in zip(original_boxes, label_boxes, scores):
        if label_box is None:
            continue

        score = float(score_value)
        accumulator.num_patch_boxes += 1
        accumulator.max_diff = max(accumulator.max_diff, score)
        accumulator.min_diff = score if accumulator.min_diff is None else min(accumulator.min_diff, score)
        accumulator.total_diff += score
        patch = DinoPatch(
            original_box_xyxy=[float(value) for value in original_box],
            label_box_xyxy=label_box,
            score=score,
        )
        accumulator.all_patches.append(patch)


def build_dinov2_patch_metadata(
    accumulator: DinoPatchDiffAccumulator,
    use_amp: bool,
    tile_batch_size: int,
    tile_geometry: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "num_tiles": accumulator.num_tiles,
        "num_patch_boxes": accumulator.num_patch_boxes,
        "tiling_mode": tile_geometry.get("mode"),
        "dinov2_input_size": tile_geometry.get("input_size"),
        "source_tile_width": tile_geometry.get("tile_width"),
        "source_tile_height": tile_geometry.get("tile_height"),
        "source_stride_x": tile_geometry.get("stride_x"),
        "source_stride_y": tile_geometry.get("stride_y"),
        "tile_width_divisor": tile_geometry.get("tile_width_divisor"),
        "tile_height_divisor": tile_geometry.get("tile_height_divisor"),
        "tile_stride_ratio": tile_geometry.get("tile_stride_ratio"),
        "max_diff": float(accumulator.max_diff),
        "min_diff": float(accumulator.min_diff) if accumulator.min_diff is not None else None,
        "mean_diff": (
            float(accumulator.total_diff / accumulator.num_patch_boxes)
            if accumulator.num_patch_boxes > 0
            else None
        ),
        "amp": bool(use_amp),
        "amp_dtype": "float16" if use_amp else None,
        "tile_batch_size": int(tile_batch_size),
    }


def compute_dinov2_patch_diffs_batch(
    model: Any,
    device: str,
    jobs: Sequence[DinoPatchDiffInput],
    tiling_config: DinoTilingConfig,
    edge_suppress_patches: int,
    tile_batch_size: int,
    use_amp: bool = False,
) -> List[DinoPatchDiffOutput]:
    import torch
    import torch.nn.functional as functional

    if not jobs:
        return []

    tile_batch_size = max(1, int(tile_batch_size))
    accumulators = [DinoPatchDiffAccumulator() for _ in jobs]
    job_states: List[Dict[str, Any]] = []
    for job in jobs:
        original_array = np.asarray(job.original_image.convert("RGB"))
        original_h, original_w = original_array.shape[:2]
        label_w, label_h = job.alignment.label_size
        tile_geometry = resolve_dinov2_tile_geometry(
            width=original_w,
            height=original_h,
            tiling_config=tiling_config,
        )
        job_states.append(
            {
                "original_array": original_array,
                "original_w": original_w,
                "original_h": original_h,
                "label_w": label_w,
                "label_h": label_h,
                "tile_geometry": tile_geometry,
                "original_to_label": np.asarray(job.alignment.matrix_original_to_label, dtype=np.float64),
                "aligned_label_array": job.aligned_label_array,
            }
        )

    pending_tiles: List[Dict[str, Any]] = []

    def flush_pending_tiles() -> None:
        if not pending_tiles:
            return

        crops: List[np.ndarray] = []
        for tile in pending_tiles:
            crops.append(tile["original_crop"])
            crops.append(tile["label_crop"])

        batch = prepare_dinov2_batch_from_crops(crops, tiling_config.input_size, device)
        with amp_inference_context(use_amp, device=device):
            tokens = extract_dinov2_patch_tokens(model, batch)
        tokens = tokens.float()
        grid = infer_square_patch_grid(int(tokens.shape[1]))
        diffs = 1.0 - functional.cosine_similarity(tokens[0::2], tokens[1::2], dim=-1)
        diffs_np = tensor_to_numpy(diffs).reshape(len(pending_tiles), grid, grid)

        for tile, tile_diffs in zip(pending_tiles, diffs_np):
            job_index = int(tile["job_index"])
            state = job_states[job_index]
            append_dinov2_tile_patches(
                accumulators[job_index],
                tile_diffs,
                x1=tile["x1"],
                y1=tile["y1"],
                x2=tile["x2"],
                y2=tile["y2"],
                image_width=state["original_w"],
                image_height=state["original_h"],
                original_to_label=state["original_to_label"],
                label_width=state["label_w"],
                label_height=state["label_h"],
                edge_suppress_patches=edge_suppress_patches,
            )
        pending_tiles.clear()

    with torch.inference_mode():
        for job_index, state in enumerate(job_states):
            original_array = state["original_array"]
            aligned_label_array = state["aligned_label_array"]
            original_h = int(state["original_h"])
            original_w = int(state["original_w"])
            tile_geometry = state["tile_geometry"]
            for x1, y1, x2, y2 in iter_tile_windows(
                original_w,
                original_h,
                tile_width=int(tile_geometry["tile_width"]),
                tile_height=int(tile_geometry["tile_height"]),
                stride_x=int(tile_geometry["stride_x"]),
                stride_y=int(tile_geometry["stride_y"]),
            ):
                original_crop = original_array[y1:y2, x1:x2]
                label_crop = aligned_label_array[y1:y2, x1:x2]
                if original_crop.size == 0 or label_crop.size == 0:
                    continue

                pending_tiles.append(
                    {
                        "job_index": job_index,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "original_crop": original_crop,
                        "label_crop": label_crop,
                    }
                )
                if len(pending_tiles) >= tile_batch_size:
                    flush_pending_tiles()
        flush_pending_tiles()

    outputs: List[DinoPatchDiffOutput] = []
    for accumulator, state in zip(accumulators, job_states):
        outputs.append(
            DinoPatchDiffOutput(
                all_patches=accumulator.all_patches,
                metadata=build_dinov2_patch_metadata(
                    accumulator,
                    use_amp=use_amp,
                    tile_batch_size=tile_batch_size,
                    tile_geometry=state["tile_geometry"],
                ),
            )
        )
    return outputs
