"""SAM 3 import, processor construction, and per-image inference."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from grape_auto_labeling.config import (
    normalize_optional_text,
    strict_config_bool,
)
from grape_auto_labeling.labeling.detections import Detection
from grape_auto_labeling.utils.devices import (
    amp_inference_context,
    convert_module_floating_dtype,
    resolve_sam3_device,
)
from grape_auto_labeling.models.sam3_compat import apply_sam3_dtype_compatibility
from grape_auto_labeling.models.sam3_exemplar import (
    Sam3ImageExemplarPredictor,
    sam3_prompt_mode_uses_exemplar,
)
from grape_auto_labeling.utils.tensor_utils import tensor_to_numpy
from grape_auto_labeling.utils.yolo_labels import clip_box


def install_pkg_resources_compat_shim() -> bool:
    try:
        import pkg_resources  # noqa: F401

        return False
    except ModuleNotFoundError:
        pass

    import importlib.resources as resources

    module = types.ModuleType("pkg_resources")

    def resource_filename(package: str, resource_name: str) -> str:
        return str(resources.files(package).joinpath(resource_name))

    module.resource_filename = resource_filename  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = module
    return True


def import_sam3_runtime(sam3_repo_dir: Optional[Path] = None):
    if sam3_repo_dir is not None and sam3_repo_dir.exists():
        repo_text = str(sam3_repo_dir.resolve(strict=False))
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)

    install_pkg_resources_compat_shim()
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ModuleNotFoundError as exc:
        if exc.name == "einops":
            raise ModuleNotFoundError(
                "SAM3 import requires einops. Install it in the active environment "
                "or run `pip install einops`."
            ) from exc
        raise

    return build_sam3_image_model, Sam3Processor


def build_sam3_processor(
    cfg: Dict[str, Any],
    confidence_threshold: float,
    use_amp: bool = False,
):
    sam3_repo_text = normalize_optional_text(cfg.get("SAM3_REPO_DIR"))
    sam3_repo_dir = (
        Path(sam3_repo_text).expanduser().resolve(strict=False)
        if sam3_repo_text
        else None
    )
    if sam3_repo_dir is not None and not sam3_repo_dir.is_dir():
        raise FileNotFoundError(f"SAM3_REPO_DIR is not a directory: {sam3_repo_dir}")

    checkpoint_text = normalize_optional_text(cfg.get("SAM3_CHECKPOINT_PATH"))
    checkpoint_path_obj = (
        Path(checkpoint_text).expanduser().resolve(strict=False)
        if checkpoint_text is not None
        else None
    )
    if checkpoint_path_obj is not None and not checkpoint_path_obj.is_file():
        raise FileNotFoundError(
            f"SAM3_CHECKPOINT_PATH is not a file: {checkpoint_path_obj}"
        )

    load_from_hf_requested = strict_config_bool(
        cfg,
        "SAM3_LOAD_FROM_HF",
        checkpoint_path_obj is None,
    )
    if checkpoint_path_obj is None and not load_from_hf_requested:
        raise ValueError(
            "SAM3_CHECKPOINT_PATH must be set when SAM3_LOAD_FROM_HF is false."
        )

    device = resolve_sam3_device(cfg.get("DEVICE", "auto"), name="DEVICE")
    build_sam3_image_model, Sam3Processor = import_sam3_runtime(sam3_repo_dir=sam3_repo_dir)
    dtype_compatibility = apply_sam3_dtype_compatibility()

    builder_load_from_hf = checkpoint_path_obj is None and load_from_hf_requested
    checkpoint_source = "local" if checkpoint_path_obj is not None else "huggingface"
    checkpoint_path = str(checkpoint_path_obj) if checkpoint_path_obj is not None else None

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=builder_load_from_hf,
        enable_segmentation=True,
    )
    if use_amp:
        import torch

        convert_module_floating_dtype(model, torch.float16)
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    return processor, {
        "device": device,
        "checkpoint_path": checkpoint_path,
        "checkpoint_source": checkpoint_source,
        "load_from_hf": load_from_hf_requested,
        "effective_builder_load_from_hf": builder_load_from_hf,
        "sam3_repo_dir": str(sam3_repo_dir) if sam3_repo_dir else None,
        "dtype_compatibility": dtype_compatibility,
    }


def extract_sam3_masks(raw_masks: Any, width: int, height: int) -> List[np.ndarray]:
    masks = tensor_to_numpy(raw_masks)
    if masks.size == 0:
        return []

    if masks.ndim == 4:
        if masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.shape[-1] == 1:
            masks = masks[..., 0]
        elif masks.shape[0] == 1:
            masks = masks[0]
        else:
            masks = masks.reshape((-1, *masks.shape[-2:]))
    elif masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim != 3:
        return []

    normalized: List[np.ndarray] = []
    for mask in masks:
        mask_bool = np.asarray(mask) > 0.5
        if mask_bool.shape != (height, width):
            resized = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L").resize(
                (width, height),
                Image.NEAREST,
            )
            mask_bool = np.asarray(resized) > 0
        normalized.append(mask_bool.astype(bool, copy=False))
    return normalized


def run_sam3_on_image(
    processor: Any,
    image: Image.Image,
    prompts: Sequence[str],
    class_id: int,
    use_amp: bool = False,
) -> List[Detection]:
    width, height = image.size
    detections: List[Detection] = []

    with amp_inference_context(use_amp, dtype="float16"):
        state = processor.set_image(image)

        for prompt in prompts:
            output = processor.set_text_prompt(prompt=prompt, state=state)
            boxes = tensor_to_numpy(output.get("boxes")).reshape(-1, 4)
            scores = tensor_to_numpy(output.get("scores")).reshape(-1)
            masks = extract_sam3_masks(output.get("masks"), width=width, height=height)
            for index, (box, score) in enumerate(zip(boxes, scores)):
                clipped = clip_box(box, width=width, height=height)
                if clipped is None:
                    continue
                detections.append(
                    Detection(
                        box_xyxy=clipped,
                        score=float(score),
                        prompt=prompt,
                        class_id=class_id,
                        mask=masks[index] if index < len(masks) else None,
                    )
                )

    return detections


def sam3_output_to_detections(
    output: Dict[str, Any],
    width: int,
    height: int,
    prompt: str,
    class_id: int,
) -> List[Detection]:
    detections: List[Detection] = []
    boxes = tensor_to_numpy(output.get("boxes")).reshape(-1, 4)
    scores = tensor_to_numpy(output.get("scores")).reshape(-1)
    masks = extract_sam3_masks(output.get("masks"), width=width, height=height)
    for index, (box, score) in enumerate(zip(boxes, scores)):
        clipped = clip_box(box, width=width, height=height)
        if clipped is None:
            continue
        detections.append(
            Detection(
                box_xyxy=clipped,
                score=float(score),
                prompt=prompt,
                class_id=class_id,
                mask=masks[index] if index < len(masks) else None,
            )
        )
    return detections


def run_sam3_image_exemplar_on_image(
    predictor: Sam3ImageExemplarPredictor,
    image: Image.Image,
    class_id: int,
    text_prompt: Optional[str] = None,
    use_amp: bool = False,
) -> List[Detection]:
    width, height = image.size
    with amp_inference_context(use_amp, dtype="float16"):
        output = predictor.predict(image, text_prompt=text_prompt)
    prompt_label = (
        f"{text_prompt} + {predictor.prompt_label}"
        if text_prompt is not None
        else predictor.prompt_label
    )
    return sam3_output_to_detections(
        output=output,
        width=width,
        height=height,
        prompt=prompt_label,
        class_id=class_id,
    )


def run_sam3_prompt_mode_on_image(
    processor: Any,
    exemplar_predictor: Optional[Sam3ImageExemplarPredictor],
    image: Image.Image,
    prompts: Sequence[str],
    class_id: int,
    prompt_mode: str,
    use_amp: bool = False,
) -> List[Detection]:
    detections: List[Detection] = []
    if prompt_mode == "text":
        detections.extend(
            run_sam3_on_image(
                processor=processor,
                image=image,
                prompts=prompts,
                class_id=class_id,
                use_amp=use_amp,
            )
        )
        return detections

    if sam3_prompt_mode_uses_exemplar(prompt_mode):
        if exemplar_predictor is None:
            raise ValueError("SAM3 image exemplar predictor is required for exemplar mode.")
        for text_prompt in prompts:
            detections.extend(
                run_sam3_image_exemplar_on_image(
                    predictor=exemplar_predictor,
                    image=image,
                    class_id=class_id,
                    text_prompt=text_prompt,
                    use_amp=use_amp,
                )
            )
    return detections
