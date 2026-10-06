from __future__ import annotations

import contextlib
import functools
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


PROMPT_MODE_TEXT = "text"
PROMPT_MODE_TEXT_AND_EXEMPLAR = "text_and_exemplar"
SAM3_PROMPT_MODES = {
    PROMPT_MODE_TEXT,
    PROMPT_MODE_TEXT_AND_EXEMPLAR,
}


@dataclass(frozen=True)
class Sam3ImageExemplarSpec:
    image_path: Path
    label_path: Path
    class_id: int


@dataclass
class LoadedSam3ImageExemplar:
    spec: Sam3ImageExemplarSpec
    image: Image.Image
    boxes_xyxy: List[List[float]]


@dataclass
class Sam3ImageExemplarEmbedding:
    label: str
    image_path: Path
    label_path: Path
    class_id: int
    num_boxes: int
    prompt_embed: Any
    prompt_mask: Any


def normalize_sam3_prompt_mode(raw_mode: Any) -> str:
    mode = str(raw_mode if raw_mode is not None else PROMPT_MODE_TEXT).strip().lower()
    if mode not in SAM3_PROMPT_MODES:
        raise ValueError(
            "SAM3_PROMPT_MODE must be one of "
            f"{', '.join(sorted(SAM3_PROMPT_MODES))}; got {raw_mode!r}."
        )
    return mode


def sam3_prompt_mode_uses_exemplar(prompt_mode: str) -> bool:
    return prompt_mode == PROMPT_MODE_TEXT_AND_EXEMPLAR


def parse_sam3_image_exemplar_specs(
    raw_exemplars: Any,
    default_class_id: int,
) -> List[Sam3ImageExemplarSpec]:
    if raw_exemplars is None:
        return []
    if isinstance(raw_exemplars, dict):
        records = [raw_exemplars]
    else:
        records = list(raw_exemplars)

    specs: List[Sam3ImageExemplarSpec] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                "Each SAM3_IMAGE_EXEMPLARS entry must be a mapping with "
                f"image_path and label_path; entry {index} is {record!r}."
            )

        image_path = _resolve_existing_path(
            record.get("image_path"),
            f"SAM3_IMAGE_EXEMPLARS[{index}].image_path",
        )
        label_path = _resolve_existing_path(
            record.get("label_path"),
            f"SAM3_IMAGE_EXEMPLARS[{index}].label_path",
        )
        class_id = int(record.get("class_id", default_class_id))
        if class_id < 0:
            raise ValueError(
                f"SAM3_IMAGE_EXEMPLARS[{index}].class_id must be non-negative; "
                f"got {class_id}."
            )
        specs.append(
            Sam3ImageExemplarSpec(
                image_path=image_path,
                label_path=label_path,
                class_id=class_id,
            )
        )
    return specs


def validate_exemplar_yolo_label(label_path: Path) -> int:
    """Validate strict normalized YOLO box syntax for an exemplar label."""
    num_boxes = 0
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Exemplar YOLO label line {line_number} in {label_path} must "
                f"contain exactly 5 fields; got {len(parts)}: {raw_line}"
            )

        try:
            class_value = float(parts[0])
            coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(
                f"Exemplar YOLO label line {line_number} in {label_path} contains "
                f"a non-numeric value: {raw_line}"
            ) from exc

        if not np.isfinite(class_value) or not np.isfinite(coordinates).all():
            raise ValueError(
                f"Exemplar YOLO label line {line_number} in {label_path} must "
                f"contain only finite values: {raw_line}"
            )
        if class_value < 0 or not float(class_value).is_integer():
            raise ValueError(
                f"Exemplar YOLO label line {line_number} in {label_path} has an "
                f"invalid class ID: {parts[0]}"
            )

        center_x, center_y, box_width, box_height = [float(value) for value in coordinates]
        if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
            raise ValueError(
                f"Exemplar YOLO label line {line_number} in {label_path} has a "
                f"center outside [0, 1]: {raw_line}"
            )
        if not 0.0 < box_width <= 1.0 or not 0.0 < box_height <= 1.0:
            raise ValueError(
                f"Exemplar YOLO label line {line_number} in {label_path} must have "
                f"width and height in (0, 1]: {raw_line}"
            )

        left = center_x - box_width / 2.0
        top = center_y - box_height / 2.0
        right = center_x + box_width / 2.0
        bottom = center_y + box_height / 2.0
        if left < 0.0 or top < 0.0 or right > 1.0 or bottom > 1.0:
            raise ValueError(
                f"Exemplar YOLO label line {line_number} in {label_path} defines "
                f"a box outside normalized image bounds: {raw_line}"
            )
        num_boxes += 1

    if num_boxes == 0:
        raise ValueError(f"Exemplar YOLO label file contains no boxes: {label_path}")
    return num_boxes


def load_sam3_image_exemplars(
    raw_exemplars: Any,
    default_class_id: int,
    load_yolo_boxes: Callable[[Path, int, int, Optional[int]], List[List[float]]],
) -> List[LoadedSam3ImageExemplar]:
    specs = parse_sam3_image_exemplar_specs(
        raw_exemplars,
        default_class_id=default_class_id,
    )
    if not specs:
        raise ValueError("SAM3_IMAGE_EXEMPLARS must contain at least one exemplar.")

    loaded: List[LoadedSam3ImageExemplar] = []
    for spec in specs:
        validate_exemplar_yolo_label(spec.label_path)
        try:
            image = Image.open(spec.image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Failed to open SAM3 exemplar image: {spec.image_path}"
            ) from exc
        width, height = image.size
        boxes = load_yolo_boxes(
            spec.label_path,
            width=width,
            height=height,
            target_class_id=spec.class_id,
        )
        if not boxes:
            raise ValueError(
                "SAM3 image exemplar has no valid YOLO boxes after class filtering: "
                f"image_path={spec.image_path}, label_path={spec.label_path}, "
                f"class_id={spec.class_id}"
            )
        loaded.append(
            LoadedSam3ImageExemplar(
                spec=spec,
                image=image,
                boxes_xyxy=boxes,
            )
        )
    return loaded


def xyxy_to_normalized_cxcywh(
    box_xyxy: Sequence[float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid xyxy box: {box_xyxy}")

    cx = ((x1 + x2) / 2.0) / float(width)
    cy = ((y1 + y2) / 2.0) / float(height)
    box_w = (x2 - x1) / float(width)
    box_h = (y2 - y1) / float(height)
    values = np.clip([cx, cy, box_w, box_h], 0.0, 1.0)
    return tuple(float(value) for value in values)


@contextlib.contextmanager
def boxes_pool_only_geometry(geometry_encoder: Any):
    has_direct = hasattr(geometry_encoder, "boxes_direct_project")
    has_pos = hasattr(geometry_encoder, "boxes_pos_enc_project")
    direct_project = getattr(geometry_encoder, "boxes_direct_project", None)
    pos_enc_project = getattr(geometry_encoder, "boxes_pos_enc_project", None)
    if has_direct:
        geometry_encoder.boxes_direct_project = None
    if has_pos:
        geometry_encoder.boxes_pos_enc_project = None
    try:
        yield
    finally:
        if has_direct:
            geometry_encoder.boxes_direct_project = direct_project
        if has_pos:
            geometry_encoder.boxes_pos_enc_project = pos_enc_project


class Sam3VisualPromptRuntimeAdapter:
    """Inject cached visual prompts while delegating inference to SAM3.

    SAM3's public image processor does not currently expose a way to pass a
    visual-prompt embedding from another image.  The model's prompt encoder
    accepts that embedding, however.  This adapter temporarily supplies the
    embedding to the prompt encoder and leaves SAM3's own grounding and
    post-processing paths in charge of the rest of inference.

    Calls through one adapter are serialized because the prompt encoder is
    replaced only for the duration of a call.  Code outside this adapter must
    not use the same SAM3 model concurrently.
    """

    _OUTPUT_KEYS = ("masks_logits", "masks", "boxes", "scores")

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self._lock = threading.RLock()
        self._active = False

    @contextlib.contextmanager
    def serialized(self):
        """Prevent other calls through this adapter during model preparation."""
        with self._lock:
            yield

    def forward(
        self,
        state: Dict[str, Any],
        visual_prompt_embed: Any,
        visual_prompt_mask: Any,
    ) -> Dict[str, Any]:
        model = self.processor.model
        encode_prompt = getattr(model, "_encode_prompt", None)
        forward_grounding = getattr(self.processor, "_forward_grounding", None)
        if not callable(encode_prompt) or not callable(forward_grounding):
            raise RuntimeError(
                "The installed SAM3 runtime does not expose the prompt encoder and "
                "image grounding hooks required by exemplar mode."
            )

        with self._lock:
            if self._active:
                raise RuntimeError("Nested SAM3 visual-prompt adapter calls are unsupported.")
            self._active = True
            try:
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = model._get_dummy_prompt()
                with self._inject_visual_prompt(
                    model=model,
                    encode_prompt=encode_prompt,
                    visual_prompt_embed=visual_prompt_embed,
                    visual_prompt_mask=visual_prompt_mask,
                ):
                    output = forward_grounding(state)
                return {key: output[key] for key in self._OUTPUT_KEYS}
            finally:
                self._active = False

    @staticmethod
    @contextlib.contextmanager
    def _inject_visual_prompt(
        model: Any,
        encode_prompt: Callable[..., Any],
        visual_prompt_embed: Any,
        visual_prompt_mask: Any,
    ):
        instance_attributes = vars(model)
        missing = object()
        previous_instance_value = instance_attributes.get("_encode_prompt", missing)

        @functools.wraps(encode_prompt)
        def encode_prompt_with_visual(*args: Any, **kwargs: Any):
            kwargs["visual_prompt_embed"] = visual_prompt_embed
            kwargs["visual_prompt_mask"] = visual_prompt_mask
            return encode_prompt(*args, **kwargs)

        model._encode_prompt = encode_prompt_with_visual
        try:
            yield
        finally:
            if previous_instance_value is missing:
                delattr(model, "_encode_prompt")
            else:
                model._encode_prompt = previous_instance_value


class Sam3ImageExemplarPredictor:
    def __init__(
        self,
        processor: Any,
        exemplars: Sequence[LoadedSam3ImageExemplar],
    ) -> None:
        if not exemplars:
            raise ValueError("At least one loaded SAM3 image exemplar is required.")
        self.processor = processor
        self.exemplars = list(exemplars)
        self.embeddings: List[Sam3ImageExemplarEmbedding] = []
        self.runtime_adapter = Sam3VisualPromptRuntimeAdapter(processor)

    @property
    def prompt_label(self) -> str:
        if len(self.embeddings) == 1:
            return self.embeddings[0].label
        stems = "+".join(embedding.image_path.stem for embedding in self.embeddings)
        return f"image_exemplar:{stems}"

    @property
    def num_boxes(self) -> int:
        return sum(len(exemplar.boxes_xyxy) for exemplar in self.exemplars)

    def metadata(self) -> Dict[str, Any]:
        return {
            "prompt_label": self.prompt_label if self.embeddings else None,
            "num_exemplars": len(self.exemplars),
            "num_exemplar_boxes": self.num_boxes,
            "exemplars": [
                {
                    "image_path": str(exemplar.spec.image_path),
                    "label_path": str(exemplar.spec.label_path),
                    "class_id": int(exemplar.spec.class_id),
                    "num_boxes": len(exemplar.boxes_xyxy),
                }
                for exemplar in self.exemplars
            ],
        }

    def build_embeddings(self) -> None:
        import torch

        with self.runtime_adapter.serialized(), torch.inference_mode():
            geometry_encoder = self.processor.model.geometry_encoder
            if getattr(geometry_encoder, "boxes_pool_project", None) is None:
                raise RuntimeError(
                    "SAM3 image exemplar mode requires geometry_encoder.boxes_pool_project."
                )

            embeddings: List[Sam3ImageExemplarEmbedding] = []
            for exemplar in self.exemplars:
                state = self.processor.set_image(exemplar.image)
                self._set_text_prompt(state, "visual")
                geometric_prompt = self._build_box_prompt(
                    boxes_xyxy=exemplar.boxes_xyxy,
                    width=exemplar.image.width,
                    height=exemplar.image.height,
                )
                with boxes_pool_only_geometry(geometry_encoder):
                    prompt_embed, prompt_mask, _ = self.processor.model._encode_prompt(
                        backbone_out=state["backbone_out"],
                        find_input=self.processor.find_stage,
                        geometric_prompt=geometric_prompt,
                        encode_text=False,
                    )
                embeddings.append(
                    Sam3ImageExemplarEmbedding(
                        label=f"image_exemplar:{exemplar.spec.image_path.stem}",
                        image_path=exemplar.spec.image_path,
                        label_path=exemplar.spec.label_path,
                        class_id=exemplar.spec.class_id,
                        num_boxes=len(exemplar.boxes_xyxy),
                        prompt_embed=prompt_embed.detach().clone(),
                        prompt_mask=prompt_mask.detach().clone(),
                    )
                )

        self.embeddings = embeddings

    def predict(self, image: Image.Image, text_prompt: Optional[str] = None) -> Dict[str, Any]:
        import torch

        if not self.embeddings:
            raise RuntimeError("Call build_embeddings() before exemplar prediction.")

        with self.runtime_adapter.serialized(), torch.inference_mode():
            state = self.processor.set_image(image)
            self._set_text_prompt(state, text_prompt or "visual")
            visual_prompt_embed, visual_prompt_mask = self._combined_visual_prompt(
                state=state,
            )
            return self.runtime_adapter.forward(
                state=state,
                visual_prompt_embed=visual_prompt_embed,
                visual_prompt_mask=visual_prompt_mask,
            )

    def _set_text_prompt(self, state: Dict[str, Any], prompt: str) -> None:
        text_outputs = self.processor.model.backbone.forward_text(
            [prompt],
            device=self.processor.device,
        )
        state["backbone_out"].update(text_outputs)

    def _build_box_prompt(
        self,
        boxes_xyxy: Sequence[Sequence[float]],
        width: int,
        height: int,
    ) -> Any:
        import torch
        from sam3.model.geometry_encoders import Prompt

        normalized_boxes = [
            xyxy_to_normalized_cxcywh(box, width=width, height=height)
            for box in boxes_xyxy
        ]
        boxes = torch.tensor(
            normalized_boxes,
            device=self.processor.device,
            dtype=torch.float32,
        ).view(len(normalized_boxes), 1, 4)
        labels = torch.ones(
            len(normalized_boxes),
            1,
            device=self.processor.device,
            dtype=torch.long,
        )
        box_mask = torch.zeros(
            1,
            len(normalized_boxes),
            device=self.processor.device,
            dtype=torch.bool,
        )
        return Prompt(
            box_embeddings=boxes,
            box_mask=box_mask,
            box_labels=labels,
        )

    def _combined_visual_prompt(self, state: Dict[str, Any]) -> Tuple[Any, Any]:
        import torch

        language_features = state["backbone_out"]["language_features"]
        target_device = language_features.device
        target_dtype = language_features.dtype
        prompt_embed = torch.cat(
            [
                embedding.prompt_embed.to(device=target_device, dtype=target_dtype)
                for embedding in self.embeddings
            ],
            dim=0,
        )
        prompt_mask = torch.cat(
            [embedding.prompt_mask.to(device=target_device) for embedding in self.embeddings],
            dim=1,
        )
        return prompt_embed, prompt_mask

def _resolve_existing_path(value: Any, key: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{key} must be set.")
    path = Path(str(value)).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"{key} does not exist: {path}")
    return path
