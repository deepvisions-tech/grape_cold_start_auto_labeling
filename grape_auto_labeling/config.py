"""YAML loading, configuration vocabulary, typed accessors, and value validation.

Every setting the pipeline reads is named in ``KNOWN_CONFIG_KEYS`` and reached
through one accessor here, so that validation and execution cannot disagree
about a default."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and require a mapping at the root."""
    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


DEFAULT_CLASS_NAMES = ["magnesium deficiency"]
SUPPORTED_ADAPTIVE_THRESHOLD_METHODS = {"otsu", "percentile_80"}
SUPPORTED_OUTPUT_EXISTING_POLICIES = {"error", "overwrite"}

# Every configuration key the pipeline reads. Preflight rejects anything else so
# that a misspelled key cannot silently fall back to a default and change the
# produced labels.
KNOWN_CONFIG_KEYS = frozenset(
    {
        "AMP",
        "AUTO_LABELING_BATCH_SIZE",
        "AUTO_LABELING_NUM_WORKERS",
        "CLASS_NAMES",
        "DEVICE",
        "DINOV2_ADAPTIVE_THRESHOLD_METHOD",
        "DINOV2_ADAPTIVE_THRESHOLD_OFFSET",
        "DINOV2_CHECKPOINT_FILENAME",
        "DINOV2_CHECKPOINT_PATH",
        "DINOV2_EDGE_SUPPRESS_PATCHES",
        "DINOV2_FILTER_ENABLED",
        "DINOV2_HUB_REF",
        "DINOV2_HUB_REPO",
        "DINOV2_INPUT_SIZE",
        "DINOV2_MODEL_NAME",
        "DINOV2_ORIGINAL_IMAGE_PATH",
        "DINOV2_REPO_DIR",
        "DINOV2_REQUIRE_PATCH_FULLY_INSIDE_SAM3_MASK",
        "DINOV2_SAVE_ARTIFACTS",
        "DINOV2_TILE_BATCH_SIZE",
        "DINOV2_TILE_HEIGHT_DIVISOR",
        "DINOV2_TILE_STRIDE_RATIO",
        "DINOV2_TILE_WIDTH_DIVISOR",
        "DISABLE_TQDM",
        "GT_LABEL_DIR",
        "LABEL_CLASS_ID",
        "LABEL_IMAGE_PATH",
        "LABEL_OUTPUT_DIR",
        "MAX_DETECTIONS_PER_IMAGE",
        "MAX_IMAGES",
        "NMS_IOU_THRESHOLD",
        "OUTPUT_EXISTING_POLICY",
        "PERFORMANCE_IOU_THRESHOLD",
        "PERFORMANCE_TEST_ENABLED",
        "SAM3_CHECKPOINT_PATH",
        "SAM3_CONFIDENCE_THRESHOLD",
        "SAM3_IMAGE_EXEMPLARS",
        "SAM3_LOAD_FROM_HF",
        "SAM3_PROMPT_MODE",
        "SAM3_REPO_DIR",
        "SAVE_VISUAL_OUTPUTS",
        "TEXT_PROMPTS",
    }
)

# Effective defaults for optional numeric settings. Preflight and the run path
# read these through the accessors below so that a single definition governs
# both validation and execution.
DEFAULT_SAM3_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_NMS_IOU_THRESHOLD = 0.6
DEFAULT_DINOV2_ADAPTIVE_THRESHOLD_OFFSET = 0.0
DEFAULT_DINOV2_EDGE_SUPPRESS_PATCHES = 0
DEFAULT_DINOV2_TILE_BATCH_SIZE = 1
DEFAULT_DINOV2_INPUT_SIZE = 518
DEFAULT_DINOV2_TILE_WIDTH_DIVISOR = 3.0
DEFAULT_DINOV2_TILE_HEIGHT_DIVISOR = 2.0
DEFAULT_DINOV2_TILE_STRIDE_RATIO = 0.5

# ViT-B/14 patch size. A DINOv2 input size that is not a multiple of this leaves
# the token grid covering only part of the resized crop, which silently shifts
# every derived patch box.
DINOV2_PATCH_SIZE = 14


def normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    text = normalize_optional_text(value)
    if text is None:
        return default
    return int(text)


def resolve_path(value: Any, key: str, must_exist: bool = False) -> Path:
    text = normalize_optional_text(value)
    if text is None:
        raise ValueError(f"{key} must be set.")
    path = Path(text).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{key} does not exist: {path}")
    return path.resolve(strict=False)


def strict_config_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    """Parse a boolean config value without treating arbitrary strings as false."""
    value = cfg.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean; got {value!r}.")


def require_config_int(
    cfg: Dict[str, Any],
    key: str,
    *,
    default: Optional[int] = None,
    minimum: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[int]:
    if key not in cfg:
        if default is not None:
            parsed = int(default)
        elif allow_none:
            return None
        else:
            raise ValueError(f"{key} must be set.")
    else:
        value = cfg.get(key)
        if normalize_optional_text(value) is None:
            if allow_none:
                return None
            if default is None:
                raise ValueError(f"{key} must be set.")
            parsed = int(default)
        elif isinstance(value, bool):
            raise ValueError(f"{key} must be an integer; got {value!r}.")
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
            parsed = int(value)
        else:
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer; got {value!r}.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{key} must be at least {minimum}; got {parsed}.")
    return parsed


def require_config_float(
    cfg: Dict[str, Any],
    key: str,
    *,
    default: Optional[float] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    allow_none: bool = False,
) -> Optional[float]:
    if key not in cfg:
        if default is not None:
            parsed = float(default)
        elif allow_none:
            return None
        else:
            raise ValueError(f"{key} must be set.")
        value = default
    else:
        value = cfg.get(key)
        if normalize_optional_text(value) is None:
            if allow_none:
                return None
            if default is None:
                raise ValueError(f"{key} must be set.")
            parsed = float(default)
        else:
            if isinstance(value, bool):
                raise ValueError(f"{key} must be numeric; got {value!r}.")
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be numeric; got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite; got {value!r}.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{key} must be at least {minimum}; got {parsed}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{key} must be at most {maximum}; got {parsed}.")
    return parsed


def config_float(
    cfg: Dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Read a validated float, treating an omitted or blank value as the default."""
    value = require_config_float(cfg, key, default=default, minimum=minimum, maximum=maximum)
    assert value is not None  # allow_none is False, so a value is always produced.
    return float(value)


def config_int(
    cfg: Dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: Optional[int] = None,
) -> int:
    """Read a validated int, treating an omitted or blank value as the default."""
    value = require_config_int(cfg, key, default=default, minimum=minimum)
    assert value is not None  # allow_none is False, so a value is always produced.
    return int(value)


def resolve_sam3_confidence_threshold(cfg: Dict[str, Any]) -> float:
    return config_float(
        cfg,
        "SAM3_CONFIDENCE_THRESHOLD",
        default=DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
        minimum=0.0,
        maximum=1.0,
    )


def resolve_nms_iou_threshold(cfg: Dict[str, Any]) -> float:
    """Resolve the proposal NMS IoU threshold.

    Suppression is always applied; 1.0 keeps every proposal, so the setting has
    no separate disabled state.
    """
    return config_float(
        cfg,
        "NMS_IOU_THRESHOLD",
        default=DEFAULT_NMS_IOU_THRESHOLD,
        minimum=0.0,
        maximum=1.0,
    )


def resolve_dinov2_adaptive_threshold_offset(cfg: Dict[str, Any]) -> float:
    return config_float(
        cfg,
        "DINOV2_ADAPTIVE_THRESHOLD_OFFSET",
        default=DEFAULT_DINOV2_ADAPTIVE_THRESHOLD_OFFSET,
        minimum=0.0,
        maximum=1.0,
    )


def resolve_dinov2_edge_suppress_patches(cfg: Dict[str, Any]) -> int:
    return config_int(
        cfg,
        "DINOV2_EDGE_SUPPRESS_PATCHES",
        default=DEFAULT_DINOV2_EDGE_SUPPRESS_PATCHES,
        minimum=0,
    )


def resolve_dinov2_tile_batch_size(cfg: Dict[str, Any]) -> int:
    return config_int(
        cfg,
        "DINOV2_TILE_BATCH_SIZE",
        default=DEFAULT_DINOV2_TILE_BATCH_SIZE,
        minimum=1,
    )


def get_dinov2_require_patch_fully_inside_sam3_mask(cfg: Dict[str, Any]) -> bool:
    return strict_config_bool(cfg, "DINOV2_REQUIRE_PATCH_FULLY_INSIDE_SAM3_MASK", True)


def validate_known_config_keys(cfg: Dict[str, Any]) -> None:
    """Reject unrecognized keys so a typo cannot silently select a default."""
    unknown = sorted(str(key) for key in cfg if str(key) not in KNOWN_CONFIG_KEYS)
    if not unknown:
        return

    import difflib

    details = []
    for key in unknown:
        close = difflib.get_close_matches(key, KNOWN_CONFIG_KEYS, n=1, cutoff=0.8)
        details.append(f"{key} (did you mean {close[0]}?)" if close else key)
    raise ValueError(
        "Unknown configuration key(s): "
        + ", ".join(details)
        + ". Remove them or correct the spelling; a misspelled key would otherwise "
        "be ignored and the documented default would be used instead."
    )


def resolve_optional_existing_path(
    value: Any,
    key: str,
    *,
    kind: str,
) -> Optional[Path]:
    text = normalize_optional_text(value)
    if text is None:
        return None
    path = Path(text).expanduser().resolve(strict=False)
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"{key} is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise FileNotFoundError(f"{key} is not a directory: {path}")
    return path


def resolve_output_existing_policy(
    cfg: Dict[str, Any],
    *,
    overwrite: Optional[bool] = None,
) -> str:
    raw_policy = cfg.get("OUTPUT_EXISTING_POLICY")
    policy = (normalize_optional_text(raw_policy) or "error").lower()
    if policy not in SUPPORTED_OUTPUT_EXISTING_POLICIES:
        raise ValueError(
            "OUTPUT_EXISTING_POLICY must be one of "
            f"{', '.join(sorted(SUPPORTED_OUTPUT_EXISTING_POLICIES))}; got {raw_policy!r}."
        )
    if overwrite:
        return "overwrite"
    return policy


def normalize_prompts(raw_prompts: Any) -> List[str]:
    if isinstance(raw_prompts, str):
        prompts = [raw_prompts]
    elif raw_prompts is None:
        prompts = []
    elif isinstance(raw_prompts, (list, tuple)):
        prompts = list(raw_prompts)
    else:
        raise ValueError("TEXT_PROMPTS must be a string or a list of strings.")

    normalized = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    if not normalized:
        raise ValueError("TEXT_PROMPTS must contain at least one non-empty prompt.")
    return normalized


def build_class_names(raw_names: Any, class_id: int) -> List[str]:
    if raw_names is None:
        names = list(DEFAULT_CLASS_NAMES)
    elif isinstance(raw_names, dict):
        names = [
            str(raw_names[key])
            for key in sorted(raw_names.keys(), key=lambda item: int(item))
        ]
    else:
        names = [str(name) for name in raw_names]

    while len(names) <= class_id:
        names.append(f"class_{len(names)}")
    return names
