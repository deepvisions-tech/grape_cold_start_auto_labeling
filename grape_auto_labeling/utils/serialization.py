"""JSON and YAML output for labeling runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from grape_auto_labeling.utils.io_paths import ensure_parent


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def yaml_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): yaml_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(
        yaml.safe_dump(
            yaml_safe_value(payload),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
