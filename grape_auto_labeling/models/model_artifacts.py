"""Small helpers for user-selected external model sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from grape_auto_labeling.config import normalize_optional_text


def build_torch_hub_repo_spec(repo: str, ref: Optional[str]) -> str:
    repo = str(repo).strip()
    if not repo:
        raise ValueError("DINOv2 Hub repository must not be empty.")
    normalized_ref = normalize_optional_text(ref)
    if normalized_ref is None:
        return repo
    if ":" in repo:
        raise ValueError(
            "DINOv2 Hub repository already contains a ref; set the base repository "
            "and DINOV2_HUB_REF separately."
        )
    return f"{repo}:{normalized_ref}"


def torch_hub_checkpoint_path(torch_module: Any, filename: str) -> Path:
    filename = Path(str(filename)).name
    if not filename:
        raise ValueError("DINOv2 checkpoint filename must not be empty.")
    return Path(torch_module.hub.get_dir()) / "checkpoints" / filename
