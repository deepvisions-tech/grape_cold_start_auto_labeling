"""Input discovery, output-path safety, and artifact pairing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def path_is_same_or_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_safe_output_path(
    *,
    image_path: Path,
    output_dir: Path,
    output_existing_policy: str,
) -> bool:
    if image_path.is_dir() and path_is_same_or_within(output_dir, image_path):
        raise ValueError(
            "LABEL_OUTPUT_DIR must not be the same as or nested inside "
            f"LABEL_IMAGE_PATH: input={image_path}, output={output_dir}"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"LABEL_OUTPUT_DIR is not a directory: {output_dir}")
    has_entries = output_dir.is_dir() and next(output_dir.iterdir(), None) is not None
    if has_entries and output_existing_policy == "error":
        raise FileExistsError(
            f"LABEL_OUTPUT_DIR is not empty: {output_dir}. "
            "Choose a new directory or pass --overwrite "
            "(OUTPUT_EXISTING_POLICY: overwrite)."
        )
    return has_entries


def validate_unique_output_targets(image_paths: Sequence[Path], input_root: Path) -> None:
    targets: Dict[str, Path] = {}
    for image_path in image_paths:
        relative_path = image_path.relative_to(input_root)
        label_target = relative_path.with_suffix(".txt")
        target_key = label_target.as_posix().casefold()
        if target_key == "classes.txt":
            raise RuntimeError(
                f"Input image {image_path} would overwrite the reserved labels/classes.txt file."
            )
        previous = targets.get(target_key)
        if previous is not None:
            raise RuntimeError(
                "Multiple input images map to the same output label path "
                f"{label_target}: {previous}, {image_path}"
            )
        targets[target_key] = image_path


def list_input_images(image_path: Path) -> Tuple[List[Path], Path]:
    if image_path.is_file():
        if image_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {image_path}")
        return [image_path], image_path.parent

    if not image_path.is_dir():
        raise FileNotFoundError(f"LABEL_IMAGE_PATH is not a file or directory: {image_path}")

    images = [
        path
        for path in image_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    images.sort()
    if not images:
        raise RuntimeError(f"No images found under LABEL_IMAGE_PATH: {image_path}")
    return images, image_path


def index_images_by_name(image_path: Path) -> Tuple[Dict[str, List[Path]], Path]:
    if image_path.is_file():
        if image_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported original image extension: {image_path}")
        image_paths = [image_path]
        root = image_path.parent
    elif image_path.is_dir():
        image_paths = sorted(
            path
            for path in image_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        root = image_path
    else:
        raise FileNotFoundError(
            f"DINOV2_ORIGINAL_IMAGE_PATH is not a file or directory: {image_path}"
        )

    index: Dict[str, List[Path]] = {}
    for path in image_paths:
        index.setdefault(path.name, []).append(path)
    return index, root


def index_yolo_labels_by_stem(label_dir: Path) -> Dict[str, List[Path]]:
    if not label_dir.is_dir():
        raise FileNotFoundError(f"GT_LABEL_DIR is not a directory: {label_dir}")

    index: Dict[str, List[Path]] = {}
    for path in sorted(label_dir.rglob("*.txt")):
        if path.name == "classes.txt":
            continue
        index.setdefault(path.stem, []).append(path)
    return index


def require_unique_original_matches(
    image_paths: Sequence[Path],
    original_index: Dict[str, List[Path]],
) -> Dict[Path, Path]:
    """Resolve every generated image to exactly one original before inference."""
    resolved: Dict[Path, Path] = {}
    for image_path in image_paths:
        matches = original_index.get(image_path.name, [])
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No original image found for generated image: {image_path}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple original images matched generated image {image_path}: "
                + ", ".join(str(path) for path in matches)
            )
        resolved[image_path] = matches[0]
    return resolved


def require_unique_gt_labels(
    image_paths: Sequence[Path],
    gt_index: Dict[str, List[Path]],
) -> None:
    """Require one existing GT file per image while allowing an empty file."""
    for image_path in image_paths:
        matches = gt_index.get(image_path.stem, [])
        if len(matches) == 0:
            raise FileNotFoundError(
                f"Missing GT label file for image: {image_path}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple GT label files matched image {image_path}: "
                + ", ".join(str(path) for path in matches)
            )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
