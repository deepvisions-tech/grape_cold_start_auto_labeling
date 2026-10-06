"""YOLO label reading, box clipping, and prediction export.

The labeling pipeline and the offline analysis path both read ground-truth
boxes. They used to do so through two independent parsers that disagreed on
box clipping, minimum box size, and field-count strictness, which meant a
single ``--gate-run-file`` comparison could evaluate its before and after
sides against differently parsed ground truth. One implementation backs both
paths so that cannot happen.

The rules that legitimately differ between the two callers are parameters:
whether a missing label file is an error, and whether a line must carry
exactly five fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

from grape_auto_labeling.utils.io_paths import ensure_parent

if TYPE_CHECKING:
    from grape_auto_labeling.labeling.detections import Detection


MISSING_FILE_EMPTY = "empty"
MISSING_FILE_ERROR = "error"


def clip_box(box_xyxy: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    """Clip a box to the image and drop it if it is degenerate.

    Returns ``None`` for a non-finite box, a box that clips away entirely, or a
    box whose clipped width or height is at most one pixel. Coordinates pass
    through float32, which is what the reported runs used.
    """
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(4)
    if not np.isfinite(box).all():
        return None

    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
        return None
    return [x1, y1, x2, y2]


def load_yolo_boxes(
    label_path: Path,
    width: int,
    height: int,
    target_class_id: Optional[int],
    *,
    require_exact_fields: bool = False,
    on_missing_file: str = MISSING_FILE_EMPTY,
) -> List[List[float]]:
    """Read normalized YOLO boxes and return clipped pixel ``xyxy`` boxes.

    ``target_class_id`` of ``None`` keeps every class. ``require_exact_fields``
    rejects lines carrying more than the five ``class cx cy w h`` fields rather
    than ignoring the extras. ``on_missing_file`` selects whether an absent file
    yields an empty list or raises.
    """
    if on_missing_file not in {MISSING_FILE_EMPTY, MISSING_FILE_ERROR}:
        raise ValueError(f"Unsupported on_missing_file value: {on_missing_file!r}")

    if not label_path.is_file():
        if on_missing_file == MISSING_FILE_ERROR:
            raise FileNotFoundError(f"GT label file does not exist: {label_path}")
        return []

    boxes: List[List[float]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) < 5 or (require_exact_fields and len(parts) != 5):
            expected = "exactly five" if require_exact_fields else "at least five"
            raise ValueError(
                f"Expected {expected} YOLO fields at {label_path}:{line_number}; "
                f"got {len(parts)}: {raw_line}"
            )

        try:
            class_id = int(float(parts[0]))
            center_x, center_y, box_width, box_height = (
                float(value) for value in parts[1:5]
            )
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from exc

        if target_class_id is not None and class_id != target_class_id:
            continue

        clipped = clip_box(
            [
                (center_x - box_width / 2.0) * float(width),
                (center_y - box_height / 2.0) * float(height),
                (center_x + box_width / 2.0) * float(width),
                (center_y + box_height / 2.0) * float(height),
            ],
            width=width,
            height=height,
        )
        if clipped is not None:
            boxes.append(clipped)
    return boxes


def save_yolo_label(label_path: Path, detections: Sequence[Detection], width: int, height: int) -> int:
    ensure_parent(label_path)
    lines = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box_xyxy
        xc = ((x1 + x2) / 2.0) / float(width)
        yc = ((y1 + y2) / 2.0) / float(height)
        box_w = (x2 - x1) / float(width)
        box_h = (y2 - y1) / float(height)
        values = np.clip([xc, yc, box_w, box_h], 0.0, 1.0)
        lines.append(
            f"{int(detection.class_id)} "
            f"{values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"
        )

    label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return len(lines)


def save_classes_txt(labels_dir: Path, class_names: Sequence[str]) -> Path:
    path = labels_dir / "classes.txt"
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return path
