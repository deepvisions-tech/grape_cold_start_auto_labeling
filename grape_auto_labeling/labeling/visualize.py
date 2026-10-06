"""Optional preview rendering for detections and DINOv2 diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from grape_auto_labeling.labeling.detections import Detection
from grape_auto_labeling.models.dinov2_runtime import DinoPatch
from grape_auto_labeling.utils.io_paths import ensure_parent


def color_for_prompt(prompt: str) -> Tuple[int, int, int]:
    palette = [
        (0, 180, 0),
        (220, 80, 40),
        (40, 120, 220),
        (190, 70, 190),
        (220, 160, 20),
        (30, 170, 170),
    ]
    index = sum(ord(char) for char in prompt) % len(palette)
    return palette[index]


def safe_draw_text(text: str, max_len: int = 48) -> str:
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text.encode("ascii", errors="replace").decode("ascii")


def draw_visualization(image: Image.Image, detections: Sequence[Detection], save_path: Path) -> None:
    ensure_parent(save_path)
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis, "RGBA")
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.box_xyxy]
        color = color_for_prompt(detection.prompt)
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=3)

        label = safe_draw_text(f"{detection.prompt} {detection.score:.2f}")
        text_x = max(0, x1)
        text_y = max(0, y1 - 14)
        text_box = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0, 180))
        draw.text((text_x, text_y), label, fill=(*color, 255), font=font)

    vis.save(save_path)


def draw_dinov2_diff_heatmap(
    image: Image.Image,
    patches: Sequence[DinoPatch],
    save_path: Path,
) -> None:
    """Draw a smooth, continuous DINOv2 difference heatmap over the image."""
    import cv2

    ensure_parent(save_path)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    score_sum = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    patch_widths: List[int] = []
    patch_heights: List[int] = []

    for patch in patches:
        x1, y1, x2, y2 = patch.label_box_xyxy
        left = max(0, min(width, int(math.floor(x1))))
        top = max(0, min(height, int(math.floor(y1))))
        right = max(0, min(width, int(math.ceil(x2))))
        bottom = max(0, min(height, int(math.ceil(y2))))
        if right <= left or bottom <= top:
            continue
        score_sum[top:bottom, left:right] += float(patch.score)
        coverage[top:bottom, left:right] += 1.0
        patch_widths.append(right - left)
        patch_heights.append(bottom - top)

    if not patch_widths:
        Image.fromarray(image_rgb, mode="RGB").save(save_path)
        return

    # Blur the weighted score and coverage separately so overlapping source tiles
    # are averaged without leaving the rectangular patch boundaries visible.
    median_patch_size = float(np.median([*patch_widths, *patch_heights]))
    sigma = max(1.0, median_patch_size * 0.35)
    smooth_score_sum = cv2.GaussianBlur(score_sum, (0, 0), sigmaX=sigma, sigmaY=sigma)
    smooth_coverage = cv2.GaussianBlur(coverage, (0, 0), sigmaX=sigma, sigmaY=sigma)
    valid = smooth_coverage > 1e-6
    score_map = np.zeros_like(smooth_score_sum)
    score_map[valid] = smooth_score_sum[valid] / smooth_coverage[valid]

    max_score = max(float(patch.score) for patch in patches)
    normalized = np.zeros_like(score_map, dtype=np.float32)
    if max_score > 0.0:
        normalized = np.clip(score_map / max_score, 0.0, 1.0)

    heatmap_bgr = cv2.applyColorMap(
        np.round(normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    alpha = np.where(valid, 0.55, 0.0).astype(np.float32)[..., None]
    blended = np.round(
        image_rgb.astype(np.float32) * (1.0 - alpha)
        + heatmap_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended, mode="RGB").save(save_path)


def draw_dinov2_box_heatmap(
    image: Image.Image,
    patches: Sequence[DinoPatch],
    save_path: Path,
) -> None:
    """Draw a red rectangular-patch heatmap."""
    ensure_parent(save_path)
    vis = image.convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    max_score = max([patch.score for patch in patches], default=0.0)
    if max_score > 0:
        for patch in patches:
            alpha = int(np.clip((patch.score / max_score) * 150.0, 0.0, 150.0))
            if alpha <= 0:
                continue
            box = [int(round(value)) for value in patch.label_box_xyxy]
            draw.rectangle(box, fill=(255, 0, 0, alpha))
    Image.alpha_composite(vis, overlay).convert("RGB").save(save_path)
