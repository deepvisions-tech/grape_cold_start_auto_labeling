"""Conversions from framework objects to plain NumPy and JSON values."""

from __future__ import annotations

from typing import Any, List

import numpy as np


def tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def matrix_to_list(matrix: np.ndarray) -> List[List[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix).reshape(3, 3)]
