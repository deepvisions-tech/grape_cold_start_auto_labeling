"""Batching and order-preserving worker mapping for image-level work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, TypeVar


T = TypeVar("T")
R = TypeVar("R")


def iter_batches(items: Sequence[Any], batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ordered_worker_map(
    executor: Optional[ThreadPoolExecutor],
    fn: Callable[[T], R],
    items: Sequence[T],
) -> List[R]:
    if not items:
        return []
    if executor is None or len(items) == 1:
        return [fn(item) for item in items]
    return list(executor.map(fn, items))
