"""Helpers for conservative CPU parallel image processing."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional


def default_worker_count(
    requested: Optional[int] = None,
    reserve_cores: int = 0,
    max_workers: Optional[int] = None,
) -> int:
    """Return a worker count for image processing."""
    cpu_count = os.cpu_count() or 1
    usable = max(1, cpu_count - max(0, reserve_cores))
    if requested is not None and requested > 0:
        return max(1, min(int(requested), usable))
    if max_workers is not None and max_workers > 0:
        return max(1, min(usable, int(max_workers)))
    return usable


@contextmanager
def limited_opencv_threads(enabled: bool = True) -> Iterator[None]:
    """Temporarily keep OpenCV from oversubscribing CPU inside Python workers."""
    if not enabled:
        yield
        return

    try:
        import cv2
    except Exception:
        yield
        return

    previous = None
    try:
        previous = cv2.getNumThreads()
        cv2.setNumThreads(1)
    except Exception:
        previous = None

    try:
        yield
    finally:
        if previous is not None:
            try:
                cv2.setNumThreads(previous)
            except Exception:
                pass
