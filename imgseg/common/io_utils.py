"""Small shared I/O and mask utilities used across the scene scripts."""

from __future__ import annotations

import os

import cv2
import numpy as np


def read_rgb(path: str) -> np.ndarray:
    """Read an image from disk as RGB, raising a clear error if missing."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(
            f"Could not read image at '{path}'. "
            f"Check that the file exists (see data/README.md)."
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return a binary mask (0/255) containing only the largest connected
    component of the input mask.
    """
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    largest_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_id, 255, 0).astype(np.uint8)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
