"""Shared binary segmentation metrics.

All four scene scripts evaluate their predicted mask against a ground-truth
mask using the same confusion-matrix-based metrics. This module centralizes
that logic so it is implemented (and tested) once.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

EPS = 1e-6


@dataclass
class SegmentationScores:
    """Container for standard binary-segmentation evaluation metrics."""

    iou: float
    dice: float
    accuracy: float
    precision: float
    recall: float
    tp: int
    tn: int
    fp: int
    fn: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"IoU={self.iou:.4f}  Dice={self.dice:.4f}  "
            f"Acc={self.accuracy:.4f}  Prec={self.precision:.4f}  "
            f"Recall={self.recall:.4f}"
        )


def evaluate(ground_truth: np.ndarray, prediction: np.ndarray) -> SegmentationScores:
    """Compute IoU, Dice/F1, Accuracy, Precision and Recall.

    Args:
        ground_truth: boolean (or 0/1 / 0/255) reference mask.
        prediction:   boolean (or 0/1 / 0/255) predicted mask, same shape.

    Returns:
        A SegmentationScores instance.
    """
    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    tp = int(np.logical_and(gt, pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())

    total = tp + tn + fp + fn
    iou = tp / (tp + fp + fn + EPS)
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    return SegmentationScores(
        iou=iou, dice=dice, accuracy=accuracy,
        precision=precision, recall=recall,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )
