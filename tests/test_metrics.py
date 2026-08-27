import numpy as np
import pytest

from imgseg.common.metrics import evaluate


def test_perfect_prediction():
    gt = np.array([[1, 1], [0, 0]], dtype=bool)
    pred = gt.copy()
    scores = evaluate(gt, pred)
    assert scores.iou == pytest.approx(1.0, abs=1e-4)
    assert scores.dice == pytest.approx(1.0, abs=1e-4)
    assert scores.accuracy == pytest.approx(1.0, abs=1e-4)
    assert scores.precision == pytest.approx(1.0, abs=1e-4)
    assert scores.recall == pytest.approx(1.0, abs=1e-4)


def test_no_overlap():
    gt = np.array([[1, 0], [0, 0]], dtype=bool)
    pred = np.array([[0, 1], [0, 0]], dtype=bool)
    scores = evaluate(gt, pred)
    assert scores.iou == pytest.approx(0.0, abs=1e-4)
    assert scores.dice == pytest.approx(0.0, abs=1e-4)
    assert scores.tp == 0
    assert scores.fp == 1
    assert scores.fn == 1


def test_empty_masks_do_not_crash():
    gt = np.zeros((10, 10), dtype=bool)
    pred = np.zeros((10, 10), dtype=bool)
    scores = evaluate(gt, pred)
    # No positive pixels anywhere: accuracy should be perfect, IoU/precision/recall
    # are degenerate (0/0 -> epsilon-guarded) but must not raise.
    assert scores.accuracy == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= scores.iou <= 1.0


def test_partial_overlap_known_confusion_matrix():
    gt = np.array([1, 1, 1, 0, 0], dtype=bool)
    pred = np.array([1, 1, 0, 0, 1], dtype=bool)
    scores = evaluate(gt, pred)
    assert scores.tp == 2
    assert scores.fp == 1
    assert scores.fn == 1
    assert scores.tn == 1
    assert scores.iou == pytest.approx(2 / 4, abs=1e-4)
    assert scores.dice == pytest.approx(2 * 2 / (2 * 2 + 1 + 1), abs=1e-4)


def test_accepts_uint8_masks():
    gt = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    pred = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    scores = evaluate(gt, pred)
    assert scores.tp == 1
    assert scores.fp == 1
    assert scores.fn == 1
