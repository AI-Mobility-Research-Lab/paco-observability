"""Transparent metric aggregation for observability validation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def ratio_of_sums(numerator: Iterable[float], denominator: Iterable[float]) -> float:
    """Return ``sum(numerator) / sum(denominator)`` after strict validation."""

    num = np.asarray(list(numerator), dtype=float)
    den = np.asarray(list(denominator), dtype=float)
    if num.shape != den.shape:
        raise ValueError("numerator and denominator must have identical shapes")
    if num.size == 0 or not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)):
        raise ValueError("ratio inputs must be non-empty and finite")
    if np.any(num < 0) or np.any(den < 0) or np.any(num > den):
        raise ValueError("counts must satisfy 0 <= numerator <= denominator")
    total = float(den.sum())
    if total <= 0:
        raise ValueError("summed denominator must be positive")
    return float(num.sum() / total)


def binary_agreement_metrics(
    predicted: Iterable[bool],
    reference: Iterable[bool],
) -> dict[str, Any]:
    """Return confusion counts and visible-class agreement metrics.

    The visible label is the positive class. Metrics with an empty denominator
    are returned as ``None`` instead of being silently defined as zero.
    """

    pred = np.asarray(list(predicted), dtype=bool)
    ref = np.asarray(list(reference), dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError("predicted and reference labels must have identical shapes")
    if pred.size == 0:
        raise ValueError("at least one label is required")

    tp = int(np.sum(pred & ref))
    tn = int(np.sum(~pred & ~ref))
    fp = int(np.sum(pred & ~ref))
    fn = int(np.sum(~pred & ref))

    def safe(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator else None

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "n": int(pred.size),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / pred.size),
        "precision_visible": precision,
        "recall_visible": recall,
        "specificity_hidden": specificity,
        "f1_visible": f1,
    }
