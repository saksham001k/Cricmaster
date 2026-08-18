"""Probability evaluation helpers for cricket match models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Weighted absolute calibration gap over equal-width probability bins."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p, edges[1:-1], right=False)

    error = 0.0
    total = len(y)
    if total == 0:
        return float("nan")

    for index in range(bins):
        mask = bucket == index
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        observed = float(y[mask].mean())
        error += (count / total) * abs(confidence - observed)

    return float(error)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Return threshold and probability-quality metrics."""

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    predicted = (p >= 0.5).astype(int)

    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10": expected_calibration_error(y, p, bins=10),
    }
