import numpy as np
import pandas as pd

from scripts.compare_models import _common_ids, safe_metrics


def test_common_ids_returns_intersection() -> None:
    left = pd.DataFrame({"match_id": ["a", "b", "c"]})
    right = pd.DataFrame({"match_id": ["b", "c", "d"]})
    assert _common_ids(left, right) == ["b", "c"]


def test_safe_metrics_handles_single_class_auc() -> None:
    metrics = safe_metrics(
        np.array([1, 1, 1]),
        np.array([0.7, 0.8, 0.9]),
    )
    assert metrics["matches"] == 3
    assert metrics["roc_auc"] is None
    assert 0.0 <= metrics["brier_score"] <= 1.0
