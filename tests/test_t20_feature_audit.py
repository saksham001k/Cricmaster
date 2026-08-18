import numpy as np
import pandas as pd

from scripts.audit_t20_features import orientation_free_auc


def test_orientation_free_auc_flips_inverse_signal() -> None:
    # Use enough observations to satisfy the audit's minimum-sample safeguard.
    y = pd.Series([0] * 5 + [1] * 5)
    x = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    assert orientation_free_auc(y, x) == 1.0


def test_orientation_free_auc_returns_none_for_missing_signal() -> None:
    y = pd.Series([0, 1] * 10)
    x = pd.Series([np.nan] * 20)

    assert orientation_free_auc(y, x) is None
