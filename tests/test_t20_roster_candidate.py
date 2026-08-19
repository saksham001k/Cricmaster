from scripts.finalize_t20_roster_candidate import _accept


def test_locked_acceptance_allows_small_noise() -> None:
    baseline = {"brier_score": 0.2400, "roc_auc": 0.6000}
    candidate = {"brier_score": 0.2410, "roc_auc": 0.5960}
    assert _accept(baseline, candidate)["accepted"] is True


def test_locked_acceptance_rejects_material_regression() -> None:
    baseline = {"brier_score": 0.2400, "roc_auc": 0.6000}
    candidate = {"brier_score": 0.2440, "roc_auc": 0.5800}
    assert _accept(baseline, candidate)["accepted"] is False
