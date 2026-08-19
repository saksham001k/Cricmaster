"""Rolling ablation of independent T20 feature groups.

No 2025+ data is read or evaluated by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.models.evaluate import classification_metrics
from cricmaster.models.posttoss import POST_TOSS_FEATURES, pair_post_toss_rows
from cricmaster.models.prematch import MODEL_FEATURES, pair_prematch_rows
from cricmaster.models.routed import symmetric_probability
from cricmaster.models.t20_experimental_features import (
    add_seasonal_elo,
    add_side_derived_features,
    add_venue_toss_interaction,
    append_pairwise_diff,
)


FOLDS = (
    ("2022", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2023-12-31", "2024-01-01", "2024-12-31"),
)


def _logistic() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    fit_intercept=False,
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def _tree() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _candidate(name: str) -> Pipeline:
    return _logistic() if name == "logistic_regression" else _tree()


def _fit(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> Pipeline:
    x = frame.loc[:, features].copy()
    for column in features:
        x[column] = pd.to_numeric(x[column], errors="coerce")

    y = frame["team_a_win"].astype(int)
    x_aug = pd.concat([x, -x], ignore_index=True)
    y_aug = pd.concat([y, 1 - y], ignore_index=True)
    model.fit(x_aug, y_aug)
    return model


def _selected_metrics(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    train_end: str,
    valid_start: str,
    valid_end: str,
) -> dict[str, Any]:
    train = frame.loc[frame["date"] <= pd.Timestamp(train_end)].copy()
    valid = frame.loc[
        (frame["date"] >= pd.Timestamp(valid_start))
        & (frame["date"] <= pd.Timestamp(valid_end))
    ].copy()

    candidate_metrics: dict[str, dict[str, Any]] = {}

    for name in ("logistic_regression", "hist_gradient_boosting"):
        model = _fit(_candidate(name), train, features)
        probability = symmetric_probability(
            {"model": model, "features": list(features)},
            valid,
        )
        candidate_metrics[name] = classification_metrics(
            valid["team_a_win"].astype(int).to_numpy(),
            probability,
        )

    selected = min(
        candidate_metrics,
        key=lambda name: (
            candidate_metrics[name]["brier_score"],
            candidate_metrics[name]["log_loss"],
        ),
    )

    return {
        "selected_model": selected,
        "metrics": candidate_metrics[selected],
        "candidates": candidate_metrics,
        "train_matches": int(len(train)),
        "validation_matches": int(len(valid)),
    }


def _evaluate_group(
    *,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_features: tuple[str, ...],
    extra_features: tuple[str, ...],
) -> dict[str, Any]:
    candidate_features = (*baseline_features, *extra_features)
    folds: list[dict[str, Any]] = []

    for label, train_end, valid_start, valid_end in FOLDS:
        base = _selected_metrics(
            baseline,
            baseline_features,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        enhanced = _selected_metrics(
            candidate,
            candidate_features,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        delta = (
            enhanced["metrics"]["brier_score"]
            - base["metrics"]["brier_score"]
        )
        folds.append(
            {
                "fold": label,
                "baseline": base,
                "candidate": enhanced,
                "brier_delta_candidate_minus_baseline": float(delta),
            }
        )

    deltas = [row["brier_delta_candidate_minus_baseline"] for row in folds]
    improved = sum(delta < 0 for delta in deltas)
    mean_delta = float(np.mean(deltas))
    worst_delta = float(max(deltas))

    return {
        "folds": folds,
        "improved_folds": improved,
        "mean_brier_delta": mean_delta,
        "worst_brier_delta": worst_delta,
        "passes_gate": bool(
            improved >= 2
            and mean_delta <= -0.001
            and worst_delta <= 0.003
        ),
        "extra_features": list(extra_features),
    }


def _print_group(name: str, result: dict[str, Any]) -> None:
    pieces = []
    for row in result["folds"]:
        pieces.append(
            f"{row['fold']}={row['brier_delta_candidate_minus_baseline']:+.4f}"
        )
    print(
        f"{name:24} "
        + " ".join(pieces)
        + f" | mean={result['mean_brier_delta']:+.4f}"
        + f" improved={result['improved_folds']}/3"
        + f" gate={'PASS' if result['passes_gate'] else 'FAIL'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate independent T20 feature groups on rolling "
            "2022/2023/2024 validation only."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/t20_expanded/prematch_features.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/processed/model_comparison/step16_feature_groups.json",
    )
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.input)
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    source = source.loc[source["date"] < pd.Timestamp("2025-01-01")].copy()

    derived = add_side_derived_features(source)

    pre_base = pair_prematch_rows(derived)
    pre_base = pre_base.loc[pre_base["format"] == "T20"].copy()

    post_base = pair_post_toss_rows(derived)
    post_base = post_base.loc[post_base["format"] == "T20"].copy()

    pre_groups: dict[str, tuple[pd.DataFrame, tuple[str, ...]]] = {}
    post_groups: dict[str, tuple[pd.DataFrame, tuple[str, ...]]] = {}

    seasonal_pre = add_seasonal_elo(pre_base, derived, mode="PRE_TOSS")
    seasonal_post = add_seasonal_elo(post_base, derived, mode="POST_TOSS")
    pre_groups["seasonal_elo"] = (
        seasonal_pre,
        ("competition_season_elo_diff",),
    )
    post_groups["seasonal_elo"] = (
        seasonal_post,
        ("competition_season_elo_diff",),
    )

    for source_col, label in (
        ("shrunk_win_rate_before", "shrunk_overall"),
        ("shrunk_h2h_win_rate", "shrunk_h2h"),
        ("shrunk_venue_win_rate", "shrunk_venue"),
        ("log_matches_before", "log_experience"),
    ):
        pre_candidate = append_pairwise_diff(
            pre_base,
            derived,
            mode="PRE_TOSS",
            source_column=source_col,
        )
        post_candidate = append_pairwise_diff(
            post_base,
            derived,
            mode="POST_TOSS",
            source_column=source_col,
        )
        feature = (f"{source_col}_diff",)
        pre_groups[label] = (pre_candidate, feature)
        post_groups[label] = (post_candidate, feature)

    post_groups["venue_toss"] = (
        add_venue_toss_interaction(post_base, derived),
        ("venue_toss_alignment",),
    )

    report: dict[str, Any] = {
        "policy": {
            "uses_2025_plus": False,
            "folds": [list(item) for item in FOLDS],
            "pass_gate": (
                "improve >=2/3 folds, mean Brier delta <= -0.001, "
                "worst fold degradation <= +0.003"
            ),
        },
        "prematch": {},
        "posttoss": {},
    }

    print("\n=== PRE_TOSS INDEPENDENT FEATURE GROUPS ===")
    for name, (candidate, features) in pre_groups.items():
        result = _evaluate_group(
            baseline=pre_base,
            candidate=candidate,
            baseline_features=MODEL_FEATURES,
            extra_features=features,
        )
        report["prematch"][name] = result
        _print_group(name, result)

    print("\n=== POST_TOSS INDEPENDENT FEATURE GROUPS ===")
    for name, (candidate, features) in post_groups.items():
        result = _evaluate_group(
            baseline=post_base,
            candidate=candidate,
            baseline_features=POST_TOSS_FEATURES,
            extra_features=features,
        )
        report["posttoss"][name] = result
        _print_group(name, result)

    passed_pre = [
        name
        for name, result in report["prematch"].items()
        if result["passes_gate"]
    ]
    passed_post = [
        name
        for name, result in report["posttoss"].items()
        if result["passes_gate"]
    ]

    print(f"\nPRE_TOSS passed groups: {passed_pre or 'none'}")
    print(f"POST_TOSS passed groups: {passed_post or 'none'}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
