"""Rolling pre-2025 validation of enhanced Cricmaster T20 features."""

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

from cricmaster.models.enhanced_t20 import (
    ENHANCED_POST_TOSS_EXTRA,
    ENHANCED_PRE_TOSS_EXTRA,
    pair_enhanced_posttoss,
    pair_enhanced_prematch,
)
from cricmaster.models.evaluate import classification_metrics
from cricmaster.models.posttoss import POST_TOSS_FEATURES, pair_post_toss_rows
from cricmaster.models.prematch import MODEL_FEATURES, pair_prematch_rows
from cricmaster.models.routed import symmetric_probability


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
    if name == "logistic_regression":
        return _logistic()
    return _tree()


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


def _metrics(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> dict[str, Any]:
    p = symmetric_probability(
        {"model": model, "features": list(features)},
        frame,
    )
    return classification_metrics(
        frame["team_a_win"].astype(int).to_numpy(),
        p,
    )


def _evaluate_fold(
    frame: pd.DataFrame,
    *,
    train_end: str,
    valid_start: str,
    valid_end: str,
    features: tuple[str, ...],
) -> dict[str, Any]:
    train = frame.loc[frame["date"] <= pd.Timestamp(train_end)].copy()
    valid = frame.loc[
        (frame["date"] >= pd.Timestamp(valid_start))
        & (frame["date"] <= pd.Timestamp(valid_end))
    ].copy()

    if train.empty or valid.empty:
        raise RuntimeError("Rolling fold produced an empty train/validation split")

    candidate_metrics: dict[str, Any] = {}
    for name in ("logistic_regression", "hist_gradient_boosting"):
        model = _fit(_candidate(name), train, features)
        candidate_metrics[name] = _metrics(model, valid, features)

    selected = min(
        candidate_metrics,
        key=lambda name: (
            candidate_metrics[name]["brier_score"],
            candidate_metrics[name]["log_loss"],
        ),
    )
    return {
        "train_matches": int(len(train)),
        "validation_matches": int(len(valid)),
        "selected_model": selected,
        "selected_metrics": candidate_metrics[selected],
        "candidate_metrics": candidate_metrics,
    }


def _run_mode(
    *,
    name: str,
    source: pd.DataFrame,
    baseline_pairer: Callable[[pd.DataFrame], pd.DataFrame],
    enhanced_pairer: Callable[[pd.DataFrame], pd.DataFrame],
    baseline_features: tuple[str, ...],
    enhanced_extra: tuple[str, ...],
) -> dict[str, Any]:
    baseline = baseline_pairer(source)
    enhanced = enhanced_pairer(source)

    baseline = baseline.loc[baseline["format"] == "T20"].copy()
    enhanced = enhanced.loc[enhanced["format"] == "T20"].copy()

    enhanced_features = (*baseline_features, *enhanced_extra)

    rows: list[dict[str, Any]] = []

    print(f"\n=== {name} ROLLING T20 VALIDATION ===")
    for label, train_end, valid_start, valid_end in FOLDS:
        base_result = _evaluate_fold(
            baseline,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            features=baseline_features,
        )
        enhanced_result = _evaluate_fold(
            enhanced,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            features=enhanced_features,
        )

        base_brier = base_result["selected_metrics"]["brier_score"]
        enhanced_brier = enhanced_result["selected_metrics"]["brier_score"]
        delta = enhanced_brier - base_brier

        print(
            f"{label}: "
            f"baseline={base_brier:.4f} "
            f"enhanced={enhanced_brier:.4f} "
            f"delta={delta:+.4f} "
            f"base_model={base_result['selected_model']} "
            f"enhanced_model={enhanced_result['selected_model']}"
        )

        rows.append(
            {
                "fold": label,
                "baseline": base_result,
                "enhanced": enhanced_result,
                "brier_delta_enhanced_minus_baseline": float(delta),
            }
        )

    deltas = [row["brier_delta_enhanced_minus_baseline"] for row in rows]
    improved_folds = sum(delta < 0 for delta in deltas)
    mean_delta = float(np.mean(deltas))

    recommendation = (
        "KEEP_CANDIDATES"
        if improved_folds >= 2 and mean_delta < 0
        else "REJECT_OR_REWORK"
    )

    print(
        f"summary: improved_folds={improved_folds}/3 "
        f"mean_brier_delta={mean_delta:+.4f}"
    )
    print(f"recommendation={recommendation}")

    return {
        "folds": rows,
        "improved_folds": improved_folds,
        "mean_brier_delta": mean_delta,
        "recommendation": recommendation,
        "enhanced_features": list(enhanced_features),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate enhanced T20 features on rolling 2022/2023/2024 "
            "validation without consulting 2025+."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/t20_expanded_step15/prematch_features.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/processed/model_comparison/step15_rolling_features.json",
    )
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.input)
    source["date"] = pd.to_datetime(source["date"], errors="raise")

    report = {
        "policy": (
            "Feature candidates are selected only from rolling 2022/2023/2024 "
            "validation; 2025+ is not evaluated by this script."
        ),
        "prematch": _run_mode(
            name="PRE_TOSS",
            source=source,
            baseline_pairer=pair_prematch_rows,
            enhanced_pairer=pair_enhanced_prematch,
            baseline_features=MODEL_FEATURES,
            enhanced_extra=ENHANCED_PRE_TOSS_EXTRA,
        ),
        "posttoss": _run_mode(
            name="POST_TOSS",
            source=source,
            baseline_pairer=pair_post_toss_rows,
            enhanced_pairer=pair_enhanced_posttoss,
            baseline_features=POST_TOSS_FEATURES,
            enhanced_extra=ENHANCED_POST_TOSS_EXTRA,
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
