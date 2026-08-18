"""Train and evaluate Cricmaster's first leakage-safe PRE_TOSS models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
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

from cricmaster.features.split import temporal_split
from cricmaster.models.evaluate import classification_metrics
from cricmaster.models.prematch import (
    ELO_DIFFERENCE_FEATURE,
    MODEL_FEATURES,
    elo_probability,
    pair_prematch_rows,
)


def _augment_signed(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Add the exact mirrored team orientation to the training set."""

    mirrored_x = -x
    mirrored_y = 1 - y.astype(int)
    return (
        pd.concat([x, mirrored_x], ignore_index=True),
        pd.concat([y.astype(int), mirrored_y], ignore_index=True),
    )


def _symmetric_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Average P(A beats B) with 1 - P(B beats A) to enforce symmetry."""

    forward = model.predict_proba(x)[:, 1]
    reverse = model.predict_proba(-x)[:, 1]
    return 0.5 * (forward + (1.0 - reverse))


def _logistic_model() -> Pipeline:
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


def _tree_model() -> Pipeline:
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


def _fit(model: Pipeline, frame: pd.DataFrame) -> Pipeline:
    x = frame.loc[:, MODEL_FEATURES]
    y = frame["team_a_win"].astype(int)
    x_aug, y_aug = _augment_signed(x, y)
    model.fit(x_aug, y_aug)
    return model


def _evaluate_model(model: Pipeline, frame: pd.DataFrame) -> dict[str, Any]:
    x = frame.loc[:, MODEL_FEATURES]
    y = frame["team_a_win"].astype(int).to_numpy()
    probabilities = _symmetric_probability(model, x)
    return classification_metrics(y, probabilities)


def _elo_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["team_a_win"].astype(int).to_numpy()
    diff = frame[ELO_DIFFERENCE_FEATURE].fillna(0.0).to_numpy()
    return classification_metrics(y, elo_probability(diff))


def _date_range(frame: pd.DataFrame) -> dict[str, str | int | None]:
    if frame.empty:
        return {"matches": 0, "start": None, "end": None}
    return {
        "matches": int(len(frame)),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
    }


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"{label:24} "
        f"n={metrics['n']:4d} "
        f"acc={metrics['accuracy']:.4f} "
        f"auc={metrics['roc_auc']:.4f} "
        f"logloss={metrics['log_loss']:.4f} "
        f"brier={metrics['brier_score']:.4f} "
        f"ece={metrics['ece_10']:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe Cricmaster PRE_TOSS baselines."
    )
    parser.add_argument(
        "--input",
        default="data/processed/t20_corpus/prematch_features.parquet",
    )
    parser.add_argument("--output", default="models/prematch")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--valid-end", default="2024-12-31")
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.input)
    paired = pair_prematch_rows(source)

    train, valid, test = temporal_split(
        paired,
        train_end=args.train_end,
        valid_end=args.valid_end,
        date_column="date",
        match_id_column="match_id",
    )

    if train.empty or valid.empty or test.empty:
        raise RuntimeError(
            "Temporal split produced an empty train/validation/test partition"
        )

    print("=== SPLIT ===")
    print("train:", _date_range(train))
    print("valid:", _date_range(valid))
    print("test :", _date_range(test))

    candidates = {
        "logistic_regression": _logistic_model(),
        "hist_gradient_boosting": _tree_model(),
    }

    validation: dict[str, dict[str, Any]] = {
        "elo_baseline": _elo_metrics(valid)
    }

    print("\n=== VALIDATION (used for selection) ===")
    _print_metrics("elo_baseline", validation["elo_baseline"])

    for name, model in candidates.items():
        fitted = _fit(model, train)
        validation[name] = _evaluate_model(fitted, valid)
        _print_metrics(name, validation[name])

    selected_name = min(
        validation,
        key=lambda name: (
            validation[name]["brier_score"],
            validation[name]["log_loss"],
        ),
    )
    print(f"\nselected_model={selected_name}")

    # Final ML candidates are trained on all information available through 2024.
    # Selection itself remains based only on the 2024 validation partition.
    development = pd.concat([train, valid], ignore_index=True)
    final_models: dict[str, Pipeline] = {}
    test_metrics: dict[str, dict[str, Any]] = {
        "elo_baseline": _elo_metrics(test)
    }

    for name, model in candidates.items():
        final_models[name] = _fit(model, development)
        test_metrics[name] = _evaluate_model(final_models[name], test)

    print("\n=== FINAL TEST (2025+) ===")
    for name in ("elo_baseline", "logistic_regression", "hist_gradient_boosting"):
        _print_metrics(name, test_metrics[name])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": final_models.get(selected_name),
        "model_name": selected_name,
        "features": (
            [ELO_DIFFERENCE_FEATURE]
            if selected_name == "elo_baseline"
            else list(MODEL_FEATURES)
        ),
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "prediction_mode": "PRE_TOSS",
        "probability_orientation": "team_a",
    }
    joblib.dump(bundle, output / "prematch_model.joblib")

    report = {
        "input": str(args.input),
        "paired_matches": int(len(paired)),
        "features": list(MODEL_FEATURES),
        "split": {
            "train": _date_range(train),
            "validation": _date_range(valid),
            "test": _date_range(test),
        },
        "validation_metrics": validation,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "selection_rule": "lowest validation Brier score, then log loss",
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nsaved {output / 'prematch_model.joblib'}")
    print(f"saved {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
