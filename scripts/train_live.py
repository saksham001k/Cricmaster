"""Train Cricmaster first-innings and chase live win-probability models."""

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

from cricmaster.models.live import (
    CHASE_FEATURES,
    FIRST_INNINGS_FEATURES,
    elo_live_baseline,
    equal_match_weights,
    live_probability_metrics,
    prepare_live_training_frame,
)


def _split(
    frame: pd.DataFrame,
    *,
    train_end: str,
    valid_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_cutoff = pd.Timestamp(train_end)
    valid_cutoff = pd.Timestamp(valid_end)

    train = frame.loc[frame["date"] <= train_cutoff].copy()
    valid = frame.loc[
        (frame["date"] > train_cutoff) & (frame["date"] <= valid_cutoff)
    ].copy()
    test = frame.loc[frame["date"] > valid_cutoff].copy()
    return train, valid, test


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
                    max_iter=1500,
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
                    max_iter=250,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _fit(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> Pipeline:
    x = frame.loc[:, features]
    y = frame["batting_team_eventual_win"].astype(int)
    weights = equal_match_weights(frame)
    model.fit(x, y, model__sample_weight=weights)
    return model


def _predict(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    return model.predict_proba(frame.loc[:, features])[:, 1]


def _range(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"states": 0, "matches": 0, "start": None, "end": None}
    return {
        "states": int(len(frame)),
        "matches": int(frame["match_id"].nunique()),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
    }


def _show(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"{label:24} "
        f"states={metrics['states']:7d} "
        f"matches={metrics['matches']:4d} "
        f"acc={metrics['accuracy']:.4f} "
        f"auc={metrics['roc_auc']:.4f} "
        f"logloss={metrics['log_loss']:.4f} "
        f"brier={metrics['brier_score']:.4f} "
        f"ece={metrics['ece_10']:.4f}"
    )


def _train_one(
    frame: pd.DataFrame,
    *,
    name: str,
    innings_number: int,
    features: tuple[str, ...],
    output: Path,
    train_end: str,
    valid_end: str,
) -> dict[str, Any]:
    subset = frame.loc[frame["innings_number"] == innings_number].copy()
    train, valid, test = _split(
        subset,
        train_end=train_end,
        valid_end=valid_end,
    )
    if train.empty or valid.empty or test.empty:
        raise RuntimeError(f"{name}: temporal split produced an empty partition")

    print(f"\n=== {name.upper()} SPLIT ===")
    print("train:", _range(train))
    print("valid:", _range(valid))
    print("test :", _range(test))

    candidates = {
        "logistic_regression": _logistic(),
        "hist_gradient_boosting": _tree(),
    }

    validation: dict[str, dict[str, Any]] = {
        "elo_baseline": live_probability_metrics(valid, elo_live_baseline(valid))
    }

    print(f"\n=== {name.upper()} VALIDATION ===")
    _show("elo_baseline", validation["elo_baseline"])

    for model_name, model in candidates.items():
        fitted = _fit(model, train, features)
        validation[model_name] = live_probability_metrics(
            valid,
            _predict(fitted, valid, features),
        )
        _show(model_name, validation[model_name])

    selected = min(
        candidates,
        key=lambda model_name: (
            validation[model_name]["brier_score"],
            validation[model_name]["log_loss"],
        ),
    )
    print(f"selected_{name}={selected}")

    development = pd.concat([train, valid], ignore_index=True)
    final_models: dict[str, Pipeline] = {}
    test_metrics: dict[str, dict[str, Any]] = {
        "elo_baseline": live_probability_metrics(test, elo_live_baseline(test))
    }

    for model_name, model in candidates.items():
        final_models[model_name] = _fit(model, development, features)
        test_metrics[model_name] = live_probability_metrics(
            test,
            _predict(final_models[model_name], test, features),
        )

    print(f"\n=== {name.upper()} FINAL TEST (2025+) ===")
    for model_name in ("elo_baseline", "logistic_regression", "hist_gradient_boosting"):
        _show(model_name, test_metrics[model_name])

    bundle = {
        "model": final_models[selected],
        "model_name": selected,
        "features": list(features),
        "innings_number": innings_number,
        "train_end": train_end,
        "valid_end": valid_end,
        "prediction_mode": "LIVE_AFTER_LEGAL_BALL",
        "probability_orientation": "batting_team",
        "match_equal_weighting": True,
    }
    model_path = output / f"{name}_model.joblib"
    joblib.dump(bundle, model_path)

    return {
        "name": name,
        "innings_number": innings_number,
        "features": list(features),
        "split": {
            "train": _range(train),
            "validation": _range(valid),
            "test": _range(test),
        },
        "validation_metrics": validation,
        "selected_model": selected,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "selection_rule": "lowest validation Brier score, then log loss",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train match-balanced Cricmaster live win-probability models."
    )
    parser.add_argument(
        "--live",
        default="data/processed/t20_corpus/live_states.parquet",
    )
    parser.add_argument(
        "--prematch",
        default="data/processed/t20_corpus/prematch_features.parquet",
    )
    parser.add_argument("--output", default="models/live")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--valid-end", default="2024-12-31")
    args = parser.parse_args(argv)

    print("Loading processed datasets ...")
    live = pd.read_parquet(args.live)
    prematch = pd.read_parquet(args.prematch)

    print("Preparing legal-ball live states with POST_TOSS context ...")
    frame = prepare_live_training_frame(live, prematch)
    print(
        f"prepared_states={len(frame)} "
        f"matches={frame['match_id'].nunique()} "
        f"innings1={(frame['innings_number'] == 1).sum()} "
        f"innings2={(frame['innings_number'] == 2).sum()}"
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    first = _train_one(
        frame,
        name="first_innings",
        innings_number=1,
        features=FIRST_INNINGS_FEATURES,
        output=output,
        train_end=args.train_end,
        valid_end=args.valid_end,
    )
    chase = _train_one(
        frame,
        name="chase",
        innings_number=2,
        features=CHASE_FEATURES,
        output=output,
        train_end=args.train_end,
        valid_end=args.valid_end,
    )

    report = {
        "prepared_states": int(len(frame)),
        "prepared_matches": int(frame["match_id"].nunique()),
        "legal_balls_only": True,
        "binary_results_only": True,
        "match_equal_weighting": True,
        "first_innings": first,
        "chase": chase,
    }
    report_path = output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nsaved {output / 'first_innings_model.joblib'}")
    print(f"saved {output / 'chase_model.joblib'}")
    print(f"saved {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
