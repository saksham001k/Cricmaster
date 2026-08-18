"""Train independently selected T20I and T20 Cricmaster routers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
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
from cricmaster.models.routed import (
    SUPPORTED_ROUTED_FORMATS,
    symmetric_probability,
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


def _augment(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    return (
        pd.concat([x, -x], ignore_index=True),
        pd.concat([y.astype(int), 1 - y.astype(int)], ignore_index=True),
    )


def _fit(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> Pipeline:
    x = frame.loc[:, features].copy()
    for column in features:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    y = frame["team_a_win"].astype(int)
    x_aug, y_aug = _augment(x, y)
    model.fit(x_aug, y_aug)
    return model


def _metrics(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> dict[str, Any]:
    bundle = {"model": model, "features": list(features)}
    probability = symmetric_probability(bundle, frame)
    y = frame["team_a_win"].astype(int).to_numpy()
    return classification_metrics(y, probability)


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


def _range(frame: pd.DataFrame) -> dict[str, Any]:
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


def _train_domain(
    frame: pd.DataFrame,
    *,
    domain: str,
    mode: str,
    features: tuple[str, ...],
    train_end: str,
    valid_end: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    domain_frame = frame.loc[frame["format"] == domain].copy()
    train, valid, test = _split(
        domain_frame,
        train_end=train_end,
        valid_end=valid_end,
    )

    if train.empty or valid.empty or test.empty:
        raise RuntimeError(
            f"{mode}/{domain}: temporal split produced an empty partition"
        )

    print(f"\n=== {mode} / {domain} ===")
    print("train:", _range(train))
    print("valid:", _range(valid))
    print("test :", _range(test))

    candidates = {
        "logistic_regression": _logistic(),
        "hist_gradient_boosting": _tree(),
    }

    validation: dict[str, dict[str, Any]] = {}

    print("\nVALIDATION")
    for name, model in candidates.items():
        fitted = _fit(model, train, features)
        validation[name] = _metrics(fitted, valid, features)
        _print_metrics(name, validation[name])

    selected = min(
        validation,
        key=lambda name: (
            validation[name]["brier_score"],
            validation[name]["log_loss"],
        ),
    )
    print(f"selected={selected}")

    development = pd.concat([train, valid], ignore_index=True)

    final_models: dict[str, Pipeline] = {}
    test_metrics: dict[str, dict[str, Any]] = {}

    print("\nFINAL TEST (2025+)")
    for name, model in candidates.items():
        final_models[name] = _fit(model, development, features)
        test_metrics[name] = _metrics(final_models[name], test, features)
        _print_metrics(name, test_metrics[name])

    bundle = {
        "model": final_models[selected],
        "model_name": selected,
        "features": list(features),
        "domain": domain,
        "prediction_mode": mode,
        "train_end": train_end,
        "valid_end": valid_end,
        "probability_orientation": "team_a",
    }

    report = {
        "domain": domain,
        "prediction_mode": mode,
        "split": {
            "train": _range(train),
            "validation": _range(valid),
            "test": _range(test),
        },
        "validation_metrics": validation,
        "selected_model": selected,
        "test_metrics": test_metrics,
    }
    return bundle, report


def _train_router(
    source: pd.DataFrame,
    *,
    mode: str,
    pairer: Callable[[pd.DataFrame], pd.DataFrame],
    features: tuple[str, ...],
    train_end: str,
    valid_end: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paired = pairer(source)

    bundles: dict[str, dict[str, Any]] = {}
    reports: dict[str, Any] = {}

    for domain in SUPPORTED_ROUTED_FORMATS:
        bundle, report = _train_domain(
            paired,
            domain=domain,
            mode=mode,
            features=features,
            train_end=train_end,
            valid_end=valid_end,
        )
        bundles[domain] = bundle
        reports[domain] = report

    router = {
        "router_type": "FORMAT",
        "prediction_mode": mode,
        "domains": list(SUPPORTED_ROUTED_FORMATS),
        "bundles": bundles,
    }
    return router, reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train format-routed expanded Cricmaster models."
    )
    parser.add_argument(
        "--input",
        default="data/processed/t20_expanded/prematch_features.parquet",
    )
    parser.add_argument("--output", default="models/routed_expanded")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--valid-end", default="2024-12-31")
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.input)

    pre_router, pre_report = _train_router(
        source,
        mode="PRE_TOSS",
        pairer=pair_prematch_rows,
        features=MODEL_FEATURES,
        train_end=args.train_end,
        valid_end=args.valid_end,
    )

    post_router, post_report = _train_router(
        source,
        mode="POST_TOSS",
        pairer=pair_post_toss_rows,
        features=POST_TOSS_FEATURES,
        train_end=args.train_end,
        valid_end=args.valid_end,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    joblib.dump(pre_router, output / "prematch_router.joblib")
    joblib.dump(post_router, output / "posttoss_router.joblib")

    report = {
        "input": str(args.input),
        "selection_rule": "lowest 2024 validation Brier score, then log loss",
        "prematch": pre_report,
        "posttoss": post_report,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nsaved {output / 'prematch_router.joblib'}")
    print(f"saved {output / 'posttoss_router.joblib'}")
    print(f"saved {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
