"""Train validation-gated competition specialists for Cricmaster T20."""

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
from cricmaster.models.routed import symmetric_probability
from cricmaster.models.specialist import competition_key


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
    if name == "hist_gradient_boosting":
        return _tree()
    raise ValueError(f"Unknown model architecture {name!r}")


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
    probability = symmetric_probability(
        {"model": model, "features": list(features)},
        frame,
    )
    return classification_metrics(
        frame["team_a_win"].astype(int).to_numpy(),
        probability,
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


def _select_architecture(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Pipeline]]:
    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for name in ("logistic_regression", "hist_gradient_boosting"):
        fitted[name] = _fit(_candidate(name), train, features)
        metrics[name] = _metrics(fitted[name], valid, features)

    selected = min(
        metrics,
        key=lambda name: (
            metrics[name]["brier_score"],
            metrics[name]["log_loss"],
        ),
    )
    return selected, metrics, fitted


def _prepare(
    source: pd.DataFrame,
    pairer: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    paired = pairer(source)
    paired = paired.loc[paired["format"] == "T20"].copy()
    paired["competition_key"] = paired["competition"].map(competition_key)
    return paired


def _train_mode(
    source: pd.DataFrame,
    *,
    mode: str,
    pairer: Callable[[pd.DataFrame], pd.DataFrame],
    features: tuple[str, ...],
    base_router: dict[str, Any],
    train_end: str,
    valid_end: str,
    min_train: int,
    min_valid: int,
    min_brier_gain: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = _prepare(source, pairer)
    train_all, valid_all, _test_all = _split(
        frame,
        train_end=train_end,
        valid_end=valid_end,
    )

    base_t20 = base_router["bundles"]["T20"]
    base_architecture = str(base_t20["model_name"])

    # Refit the fallback architecture using train <= 2023 only so the 2024
    # validation comparison is leakage-safe.
    fallback_dev = _fit(
        _candidate(base_architecture),
        train_all,
        features,
    )

    approved: dict[str, dict[str, Any]] = {}
    decisions: dict[str, Any] = {}

    keys = sorted(
        key
        for key in frame["competition_key"].dropna().unique()
        if str(key).strip()
    )

    print(f"\n=== {mode} COMPETITION GATING ===")
    print(
        f"fallback={base_architecture} "
        f"min_train={min_train} min_valid={min_valid} "
        f"min_brier_gain={min_brier_gain:.4f}"
    )

    for key in keys:
        competition = frame.loc[frame["competition_key"] == key].copy()
        train, valid, test = _split(
            competition,
            train_end=train_end,
            valid_end=valid_end,
        )

        decision: dict[str, Any] = {
            "competition": key,
            "train_matches": int(len(train)),
            "validation_matches": int(len(valid)),
            "test_matches": int(len(test)),
            "approved": False,
            "reason": None,
        }

        if len(train) < min_train:
            decision["reason"] = "insufficient_train"
            decisions[key] = decision
            continue

        if len(valid) < min_valid:
            decision["reason"] = "insufficient_validation"
            decisions[key] = decision
            continue

        selected, candidate_metrics, _candidate_models = _select_architecture(
            train,
            valid,
            features,
        )
        specialist_valid = candidate_metrics[selected]
        fallback_valid = _metrics(fallback_dev, valid, features)
        gain = (
            fallback_valid["brier_score"]
            - specialist_valid["brier_score"]
        )

        decision.update(
            {
                "selected_model": selected,
                "candidate_validation_metrics": candidate_metrics,
                "fallback_validation_metrics": fallback_valid,
                "specialist_validation_metrics": specialist_valid,
                "validation_brier_gain": float(gain),
            }
        )

        if gain < min_brier_gain:
            decision["reason"] = "insufficient_validation_gain"
            decisions[key] = decision
            print(
                f"{key:28} rejected "
                f"train={len(train):4d} valid={len(valid):3d} "
                f"gain={gain:+.4f}"
            )
            continue

        development = pd.concat([train, valid], ignore_index=True)
        final_model = _fit(
            _candidate(selected),
            development,
            features,
        )
        bundle = {
            "model": final_model,
            "model_name": selected,
            "features": list(features),
            "domain": "T20",
            "competition": key,
            "prediction_mode": mode,
            "train_end": train_end,
            "valid_end": valid_end,
            "probability_orientation": "team_a",
        }

        decision["approved"] = True
        decision["reason"] = "validation_approved"
        approved[key] = bundle
        decisions[key] = decision

        print(
            f"{key:28} APPROVED "
            f"train={len(train):4d} valid={len(valid):3d} "
            f"model={selected:22} gain={gain:+.4f}"
        )

    router = {
        "router_type": "FORMAT_PLUS_COMPETITION",
        "prediction_mode": mode,
        "base_router": base_router,
        "specialists": approved,
        "gating": {
            "min_train": min_train,
            "min_valid": min_valid,
            "min_brier_gain": min_brier_gain,
            "validation_end": valid_end,
        },
    }
    report = {
        "prediction_mode": mode,
        "fallback_architecture": base_architecture,
        "approved_specialists": sorted(approved),
        "decisions": decisions,
    }
    return router, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train validation-gated competition specialists."
    )
    parser.add_argument(
        "--input",
        default="data/processed/t20_expanded/prematch_features.parquet",
    )
    parser.add_argument(
        "--base-prematch-router",
        default="models/routed_expanded/prematch_router.joblib",
    )
    parser.add_argument(
        "--base-posttoss-router",
        default="models/routed_expanded/posttoss_router.joblib",
    )
    parser.add_argument("--output", default="models/specialist_expanded")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--valid-end", default="2024-12-31")
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--min-valid", type=int, default=25)
    parser.add_argument("--min-brier-gain", type=float, default=0.005)
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.input)
    base_pre = joblib.load(args.base_prematch_router)
    base_post = joblib.load(args.base_posttoss_router)

    pre_router, pre_report = _train_mode(
        source,
        mode="PRE_TOSS",
        pairer=pair_prematch_rows,
        features=MODEL_FEATURES,
        base_router=base_pre,
        train_end=args.train_end,
        valid_end=args.valid_end,
        min_train=args.min_train,
        min_valid=args.min_valid,
        min_brier_gain=args.min_brier_gain,
    )

    post_router, post_report = _train_mode(
        source,
        mode="POST_TOSS",
        pairer=pair_post_toss_rows,
        features=POST_TOSS_FEATURES,
        base_router=base_post,
        train_end=args.train_end,
        valid_end=args.valid_end,
        min_train=args.min_train,
        min_valid=args.min_valid,
        min_brier_gain=args.min_brier_gain,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    joblib.dump(pre_router, output / "prematch_specialist_router.joblib")
    joblib.dump(post_router, output / "posttoss_specialist_router.joblib")

    report = {
        "selection_policy": {
            "architecture": "lowest validation Brier score, then log loss",
            "specialist_gate": (
                "specialist must meet train/validation minimums and beat the "
                "generic T20 fallback on the same 2024 competition rows"
            ),
        },
        "prematch": pre_report,
        "posttoss": post_report,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nsaved {output / 'prematch_specialist_router.joblib'}")
    print(f"saved {output / 'posttoss_specialist_router.joblib'}")
    print(f"saved {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
