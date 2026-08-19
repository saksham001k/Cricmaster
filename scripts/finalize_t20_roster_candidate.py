"""Final locked sanity check for the roster-aware T20 production candidate.

Feature selection is frozen from Step 17:
- PRE_TOSS: previous-XI core strength
- POST_TOSS: previous-XI core strength

2025+ is evaluated once here. This script must not be used for further tuning.
"""

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

from cricmaster.models.evaluate import classification_metrics
from cricmaster.models.posttoss import POST_TOSS_FEATURES, pair_post_toss_rows
from cricmaster.models.prematch import MODEL_FEATURES, pair_prematch_rows
from cricmaster.models.roster_features import (
    PREVIOUS_CORE_DIFFS,
    append_roster_differences,
    build_roster_side_features,
)
from cricmaster.models.routed import symmetric_probability


TRAIN_END = pd.Timestamp("2023-12-31")
VALID_START = pd.Timestamp("2024-01-01")
VALID_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")

# Locked before viewing Step 18 output.
MAX_BRIER_DEGRADATION = 0.003
MAX_AUC_DEGRADATION = 0.010


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
    raise ValueError(name)


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


def _select_on_2024(
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[str, dict[str, dict[str, Any]]]:
    train = frame.loc[frame["date"] <= TRAIN_END].copy()
    valid = frame.loc[
        (frame["date"] >= VALID_START)
        & (frame["date"] <= VALID_END)
    ].copy()

    results: dict[str, dict[str, Any]] = {}
    for name in ("logistic_regression", "hist_gradient_boosting"):
        model = _fit(_candidate(name), train, features)
        results[name] = _metrics(model, valid, features)

    selected = min(
        results,
        key=lambda name: (
            results[name]["brier_score"],
            results[name]["log_loss"],
        ),
    )
    return selected, results


def _fit_through_2024_and_test(
    frame: pd.DataFrame,
    *,
    architecture: str,
    features: tuple[str, ...],
) -> tuple[Pipeline, dict[str, Any]]:
    development = frame.loc[frame["date"] <= VALID_END].copy()
    test = frame.loc[frame["date"] >= TEST_START].copy()

    model = _fit(_candidate(architecture), development, features)
    return model, _metrics(model, test, features)


def _accept(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    brier_delta = candidate["brier_score"] - baseline["brier_score"]
    auc_delta = candidate["roc_auc"] - baseline["roc_auc"]

    accepted = (
        brier_delta <= MAX_BRIER_DEGRADATION
        and auc_delta >= -MAX_AUC_DEGRADATION
    )

    return {
        "accepted": bool(accepted),
        "brier_delta_candidate_minus_baseline": float(brier_delta),
        "auc_delta_candidate_minus_baseline": float(auc_delta),
        "rules": {
            "max_brier_degradation": MAX_BRIER_DEGRADATION,
            "max_auc_degradation": MAX_AUC_DEGRADATION,
        },
    }


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"{label:28} "
        f"n={metrics['n']:4d} "
        f"acc={metrics['accuracy']:.4f} "
        f"auc={metrics['roc_auc']:.4f} "
        f"logloss={metrics['log_loss']:.4f} "
        f"brier={metrics['brier_score']:.4f} "
        f"ece={metrics['ece_10']:.4f}"
    )


def _run_mode(
    *,
    name: str,
    frame: pd.DataFrame,
    baseline_features: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_features = (*baseline_features, *PREVIOUS_CORE_DIFFS)

    base_arch, base_valid = _select_on_2024(frame, baseline_features)
    candidate_arch, candidate_valid = _select_on_2024(frame, candidate_features)

    base_model, base_test = _fit_through_2024_and_test(
        frame,
        architecture=base_arch,
        features=baseline_features,
    )
    candidate_model, candidate_test = _fit_through_2024_and_test(
        frame,
        architecture=candidate_arch,
        features=candidate_features,
    )

    decision = _accept(base_test, candidate_test)

    print(f"\n=== {name} LOCKED 2025+ SANITY CHECK ===")
    print(f"baseline_architecture={base_arch}")
    print(f"candidate_architecture={candidate_arch}")
    _print_metrics("baseline", base_test)
    _print_metrics("roster candidate", candidate_test)
    print(
        "decision="
        + ("ACCEPT" if decision["accepted"] else "REJECT")
        + f" brier_delta={decision['brier_delta_candidate_minus_baseline']:+.4f}"
        + f" auc_delta={decision['auc_delta_candidate_minus_baseline']:+.4f}"
    )

    report = {
        "baseline_features": list(baseline_features),
        "candidate_features": list(candidate_features),
        "baseline_architecture": base_arch,
        "candidate_architecture": candidate_arch,
        "validation_2024": {
            "baseline": base_valid,
            "candidate": candidate_valid,
        },
        "test_2025_plus": {
            "baseline": base_test,
            "candidate": candidate_test,
        },
        "decision": decision,
    }

    bundle = {
        "model": candidate_model,
        "model_name": candidate_arch,
        "features": list(candidate_features),
        "prediction_mode": name,
        "format": "T20",
        "feature_family": "previous_xi_core_strength",
        "trained_through": "2024-12-31",
    }
    return report, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        default="data/raw/cricsheet/t20_expanded",
    )
    parser.add_argument(
        "--data",
        default="data/processed/t20_expanded/prematch_features.parquet",
    )
    parser.add_argument(
        "--roster-cache",
        default="data/processed/model_comparison/step18_roster_side_all.parquet",
    )
    parser.add_argument(
        "--output",
        default="models/roster_candidate",
    )
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.data)
    source["date"] = pd.to_datetime(source["date"], errors="raise")

    cache = Path(args.roster_cache)
    if cache.exists():
        print(f"Loading roster cache: {cache}")
        side = pd.read_parquet(cache)
        side["date"] = pd.to_datetime(side["date"], errors="raise")
    else:
        print("Building leakage-safe roster history through full corpus ...")
        side = build_roster_side_features(
            args.raw,
            cutoff="2099-12-31",
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        side.to_parquet(cache, index=False)
        print(f"saved roster cache {cache}")

    pre = pair_prematch_rows(source)
    pre = pre.loc[pre["format"] == "T20"].copy()
    pre = append_roster_differences(pre, side)

    post = pair_post_toss_rows(source)
    post = post.loc[post["format"] == "T20"].copy()
    post = append_roster_differences(post, side)

    pre_report, pre_bundle = _run_mode(
        name="PRE_TOSS",
        frame=pre,
        baseline_features=MODEL_FEATURES,
    )
    post_report, post_bundle = _run_mode(
        name="POST_TOSS",
        frame=post,
        baseline_features=POST_TOSS_FEATURES,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "locked_policy": {
            "feature_selection_source": "Step 17 rolling 2022/2023/2024 validation",
            "no_further_tuning_after_this_test": True,
            "acceptance_rules": {
                "max_brier_degradation": MAX_BRIER_DEGRADATION,
                "max_auc_degradation": MAX_AUC_DEGRADATION,
            },
        },
        "prematch": pre_report,
        "posttoss": post_report,
    }

    (output / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if pre_report["decision"]["accepted"]:
        joblib.dump(pre_bundle, output / "prematch_t20_roster.joblib")
        print(f"saved {output / 'prematch_t20_roster.joblib'}")
    else:
        print("PRE_TOSS candidate rejected; model artifact not promoted.")

    if post_report["decision"]["accepted"]:
        joblib.dump(post_bundle, output / "posttoss_t20_roster.joblib")
        print(f"saved {output / 'posttoss_t20_roster.joblib'}")
    else:
        print("POST_TOSS candidate rejected; model artifact not promoted.")

    print(f"saved {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
