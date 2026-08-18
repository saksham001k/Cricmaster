"""Apples-to-apples benchmark for Cricmaster narrow vs expanded models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.models.prematch import elo_probability, pair_prematch_rows
from cricmaster.models.posttoss import pair_post_toss_rows


def safe_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    predicted = (p >= 0.5).astype(int)

    auc: float | None
    if len(np.unique(y)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y, p))

    return {
        "matches": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "roc_auc": auc,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "positive_rate": float(y.mean()) if len(y) else None,
    }


def bundle_probability(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    name = str(bundle["model_name"])
    features = list(bundle["features"])

    if name == "elo_baseline":
        diff = frame.loc[:, features[0]].fillna(0.0).to_numpy(dtype=float)
        return np.asarray(elo_probability(diff), dtype=float)

    model = bundle.get("model")
    if model is None:
        raise ValueError(f"Bundle {name!r} contains no fitted model")

    x = frame.loc[:, features].copy()
    # Runtime follows the training pipelines' constant-zero missing policy.
    for column in features:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    forward = model.predict_proba(x)[:, 1]
    reverse = model.predict_proba(-x)[:, 1]
    return 0.5 * (forward + (1.0 - reverse))


def _test_only(frame: pd.DataFrame, start: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(start)
    return frame.loc[frame["date"] >= cutoff].copy()


def _common_ids(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return sorted(set(left["match_id"]) & set(right["match_id"]))


def _assert_common_targets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    common_ids: list[str],
) -> None:
    left_target = left.set_index("match_id").loc[common_ids, "team_a_win"].astype(int)
    right_target = right.set_index("match_id").loc[common_ids, "team_a_win"].astype(int)
    if not left_target.equals(right_target):
        mismatches = left_target[left_target != right_target]
        raise ValueError(
            f"Common-match targets disagree for {len(mismatches)} matches"
        )


def _evaluate(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    y = frame["team_a_win"].astype(int).to_numpy()
    p = bundle_probability(bundle, frame)
    return safe_metrics(y, p)


def _subgroups(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    column: str,
    *,
    min_matches: int = 1,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = frame.assign(
        **{column: frame[column].fillna("(none)").astype(str)}
    ).groupby(column, dropna=False)

    for value, group in grouped:
        if len(group) < min_matches:
            continue
        metrics = _evaluate(bundle, group)
        rows.append({"group": str(value), **metrics})

    rows.sort(key=lambda item: item["matches"], reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def _compare_common_by_group(
    old_frame: pd.DataFrame,
    expanded_frame: pd.DataFrame,
    old_bundle: dict[str, Any],
    expanded_bundle: dict[str, Any],
    common_ids: list[str],
    *,
    column: str,
    min_matches: int,
) -> list[dict[str, Any]]:
    old_indexed = old_frame.set_index("match_id", drop=False)
    expanded_indexed = expanded_frame.set_index("match_id", drop=False)

    rows: list[dict[str, Any]] = []
    values = expanded_indexed.loc[common_ids, column].fillna("(none)").astype(str)

    for value, ids in values.groupby(values).groups.items():
        match_ids = list(ids)
        if len(match_ids) < min_matches:
            continue

        old_group = old_indexed.loc[match_ids].reset_index(drop=True)
        expanded_group = expanded_indexed.loc[match_ids].reset_index(drop=True)

        old_metrics = _evaluate(old_bundle, old_group)
        expanded_metrics = _evaluate(expanded_bundle, expanded_group)

        rows.append(
            {
                "group": str(value),
                "matches": len(match_ids),
                "old_brier": old_metrics["brier_score"],
                "expanded_brier": expanded_metrics["brier_score"],
                "brier_delta_expanded_minus_old": (
                    expanded_metrics["brier_score"] - old_metrics["brier_score"]
                ),
                "old_accuracy": old_metrics["accuracy"],
                "expanded_accuracy": expanded_metrics["accuracy"],
                "accuracy_delta_expanded_minus_old": (
                    expanded_metrics["accuracy"] - old_metrics["accuracy"]
                ),
            }
        )

    rows.sort(key=lambda item: item["matches"], reverse=True)
    return rows


def _print_metric_line(label: str, metrics: dict[str, Any]) -> None:
    auc = metrics["roc_auc"]
    auc_text = "N/A" if auc is None else f"{auc:.4f}"
    print(
        f"{label:28} "
        f"n={metrics['matches']:4d} "
        f"acc={metrics['accuracy']:.4f} "
        f"auc={auc_text} "
        f"logloss={metrics['log_loss']:.4f} "
        f"brier={metrics['brier_score']:.4f}"
    )


def _run_mode(
    *,
    name: str,
    pairer: Callable[[pd.DataFrame], pd.DataFrame],
    old_source: pd.DataFrame,
    expanded_source: pd.DataFrame,
    old_bundle: dict[str, Any],
    expanded_bundle: dict[str, Any],
    test_start: str,
    competition_min: int,
) -> dict[str, Any]:
    old = _test_only(pairer(old_source), test_start)
    expanded = _test_only(pairer(expanded_source), test_start)

    common_ids = _common_ids(old, expanded)
    _assert_common_targets(old, expanded, common_ids)

    old_common = (
        old.set_index("match_id", drop=False)
        .loc[common_ids]
        .reset_index(drop=True)
    )
    expanded_common = (
        expanded.set_index("match_id", drop=False)
        .loc[common_ids]
        .reset_index(drop=True)
    )

    old_common_metrics = _evaluate(old_bundle, old_common)
    expanded_common_metrics = _evaluate(expanded_bundle, expanded_common)
    expanded_all_metrics = _evaluate(expanded_bundle, expanded)

    print(f"\n=== {name} COMMON 2025+ MATCHES ===")
    print(
        f"old_test={len(old)} expanded_test={len(expanded)} "
        f"common={len(common_ids)} expanded_only={len(expanded) - len(common_ids)}"
    )
    _print_metric_line("old model / old features", old_common_metrics)
    _print_metric_line("expanded model / expanded", expanded_common_metrics)

    print(f"\n=== {name} EXPANDED FULL 2025+ ===")
    _print_metric_line("expanded selected model", expanded_all_metrics)

    common_format = _compare_common_by_group(
        old,
        expanded,
        old_bundle,
        expanded_bundle,
        common_ids,
        column="format",
        min_matches=20,
    )
    common_gender = _compare_common_by_group(
        old,
        expanded,
        old_bundle,
        expanded_bundle,
        common_ids,
        column="gender",
        min_matches=20,
    )

    competition = _subgroups(
        expanded,
        expanded_bundle,
        "competition",
        min_matches=competition_min,
        top_n=40,
    )

    print(f"\n{name} expanded test — largest competition groups:")
    for item in competition[:15]:
        auc = item["roc_auc"]
        auc_text = "N/A" if auc is None else f"{auc:.3f}"
        print(
            f"  {item['group'][:38]:38} "
            f"n={item['matches']:4d} "
            f"acc={item['accuracy']:.3f} "
            f"auc={auc_text} "
            f"brier={item['brier_score']:.3f}"
        )

    return {
        "coverage": {
            "old_test_matches": int(len(old)),
            "expanded_test_matches": int(len(expanded)),
            "common_test_matches": int(len(common_ids)),
            "expanded_only_test_matches": int(len(expanded) - len(common_ids)),
        },
        "old_common_metrics": old_common_metrics,
        "expanded_common_metrics": expanded_common_metrics,
        "expanded_full_test_metrics": expanded_all_metrics,
        "common_format_comparison": common_format,
        "common_gender_comparison": common_gender,
        "expanded_competition_metrics": competition,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Cricmaster narrow and expanded PRE_TOSS/POST_TOSS models "
            "on identical holdout matches and report expanded coverage."
        )
    )
    parser.add_argument(
        "--old-data",
        default="data/processed/t20_corpus/prematch_features.parquet",
    )
    parser.add_argument(
        "--expanded-data",
        default="data/processed/t20_expanded/prematch_features.parquet",
    )
    parser.add_argument(
        "--old-prematch-model",
        default="models/prematch/prematch_model.joblib",
    )
    parser.add_argument(
        "--expanded-prematch-model",
        default="models/prematch_expanded/prematch_model.joblib",
    )
    parser.add_argument(
        "--old-posttoss-model",
        default="models/posttoss/posttoss_model.joblib",
    )
    parser.add_argument(
        "--expanded-posttoss-model",
        default="models/posttoss_expanded/posttoss_model.joblib",
    )
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--competition-min", type=int, default=20)
    parser.add_argument(
        "--output",
        default="data/processed/model_comparison/step11_metrics.json",
    )
    args = parser.parse_args(argv)

    print("Loading old and expanded feature tables ...")
    old_source = pd.read_parquet(args.old_data)
    expanded_source = pd.read_parquet(args.expanded_data)

    old_pre = joblib.load(args.old_prematch_model)
    expanded_pre = joblib.load(args.expanded_prematch_model)
    old_post = joblib.load(args.old_posttoss_model)
    expanded_post = joblib.load(args.expanded_posttoss_model)

    report = {
        "test_start": args.test_start,
        "prematch": _run_mode(
            name="PRE_TOSS",
            pairer=pair_prematch_rows,
            old_source=old_source,
            expanded_source=expanded_source,
            old_bundle=old_pre,
            expanded_bundle=expanded_pre,
            test_start=args.test_start,
            competition_min=args.competition_min,
        ),
        "posttoss": _run_mode(
            name="POST_TOSS",
            pairer=pair_post_toss_rows,
            old_source=old_source,
            expanded_source=expanded_source,
            old_bundle=old_post,
            expanded_bundle=expanded_post,
            test_start=args.test_start,
            competition_min=args.competition_min,
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
