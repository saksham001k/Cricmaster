"""Evaluate format-routed models against the single expanded global models."""

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

from cricmaster.models.posttoss import pair_post_toss_rows
from cricmaster.models.prematch import pair_prematch_rows
from cricmaster.models.routed import routed_probability, symmetric_probability


def safe_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    predicted = (p >= 0.5).astype(int)

    auc = None if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))

    return {
        "matches": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "roc_auc": auc,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
    }


def _metrics_global(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    return safe_metrics(
        frame["team_a_win"].astype(int).to_numpy(),
        symmetric_probability(bundle, frame),
    )


def _metrics_routed(
    router: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    return safe_metrics(
        frame["team_a_win"].astype(int).to_numpy(),
        routed_probability(router, frame),
    )


def _competition_rows(
    frame: pd.DataFrame,
    global_bundle: dict[str, Any],
    router: dict[str, Any],
    *,
    min_matches: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    work = frame.copy()
    work["competition"] = work["competition"].fillna("(none)").astype(str)

    for competition, group in work.groupby("competition"):
        if len(group) < min_matches:
            continue

        global_metrics = _metrics_global(global_bundle, group)
        routed_metrics = _metrics_routed(router, group)

        rows.append(
            {
                "competition": str(competition),
                "matches": int(len(group)),
                "global_brier": global_metrics["brier_score"],
                "routed_brier": routed_metrics["brier_score"],
                "brier_delta_routed_minus_global": (
                    routed_metrics["brier_score"] - global_metrics["brier_score"]
                ),
                "global_accuracy": global_metrics["accuracy"],
                "routed_accuracy": routed_metrics["accuracy"],
            }
        )

    rows.sort(key=lambda item: item["matches"], reverse=True)
    return rows


def _print_line(label: str, metrics: dict[str, Any]) -> None:
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
    source: pd.DataFrame,
    global_bundle: dict[str, Any],
    router: dict[str, Any],
    test_start: str,
    competition_min: int,
) -> dict[str, Any]:
    frame = pairer(source)
    frame = frame.loc[frame["date"] >= pd.Timestamp(test_start)].copy()

    global_metrics = _metrics_global(global_bundle, frame)
    routed_metrics = _metrics_routed(router, frame)

    print(f"\n=== {name} FULL EXPANDED 2025+ ===")
    _print_line("single expanded model", global_metrics)
    _print_line("format-routed model", routed_metrics)

    by_format: dict[str, Any] = {}
    print(f"\n{name} BY FORMAT")
    for domain, group in frame.groupby("format"):
        global_domain = _metrics_global(global_bundle, group)
        routed_domain = _metrics_routed(router, group)
        by_format[str(domain)] = {
            "global": global_domain,
            "routed": routed_domain,
        }
        print(f"\n{domain}")
        _print_line("single expanded model", global_domain)
        _print_line("format-routed model", routed_domain)

    competitions = _competition_rows(
        frame,
        global_bundle,
        router,
        min_matches=competition_min,
    )

    print(f"\n{name} largest competition groups")
    for row in competitions[:20]:
        print(
            f"  {row['competition'][:36]:36} "
            f"n={row['matches']:4d} "
            f"global={row['global_brier']:.3f} "
            f"routed={row['routed_brier']:.3f} "
            f"delta={row['brier_delta_routed_minus_global']:+.3f}"
        )

    return {
        "global_metrics": global_metrics,
        "routed_metrics": routed_metrics,
        "by_format": by_format,
        "competition_comparison": competitions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate expanded global vs T20I/T20 routed models."
    )
    parser.add_argument(
        "--data",
        default="data/processed/t20_expanded/prematch_features.parquet",
    )
    parser.add_argument(
        "--global-prematch",
        default="models/prematch_expanded/prematch_model.joblib",
    )
    parser.add_argument(
        "--global-posttoss",
        default="models/posttoss_expanded/posttoss_model.joblib",
    )
    parser.add_argument(
        "--prematch-router",
        default="models/routed_expanded/prematch_router.joblib",
    )
    parser.add_argument(
        "--posttoss-router",
        default="models/routed_expanded/posttoss_router.joblib",
    )
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--competition-min", type=int, default=20)
    parser.add_argument(
        "--output",
        default="data/processed/model_comparison/step12_routed_metrics.json",
    )
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.data)

    global_pre = joblib.load(args.global_prematch)
    global_post = joblib.load(args.global_posttoss)
    pre_router = joblib.load(args.prematch_router)
    post_router = joblib.load(args.posttoss_router)

    report = {
        "test_start": args.test_start,
        "prematch": _run_mode(
            name="PRE_TOSS",
            pairer=pair_prematch_rows,
            source=source,
            global_bundle=global_pre,
            router=pre_router,
            test_start=args.test_start,
            competition_min=args.competition_min,
        ),
        "posttoss": _run_mode(
            name="POST_TOSS",
            pairer=pair_post_toss_rows,
            source=source,
            global_bundle=global_post,
            router=post_router,
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
