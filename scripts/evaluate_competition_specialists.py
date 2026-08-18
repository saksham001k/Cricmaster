"""Evaluate Cricmaster competition specialists on the frozen 2025+ holdout."""

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
from cricmaster.models.routed import routed_probability
from cricmaster.models.specialist import (
    competition_key,
    specialist_bundle,
    specialist_probability,
)


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


def _metrics_base(
    router: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    return safe_metrics(
        frame["team_a_win"].astype(int).to_numpy(),
        routed_probability(router, frame),
    )


def _metrics_specialist(
    router: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    return safe_metrics(
        frame["team_a_win"].astype(int).to_numpy(),
        specialist_probability(router, frame),
    )


def _line(label: str, metrics: dict[str, Any]) -> None:
    auc = metrics["roc_auc"]
    auc_text = "N/A" if auc is None else f"{auc:.4f}"
    print(
        f"{label:30} "
        f"n={metrics['matches']:4d} "
        f"acc={metrics['accuracy']:.4f} "
        f"auc={auc_text} "
        f"logloss={metrics['log_loss']:.4f} "
        f"brier={metrics['brier_score']:.4f}"
    )


def _run_mode(
    *,
    name: str,
    source: pd.DataFrame,
    pairer: Callable[[pd.DataFrame], pd.DataFrame],
    base_router: dict[str, Any],
    specialist_router: dict[str, Any],
    test_start: str,
    competition_min: int,
) -> dict[str, Any]:
    frame = pairer(source)
    frame = frame.loc[frame["date"] >= pd.Timestamp(test_start)].copy()
    frame["competition_key"] = frame["competition"].map(competition_key)

    base_metrics = _metrics_base(base_router, frame)
    specialist_metrics = _metrics_specialist(specialist_router, frame)

    print(f"\n=== {name} FULL EXPANDED 2025+ ===")
    _line("format router", base_metrics)
    _line("competition specialists", specialist_metrics)

    t20 = frame.loc[frame["format"] == "T20"].copy()
    base_t20 = _metrics_base(base_router, t20)
    specialist_t20 = _metrics_specialist(specialist_router, t20)

    print(f"\n=== {name} T20 ONLY 2025+ ===")
    _line("format router", base_t20)
    _line("competition specialists", specialist_t20)

    competition_rows: list[dict[str, Any]] = []

    for key, group in t20.groupby("competition_key", dropna=False):
        if len(group) < competition_min:
            continue

        base = _metrics_base(base_router, group)
        specialist = _metrics_specialist(specialist_router, group)

        first = group.iloc[0]
        _bundle, route = specialist_bundle(
            specialist_router,
            match_format=str(first["format"]),
            competition=first["competition"],
        )

        row = {
            "competition": "(none)" if pd.isna(key) else str(key),
            "route": route,
            "matches": int(len(group)),
            "base_brier": base["brier_score"],
            "specialist_brier": specialist["brier_score"],
            "brier_delta_specialist_minus_base": (
                specialist["brier_score"] - base["brier_score"]
            ),
            "base_accuracy": base["accuracy"],
            "specialist_accuracy": specialist["accuracy"],
        }
        competition_rows.append(row)

    competition_rows.sort(key=lambda item: item["matches"], reverse=True)

    print(f"\n{name} T20 COMPETITIONS")
    for row in competition_rows[:25]:
        print(
            f"  {row['competition'][:28]:28} "
            f"n={row['matches']:4d} "
            f"route={row['route'][:24]:24} "
            f"base={row['base_brier']:.3f} "
            f"special={row['specialist_brier']:.3f} "
            f"delta={row['brier_delta_specialist_minus_base']:+.3f}"
        )

    return {
        "full": {
            "base": base_metrics,
            "specialist": specialist_metrics,
        },
        "t20_only": {
            "base": base_t20,
            "specialist": specialist_t20,
        },
        "competition_comparison": competition_rows,
        "approved_specialists": sorted(
            (specialist_router.get("specialists") or {}).keys()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate competition specialists against format routing."
    )
    parser.add_argument(
        "--data",
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
    parser.add_argument(
        "--prematch-specialist-router",
        default="models/specialist_expanded/prematch_specialist_router.joblib",
    )
    parser.add_argument(
        "--posttoss-specialist-router",
        default="models/specialist_expanded/posttoss_specialist_router.joblib",
    )
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--competition-min", type=int, default=20)
    parser.add_argument(
        "--output",
        default="data/processed/model_comparison/step13_specialist_metrics.json",
    )
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.data)
    base_pre = joblib.load(args.base_prematch_router)
    base_post = joblib.load(args.base_posttoss_router)
    specialist_pre = joblib.load(args.prematch_specialist_router)
    specialist_post = joblib.load(args.posttoss_specialist_router)

    report = {
        "test_start": args.test_start,
        "prematch": _run_mode(
            name="PRE_TOSS",
            source=source,
            pairer=pair_prematch_rows,
            base_router=base_pre,
            specialist_router=specialist_pre,
            test_start=args.test_start,
            competition_min=args.competition_min,
        ),
        "posttoss": _run_mode(
            name="POST_TOSS",
            source=source,
            pairer=pair_post_toss_rows,
            base_router=base_post,
            specialist_router=specialist_post,
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
