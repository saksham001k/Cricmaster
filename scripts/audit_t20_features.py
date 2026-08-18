"""Audit why Cricmaster domestic/franchise T20 prediction is weak.

This script is diagnostic only. It does not train or replace models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.models.posttoss import POST_TOSS_FEATURES, pair_post_toss_rows
from cricmaster.models.prematch import MODEL_FEATURES, pair_prematch_rows


UNUSED_CANDIDATE_FEATURES = (
    "opponent_matches_at_venue",
    "opponent_win_rate_at_venue",
    "venue_batting_first_win_rate",
    "venue_chasing_win_rate",
    "venue_decided_matches",
    "historical_first_innings_average",
    "historical_first_innings_matches",
    "home_away",
    "season",
)


def orientation_free_auc(
    y_true: pd.Series | np.ndarray,
    feature: pd.Series | np.ndarray,
) -> float | None:
    """Measure univariate discrimination without assuming coefficient sign."""

    y = np.asarray(y_true, dtype=int)
    x = pd.to_numeric(pd.Series(feature), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x)

    if int(mask.sum()) < 10 or len(np.unique(y[mask])) < 2:
        return None

    auc = float(roc_auc_score(y[mask], x[mask]))
    return max(auc, 1.0 - auc)


def _pct(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def _safe_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _coverage_for_group(
    pre_rows: pd.DataFrame,
    post_rows: pd.DataFrame,
) -> dict[str, Any]:
    match_count = int(pre_rows["match_id"].nunique())

    post_match_ids = set(post_rows["match_id"])
    pre_for_post = pre_rows.loc[pre_rows["match_id"].isin(post_match_ids)]

    lineup_known = (
        post_rows["lineup_status"].astype(str).eq("LINEUP_KNOWN")
        if "lineup_status" in post_rows.columns
        else pd.Series(False, index=post_rows.index)
    )

    return {
        "matches": match_count,
        "teams": int(pre_rows["team"].nunique()),
        "median_matches_before_per_side": _safe_median(pre_rows["matches_before"]),
        "pct_side_rows_lt_5_prior_matches": _pct(pre_rows["matches_before"] < 5),
        "pct_side_rows_lt_10_prior_matches": _pct(pre_rows["matches_before"] < 10),
        "pct_side_rows_lt_20_prior_matches": _pct(pre_rows["matches_before"] < 20),
        "pct_side_rows_no_h2h": _pct(pre_rows["h2h_matches_before"] == 0),
        "median_h2h_matches_before": _safe_median(pre_rows["h2h_matches_before"]),
        "pct_side_rows_no_team_venue_history": _pct(
            pre_rows["team_matches_at_venue"] == 0
        ),
        "median_team_matches_at_venue": _safe_median(
            pre_rows["team_matches_at_venue"]
        ),
        "pct_home_away_present": _pct(pre_rows["home_away"].notna()),
        "posttoss_matches_available": int(pre_for_post["match_id"].nunique()),
        "pct_posttoss_side_rows_lineup_known": _pct(lineup_known),
        "median_xi_batters_with_history": _safe_median(
            post_rows["xi_batters_with_history"]
        ),
        "median_xi_bowlers_with_history": _safe_median(
            post_rows["xi_bowlers_with_history"]
        ),
        "pct_xi_batting_average_missing": _pct(
            post_rows["xi_mean_batting_average"].isna()
        ),
        "pct_xi_bowling_economy_missing": _pct(
            post_rows["xi_mean_bowling_economy"].isna()
        ),
    }


def competition_coverage(
    source: pd.DataFrame,
    *,
    start_date: str,
    min_matches: int,
) -> pd.DataFrame:
    """Summarize historical-feature depth for T20 competitions."""

    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.loc[
        (frame["format"] == "T20")
        & (frame["date"] >= pd.Timestamp(start_date))
    ].copy()

    pre = frame.loc[frame["prediction_mode"] == "PRE_TOSS"].copy()
    post = frame.loc[frame["prediction_mode"] == "POST_TOSS"].copy()

    rows: list[dict[str, Any]] = []

    for competition, pre_group in pre.groupby("competition", dropna=False):
        matches = int(pre_group["match_id"].nunique())
        if matches < min_matches:
            continue

        post_group = post.loc[
            post["match_id"].isin(pre_group["match_id"].unique())
        ].copy()

        row = {
            "competition": "(none)" if pd.isna(competition) else str(competition),
            **_coverage_for_group(pre_group, post_group),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        ["matches", "competition"],
        ascending=[False, True],
    ).reset_index(drop=True)


def team_cold_start(
    source: pd.DataFrame,
    *,
    start_date: str,
    min_matches: int,
) -> pd.DataFrame:
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.loc[
        (frame["format"] == "T20")
        & (frame["prediction_mode"] == "PRE_TOSS")
        & (frame["date"] >= pd.Timestamp(start_date))
    ].copy()

    rows: list[dict[str, Any]] = []
    for team, group in frame.groupby("team"):
        matches = int(group["match_id"].nunique())
        if matches < min_matches:
            continue

        competitions = sorted(
            str(item)
            for item in group["competition"].dropna().unique()
        )
        rows.append(
            {
                "team": str(team),
                "matches": matches,
                "competitions": ", ".join(competitions),
                "median_prior_matches": _safe_median(group["matches_before"]),
                "min_prior_matches": int(group["matches_before"].min()),
                "pct_rows_lt_10_prior": _pct(group["matches_before"] < 10),
                "pct_rows_no_h2h": _pct(group["h2h_matches_before"] == 0),
                "pct_rows_no_venue_history": _pct(
                    group["team_matches_at_venue"] == 0
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        ["median_prior_matches", "matches"],
        ascending=[True, False],
    ).reset_index(drop=True)


def feature_signal(
    paired: pd.DataFrame,
    *,
    features: tuple[str, ...],
    valid_start: str,
    valid_end: str,
    test_start: str,
) -> pd.DataFrame:
    """Compare univariate feature signal on 2024 validation vs 2025+ test."""

    valid = paired.loc[
        (paired["date"] >= pd.Timestamp(valid_start))
        & (paired["date"] <= pd.Timestamp(valid_end))
    ].copy()
    test = paired.loc[paired["date"] >= pd.Timestamp(test_start)].copy()

    rows: list[dict[str, Any]] = []

    for feature in features:
        valid_auc = orientation_free_auc(
            valid["team_a_win"],
            valid[feature],
        )
        test_auc = orientation_free_auc(
            test["team_a_win"],
            test[feature],
        )

        rows.append(
            {
                "feature": feature,
                "validation_auc_orientation_free": valid_auc,
                "test_auc_orientation_free": test_auc,
                "auc_change_test_minus_validation": (
                    None
                    if valid_auc is None or test_auc is None
                    else float(test_auc - valid_auc)
                ),
                "validation_missing_pct": float(valid[feature].isna().mean()),
                "test_missing_pct": float(test[feature].isna().mean()),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        "test_auc_orientation_free",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def unused_feature_availability(
    source: pd.DataFrame,
    *,
    start_date: str,
) -> pd.DataFrame:
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.loc[
        (frame["format"] == "T20")
        & (frame["prediction_mode"] == "PRE_TOSS")
        & (frame["date"] >= pd.Timestamp(start_date))
    ].copy()

    rows: list[dict[str, Any]] = []
    for feature in UNUSED_CANDIDATE_FEATURES:
        if feature not in frame.columns:
            rows.append(
                {
                    "feature": feature,
                    "present_in_dataset": False,
                    "non_null_pct": 0.0,
                    "unique_values": 0,
                }
            )
            continue

        series = frame[feature]
        rows.append(
            {
                "feature": feature,
                "present_in_dataset": True,
                "non_null_pct": float(series.notna().mean()),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def _print_table(title: str, frame: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("(no rows)")
        return
    print(frame.loc[:, columns].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit feature quality for expanded domestic/franchise T20."
    )
    parser.add_argument(
        "--input",
        default="data/processed/t20_expanded/prematch_features.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/processed/feature_audit",
    )
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--valid-start", default="2024-01-01")
    parser.add_argument("--valid-end", default="2024-12-31")
    parser.add_argument("--competition-min", type=int, default=20)
    parser.add_argument("--team-min", type=int, default=5)
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.input)
    source["date"] = pd.to_datetime(source["date"], errors="raise")

    pre_paired = pair_prematch_rows(source)
    pre_t20 = pre_paired.loc[pre_paired["format"] == "T20"].copy()

    post_paired = pair_post_toss_rows(source)
    post_t20 = post_paired.loc[post_paired["format"] == "T20"].copy()

    coverage = competition_coverage(
        source,
        start_date=args.test_start,
        min_matches=args.competition_min,
    )
    cold = team_cold_start(
        source,
        start_date=args.test_start,
        min_matches=args.team_min,
    )
    pre_signal = feature_signal(
        pre_t20,
        features=MODEL_FEATURES,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
    )
    post_signal = feature_signal(
        post_t20,
        features=POST_TOSS_FEATURES,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
    )
    unused = unused_feature_availability(
        source,
        start_date=args.test_start,
    )

    _print_table(
        "2025+ T20 COMPETITION COVERAGE",
        coverage,
        [
            "competition",
            "matches",
            "teams",
            "median_matches_before_per_side",
            "pct_side_rows_lt_10_prior_matches",
            "pct_side_rows_no_h2h",
            "pct_side_rows_no_team_venue_history",
            "pct_posttoss_side_rows_lineup_known",
            "median_xi_batters_with_history",
            "median_xi_bowlers_with_history",
        ],
    )

    _print_table(
        "PRE_TOSS FEATURE SIGNAL",
        pre_signal,
        [
            "feature",
            "validation_auc_orientation_free",
            "test_auc_orientation_free",
            "auc_change_test_minus_validation",
            "test_missing_pct",
        ],
    )

    _print_table(
        "POST_TOSS FEATURE SIGNAL",
        post_signal,
        [
            "feature",
            "validation_auc_orientation_free",
            "test_auc_orientation_free",
            "auc_change_test_minus_validation",
            "test_missing_pct",
        ],
    )

    _print_table(
        "AVAILABLE BUT CURRENTLY UNUSED CANDIDATES",
        unused,
        [
            "feature",
            "present_in_dataset",
            "non_null_pct",
            "unique_values",
        ],
    )

    print("\n=== LOW-HISTORY T20 TEAMS (first 30) ===")
    if cold.empty:
        print("(no rows)")
    else:
        print(cold.head(30).to_string(index=False))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    coverage.to_csv(output / "competition_coverage.csv", index=False)
    cold.to_csv(output / "team_cold_start.csv", index=False)
    pre_signal.to_csv(output / "prematch_feature_signal.csv", index=False)
    post_signal.to_csv(output / "posttoss_feature_signal.csv", index=False)
    unused.to_csv(output / "unused_feature_availability.csv", index=False)

    summary = {
        "input": args.input,
        "test_start": args.test_start,
        "t20_test_matches": int(
            pre_t20.loc[pre_t20["date"] >= pd.Timestamp(args.test_start), "match_id"]
            .nunique()
        ),
        "competition_rows": int(len(coverage)),
        "teams_audited": int(len(cold)),
        "home_away_non_null_pct_2025_t20": (
            float(
                source.loc[
                    (source["format"] == "T20")
                    & (source["prediction_mode"] == "PRE_TOSS")
                    & (source["date"] >= pd.Timestamp(args.test_start)),
                    "home_away",
                ].notna().mean()
            )
            if "home_away" in source.columns
            else None
        ),
        "notes": [
            "Orientation-free AUC near 0.50 means weak standalone discrimination.",
            "Large validation-to-test AUC drops indicate feature instability or season shift.",
            "Coverage metrics are side-row percentages, not match percentages.",
            "This audit does not use 2025+ results for model selection; it diagnoses the already-open holdout.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nsaved audit files under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
