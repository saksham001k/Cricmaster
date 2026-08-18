"""Pre-match model dataset preparation and Elo baseline utilities."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

SIGNED_SOURCE_FEATURES: Final[tuple[str, ...]] = (
    "matches_before",
    "win_rate_before",
    "wins_last_5",
    "win_rate_last_5",
    "wins_last_10",
    "win_rate_last_10",
    "wins_last_20",
    "win_rate_last_20",
    "h2h_team_wins",
    "h2h_team_win_rate",
    "h2h_last_5_win_rate",
    "team_matches_at_venue",
    "team_win_rate_at_venue",
    "team_elo_before",
)

MODEL_FEATURES: Final[tuple[str, ...]] = tuple(
    f"{name}_diff" for name in SIGNED_SOURCE_FEATURES
)
ELO_DIFFERENCE_FEATURE: Final[str] = "team_elo_before_diff"


def _numeric(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def pair_prematch_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert two team-perspective PRE_TOSS rows into one row per match.

    Team A is chosen deterministically by canonical team name. Every model
    feature is a signed Team-A minus Team-B difference. This lets downstream
    models be symmetrized so swapping the teams produces complementary
    probabilities.
    """

    required = {
        "match_id",
        "date",
        "team",
        "team_win",
        "prediction_mode",
        "format",
        "competition",
        "gender",
        *SIGNED_SOURCE_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required pre-match columns: {missing}")

    pre_toss = frame.loc[frame["prediction_mode"] == "PRE_TOSS"].copy()
    if pre_toss.empty:
        raise ValueError("No PRE_TOSS rows found")

    records: list[dict[str, object]] = []

    for match_id, group in pre_toss.groupby("match_id", sort=False):
        if len(group) != 2:
            raise ValueError(
                f"Match {match_id} has {len(group)} PRE_TOSS rows; expected exactly 2"
            )

        ordered = group.sort_values("team", kind="stable").reset_index(drop=True)
        team_a = ordered.iloc[0]
        team_b = ordered.iloc[1]

        a_label = int(team_a["team_win"])
        b_label = int(team_b["team_win"])
        if a_label + b_label != 1:
            raise ValueError(f"Match {match_id} does not contain exactly one winner")

        record: dict[str, object] = {
            "match_id": str(match_id),
            "date": team_a["date"],
            "format": team_a["format"],
            "competition": team_a["competition"],
            "gender": team_a["gender"],
            "team_a": team_a["team"],
            "team_b": team_b["team"],
            "team_a_win": a_label,
        }

        for source in SIGNED_SOURCE_FEATURES:
            a_value = _numeric(team_a[source])
            b_value = _numeric(team_b[source])
            record[f"{source}_diff"] = (
                a_value - b_value
                if not (np.isnan(a_value) or np.isnan(b_value))
                else np.nan
            )

        records.append(record)

    paired = pd.DataFrame.from_records(records)
    paired["date"] = pd.to_datetime(paired["date"], errors="raise")
    return paired.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)


def elo_probability(elo_difference: pd.Series | np.ndarray | float) -> np.ndarray:
    """Convert a Team-A minus Team-B Elo difference into Team-A win probability."""

    diff = np.asarray(elo_difference, dtype=float)
    return 1.0 / (1.0 + np.power(10.0, -diff / 400.0))
