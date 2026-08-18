"""POST_TOSS model preparation with toss and optional playing-XI strength."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from cricmaster.models.prematch import MODEL_FEATURES as PRE_TOSS_FEATURES
from cricmaster.models.prematch import SIGNED_SOURCE_FEATURES

XI_SOURCE_FEATURES: Final[tuple[str, ...]] = (
    "xi_batters_with_history",
    "xi_mean_batting_average",
    "xi_mean_recent_runs",
    "xi_bowlers_with_history",
    "xi_mean_bowling_economy",
    "xi_mean_recent_wickets",
)

XI_DIFF_FEATURES: Final[tuple[str, ...]] = tuple(
    f"{name}_diff" for name in XI_SOURCE_FEATURES
)

TOSS_FEATURES: Final[tuple[str, ...]] = (
    "toss_bat_advantage",
    "toss_field_advantage",
)

POST_TOSS_FEATURES: Final[tuple[str, ...]] = (
    *PRE_TOSS_FEATURES,
    *XI_DIFF_FEATURES,
    *TOSS_FEATURES,
)


def _numeric(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def pair_post_toss_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert two POST_TOSS team rows into one symmetric match row."""

    required = {
        "match_id",
        "date",
        "team",
        "team_win",
        "prediction_mode",
        "format",
        "competition",
        "gender",
        "toss_winner",
        "toss_decision",
        "team_won_toss",
        *SIGNED_SOURCE_FEATURES,
        *XI_SOURCE_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required POST_TOSS columns: {missing}")

    post_toss = frame.loc[frame["prediction_mode"] == "POST_TOSS"].copy()
    if post_toss.empty:
        raise ValueError("No POST_TOSS rows found")

    records: list[dict[str, object]] = []

    for match_id, group in post_toss.groupby("match_id", sort=False):
        if len(group) != 2:
            raise ValueError(
                f"Match {match_id} has {len(group)} POST_TOSS rows; expected exactly 2"
            )

        ordered = group.sort_values("team", kind="stable").reset_index(drop=True)
        team_a = ordered.iloc[0]
        team_b = ordered.iloc[1]

        a_label = int(team_a["team_win"])
        b_label = int(team_b["team_win"])
        if a_label + b_label != 1:
            raise ValueError(f"Match {match_id} does not contain exactly one winner")

        a_won_toss = bool(team_a["team_won_toss"])
        b_won_toss = bool(team_b["team_won_toss"])
        if a_won_toss == b_won_toss:
            raise ValueError(f"Match {match_id} has inconsistent toss ownership")

        decision = str(team_a["toss_decision"] or "").strip().lower()
        toss_sign = 1.0 if a_won_toss else -1.0

        record: dict[str, object] = {
            "match_id": str(match_id),
            "date": team_a["date"],
            "format": team_a["format"],
            "competition": team_a["competition"],
            "gender": team_a["gender"],
            "team_a": team_a["team"],
            "team_b": team_b["team"],
            "team_a_win": a_label,
            "toss_bat_advantage": toss_sign if decision == "bat" else 0.0,
            "toss_field_advantage": toss_sign
            if decision in {"field", "bowl"}
            else 0.0,
        }

        for source in (*SIGNED_SOURCE_FEATURES, *XI_SOURCE_FEATURES):
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
