"""Experimental T20 feature groups evaluated only on pre-2025 rolling folds."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


INITIAL_ELO = 1500.0
DEFAULT_K = 20.0
DEFAULT_SEASON_RETENTION = 0.5
PRIOR_MATCHES = 10.0
PRIOR_RATE = 0.5


def shrunk_rate(
    wins: object,
    matches: object,
    *,
    prior_matches: float = PRIOR_MATCHES,
    prior_rate: float = PRIOR_RATE,
) -> float:
    try:
        w = float(wins)
        n = float(matches)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(w) or not np.isfinite(n) or n < 0:
        return float("nan")

    return (w + prior_matches * prior_rate) / (n + prior_matches)


def _elo_expected(team_rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - team_rating) / 400.0))


def _competition_key(value: object) -> str:
    if value is None or pd.isna(value):
        return "(none)"
    cleaned = " ".join(str(value).strip().split())
    return cleaned or "(none)"


def _season_key(value: object, date_value: object) -> str:
    if value is not None and not pd.isna(value):
        cleaned = " ".join(str(value).strip().split())
        if cleaned:
            return cleaned
    return str(pd.Timestamp(date_value).year)


def competition_season_elo(
    source: pd.DataFrame,
    *,
    mode: str = "PRE_TOSS",
    retention: float = DEFAULT_SEASON_RETENTION,
    k_factor: float = DEFAULT_K,
) -> pd.DataFrame:
    """Return one leakage-safe competition-season Elo differential per match.

    Ratings are competition/gender/team specific. At a team's first appearance
    in a new season, its latest competition rating is regressed toward 1500.
    """

    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must be between 0 and 1")

    required = {
        "match_id",
        "date",
        "format",
        "competition",
        "season",
        "gender",
        "team",
        "team_win",
        "prediction_mode",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing seasonal Elo columns: {missing}")

    work = source.loc[
        (source["prediction_mode"] == mode)
        & (source["format"] == "T20")
    ].copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work = work.sort_values(["date", "match_id", "team"], kind="stable")

    latest_rating: dict[tuple[str, str, str], float] = defaultdict(
        lambda: INITIAL_ELO
    )
    season_rating: dict[tuple[str, str, str, str], float] = {}
    records: list[dict[str, Any]] = []

    for match_id, group in work.groupby("match_id", sort=False):
        if len(group) != 2:
            raise ValueError(
                f"Match {match_id} has {len(group)} {mode} rows; expected 2"
            )

        ordered = group.sort_values("team", kind="stable").reset_index(drop=True)
        a = ordered.iloc[0]
        b = ordered.iloc[1]

        competition = _competition_key(a["competition"])
        gender = str(a["gender"] or "")
        season = _season_key(a["season"], a["date"])
        team_a = str(a["team"])
        team_b = str(b["team"])

        def rating(team: str) -> float:
            season_key = (competition, gender, season, team)
            if season_key not in season_rating:
                prior = latest_rating[(competition, gender, team)]
                season_rating[season_key] = (
                    INITIAL_ELO + retention * (prior - INITIAL_ELO)
                )
            return season_rating[season_key]

        rating_a = rating(team_a)
        rating_b = rating(team_b)

        records.append(
            {
                "match_id": str(match_id),
                "competition_season_elo_diff": rating_a - rating_b,
            }
        )

        score_a = int(a["team_win"])
        expected_a = _elo_expected(rating_a, rating_b)

        new_a = rating_a + k_factor * (score_a - expected_a)
        new_b = rating_b + k_factor * ((1 - score_a) - (1.0 - expected_a))

        season_rating[(competition, gender, season, team_a)] = new_a
        season_rating[(competition, gender, season, team_b)] = new_b
        latest_rating[(competition, gender, team_a)] = new_a
        latest_rating[(competition, gender, team_b)] = new_b

    return pd.DataFrame.from_records(records)


def add_side_derived_features(source: pd.DataFrame) -> pd.DataFrame:
    """Add experimental side-perspective features without changing source."""

    work = source.copy()

    work["shrunk_win_rate_before"] = [
        shrunk_rate(w, n)
        for w, n in zip(work["wins_before"], work["matches_before"])
    ]
    work["shrunk_h2h_win_rate"] = [
        shrunk_rate(w, n)
        for w, n in zip(work["h2h_team_wins"], work["h2h_matches_before"])
    ]
    work["shrunk_venue_win_rate"] = [
        shrunk_rate(w, n)
        for w, n in zip(work["team_wins_at_venue"], work["team_matches_at_venue"])
    ]
    work["log_matches_before"] = np.log1p(
        pd.to_numeric(work["matches_before"], errors="coerce").clip(lower=0)
    )

    return work


def append_pairwise_diff(
    paired: pd.DataFrame,
    source: pd.DataFrame,
    *,
    mode: str,
    source_column: str,
) -> pd.DataFrame:
    """Append Team-A minus Team-B difference for one side-level source column."""

    work = source.loc[
        (source["prediction_mode"] == mode)
        & (source["format"] == "T20")
    ].copy()

    lookup: dict[str, float] = {}

    for match_id, group in work.groupby("match_id", sort=False):
        if len(group) != 2:
            continue
        ordered = group.sort_values("team", kind="stable").reset_index(drop=True)
        a = pd.to_numeric(
            pd.Series([ordered.iloc[0][source_column]]),
            errors="coerce",
        ).iloc[0]
        b = pd.to_numeric(
            pd.Series([ordered.iloc[1][source_column]]),
            errors="coerce",
        ).iloc[0]
        lookup[str(match_id)] = (
            float(a - b)
            if pd.notna(a) and pd.notna(b)
            else float("nan")
        )

    result = paired.copy()
    result[f"{source_column}_diff"] = [
        lookup.get(str(match_id), float("nan"))
        for match_id in result["match_id"]
    ]
    return result


def add_seasonal_elo(
    paired: pd.DataFrame,
    source: pd.DataFrame,
    *,
    mode: str,
    retention: float = DEFAULT_SEASON_RETENTION,
) -> pd.DataFrame:
    seasonal = competition_season_elo(
        source,
        mode=mode,
        retention=retention,
    )
    return paired.merge(seasonal, on="match_id", how="left", validate="one_to_one")


def add_venue_toss_interaction(
    paired_posttoss: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Add signed venue preference aligned with the toss-winning side.

    Positive means Team A's toss outcome aligns with the venue's historical
    tendency; negative favors Team B. It is symmetric under team swapping.
    """

    work = source.loc[
        (source["prediction_mode"] == "POST_TOSS")
        & (source["format"] == "T20")
    ].copy()

    venue_lookup: dict[str, tuple[float, float]] = {}
    for match_id, group in work.groupby("match_id", sort=False):
        first = group.iloc[0]
        bat = pd.to_numeric(
            pd.Series([first["venue_batting_first_win_rate"]]),
            errors="coerce",
        ).iloc[0]
        chase = pd.to_numeric(
            pd.Series([first["venue_chasing_win_rate"]]),
            errors="coerce",
        ).iloc[0]
        venue_lookup[str(match_id)] = (
            float(bat) if pd.notna(bat) else float("nan"),
            float(chase) if pd.notna(chase) else float("nan"),
        )

    result = paired_posttoss.copy()
    values: list[float] = []

    for _, row in result.iterrows():
        bat_rate, chase_rate = venue_lookup.get(
            str(row["match_id"]),
            (float("nan"), float("nan")),
        )
        if not np.isfinite(bat_rate) or not np.isfinite(chase_rate):
            values.append(float("nan"))
            continue

        bat_tendency = bat_rate - 0.5
        chase_tendency = chase_rate - 0.5
        signed = (
            float(row["toss_bat_advantage"]) * bat_tendency
            + float(row["toss_field_advantage"]) * chase_tendency
        )
        values.append(signed)

    result["venue_toss_alignment"] = values
    return result
