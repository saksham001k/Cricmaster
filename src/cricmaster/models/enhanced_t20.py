"""Enhanced T20 candidate feature construction."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from cricmaster.models.posttoss import pair_post_toss_rows
from cricmaster.models.prematch import pair_prematch_rows


SHRINKAGE_PRIOR_MATCHES: Final[float] = 10.0
PRIOR_WIN_RATE: Final[float] = 0.5

NEW_XI_SOURCE_FEATURES: Final[tuple[str, ...]] = (
    "xi_mean_batting_strike_rate",
    "xi_mean_recent_strike_rate",
    "xi_mean_recent_bowling_economy",
)

NEW_XI_DIFF_FEATURES: Final[tuple[str, ...]] = tuple(
    f"{name}_diff" for name in NEW_XI_SOURCE_FEATURES
)

SHRUNK_DIFF_FEATURES: Final[tuple[str, ...]] = (
    "shrunk_win_rate_before_diff",
    "shrunk_h2h_win_rate_diff",
    "shrunk_venue_win_rate_diff",
)

ENHANCED_PRE_TOSS_EXTRA: Final[tuple[str, ...]] = SHRUNK_DIFF_FEATURES
ENHANCED_POST_TOSS_EXTRA: Final[tuple[str, ...]] = (
    *SHRUNK_DIFF_FEATURES,
    *NEW_XI_DIFF_FEATURES,
)


def shrunk_rate(
    wins: object,
    matches: object,
    *,
    prior_matches: float = SHRINKAGE_PRIOR_MATCHES,
    prior_rate: float = PRIOR_WIN_RATE,
) -> float:
    try:
        n = float(matches)
        w = float(wins)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(n) or not np.isfinite(w) or n < 0:
        return float("nan")

    return (w + prior_matches * prior_rate) / (n + prior_matches)


def _pair_extra_rates(source: pd.DataFrame) -> pd.DataFrame:
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
    return work


def _append_pairwise_differences(
    paired: pd.DataFrame,
    source: pd.DataFrame,
    *,
    mode: str,
    source_columns: tuple[str, ...],
) -> pd.DataFrame:
    work = source.loc[source["prediction_mode"] == mode].copy()

    lookup: dict[str, dict[str, float]] = {}
    for match_id, group in work.groupby("match_id", sort=False):
        if len(group) != 2:
            continue
        ordered = group.sort_values("team", kind="stable").reset_index(drop=True)
        row: dict[str, float] = {}
        for column in source_columns:
            if column not in ordered.columns:
                raise ValueError(f"Missing enhanced source column: {column}")
            a = pd.to_numeric(
                pd.Series([ordered.iloc[0][column]]),
                errors="coerce",
            ).iloc[0]
            b = pd.to_numeric(
                pd.Series([ordered.iloc[1][column]]),
                errors="coerce",
            ).iloc[0]
            row[f"{column}_diff"] = (
                float(a - b)
                if pd.notna(a) and pd.notna(b)
                else float("nan")
            )
        lookup[str(match_id)] = row

    result = paired.copy()
    for column in source_columns:
        diff_col = f"{column}_diff"
        result[diff_col] = [
            lookup.get(str(match_id), {}).get(diff_col, float("nan"))
            for match_id in result["match_id"]
        ]
    return result


def pair_enhanced_prematch(source: pd.DataFrame) -> pd.DataFrame:
    work = _pair_extra_rates(source)
    paired = pair_prematch_rows(work)
    return _append_pairwise_differences(
        paired,
        work,
        mode="PRE_TOSS",
        source_columns=(
            "shrunk_win_rate_before",
            "shrunk_h2h_win_rate",
            "shrunk_venue_win_rate",
        ),
    )


def pair_enhanced_posttoss(source: pd.DataFrame) -> pd.DataFrame:
    work = _pair_extra_rates(source)
    paired = pair_post_toss_rows(work)
    return _append_pairwise_differences(
        paired,
        work,
        mode="POST_TOSS",
        source_columns=(
            "shrunk_win_rate_before",
            "shrunk_h2h_win_rate",
            "shrunk_venue_win_rate",
            *NEW_XI_SOURCE_FEATURES,
        ),
    )
