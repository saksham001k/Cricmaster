"""Legal-ball arithmetic and null-safe cricket rate helpers.

Cricket over notation is not decimal arithmetic. ``17.4`` means 17 overs
and 4 balls, never 17.4 × 6.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cricmaster.data.formats import (
    LIMITED_OVERS_FORMATS,
    MatchFormat,
    UNLIMITED_OVERS_FORMATS,
)
from cricmaster.data.models import Delivery, MatchState

FORM_WINDOWS = (5, 10, 20)


def is_legal_delivery(*, is_wide: bool, is_noball: bool) -> bool:
    return not is_wide and not is_noball


def cricket_overs_to_balls(overs: float | int, balls_per_over: int) -> int:
    """Convert cricket over notation (e.g. 17.4) into legal balls."""

    if balls_per_over <= 0:
        balls_per_over = 6
    whole = int(overs)
    fractional_balls = int(round((float(overs) - whole) * 10))
    return whole * balls_per_over + fractional_balls


def balls_to_over_parts(legal_balls: int, balls_per_over: int) -> tuple[int, int]:
    if balls_per_over <= 0:
        balls_per_over = 6
    return divmod(max(legal_balls, 0), balls_per_over)


def overs_notation(legal_balls: int, balls_per_over: int) -> float:
    overs, balls = balls_to_over_parts(legal_balls, balls_per_over)
    return overs + (balls / 10.0)


def rate(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def innings_ball_limit(
    match_format: MatchFormat,
    *,
    scheduled_overs: int | None = None,
    balls_per_over: int | None = None,
    target_overs: float | int | None = None,
) -> int | None:
    """Return scheduled legal balls for a limited-overs innings, else None."""

    if match_format in UNLIMITED_OVERS_FORMATS or match_format is MatchFormat.OTHER:
        return None
    bpo = balls_per_over or (5 if match_format is MatchFormat.HUNDRED else 6)
    if target_overs is not None:
        return cricket_overs_to_balls(target_overs, bpo)
    if match_format is MatchFormat.HUNDRED:
        if scheduled_overs and bpo:
            product = scheduled_overs * bpo
            if product in {100, 50}:
                return product
            if scheduled_overs == 100:
                return 100
        return 100
    if scheduled_overs:
        return scheduled_overs * bpo
    defaults = {
        MatchFormat.T20: 120,
        MatchFormat.T20I: 120,
        MatchFormat.T10: 60,
        MatchFormat.ODI: 300,
        MatchFormat.LIST_A: 300,
    }
    return defaults.get(match_format)


def supports_live_states(match_format: MatchFormat) -> bool:
    return match_format in LIMITED_OVERS_FORMATS


def chronological_key(match: MatchState) -> tuple[date, str]:
    match_date = match.metadata.date or date.min
    return (match_date, match.metadata.match_id)


def sort_matches(matches: list[MatchState]) -> list[MatchState]:
    return sorted(matches, key=chronological_key)


def completed_binary_result(match: MatchState) -> str | None:
    """Return 'win' when a competing team won; otherwise None."""

    winner = match.metadata.winner
    if match.metadata.result_type != "win" or not winner:
        return None
    teams = {match.metadata.team1, match.metadata.team2}
    if winner not in teams:
        return None
    return "win"


def exclusion_reason(match: MatchState) -> str | None:
    if not match.metadata.date:
        return "missing_date"
    result = match.metadata.result_type or "unknown_result"
    if completed_binary_result(match) is None:
        return result if result != "win" else "winner_not_in_teams"
    return None


def win_rate(wins: int, matches: int) -> float | None:
    return rate(wins, matches)


def recent_window_stats(results: list[int], window: int) -> dict[str, Any]:
    """results are 1 (win) or 0 (not a win) in chronological order."""

    sample = results[-window:]
    played = len(sample)
    wins = int(sum(sample))
    return {
        "played": played,
        "wins": wins,
        "win_rate": win_rate(wins, played),
    }
