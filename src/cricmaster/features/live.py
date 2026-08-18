"""Ball-by-ball live states for limited-overs cricket.

Targets (eventual winner) are labels. In-match fields use only information
available after the current delivery.
"""

from __future__ import annotations

from typing import Any, Iterator

from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import InningsState, MatchState
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.utils import (
    innings_ball_limit,
    overs_notation,
    rate,
    supports_live_states,
)

WICKETS_AVAILABLE = 10


def compute_live_metrics(
    *,
    runs: int,
    wickets: int,
    legal_balls: int,
    target: int | None,
    ball_limit: int | None,
    balls_per_over: int,
) -> dict[str, Any]:
    """Scoreboard metrics using only information available after this ball."""

    bpo = balls_per_over if balls_per_over > 0 else 6
    balls_remaining = (ball_limit - legal_balls) if ball_limit is not None else None
    current_rr = rate(runs, legal_balls / bpo) if legal_balls else None
    if target is None:
        runs_required = None
        required_run_rate = None
        run_rate_difference = None
    else:
        runs_required = target - runs
        overs_remaining = (
            (balls_remaining / bpo)
            if balls_remaining is not None and balls_remaining > 0
            else None
        )
        required_run_rate = rate(runs_required, overs_remaining) if overs_remaining else None
        run_rate_difference = (
            (current_rr - required_run_rate)
            if current_rr is not None and required_run_rate is not None
            else None
        )
    return {
        "runs": runs,
        "wickets": wickets,
        "legal_balls_bowled": legal_balls,
        "overs": overs_notation(legal_balls, bpo),
        "balls_remaining": balls_remaining,
        "current_run_rate": current_rr,
        "target": target,
        "runs_required": runs_required,
        "required_run_rate": required_run_rate,
        "run_rate_difference": run_rate_difference,
        "wickets_in_hand": WICKETS_AVAILABLE - wickets,
    }


def _innings_lookup(match: MatchState) -> dict[int, InningsState]:
    return {item.innings_number: item for item in match.innings_history}


def _chase_target(innings: InningsState | None) -> int | None:
    if innings is None or innings.target is None:
        return None
    return innings.target


def iter_live_states(match: MatchState) -> Iterator[dict[str, Any]]:
    if not supports_live_states(match.metadata.format):
        return
    meta = match.metadata
    winner = canonicalize_team(meta.winner) if meta.winner else None
    lookup = _innings_lookup(match)
    current_innings = 0
    runs = 0
    wickets = 0
    legal_balls = 0

    for delivery in match.deliveries:
        if delivery.innings != current_innings:
            current_innings = delivery.innings
            runs = 0
            wickets = 0
            legal_balls = 0
        runs += delivery.runs_total
        if delivery.wicket:
            wickets += 1
        if delivery.is_legal:
            legal_balls += 1

        innings_meta = lookup.get(delivery.innings)
        batting_team = canonicalize_team(delivery.batting_team)
        bowling_team = canonicalize_team(
            innings_meta.bowling_team
            if innings_meta and innings_meta.bowling_team
            else (
                meta.team2
                if canonicalize_team(meta.team1) == batting_team
                else meta.team1
            )
        )
        bpo = meta.balls_per_over or (5 if meta.format is MatchFormat.HUNDRED else 6)
        target = _chase_target(innings_meta)
        ball_limit = innings_ball_limit(
            meta.format,
            scheduled_overs=meta.scheduled_overs,
            balls_per_over=bpo,
            target_overs=innings_meta.target_overs if innings_meta is not None else None,
        )
        metrics = compute_live_metrics(
            runs=runs,
            wickets=wickets,
            legal_balls=legal_balls,
            target=target,
            ball_limit=ball_limit,
            balls_per_over=bpo,
        )
        batting_win = None if winner is None else int(batting_team == winner)
        yield {
            "match_id": meta.match_id,
            "date": meta.date.isoformat() if meta.date else None,
            "format": str(meta.format),
            "competition": meta.competition,
            "innings_number": delivery.innings,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            **metrics,
            "is_wide": delivery.is_wide,
            "is_noball": delivery.is_noball,
            "is_wicket": delivery.wicket,
            "is_legal": delivery.is_legal,
            "runs_this_ball": delivery.runs_total,
            "striker": delivery.striker or None,
            "non_striker": delivery.non_striker or None,
            "bowler": delivery.bowler or None,
            "eventual_winner": winner,
            "batting_team_eventual_win": batting_win,
        }
