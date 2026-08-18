"""Team-perspective pre-match rows. Labels may use the result; features may not."""

from __future__ import annotations

from typing import Any, Iterable

from cricmaster.data.models import MatchState
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.history import HistoricalState
from cricmaster.features.toss import PredictionMode, toss_features
from cricmaster.features.utils import completed_binary_result, exclusion_reason

PREMATCH_MODES = (PredictionMode.PRE_TOSS, PredictionMode.POST_TOSS)


def build_prematch_rows(
    match: MatchState,
    state: HistoricalState,
    *,
    modes: Iterable[PredictionMode] = PREMATCH_MODES,
) -> tuple[list[dict[str, Any]], str | None]:
    reason = exclusion_reason(match)
    if reason:
        return [], reason
    if completed_binary_result(match) is None:
        return [], exclusion_reason(match) or "not_binary_result"

    meta = match.metadata
    winner = canonicalize_team(meta.winner) if meta.winner else ""
    rows: list[dict[str, Any]] = []
    sides = (
        (meta.team1, meta.team2, meta.team1_players),
        (meta.team2, meta.team1, meta.team2_players),
    )
    for raw_team, raw_opponent, _players in sides:
        team = canonicalize_team(raw_team)
        opponent = canonicalize_team(raw_opponent)
        history = state.features_for(match, team, opponent)
        base = {
            "match_id": meta.match_id,
            "date": meta.date.isoformat() if meta.date else None,
            "format": str(meta.format),
            "competition": meta.competition,
            "season": meta.season,
            "venue": meta.venue,
            "city": meta.city,
            "gender": meta.gender,
            "home_away": None,
            "raw_team_name": raw_team,
            "team": team,
            "raw_opponent_name": raw_opponent,
            "opponent": opponent,
            "team_win": 1 if team == winner else 0,
        }
        base.update(history)
        for mode in modes:
            row = dict(base)
            row.update(toss_features(meta, team, mode))
            rows.append(row)
    return rows, None
