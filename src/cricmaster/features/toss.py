"""Toss feature helpers. PRE_TOSS rows must leave these fields null."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from cricmaster.data.models import MatchMetadata
from cricmaster.data.team_aliases import canonicalize_team


class PredictionMode(StrEnum):
    PRE_TOSS = "PRE_TOSS"
    POST_TOSS = "POST_TOSS"


def toss_features(
    metadata: MatchMetadata,
    team: str,
    mode: PredictionMode,
) -> dict[str, Any]:
    if mode is PredictionMode.PRE_TOSS:
        return {
            "prediction_mode": mode.value,
            "toss_winner": None,
            "toss_decision": None,
            "team_won_toss": None,
        }
    toss_winner = metadata.toss_winner
    return {
        "prediction_mode": mode.value,
        "toss_winner": toss_winner,
        "toss_decision": metadata.toss_decision,
        "team_won_toss": (
            None
            if not toss_winner
            else canonicalize_team(toss_winner) == canonicalize_team(team)
        ),
    }
