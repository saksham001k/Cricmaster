"""Batter summaries extracted from completed historical matches."""

from __future__ import annotations

from dataclasses import dataclass

from cricmaster.data.models import MatchState
from cricmaster.data.team_aliases import canonicalize_team


@dataclass
class BattingInnings:
    player: str
    runs: int
    balls: int
    out: bool


def extract_batting_innings(match: MatchState) -> list[BattingInnings]:
    """One row per batter innings using only this match's deliveries."""

    grouped: dict[tuple[int, str], dict[str, int | bool | str]] = {}
    for delivery in match.deliveries:
        if not delivery.striker:
            continue
        key = (delivery.innings, delivery.striker)
        row = grouped.setdefault(
            key,
            {"player": delivery.striker, "runs": 0, "balls": 0, "out": False},
        )
        row["runs"] = int(row["runs"]) + delivery.runs_batter
        if not delivery.is_wide:
            row["balls"] = int(row["balls"]) + 1
        if delivery.wicket and delivery.player_out == delivery.striker:
            row["out"] = True
    return [
        BattingInnings(
            player=str(row["player"]),
            runs=int(row["runs"]),
            balls=int(row["balls"]),
            out=bool(row["out"]),
        )
        for row in grouped.values()
    ]


def team_xi(match: MatchState, team: str) -> list[str] | None:
    meta = match.metadata
    if canonicalize_team(team) == canonicalize_team(meta.team1):
        return list(meta.team1_players) if meta.team1_players else None
    if canonicalize_team(team) == canonicalize_team(meta.team2):
        return list(meta.team2_players) if meta.team2_players else None
    return None
