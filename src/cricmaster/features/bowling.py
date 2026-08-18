"""Bowler summaries extracted from completed historical matches."""

from __future__ import annotations

from dataclasses import dataclass

from cricmaster.data.models import MatchState

NON_BOWLER_WICKETS = {
    "run out",
    "obstructing the field",
    "retired hurt",
    "retired out",
    "timed out",
    "absent hurt",
    "absent",
    "handled the ball",
}


@dataclass
class BowlingSpell:
    player: str
    balls: int
    runs: int
    wickets: int


def extract_bowling_spells(match: MatchState) -> list[BowlingSpell]:
    grouped: dict[tuple[int, str], dict[str, int | str]] = {}
    for delivery in match.deliveries:
        if not delivery.bowler:
            continue
        key = (delivery.innings, delivery.bowler)
        row = grouped.setdefault(
            key, {"player": delivery.bowler, "balls": 0, "runs": 0, "wickets": 0}
        )
        row["runs"] = int(row["runs"]) + delivery.runs_total
        if delivery.is_legal:
            row["balls"] = int(row["balls"]) + 1
        if delivery.wicket and (delivery.wicket_type or "").lower() not in NON_BOWLER_WICKETS:
            row["wickets"] = int(row["wickets"]) + 1
    return [
        BowlingSpell(
            player=str(row["player"]),
            balls=int(row["balls"]),
            runs=int(row["runs"]),
            wickets=int(row["wickets"]),
        )
        for row in grouped.values()
    ]
