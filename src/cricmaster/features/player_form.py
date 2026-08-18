"""Conservative prior-match player form. Career totals are never used."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from cricmaster.data.formats import MatchFormat
from cricmaster.features.batting import BattingInnings
from cricmaster.features.bowling import BowlingSpell
from cricmaster.features.utils import rate

RECENT_PLAYER = 5


class PlayerFormBook:
    def __init__(self) -> None:
        self._bat_innings: dict[tuple[str, str], int] = defaultdict(int)
        self._bat_runs: dict[tuple[str, str], int] = defaultdict(int)
        self._bat_balls: dict[tuple[str, str], int] = defaultdict(int)
        self._bat_outs: dict[tuple[str, str], int] = defaultdict(int)
        self._recent_runs: dict[tuple[str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=RECENT_PLAYER)
        )
        self._recent_bat_balls: dict[tuple[str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=RECENT_PLAYER)
        )
        self._bowl_balls: dict[tuple[str, str], int] = defaultdict(int)
        self._bowl_runs: dict[tuple[str, str], int] = defaultdict(int)
        self._bowl_wickets: dict[tuple[str, str], int] = defaultdict(int)
        self._recent_wickets: dict[tuple[str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=RECENT_PLAYER)
        )
        self._recent_bowl_runs: dict[tuple[str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=RECENT_PLAYER)
        )
        self._recent_bowl_balls: dict[tuple[str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=RECENT_PLAYER)
        )

    def _key(self, match_format: MatchFormat | str, player: str) -> tuple[str, str]:
        return (str(match_format), player)

    def batter_snapshot(self, match_format: MatchFormat | str, player: str) -> dict[str, Any]:
        key = self._key(match_format, player)
        innings = self._bat_innings[key]
        runs = self._bat_runs[key]
        balls = self._bat_balls[key]
        outs = self._bat_outs[key]
        recent_runs = list(self._recent_runs[key])
        recent_balls = list(self._recent_bat_balls[key])
        return {
            "innings": innings,
            "runs": runs,
            "balls": balls,
            "average": rate(runs, outs),
            "strike_rate": (rate(runs, balls) * 100) if balls else None,
            "recent_runs": sum(recent_runs) if recent_runs else None,
            "recent_strike_rate": (
                rate(sum(recent_runs), sum(recent_balls)) * 100
                if recent_balls and sum(recent_balls) > 0
                else None
            ),
        }

    def bowler_snapshot(self, match_format: MatchFormat | str, player: str) -> dict[str, Any]:
        key = self._key(match_format, player)
        balls = self._bowl_balls[key]
        runs = self._bowl_runs[key]
        wickets = self._bowl_wickets[key]
        recent_wickets = list(self._recent_wickets[key])
        recent_runs = list(self._recent_bowl_runs[key])
        recent_balls = list(self._recent_bowl_balls[key])
        overs = balls / 6 if balls else 0
        return {
            "balls": balls,
            "runs_conceded": runs,
            "wickets": wickets,
            "economy": rate(runs, overs) if overs else None,
            "recent_wickets": sum(recent_wickets) if recent_wickets else None,
            "recent_economy": (
                rate(sum(recent_runs), (sum(recent_balls) / 6))
                if recent_balls and sum(recent_balls) > 0
                else None
            ),
        }

    def lineup_snapshot(
        self, match_format: MatchFormat | str, players: list[str] | None
    ) -> dict[str, Any]:
        if not players:
            return {
                "lineup_status": "LINEUP_UNKNOWN",
                "xi_batters_with_history": None,
                "xi_mean_batting_average": None,
                "xi_mean_recent_runs": None,
                "xi_bowlers_with_history": None,
                "xi_mean_bowling_economy": None,
                "xi_mean_recent_wickets": None,
            }
        batting = [self.batter_snapshot(match_format, player) for player in players]
        bowling = [self.bowler_snapshot(match_format, player) for player in players]
        bat_avgs = [row["average"] for row in batting if row["average"] is not None]
        recent_runs = [row["recent_runs"] for row in batting if row["recent_runs"] is not None]
        economies = [row["economy"] for row in bowling if row["economy"] is not None]
        recent_wickets = [row["recent_wickets"] for row in bowling if row["recent_wickets"] is not None]
        return {
            "lineup_status": "LINEUP_KNOWN",
            "xi_batters_with_history": len(bat_avgs),
            "xi_mean_batting_average": (sum(bat_avgs) / len(bat_avgs)) if bat_avgs else None,
            "xi_mean_recent_runs": (sum(recent_runs) / len(recent_runs)) if recent_runs else None,
            "xi_bowlers_with_history": len(economies),
            "xi_mean_bowling_economy": (sum(economies) / len(economies)) if economies else None,
            "xi_mean_recent_wickets": (
                sum(recent_wickets) / len(recent_wickets) if recent_wickets else None
            ),
        }

    def update_batting(
        self, match_format: MatchFormat | str, innings: list[BattingInnings]
    ) -> None:
        for item in innings:
            key = self._key(match_format, item.player)
            self._bat_innings[key] += 1
            self._bat_runs[key] += item.runs
            self._bat_balls[key] += item.balls
            if item.out:
                self._bat_outs[key] += 1
            self._recent_runs[key].append(item.runs)
            self._recent_bat_balls[key].append(item.balls)

    def update_bowling(
        self, match_format: MatchFormat | str, spells: list[BowlingSpell]
    ) -> None:
        for item in spells:
            key = self._key(match_format, item.player)
            self._bowl_balls[key] += item.balls
            self._bowl_runs[key] += item.runs
            self._bowl_wickets[key] += item.wickets
            self._recent_wickets[key].append(item.wickets)
            self._recent_bowl_runs[key].append(item.runs)
            self._recent_bowl_balls[key].append(item.balls)
