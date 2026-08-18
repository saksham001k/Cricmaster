"""Format-specific head-to-head history."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from cricmaster.data.formats import MatchFormat
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.utils import rate, recent_window_stats

RECENT_H2H = 5


class HeadToHeadBook:
    def __init__(self) -> None:
        self._matches: dict[tuple[str, str, str, str], int] = defaultdict(int)
        self._wins: dict[tuple[str, str, str, str], int] = defaultdict(int)
        self._recent: dict[tuple[str, str, str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=RECENT_H2H)
        )

    def _key(
        self,
        match_format: MatchFormat | str,
        team: str,
        opponent: str,
        gender: str | None = None,
    ) -> tuple[str, str, str, str]:
        return (str(match_format), gender or "", canonicalize_team(team), canonicalize_team(opponent))

    def snapshot(
        self,
        match_format: MatchFormat | str,
        team: str,
        opponent: str,
        gender: str | None = None,
    ) -> dict[str, Any]:
        team_key = self._key(match_format, team, opponent, gender)
        opp_key = self._key(match_format, opponent, team, gender)
        matches = self._matches[team_key] + self._matches[opp_key]
        team_wins = self._wins[team_key]
        opponent_wins = self._wins[opp_key]
        recent = recent_window_stats(list(self._recent[team_key]), RECENT_H2H)
        return {
            "h2h_matches_before": matches,
            "h2h_team_wins": team_wins,
            "h2h_opponent_wins": opponent_wins,
            "h2h_team_win_rate": rate(team_wins, matches),
            "h2h_last_5_matches": recent["played"],
            "h2h_last_5_win_rate": recent["win_rate"],
        }

    def update(
        self,
        match_format: MatchFormat | str,
        winner: str,
        loser: str,
        gender: str | None = None,
    ) -> None:
        key = self._key(match_format, winner, loser, gender)
        self._matches[key] += 1
        self._wins[key] += 1
        self._recent[key].append(1)
        self._recent[self._key(match_format, loser, winner, gender)].append(0)
