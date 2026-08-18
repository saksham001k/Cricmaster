"""Recent and long-term team form from already-observed results only."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from cricmaster.data.formats import MatchFormat
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.utils import FORM_WINDOWS, recent_window_stats, win_rate

MAX_RECENT = max(FORM_WINDOWS)


class TeamFormBook:
    def __init__(self) -> None:
        self._results: dict[tuple[str, str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=MAX_RECENT)
        )
        self._matches: dict[tuple[str, str, str], int] = defaultdict(int)
        self._wins: dict[tuple[str, str, str], int] = defaultdict(int)

    def _key(
        self, match_format: MatchFormat | str, team: str, gender: str | None = None
    ) -> tuple[str, str, str]:
        return (str(match_format), gender or "", canonicalize_team(team))

    def snapshot(
        self,
        match_format: MatchFormat | str,
        team: str,
        gender: str | None = None,
    ) -> dict[str, Any]:
        key = self._key(match_format, team, gender)
        results = list(self._results[key])
        matches = self._matches[key]
        wins = self._wins[key]
        features: dict[str, Any] = {
            "matches_before": matches,
            "wins_before": wins,
            "win_rate_before": win_rate(wins, matches),
        }
        for window in FORM_WINDOWS:
            stats = recent_window_stats(results, window)
            features[f"matches_last_{window}"] = stats["played"]
            features[f"wins_last_{window}"] = stats["wins"]
            features[f"win_rate_last_{window}"] = stats["win_rate"]
        return features

    def update(
        self,
        match_format: MatchFormat | str,
        team: str,
        *,
        won: bool,
        gender: str | None = None,
    ) -> None:
        key = self._key(match_format, team, gender)
        self._matches[key] += 1
        if won:
            self._wins[key] += 1
        self._results[key].append(1 if won else 0)
