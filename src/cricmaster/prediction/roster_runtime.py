"""Runtime previous-XI features for the frozen T20 roster family.

PRE_TOSS must never use the current playing XI. Only the last previously known
XI for each team is used, reconstructed from matches strictly before the
prediction date.

If that history is missing, features are left as NaN (later zero-imputed by the
trained pipeline) and a warning is emitted. Players are never invented.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import MatchState
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.batting import extract_batting_innings, team_xi
from cricmaster.features.bowling import extract_bowling_spells
from cricmaster.features.player_form import PlayerFormBook
from cricmaster.models.roster_features import CORE_STRENGTH_FIELDS, PREVIOUS_CORE_DIFFS, _lineup_strength


class RosterRuntime:
    """Leakage-safe previous-XI book matching training semantics."""

    def __init__(self) -> None:
        self.players = PlayerFormBook()
        self._xi_history: dict[tuple[str, str, str], deque[list[str]]] = defaultdict(
            lambda: deque(maxlen=2)
        )

    def _key(self, match_format: MatchFormat | str, gender: str, team: str) -> tuple[str, str, str]:
        return (str(match_format), gender or "", canonicalize_team(team))

    def observe_completed_match(self, match: MatchState) -> None:
        """Update after a historical match. Training only used T20 matches."""

        if match.metadata.format is not MatchFormat.T20:
            return

        meta = match.metadata
        gender = meta.gender or ""
        self.players.update_batting(meta.format, extract_batting_innings(match))
        self.players.update_bowling(meta.format, extract_bowling_spells(match))
        for raw_team in (meta.team1, meta.team2):
            current_xi = team_xi(match, raw_team)
            if not current_xi:
                continue
            key = self._key(meta.format, gender, raw_team)
            self._xi_history[key].append(list(current_xi))

    def previous_xi(
        self,
        match_format: MatchFormat | str,
        gender: str,
        team: str,
    ) -> list[str] | None:
        history = self._xi_history[self._key(match_format, gender, team)]
        if not history:
            return None
        return list(history[-1])

    def previous_core_strength(
        self,
        match_format: MatchFormat | str,
        gender: str,
        team: str,
    ) -> dict[str, float]:
        lineup = self.previous_xi(match_format, gender, team)
        strength = _lineup_strength(self.players, match_format, lineup)
        return {f"previous_xi_{name}": strength[name] for name in CORE_STRENGTH_FIELDS}


def _finite_diff(left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(left) - float(right)


def previous_xi_core_differences(
    roster: RosterRuntime,
    *,
    match_format: MatchFormat,
    gender: str,
    team1: str,
    team2: str,
) -> tuple[dict[str, float], bool, bool]:
    """Team1 minus Team2 previous-XI core diffs. Ignores any current XI."""

    left_xi = roster.previous_xi(match_format, gender, team1)
    right_xi = roster.previous_xi(match_format, gender, team2)
    left = roster.previous_core_strength(match_format, gender, team1)
    right = roster.previous_core_strength(match_format, gender, team2)

    row: dict[str, float] = {}
    for name in CORE_STRENGTH_FIELDS:
        row[f"previous_xi_{name}_diff"] = _finite_diff(
            left[f"previous_xi_{name}"],
            right[f"previous_xi_{name}"],
        )
    return row, left_xi is not None, right_xi is not None


def append_previous_xi_core_features(
    frame: pd.DataFrame,
    roster_diffs: dict[str, float],
) -> pd.DataFrame:
    result = frame.copy()
    for name in PREVIOUS_CORE_DIFFS:
        result[name] = roster_diffs.get(name, float("nan"))
    return result


def features_for_bundle(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Select artifact features in trained order. Missing values stay NaN."""

    trained = list(bundle["features"])
    result = frame.copy()
    for name in trained:
        if name not in result.columns:
            result[name] = float("nan")
    return result.loc[:, trained]
