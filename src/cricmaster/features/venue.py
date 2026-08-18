"""Venue records using only matches completed before time T."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from cricmaster.data.formats import MatchFormat
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.utils import rate


class VenueBook:
    def __init__(self) -> None:
        self._team: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        self._bat_first_wins: dict[tuple[str, str, str], int] = defaultdict(int)
        self._chase_wins: dict[tuple[str, str, str], int] = defaultdict(int)
        self._decided: dict[tuple[str, str, str], int] = defaultdict(int)
        self._first_innings_totals: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    def _venue_key(
        self,
        match_format: MatchFormat | str,
        venue: str | None,
        gender: str | None = None,
    ) -> tuple[str, str, str] | None:
        if not venue:
            return None
        return (str(match_format), gender or "", venue.strip())

    def snapshot(
        self,
        match_format: MatchFormat | str,
        venue: str | None,
        team: str,
        opponent: str,
        gender: str | None = None,
    ) -> dict[str, Any]:
        venue_key = self._venue_key(match_format, venue, gender)
        empty = {
            "team_matches_at_venue": 0,
            "team_wins_at_venue": 0,
            "team_win_rate_at_venue": None,
            "opponent_matches_at_venue": 0,
            "opponent_wins_at_venue": 0,
            "opponent_win_rate_at_venue": None,
            "venue_batting_first_win_rate": None,
            "venue_chasing_win_rate": None,
            "venue_decided_matches": 0,
            "historical_first_innings_average": None,
            "historical_first_innings_matches": 0,
        }
        if venue_key is None:
            return empty
        team_results = self._team[(venue_key[0], venue_key[1], venue_key[2], canonicalize_team(team))]
        opp_results = self._team[(venue_key[0], venue_key[1], venue_key[2], canonicalize_team(opponent))]
        team_wins = int(sum(team_results))
        opp_wins = int(sum(opp_results))
        decided = self._decided[venue_key]
        first_totals = self._first_innings_totals[venue_key]
        return {
            "team_matches_at_venue": len(team_results),
            "team_wins_at_venue": team_wins,
            "team_win_rate_at_venue": rate(team_wins, len(team_results)),
            "opponent_matches_at_venue": len(opp_results),
            "opponent_wins_at_venue": opp_wins,
            "opponent_win_rate_at_venue": rate(opp_wins, len(opp_results)),
            "venue_batting_first_win_rate": rate(self._bat_first_wins[venue_key], decided),
            "venue_chasing_win_rate": rate(self._chase_wins[venue_key], decided),
            "venue_decided_matches": decided,
            "historical_first_innings_average": (
                sum(first_totals) / len(first_totals) if first_totals else None
            ),
            "historical_first_innings_matches": len(first_totals),
        }

    def update(
        self,
        match_format: MatchFormat | str,
        venue: str | None,
        *,
        team1: str,
        team2: str,
        winner: str | None,
        batting_first: str | None,
        first_innings_runs: int | None,
        gender: str | None = None,
    ) -> None:
        venue_key = self._venue_key(match_format, venue, gender)
        if venue_key is None:
            return
        if first_innings_runs is not None:
            self._first_innings_totals[venue_key].append(first_innings_runs)
        if not winner:
            return
        for team in (team1, team2):
            self._team[(venue_key[0], venue_key[1], venue_key[2], canonicalize_team(team))].append(
                1 if canonicalize_team(team) == canonicalize_team(winner) else 0
            )
        self._decided[venue_key] += 1
        if batting_first:
            if canonicalize_team(winner) == canonicalize_team(batting_first):
                self._bat_first_wins[venue_key] += 1
            else:
                self._chase_wins[venue_key] += 1
