"""Chronological historical state. Features must be read before update()."""

from __future__ import annotations

from typing import Any

from cricmaster.data.models import MatchState
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.batting import extract_batting_innings, team_xi
from cricmaster.features.bowling import extract_bowling_spells
from cricmaster.features.elo import FormatElo
from cricmaster.features.head_to_head import HeadToHeadBook
from cricmaster.features.player_form import PlayerFormBook
from cricmaster.features.team_form import TeamFormBook
from cricmaster.features.venue import VenueBook


class HistoricalState:
    """Knowledge available strictly before the next match is applied."""

    def __init__(self, *, elo_k: float = 20.0, elo_initial: float = 1500.0) -> None:
        self.form = TeamFormBook()
        self.h2h = HeadToHeadBook()
        self.venue = VenueBook()
        self.elo = FormatElo(initial=elo_initial, k_factor=elo_k)
        self.players = PlayerFormBook()

    def features_for(self, match: MatchState, team: str, opponent: str) -> dict[str, Any]:
        meta = match.metadata
        match_format = meta.format
        gender = meta.gender
        snapshot: dict[str, Any] = {}
        snapshot.update(self.form.snapshot(match_format, team, gender=gender))
        snapshot.update(self.h2h.snapshot(match_format, team, opponent, gender=gender))
        snapshot.update(self.venue.snapshot(match_format, meta.venue, team, opponent, gender=gender))
        snapshot.update(self.elo.snapshot(match_format, team, opponent, gender=gender))
        snapshot.update(self.players.lineup_snapshot(match_format, team_xi(match, team)))
        return snapshot

    def update(self, match: MatchState) -> None:
        meta = match.metadata
        gender = meta.gender
        team1 = canonicalize_team(meta.team1)
        team2 = canonicalize_team(meta.team2)
        winner = canonicalize_team(meta.winner) if meta.winner else ""
        batting_first = None
        first_innings_runs = None
        if match.innings_history:
            batting_first = canonicalize_team(match.innings_history[0].batting_team)
            first_innings_runs = match.innings_history[0].runs

        decisive = winner in {team1, team2}
        draw_or_tie = (meta.result_type or "").lower() in {"draw", "tie"}
        if decisive or draw_or_tie:
            self.form.update(meta.format, team1, won=winner == team1, gender=gender)
            self.form.update(meta.format, team2, won=winner == team2, gender=gender)
            if decisive:
                loser = team2 if winner == team1 else team1
                self.h2h.update(meta.format, winner, loser, gender=gender)
                self.elo.update(meta.format, winner, loser, team_score=1.0, gender=gender)
            else:
                self.elo.update(meta.format, team1, team2, team_score=0.5, gender=gender)
            self.venue.update(
                meta.format,
                meta.venue,
                team1=team1,
                team2=team2,
                winner=winner or None,
                batting_first=batting_first,
                first_innings_runs=first_innings_runs,
                gender=gender,
            )
        elif first_innings_runs is not None:
            self.venue.update(
                meta.format,
                meta.venue,
                team1=team1,
                team2=team2,
                winner=None,
                batting_first=batting_first,
                first_innings_runs=first_innings_runs,
                gender=gender,
            )

        self.players.update_batting(meta.format, extract_batting_innings(match))
        self.players.update_bowling(meta.format, extract_bowling_spells(match))
