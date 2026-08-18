"""Synthetic matches for unit tests."""

from __future__ import annotations

from datetime import date

from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import Delivery, InningsState, MatchMetadata, MatchState


def delivery(
    *,
    innings: int = 1,
    over: int = 0,
    ball: int = 1,
    batting_team: str = "India",
    striker: str = "Batter A",
    non_striker: str = "Batter B",
    bowler: str = "Bowler C",
    runs_batter: int = 0,
    runs_extras: int = 0,
    runs_total: int | None = None,
    wicket: bool = False,
    wicket_type: str | None = None,
    player_out: str | None = None,
    is_wide: bool = False,
    is_noball: bool = False,
) -> Delivery:
    total = runs_batter + runs_extras if runs_total is None else runs_total
    return Delivery(
        innings=innings,
        over=over,
        ball=ball,
        batting_team=batting_team,
        striker=striker,
        non_striker=non_striker,
        bowler=bowler,
        runs_batter=runs_batter,
        runs_extras=runs_extras,
        runs_total=total,
        wicket=wicket,
        wicket_type=wicket_type,
        player_out=player_out,
        is_wide=is_wide,
        is_noball=is_noball,
    )


def completed_t20(
    match_id: str,
    match_date: date,
    team1: str,
    team2: str,
    winner: str,
    *,
    venue: str = "MCG",
    deliveries: list[Delivery] | None = None,
    team1_players: list[str] | None = None,
    team2_players: list[str] | None = None,
    result_type: str = "win",
) -> MatchState:
    innings = [
        InningsState(
            batting_team=team1,
            bowling_team=team2,
            innings_number=1,
            runs=160,
            wickets=5,
            balls=120,
            overs=20.0,
        ),
        InningsState(
            batting_team=team2,
            bowling_team=team1,
            innings_number=2,
            runs=140,
            wickets=10,
            balls=115,
            overs=19.1,
            target=161,
            target_overs=20,
        ),
    ]
    return MatchState(
        metadata=MatchMetadata(
            match_id=match_id,
            format=MatchFormat.T20I,
            competition="Example Series",
            season="2024",
            date=match_date,
            venue=venue,
            city="Melbourne",
            team1=team1,
            team2=team2,
            toss_winner=team1,
            toss_decision="bat",
            winner=winner if result_type == "win" else None,
            result_type=result_type,
            source="unit-test",
            balls_per_over=6,
            scheduled_overs=20,
            team1_players=team1_players,
            team2_players=team2_players,
        ),
        innings_history=innings,
        deliveries=deliveries or [],
        source="unit-test",
    )
