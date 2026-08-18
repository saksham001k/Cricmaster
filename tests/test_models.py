from datetime import date

from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import Delivery, InningsState, MatchMetadata, MatchState


def test_match_metadata_creation() -> None:
    metadata = MatchMetadata(
        match_id="m1",
        format=MatchFormat.T20,
        competition="IPL",
        season="2024",
        match_number=1,
        date=date(2024, 4, 1),
        venue="Wankhede Stadium",
        city="Mumbai",
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        toss_winner="Mumbai Indians",
        toss_decision="field",
        winner=None,
        result_type=None,
        player_of_match=None,
        source="unit-test",
    )
    assert metadata.format is MatchFormat.T20
    assert metadata.competition == "IPL"
    assert metadata.winner is None


def test_innings_optional_chase_fields() -> None:
    innings = InningsState(
        batting_team="Mumbai Indians",
        bowling_team="Chennai Super Kings",
        innings_number=2,
        runs=10,
        wickets=0,
        overs=1.3,
        balls=9,
        target=180,
        required_runs=170,
        balls_remaining=111,
        current_run_rate=6.67,
        required_run_rate=9.19,
    )
    assert innings.target == 180
    assert innings.required_run_rate == 9.19


def test_match_state_combines_components() -> None:
    metadata = MatchMetadata(
        match_id="m1",
        format=MatchFormat.ODI,
        team1="India",
        team2="Australia",
        source="unit-test",
    )
    delivery = Delivery(
        innings=1,
        over=0,
        ball=1,
        batting_team="India",
        striker="V Kohli",
        non_striker="RG Sharma",
        bowler="MA Starc",
        runs_batter=1,
        runs_total=1,
    )
    state = MatchState(
        metadata=metadata,
        innings_history=[],
        deliveries=[delivery],
        source="unit-test",
    )
    assert state.current_innings is None
    assert state.deliveries[0].striker == "V Kohli"
