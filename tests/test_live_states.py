from datetime import date

from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import InningsState, MatchMetadata, MatchState
from cricmaster.features.live import compute_live_metrics, iter_live_states
from tests.helpers import delivery


def test_chase_metrics_example() -> None:
    metrics = compute_live_metrics(
        runs=150,
        wickets=4,
        legal_balls=90,
        target=181,
        ball_limit=120,
        balls_per_over=6,
    )
    assert metrics["balls_remaining"] == 30
    assert metrics["runs_required"] == 31
    assert metrics["wickets_in_hand"] == 6
    assert metrics["required_run_rate"] == 6.2


def test_first_innings_has_null_target_fields() -> None:
    metrics = compute_live_metrics(
        runs=45,
        wickets=1,
        legal_balls=30,
        target=None,
        ball_limit=120,
        balls_per_over=6,
    )
    assert metrics["target"] is None
    assert metrics["runs_required"] is None
    assert metrics["required_run_rate"] is None
    assert metrics["balls_remaining"] == 90


def test_live_states_handle_wide_noball_boundary_and_wicket() -> None:
    deliveries = [
        delivery(ball=1, batting_team="Mumbai Indians", runs_extras=1, runs_total=1, is_wide=True),
        delivery(ball=1, batting_team="Mumbai Indians", runs_batter=4, runs_total=4),
        delivery(ball=2, batting_team="Mumbai Indians", runs_batter=6, runs_total=6),
        delivery(
            ball=3,
            batting_team="Mumbai Indians",
            wicket=True,
            wicket_type="bowled",
            player_out="Batter A",
        ),
        delivery(
            ball=4,
            batting_team="Mumbai Indians",
            runs_extras=1,
            runs_total=1,
            is_noball=True,
            runs_batter=0,
        ),
    ]
    match = MatchState(
        metadata=MatchMetadata(
            match_id="live1",
            format=MatchFormat.T20,
            competition="IPL",
            date=date(2024, 4, 1),
            venue="Wankhede Stadium",
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            winner="Mumbai Indians",
            result_type="win",
            source="unit-test",
            balls_per_over=6,
            scheduled_overs=20,
        ),
        innings_history=[
            InningsState(
                batting_team="Mumbai Indians",
                bowling_team="Chennai Super Kings",
                innings_number=1,
                runs=12,
                wickets=1,
                balls=2,
            )
        ],
        deliveries=deliveries,
        source="unit-test",
    )
    rows = list(iter_live_states(match))
    assert rows[0]["is_wide"] is True
    assert rows[0]["legal_balls_bowled"] == 0
    assert rows[0]["runs"] == 1
    assert rows[1]["runs"] == 5
    assert rows[1]["legal_balls_bowled"] == 1
    assert rows[2]["runs"] == 11
    assert rows[3]["is_wicket"] is True
    assert rows[3]["wickets"] == 1
    assert rows[3]["wickets_in_hand"] == 9
    assert rows[4]["is_noball"] is True
    assert rows[4]["legal_balls_bowled"] == 3
    assert rows[4]["target"] is None
    assert rows[-1]["batting_team_eventual_win"] == 1


def test_test_matches_do_not_emit_live_states() -> None:
    match = MatchState(
        metadata=MatchMetadata(
            match_id="test1",
            format=MatchFormat.TEST,
            team1="India",
            team2="Australia",
            winner="India",
            result_type="win",
            source="unit-test",
            date=date(2024, 1, 1),
        ),
        deliveries=[delivery()],
        source="unit-test",
    )
    assert list(iter_live_states(match)) == []
