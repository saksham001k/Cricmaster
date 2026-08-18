from datetime import date

from cricmaster.features.history import HistoricalState
from cricmaster.features.prematch import build_prematch_rows
from cricmaster.features.toss import PredictionMode
from tests.helpers import completed_t20, delivery


def test_second_match_cannot_see_its_own_or_future_results() -> None:
    first = completed_t20("m1", date(2024, 6, 1), "India", "Australia", "India")
    second = completed_t20("m2", date(2024, 6, 8), "India", "Australia", "Australia")
    state = HistoricalState()

    rows1, reason1 = build_prematch_rows(first, state, modes=[PredictionMode.PRE_TOSS])
    assert reason1 is None
    india1 = next(row for row in rows1 if row["team"] == "India")
    assert india1["matches_before"] == 0
    assert india1["wins_before"] == 0
    assert india1["h2h_matches_before"] == 0
    assert india1["team_elo_before"] == 1500
    assert india1["team_matches_at_venue"] == 0
    assert india1["team_win"] == 1
    state.update(first)

    rows2, reason2 = build_prematch_rows(second, state, modes=[PredictionMode.PRE_TOSS])
    assert reason2 is None
    india2 = next(row for row in rows2 if row["team"] == "India")
    assert india2["matches_before"] == 1
    assert india2["wins_before"] == 1
    assert india2["wins_last_5"] == 1
    assert india2["h2h_matches_before"] == 1
    assert india2["h2h_team_wins"] == 1
    assert india2["team_elo_before"] > 1500
    assert india2["team_matches_at_venue"] == 1
    assert india2["team_wins_at_venue"] == 1
    assert india2["team_win"] == 0
    # Future match m2 result must not appear in its own features.
    assert india2["wins_before"] == 1


def test_player_stats_exclude_future_and_current_match() -> None:
    match1 = completed_t20(
        "p1",
        date(2024, 1, 1),
        "India",
        "Australia",
        "India",
        team1_players=["Batter A"],
        deliveries=[
            delivery(runs_batter=50, runs_total=50, striker="Batter A"),
            delivery(over=0, ball=2, runs_batter=4, runs_total=4, striker="Batter A"),
        ],
    )
    match2 = completed_t20(
        "p2",
        date(2024, 1, 8),
        "India",
        "Australia",
        "India",
        team1_players=["Batter A"],
        deliveries=[delivery(runs_batter=80, runs_total=80, striker="Batter A")],
    )
    state = HistoricalState()
    rows1, _ = build_prematch_rows(match1, state, modes=[PredictionMode.PRE_TOSS])
    india1 = next(row for row in rows1 if row["team"] == "India")
    assert india1["lineup_status"] == "LINEUP_KNOWN"
    assert india1["xi_mean_recent_runs"] is None
    state.update(match1)
    rows2, _ = build_prematch_rows(match2, state, modes=[PredictionMode.PRE_TOSS])
    india2 = next(row for row in rows2 if row["team"] == "India")
    assert india2["xi_mean_recent_runs"] == 54
    state.update(match2)
    # Updating match 2 must not rewrite match 1 features already emitted.
    assert india1["xi_mean_recent_runs"] is None


def test_pre_toss_rows_do_not_include_toss_information() -> None:
    match = completed_t20("toss1", date(2024, 2, 1), "India", "Australia", "India")
    rows, _ = build_prematch_rows(match, HistoricalState())
    pre = next(row for row in rows if row["prediction_mode"] == "PRE_TOSS")
    post = next(row for row in rows if row["prediction_mode"] == "POST_TOSS" and row["team"] == "India")
    assert pre["toss_winner"] is None
    assert pre["team_won_toss"] is None
    assert post["toss_winner"] == "India"
    assert post["team_won_toss"] is True
