import numpy as np
import pandas as pd

from cricmaster.models.live import (
    equal_match_weights,
    prepare_live_training_frame,
)


def test_equal_match_weights_equalize_total_match_weight() -> None:
    frame = pd.DataFrame(
        {
            "match_id": ["a", "a", "a", "b"],
        }
    )
    weights = equal_match_weights(frame)

    a_total = weights[:3].sum()
    b_total = weights[3:].sum()
    assert np.isclose(a_total, b_total)


def test_prepare_live_training_keeps_legal_binary_states_only() -> None:
    live = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-01-01",
                "format": "T20I",
                "innings_number": 1,
                "batting_team": "Alpha",
                "runs": 1,
                "wickets": 0,
                "legal_balls_bowled": 1,
                "balls_remaining": 119,
                "current_run_rate": 6.0,
                "target": None,
                "runs_required": None,
                "required_run_rate": None,
                "run_rate_difference": None,
                "wickets_in_hand": 10,
                "is_legal": True,
                "batting_team_eventual_win": 1,
            },
            {
                "match_id": "m1",
                "date": "2024-01-01",
                "format": "T20I",
                "innings_number": 1,
                "batting_team": "Alpha",
                "runs": 2,
                "wickets": 0,
                "legal_balls_bowled": 1,
                "balls_remaining": 119,
                "current_run_rate": 12.0,
                "target": None,
                "runs_required": None,
                "required_run_rate": None,
                "run_rate_difference": None,
                "wickets_in_hand": 10,
                "is_legal": False,
                "batting_team_eventual_win": 1,
            },
        ]
    )

    prematch = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team": "Alpha",
                "prediction_mode": "POST_TOSS",
                "gender": "male",
                "toss_decision": "bat",
                "team_won_toss": True,
                "matches_before": 10,
                "win_rate_before": 0.6,
                "win_rate_last_5": 0.6,
                "win_rate_last_10": 0.6,
                "win_rate_last_20": 0.6,
                "h2h_team_win_rate": 0.5,
                "team_win_rate_at_venue": 0.5,
                "elo_difference": 100.0,
                "xi_batters_with_history": None,
                "xi_mean_batting_average": None,
                "xi_mean_recent_runs": None,
                "xi_bowlers_with_history": None,
                "xi_mean_bowling_economy": None,
                "xi_mean_recent_wickets": None,
            }
        ]
    )

    prepared = prepare_live_training_frame(live, prematch)

    assert len(prepared) == 1
    assert prepared.loc[0, "legal_balls_bowled"] == 1
    assert prepared.loc[0, "toss_bat_signed"] == 1.0
    assert prepared.loc[0, "is_t20i"] == 1.0
    assert prepared.loc[0, "is_male"] == 1.0
