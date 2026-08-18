import numpy as np
import pandas as pd

from cricmaster.models.posttoss import pair_post_toss_rows


BASE = {
    "match_id": "m1",
    "date": "2024-01-01",
    "format": "T20I",
    "competition": "Example",
    "gender": "male",
    "prediction_mode": "POST_TOSS",
    "matches_before": 10,
    "win_rate_before": 0.5,
    "wins_last_5": 3,
    "win_rate_last_5": 0.6,
    "wins_last_10": 6,
    "win_rate_last_10": 0.6,
    "wins_last_20": 12,
    "win_rate_last_20": 0.6,
    "h2h_team_wins": 2,
    "h2h_team_win_rate": 0.5,
    "h2h_last_5_win_rate": 0.5,
    "team_matches_at_venue": 4,
    "team_win_rate_at_venue": 0.5,
    "team_elo_before": 1500,
    "xi_batters_with_history": 6,
    "xi_mean_batting_average": 25.0,
    "xi_mean_recent_runs": 22.0,
    "xi_bowlers_with_history": 5,
    "xi_mean_bowling_economy": 7.5,
    "xi_mean_recent_wickets": 1.2,
}


def _row(team: str, opponent: str, won: int, toss: bool) -> dict:
    row = dict(BASE)
    row.update(
        {
            "team": team,
            "opponent": opponent,
            "team_win": won,
            "toss_winner": "Alpha",
            "toss_decision": "field",
            "team_won_toss": toss,
        }
    )
    return row


def test_post_toss_pair_has_antisymmetric_toss_feature() -> None:
    frame = pd.DataFrame(
        [
            _row("Alpha", "Beta", 1, True),
            _row("Beta", "Alpha", 0, False),
        ]
    )
    paired = pair_post_toss_rows(frame)

    assert len(paired) == 1
    assert paired.loc[0, "team_a"] == "Alpha"
    assert paired.loc[0, "toss_bat_advantage"] == 0.0
    assert paired.loc[0, "toss_field_advantage"] == 1.0


def test_post_toss_pair_xi_difference() -> None:
    left = _row("Alpha", "Beta", 1, True)
    right = _row("Beta", "Alpha", 0, False)
    left["xi_mean_recent_runs"] = 30.0
    right["xi_mean_recent_runs"] = 20.0

    paired = pair_post_toss_rows(pd.DataFrame([left, right]))

    assert np.isclose(paired.loc[0, "xi_mean_recent_runs_diff"], 10.0)
