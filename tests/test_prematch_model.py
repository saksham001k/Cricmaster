import numpy as np
import pandas as pd

from cricmaster.models.prematch import elo_probability, pair_prematch_rows


def _row(team: str, opponent: str, won: int, elo: float, rate: float) -> dict:
    return {
        "match_id": "m1",
        "date": "2024-01-01",
        "format": "T20I",
        "competition": "Example",
        "gender": "male",
        "team": team,
        "opponent": opponent,
        "team_win": won,
        "prediction_mode": "PRE_TOSS",
        "matches_before": 10,
        "win_rate_before": rate,
        "wins_last_5": 3,
        "win_rate_last_5": rate,
        "wins_last_10": 6,
        "win_rate_last_10": rate,
        "wins_last_20": 6,
        "win_rate_last_20": rate,
        "h2h_team_wins": 2 if won else 1,
        "h2h_team_win_rate": rate,
        "h2h_last_5_win_rate": rate,
        "team_matches_at_venue": 4,
        "team_win_rate_at_venue": rate,
        "team_elo_before": elo,
    }


def test_pair_prematch_rows_builds_one_signed_match_row() -> None:
    frame = pd.DataFrame(
        [
            _row("Beta", "Alpha", 0, 1400.0, 0.4),
            _row("Alpha", "Beta", 1, 1600.0, 0.6),
        ]
    )

    paired = pair_prematch_rows(frame)

    assert len(paired) == 1
    assert paired.loc[0, "team_a"] == "Alpha"
    assert paired.loc[0, "team_b"] == "Beta"
    assert paired.loc[0, "team_a_win"] == 1
    assert paired.loc[0, "team_elo_before_diff"] == 200.0
    assert np.isclose(paired.loc[0, "win_rate_before_diff"], 0.2)


def test_elo_probability_is_symmetric() -> None:
    p = float(elo_probability(400.0))
    reverse = float(elo_probability(-400.0))

    assert np.isclose(p, 10.0 / 11.0)
    assert np.isclose(p + reverse, 1.0)
    assert np.isclose(float(elo_probability(0.0)), 0.5)
