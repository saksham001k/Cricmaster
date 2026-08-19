import pandas as pd

from cricmaster.models.t20_experimental_features import (
    competition_season_elo,
    shrunk_rate,
)


def test_shrunk_rate_regresses_small_samples() -> None:
    assert shrunk_rate(1, 1) < 1.0
    assert shrunk_rate(1, 1) > 0.5
    assert shrunk_rate(0, 0) == 0.5


def test_competition_season_elo_is_prematch_and_regresses_between_seasons() -> None:
    rows = []

    def add_match(match_id, date, season, winner):
        for team in ("Alpha", "Beta"):
            rows.append(
                {
                    "match_id": match_id,
                    "date": date,
                    "format": "T20",
                    "competition": "League",
                    "season": season,
                    "gender": "male",
                    "team": team,
                    "team_win": 1 if team == winner else 0,
                    "prediction_mode": "PRE_TOSS",
                }
            )

    add_match("m1", "2022-01-01", "2022", "Alpha")
    add_match("m2", "2022-01-02", "2022", "Alpha")
    add_match("m3", "2023-01-01", "2023", "Alpha")

    frame = pd.DataFrame(rows)
    result = competition_season_elo(frame, retention=0.5)

    first = float(result.loc[result["match_id"] == "m1", "competition_season_elo_diff"].iloc[0])
    second = float(result.loc[result["match_id"] == "m2", "competition_season_elo_diff"].iloc[0])
    next_season = float(result.loc[result["match_id"] == "m3", "competition_season_elo_diff"].iloc[0])

    assert first == 0.0
    assert second > 0.0
    assert 0.0 < next_season < second
