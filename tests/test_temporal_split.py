from datetime import date

import pandas as pd

from cricmaster.features.split import temporal_split


def test_temporal_split_keeps_match_rows_together() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2024-01-01", "team": "India", "team_win": 1},
            {"match_id": "m1", "date": "2024-01-01", "team": "Australia", "team_win": 0},
            {"match_id": "m2", "date": "2024-06-01", "team": "India", "team_win": 0},
            {"match_id": "m2", "date": "2024-06-01", "team": "Australia", "team_win": 1},
            {"match_id": "m3", "date": "2024-12-01", "team": "India", "team_win": 1},
            {"match_id": "m3", "date": "2024-12-01", "team": "Australia", "team_win": 0},
        ]
    )
    train, valid, test = temporal_split(
        frame,
        train_end=date(2024, 3, 1),
        valid_end=date(2024, 9, 1),
    )
    assert set(train["match_id"]) == {"m1"}
    assert set(valid["match_id"]) == {"m2"}
    assert set(test["match_id"]) == {"m3"}
    assert len(train) == 2
    assert len(valid) == 2
    assert len(test) == 2


def test_no_random_split_of_same_match() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2024-01-01", "team": "A"},
            {"match_id": "m1", "date": "2024-01-01", "team": "B"},
        ]
    )
    train, valid, test = temporal_split(frame, train_end="2024-01-01")
    assert valid.empty
    assert test.empty
    assert set(train["team"]) == {"A", "B"}
