from datetime import date

import pytest

from cricmaster.data.formats import MatchFormat
from cricmaster.prediction.live import (
    LivePredictionRequest,
    _terminal_chase_probability,
    parse_cricket_overs,
)


def test_parse_cricket_overs() -> None:
    assert parse_cricket_overs("15.3") == 93
    assert parse_cricket_overs("20") == 120
    assert parse_cricket_overs("0.5") == 5


def test_parse_cricket_overs_rejects_invalid_ball_component() -> None:
    with pytest.raises(ValueError):
        parse_cricket_overs("15.6")


def _request(*, runs: int, wickets: int, target: int) -> LivePredictionRequest:
    return LivePredictionRequest(
        team1="Alpha",
        team2="Beta",
        batting_team="Alpha",
        match_format=MatchFormat.T20I,
        match_date=date(2026, 1, 1),
        gender="male",
        innings_number=2,
        runs=runs,
        wickets=wickets,
        legal_balls=100,
        target=target,
    )


def test_terminal_chase_target_reached() -> None:
    probability, reason = _terminal_chase_probability(
        _request(runs=181, wickets=4, target=181),
        {"balls_remaining": 20},
    )
    assert probability == 1.0
    assert reason == "target reached"


def test_terminal_chase_all_out() -> None:
    probability, reason = _terminal_chase_probability(
        _request(runs=170, wickets=10, target=181),
        {"balls_remaining": 20},
    )
    assert probability == 0.0
    assert reason == "all out before target"

def test_live_model_input_sanitizes_missing_values() -> None:
    import numpy as np
    import pandas as pd

    from cricmaster.prediction.live import _sanitize_model_input

    frame = pd.DataFrame(
        {
            "a": [np.nan],
            "b": [None],
            "c": [np.inf],
            "d": [-np.inf],
            "e": [5.5],
        }
    )

    cleaned = _sanitize_model_input(
        frame,
        ["a", "b", "c", "d", "e"],
    )

    assert cleaned.loc[0, "a"] == 0.0
    assert cleaned.loc[0, "b"] == 0.0
    assert cleaned.loc[0, "c"] == 0.0
    assert cleaned.loc[0, "d"] == 0.0
    assert cleaned.loc[0, "e"] == 5.5
    assert np.isfinite(cleaned.to_numpy()).all()
