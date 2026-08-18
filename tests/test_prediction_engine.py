from datetime import date

import numpy as np
import pandas as pd
import pytest

from cricmaster.data.formats import MatchFormat
from cricmaster.prediction.prematch import (
    PredictionRequest,
    difference_features,
    parse_match_format,
    prediction_edge,
    symmetric_probability,
)


def test_difference_features_uses_team1_minus_team2() -> None:
    left = {
        "matches_before": 20,
        "win_rate_before": 0.70,
        "team_elo_before": 1600,
    }
    right = {
        "matches_before": 10,
        "win_rate_before": 0.50,
        "team_elo_before": 1500,
    }

    frame = difference_features(left, right)

    assert frame.loc[0, "matches_before_diff"] == 10
    assert np.isclose(frame.loc[0, "win_rate_before_diff"], 0.20)
    assert frame.loc[0, "team_elo_before_diff"] == 100


def test_elo_runtime_probability_is_complementary() -> None:
    features = pd.DataFrame(
        [
            {
                "team_elo_before_diff": 400.0,
            }
        ]
    )
    bundle = {
        "model_name": "elo_baseline",
        "features": ["team_elo_before_diff"],
        "prediction_mode": "PRE_TOSS",
    }

    probability = symmetric_probability(bundle, features)

    assert np.isclose(probability, 10.0 / 11.0)
    assert np.isclose(probability + (1.0 - probability), 1.0)


def test_format_guard_rejects_untrained_formats() -> None:
    assert parse_match_format("T20I") is MatchFormat.T20I
    assert parse_match_format("T20") is MatchFormat.T20

    with pytest.raises(ValueError):
        parse_match_format("ODI")


def test_prediction_edge_labels() -> None:
    assert prediction_edge(0.52) == "very close"
    assert prediction_edge(0.58) == "slight"
    assert prediction_edge(0.67) == "moderate"
    assert prediction_edge(0.80) == "strong"
