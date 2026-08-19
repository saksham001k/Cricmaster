from __future__ import annotations

from datetime import date

from cricmaster.chatbot.formatter import explain_prediction, format_chat_prediction
from cricmaster.chatbot.intents import ChatIntent, parse_intent_name
from cricmaster.chatbot.parser import (
    extract_format,
    extract_teams,
    infer_intent_from_message,
)
from cricmaster.prediction.prematch import Driver
from cricmaster.prediction.result import ProductionPredictionResult


def _result(*, drivers: tuple[Driver, ...] = ()) -> ProductionPredictionResult:
    return ProductionPredictionResult(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        team1_probability=0.53,
        team2_probability=0.47,
        predicted_team="Mumbai Indians",
        edge="very close",
        confidence="LOW",
        prediction_mode="PRE_TOSS",
        match_format="T20",
        model_name="hist_gradient_boosting",
        model_family="roster-aware T20",
        warnings=("franchise PRE_TOSS model has limited discriminatory power",),
        drivers=drivers,
        matches_applied=10,
        team1_history_matches=80,
        team2_history_matches=80,
        historical_sample={"matches_applied": 10},
    )


def test_parse_intent_aliases() -> None:
    assert parse_intent_name("predict_match") is ChatIntent.PREDICT_MATCH
    assert parse_intent_name("explain") is ChatIntent.EXPLAIN_PREDICTION
    assert infer_intent_from_message("What formats can you predict?") is ChatIntent.MODEL_CAPABILITIES


def test_extract_teams_from_simple_text() -> None:
    assert extract_teams("Predict MI vs CSK before toss") == (
        "Mumbai Indians",
        "Chennai Super Kings",
    )
    assert extract_teams("Who will win India vs Australia tomorrow in T20I?") == (
        "India",
        "Australia",
    )
    assert extract_format("Who will win India vs Australia tomorrow in T20I?") == "T20I"


def test_chat_message_does_not_guarantee_winners() -> None:
    text = format_chat_prediction(_result())
    assert "53%" in text or "estimated win probability" in text
    assert "definitely" not in text.lower()
    assert "LOW" in text


def test_explanation_does_not_fabricate_drivers() -> None:
    text = explain_prediction(_result(drivers=()))
    assert "does not expose reliable local feature contributions" in text
    assert "Rohit" not in text
    assert "causal" not in text.lower() or "not causal" in text.lower()

    with_drivers = explain_prediction(
        _result(
            drivers=(
                Driver(
                    feature="team_elo_before_diff",
                    label="Elo rating",
                    raw_difference=12.0,
                    contribution=0.2,
                    supports="Mumbai Indians",
                ),
            )
        )
    )
    assert "Elo rating supporting Mumbai Indians" in with_drivers
    assert "statistical associations" in with_drivers
    assert "not causal" in with_drivers
