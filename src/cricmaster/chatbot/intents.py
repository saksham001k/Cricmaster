"""Deterministic chatbot intents. This is not a free-form cricket expert."""

from __future__ import annotations

from enum import StrEnum


class ChatIntent(StrEnum):
    PREDICT_MATCH = "predict_match"
    LIVE_PREDICTION = "live_prediction"
    EXPLAIN_PREDICTION = "explain_prediction"
    MODEL_CAPABILITIES = "model_capabilities"
    HELP = "help"
    UNKNOWN = "unknown"


INTENT_ALIASES = {
    "predict_match": ChatIntent.PREDICT_MATCH,
    "predict": ChatIntent.PREDICT_MATCH,
    "pre_toss": ChatIntent.PREDICT_MATCH,
    "live_prediction": ChatIntent.LIVE_PREDICTION,
    "live": ChatIntent.LIVE_PREDICTION,
    "explain_prediction": ChatIntent.EXPLAIN_PREDICTION,
    "explain": ChatIntent.EXPLAIN_PREDICTION,
    "why": ChatIntent.EXPLAIN_PREDICTION,
    "model_capabilities": ChatIntent.MODEL_CAPABILITIES,
    "capabilities": ChatIntent.MODEL_CAPABILITIES,
    "help": ChatIntent.HELP,
}


def parse_intent_name(value: str | None) -> ChatIntent | None:
    if not value:
        return None
    key = value.strip().lower().replace("-", "_")
    return INTENT_ALIASES.get(key)
