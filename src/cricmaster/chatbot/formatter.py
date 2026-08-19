"""Deterministic chat text. Never invent drivers, players, or guarantees."""

from __future__ import annotations

from cricmaster.prediction.result import ProductionPredictionResult


def other_team(result: ProductionPredictionResult) -> str:
    if result.predicted_team == result.team1:
        return result.team2
    return result.team1


def format_chat_prediction(result: ProductionPredictionResult) -> str:
    favourite_p = max(result.team1_probability, result.team2_probability) * 100
    text = (
        f"Cricmaster gives {result.predicted_team} a {favourite_p:.0f}% estimated "
        f"win probability against {other_team(result)}. "
        f"Edge is {result.edge}. Confidence is {result.confidence}."
    )
    if result.warnings:
        text += " " + result.warnings[0]
    text += " These probabilities are statistical estimates, not guarantees."
    return text


def explain_prediction(result: ProductionPredictionResult) -> str:
    favourite_p = max(result.team1_probability, result.team2_probability) * 100
    parts = [
        (
            f"Cricmaster gives {result.predicted_team} a {result.edge} statistical "
            f"edge ({favourite_p:.1f}% estimated win probability) over "
            f"{other_team(result)}. Confidence is {result.confidence}."
        )
    ]

    limitation = next(
        (
            warning
            for warning in result.warnings
            if "limited discriminatory power" in warning or "prediction is close" in warning
        ),
        None,
    )
    if limitation:
        parts.append(limitation)

    if result.drivers:
        details = []
        for driver in result.drivers[:5]:
            details.append(
                f"{driver.label} supporting {driver.supports}"
            )
        parts.append(
            "The strongest available historical drivers were: "
            + "; ".join(details)
            + ". These are statistical associations learned from historical data, "
            "not causal explanations."
        )
    else:
        parts.append(
            "The selected model does not expose reliable local feature contributions."
        )

    parts.append("This is not a guarantee of the match result.")
    return " ".join(parts)


def capabilities_message() -> str:
    return (
        "Cricmaster v1 can estimate T20I and franchise/domestic T20 matches in "
        "PRE_TOSS, POST_TOSS, and LIVE modes. The Hundred, ODI, Test, and T10 "
        "are not supported. Probabilities are estimates, not guarantees. "
        "Franchise PRE_TOSS discrimination is modest, so close games usually "
        "receive LOW confidence."
    )


def help_message() -> str:
    return (
        "Send a structured chat request with intent predict_match, live_prediction, "
        "explain_prediction, model_capabilities, or help. "
        "Example: India vs Australia, format T20I, mode PRE_TOSS, and a date. "
        "Cricmaster will not invent missing scores, lineups, or winners."
    )


def suggestions_for(intent: str, result: ProductionPredictionResult | None) -> list[str]:
    if intent == "predict_match":
        items = ["Ask for the main prediction drivers"]
        if result and result.prediction_mode == "PRE_TOSS":
            items.append("Switch to post-toss once the XI is known")
        return items
    if intent == "live_prediction":
        return ["Refresh after the next over", "Ask for the main live drivers"]
    if intent == "explain_prediction":
        return ["Run a post-toss prediction if lineups are known"]
    if intent == "model_capabilities":
        return ["Predict a T20I match before toss"]
    return ["Ask what formats Cricmaster can predict"]
