"""Unified production prediction result.

Existing PRE_TOSS / POST_TOSS / LIVE dataclasses remain for their original
CLIs. Production routing returns this single structure so callers do not have
to branch on incompatible result types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cricmaster.prediction.prematch import Driver, prediction_edge


CONFIDENCE_LABELS = ("LOW", "MEDIUM", "HIGH")


@dataclass(frozen=True)
class ProductionPredictionResult:
    team1: str
    team2: str
    team1_probability: float
    team2_probability: float
    predicted_team: str
    edge: str
    confidence: str
    prediction_mode: str
    match_format: str
    model_name: str
    model_family: str
    warnings: tuple[str, ...]
    drivers: tuple[Driver, ...]
    matches_applied: int
    team1_history_matches: int
    team2_history_matches: int
    historical_sample: dict[str, Any]
    competition: str | None = None
    venue: str | None = None
    lineup_mode: str | None = None
    previous_xi_team1_known: bool | None = None
    previous_xi_team2_known: bool | None = None
    innings_number: int | None = None
    model_kind: str | None = None
    terminal: bool | None = None


def complementary_probabilities(team1_probability: float) -> tuple[float, float]:
    """Return (P(team1), P(team2)) with P1 + P2 = 1."""

    p1 = float(team1_probability)
    p2 = 1.0 - p1
    return p1, p2


def predicted_team_from_probabilities(
    team1: str,
    team2: str,
    team1_probability: float,
    team2_probability: float,
) -> str:
    if team1_probability >= team2_probability:
        return team1
    return team2


def format_prediction_report(result: ProductionPredictionResult) -> str:
    """Human-readable production report. Probabilities are estimates."""

    lines = [
        f"{result.team1:<22} {result.team1_probability * 100:5.1f}%",
        f"{result.team2:<22} {result.team2_probability * 100:5.1f}%",
        "",
        f"Prediction: {result.predicted_team}",
        f"Edge: {result.edge}",
        f"Confidence: {result.confidence}",
        f"Mode: {result.prediction_mode}",
        f"Model: {result.model_family}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append(
        "Note: probabilities are statistical estimates, not guarantees."
    )
    return "\n".join(lines)


__all__ = [
    "CONFIDENCE_LABELS",
    "Driver",
    "ProductionPredictionResult",
    "complementary_probabilities",
    "format_prediction_report",
    "predicted_team_from_probabilities",
    "prediction_edge",
]
