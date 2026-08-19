"""Explicit, conservative confidence labels for production predictions.

Confidence is not max(probability). A 53/47 split is a close estimate, not a
high-confidence call, especially for franchise/domestic T20 PRE_TOSS where
locked evaluation showed only weak discrimination.

Labels are LOW / MEDIUM / HIGH. They are not percentages of correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cricmaster.data.formats import MatchFormat


class Confidence(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_RANK = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}


@dataclass(frozen=True)
class ConfidenceInputs:
    winner_probability: float
    match_format: MatchFormat
    prediction_mode: str
    team1_history_matches: int
    team2_history_matches: int
    venue_known: bool
    previous_xi_complete: bool | None = None
    lineup_complete: bool | None = None
    parse_errors: int = 0
    terminal: bool = False
    innings_number: int | None = None
    legal_balls: int | None = None


@dataclass(frozen=True)
class ConfidenceAssessment:
    label: Confidence
    reasons: tuple[str, ...]


def _cap(current: Confidence, ceiling: Confidence) -> Confidence:
    if _RANK[current] <= _RANK[ceiling]:
        return current
    return ceiling


def _probability_cap(winner_probability: float) -> tuple[Confidence, str]:
    p = max(float(winner_probability), 1.0 - float(winner_probability))
    if p < 0.55:
        return Confidence.LOW, "probability is close to 50%"
    if p < 0.62:
        return Confidence.MEDIUM, "probability edge is only slight"
    if p < 0.72:
        return Confidence.MEDIUM, "probability edge is moderate"
    return Confidence.HIGH, "probability edge is strong"


def _domain_cap(
    *,
    match_format: MatchFormat,
    prediction_mode: str,
    winner_probability: float,
    terminal: bool,
    innings_number: int | None,
    legal_balls: int | None,
) -> tuple[Confidence, str]:
    p = max(float(winner_probability), 1.0 - float(winner_probability))
    mode = prediction_mode.upper()

    if mode == "LIVE":
        if terminal and p >= 0.95:
            return Confidence.HIGH, "terminal live state is nearly decided"
        if innings_number == 1 and (legal_balls or 0) < 12:
            return Confidence.MEDIUM, "first innings is still early"
        return Confidence.HIGH, "live model can use current match state"

    if match_format is MatchFormat.T20I:
        return Confidence.HIGH, "T20I models have stronger historical discrimination"

    if match_format is MatchFormat.T20 and mode == "PRE_TOSS":
        # Locked franchise PRE_TOSS AUC is only ~0.56. Never emit HIGH.
        if p < 0.62:
            return (
                Confidence.LOW,
                "franchise PRE_TOSS model has limited discriminatory power",
            )
        return (
            Confidence.MEDIUM,
            "franchise PRE_TOSS model has limited discriminatory power",
        )

    if match_format is MatchFormat.T20:
        return (
            Confidence.MEDIUM,
            "franchise POST_TOSS model has limited discriminatory power",
        )

    return Confidence.LOW, "unsupported domain defaults to low confidence"


def _evidence_cap(
    *,
    team1_history_matches: int,
    team2_history_matches: int,
    venue_known: bool,
    previous_xi_complete: bool | None,
    lineup_complete: bool | None,
    parse_errors: int,
    prediction_mode: str,
) -> tuple[Confidence, tuple[str, ...]]:
    reasons: list[str] = []
    sample = min(int(team1_history_matches), int(team2_history_matches))
    if sample < 5:
        cap = Confidence.LOW
        reasons.append("historical sample is sparse")
    elif sample < 15:
        cap = Confidence.MEDIUM
        reasons.append("historical sample is modest")
    else:
        cap = Confidence.HIGH
        reasons.append("both teams have usable historical depth")

    if previous_xi_complete is False:
        cap = _cap(cap, Confidence.LOW)
        reasons.append("previous-XI history is missing or incomplete")

    if lineup_complete is False and prediction_mode.upper() in {"POST_TOSS", "LIVE"}:
        cap = _cap(cap, Confidence.MEDIUM)
        reasons.append("current playing XI is incomplete")

    if not venue_known and prediction_mode.upper() != "LIVE":
        cap = _cap(cap, Confidence.MEDIUM)
        reasons.append("venue is unknown")

    if parse_errors:
        cap = _cap(cap, Confidence.MEDIUM)
        reasons.append("some historical files failed to parse")

    return cap, tuple(reasons)


def assess_confidence(inputs: ConfidenceInputs) -> ConfidenceAssessment:
    """Combine probability separation, domain reliability, and evidence."""

    sep_cap, sep_reason = _probability_cap(inputs.winner_probability)
    domain_cap, domain_reason = _domain_cap(
        match_format=inputs.match_format,
        prediction_mode=inputs.prediction_mode,
        winner_probability=inputs.winner_probability,
        terminal=inputs.terminal,
        innings_number=inputs.innings_number,
        legal_balls=inputs.legal_balls,
    )
    evidence_cap, evidence_reasons = _evidence_cap(
        team1_history_matches=inputs.team1_history_matches,
        team2_history_matches=inputs.team2_history_matches,
        venue_known=inputs.venue_known,
        previous_xi_complete=inputs.previous_xi_complete,
        lineup_complete=inputs.lineup_complete,
        parse_errors=inputs.parse_errors,
        prediction_mode=inputs.prediction_mode,
    )

    label = _cap(_cap(sep_cap, domain_cap), evidence_cap)
    reasons = (sep_reason, domain_reason, *evidence_reasons)
    return ConfidenceAssessment(label=label, reasons=reasons)


def applicable_confidence_warnings(
    *,
    assessment: ConfidenceAssessment,
    match_format: MatchFormat,
    prediction_mode: str,
    winner_probability: float,
    previous_xi_complete: bool | None,
    ignored_current_xi: bool,
) -> tuple[str, ...]:
    """User-facing warnings that actually apply to this prediction."""

    warnings: list[str] = []
    p = max(float(winner_probability), 1.0 - float(winner_probability))
    if p < 0.55:
        warnings.append("prediction is close")

    mode = prediction_mode.upper()
    if match_format is MatchFormat.T20 and mode == "PRE_TOSS":
        warnings.append(
            "franchise PRE_TOSS model has limited discriminatory power"
        )
    elif match_format is MatchFormat.T20 and mode == "POST_TOSS":
        warnings.append(
            "franchise POST_TOSS model has limited discriminatory power"
        )

    if previous_xi_complete is False:
        warnings.append(
            "historical previous-XI information is missing or incomplete; "
            "roster features were zero-imputed"
        )
    if ignored_current_xi:
        warnings.append(
            "current playing XI was ignored because PRE_TOSS cannot use the current lineup"
        )
    return tuple(warnings)
