"""Runtime PRE_TOSS prediction engine backed by historical Cricsheet state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cricmaster.data.cricsheet import CricsheetParseError, discover_match_files, load_match
from cricmaster.data.formats import MatchFormat, normalize_competition
from cricmaster.data.models import MatchMetadata, MatchState
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.history import HistoricalState
from cricmaster.models.prematch import (
    ELO_DIFFERENCE_FEATURE,
    MODEL_FEATURES,
    SIGNED_SOURCE_FEATURES,
    elo_probability,
)

SUPPORTED_FORMATS = {MatchFormat.T20I, MatchFormat.T20}

FEATURE_LABELS = {
    "matches_before_diff": "historical match experience",
    "win_rate_before_diff": "historical win rate",
    "wins_last_5_diff": "wins in last 5",
    "win_rate_last_5_diff": "last-5 win rate",
    "wins_last_10_diff": "wins in last 10",
    "win_rate_last_10_diff": "last-10 win rate",
    "wins_last_20_diff": "wins in last 20",
    "win_rate_last_20_diff": "last-20 win rate",
    "h2h_team_wins_diff": "head-to-head wins",
    "h2h_team_win_rate_diff": "head-to-head win rate",
    "h2h_last_5_win_rate_diff": "recent head-to-head form",
    "team_matches_at_venue_diff": "venue experience",
    "team_win_rate_at_venue_diff": "venue win rate",
    "team_elo_before_diff": "Elo rating",
}


@dataclass(frozen=True)
class PredictionRequest:
    team1: str
    team2: str
    match_format: MatchFormat
    match_date: date
    gender: str
    venue: str | None = None
    competition: str | None = None


@dataclass(frozen=True)
class HistoryBuild:
    state: HistoricalState
    matches_applied: int
    parse_errors: int


@dataclass(frozen=True)
class Driver:
    feature: str
    label: str
    raw_difference: float | None
    contribution: float
    supports: str


@dataclass(frozen=True)
class PredictionResult:
    team1: str
    team2: str
    team1_probability: float
    team2_probability: float
    model_name: str
    prediction_mode: str
    edge: str
    matches_applied: int
    team1_history_matches: int
    team2_history_matches: int
    drivers: tuple[Driver, ...]
    warnings: tuple[str, ...]


def parse_match_format(value: str) -> MatchFormat:
    try:
        match_format = MatchFormat(value.strip().upper())
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in SUPPORTED_FORMATS))
        raise ValueError(f"Unsupported model format '{value}'. Allowed: {allowed}") from exc

    if match_format not in SUPPORTED_FORMATS:
        allowed = ", ".join(sorted(item.value for item in SUPPORTED_FORMATS))
        raise ValueError(
            f"The current PRE_TOSS model was not trained for {match_format.value}. "
            f"Allowed: {allowed}"
        )
    return match_format


def build_historical_state(
    raw_dir: str | Path,
    *,
    cutoff: date,
) -> HistoryBuild:
    """Reconstruct knowledge using only matches strictly before cutoff."""

    state = HistoricalState()
    matches: list[MatchState] = []
    parse_errors = 0

    for path in discover_match_files(raw_dir):
        try:
            match = load_match(path)
        except CricsheetParseError:
            parse_errors += 1
            continue

        match_date = match.metadata.date
        if match_date is None or match_date >= cutoff:
            continue
        if match.metadata.format not in SUPPORTED_FORMATS:
            continue
        matches.append(match)

    matches.sort(
        key=lambda item: (
            item.metadata.date or date.min,
            item.metadata.match_id,
        )
    )

    for match in matches:
        state.update(match)

    return HistoryBuild(
        state=state,
        matches_applied=len(matches),
        parse_errors=parse_errors,
    )


def _number(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def difference_features(
    team1_features: dict[str, Any],
    team2_features: dict[str, Any],
) -> pd.DataFrame:
    """Build Team-1 minus Team-2 features in the trained model schema."""

    row: dict[str, float] = {}
    for source in SIGNED_SOURCE_FEATURES:
        left = _number(team1_features.get(source))
        right = _number(team2_features.get(source))
        output = f"{source}_diff"
        row[output] = (
            left - right
            if not (np.isnan(left) or np.isnan(right))
            else float("nan")
        )

    return pd.DataFrame([row], columns=MODEL_FEATURES)


def request_features(
    state: HistoricalState,
    request: PredictionRequest,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Create leakage-safe PRE_TOSS features for an unplayed match."""

    team1 = canonicalize_team(request.team1)
    team2 = canonicalize_team(request.team2)
    if team1 == team2:
        raise ValueError("team1 and team2 must be different teams")

    metadata = MatchMetadata(
        match_id=f"prediction:{request.match_date.isoformat()}:{team1}:{team2}",
        format=request.match_format,
        competition=normalize_competition(request.competition),
        date=request.match_date,
        venue=request.venue,
        city=None,
        team1=request.team1,
        team2=request.team2,
        toss_winner=None,
        toss_decision=None,
        winner=None,
        result_type=None,
        player_of_match=None,
        source="prediction-request",
        gender=request.gender,
        team_type="international" if request.match_format is MatchFormat.T20I else None,
        balls_per_over=6,
        scheduled_overs=20,
        team1_players=None,
        team2_players=None,
    )
    match = MatchState(
        metadata=metadata,
        current_innings=None,
        innings_history=[],
        deliveries=[],
        current_players=None,
        source="prediction-request",
        retrieved_at=None,
    )

    team1_features = state.features_for(match, team1, team2)
    team2_features = state.features_for(match, team2, team1)
    return (
        difference_features(team1_features, team2_features),
        team1_features,
        team2_features,
    )


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"model_name", "features", "prediction_mode"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing fields: {sorted(missing)}")
    if bundle["prediction_mode"] != "PRE_TOSS":
        raise ValueError(
            f"Expected PRE_TOSS model, got {bundle['prediction_mode']!r}"
        )
    return bundle


def symmetric_probability(
    bundle: dict[str, Any],
    features: pd.DataFrame,
) -> float:
    """Return Team-1 probability and enforce P(A,B) + P(B,A) = 1."""

    name = str(bundle["model_name"])
    if name == "elo_baseline":
        diff = float(features.iloc[0][ELO_DIFFERENCE_FEATURE])
        if np.isnan(diff):
            diff = 0.0
        return float(elo_probability(diff))

    model = bundle.get("model")
    if model is None:
        raise ValueError(f"Model bundle '{name}' contains no fitted estimator")

    trained_features = list(bundle["features"])
    x = features.loc[:, trained_features]
    forward = float(model.predict_proba(x)[:, 1][0])
    reverse = float(model.predict_proba(-x)[:, 1][0])
    probability = 0.5 * (forward + (1.0 - reverse))
    return float(np.clip(probability, 0.0, 1.0))


def prediction_edge(winner_probability: float) -> str:
    p = max(float(winner_probability), 1.0 - float(winner_probability))
    if p < 0.55:
        return "very close"
    if p < 0.62:
        return "slight"
    if p < 0.72:
        return "moderate"
    return "strong"


def logistic_drivers(
    bundle: dict[str, Any],
    features: pd.DataFrame,
    *,
    team1: str,
    team2: str,
    top_n: int = 5,
) -> tuple[Driver, ...]:
    """Extract local log-odds contributions from the selected logistic pipeline."""

    if bundle.get("model_name") != "logistic_regression":
        return ()

    pipeline = bundle.get("model")
    if pipeline is None:
        return ()

    try:
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scale"]
        estimator = pipeline.named_steps["model"]
    except (AttributeError, KeyError):
        return ()

    trained_features = list(bundle["features"])
    x = features.loc[:, trained_features]
    imputed = imputer.transform(x)
    scaled = scaler.transform(imputed)
    coefficients = np.asarray(estimator.coef_[0], dtype=float)
    contributions = np.asarray(scaled[0], dtype=float) * coefficients

    drivers: list[Driver] = []
    for index, feature in enumerate(trained_features):
        raw = _number(x.iloc[0][feature])
        contribution = float(contributions[index])
        if abs(contribution) < 1e-12:
            continue
        drivers.append(
            Driver(
                feature=feature,
                label=FEATURE_LABELS.get(feature, feature),
                raw_difference=None if np.isnan(raw) else raw,
                contribution=contribution,
                supports=team1 if contribution > 0 else team2,
            )
        )

    drivers.sort(key=lambda item: abs(item.contribution), reverse=True)
    return tuple(drivers[: max(top_n, 0)])


def predict_prematch(
    request: PredictionRequest,
    *,
    raw_dir: str | Path,
    model_path: str | Path,
    top_drivers: int = 5,
) -> PredictionResult:
    history = build_historical_state(raw_dir, cutoff=request.match_date)
    features, team1_history, team2_history = request_features(history.state, request)
    bundle = load_model_bundle(model_path)
    p1 = symmetric_probability(bundle, features)
    p2 = 1.0 - p1

    team1 = canonicalize_team(request.team1)
    team2 = canonicalize_team(request.team2)
    warnings: list[str] = []

    team1_matches = int(team1_history.get("matches_before") or 0)
    team2_matches = int(team2_history.get("matches_before") or 0)

    if team1_matches < 5:
        warnings.append(
            f"{team1} has only {team1_matches} prior {request.match_format.value} "
            "matches in the loaded corpus."
        )
    if team2_matches < 5:
        warnings.append(
            f"{team2} has only {team2_matches} prior {request.match_format.value} "
            "matches in the loaded corpus."
        )
    if request.venue is None:
        warnings.append("No venue supplied; venue-specific features are unavailable.")
    if history.parse_errors:
        warnings.append(
            f"{history.parse_errors} historical files could not be parsed and were skipped."
        )

    return PredictionResult(
        team1=team1,
        team2=team2,
        team1_probability=p1,
        team2_probability=p2,
        model_name=str(bundle["model_name"]),
        prediction_mode=str(bundle["prediction_mode"]),
        edge=prediction_edge(max(p1, p2)),
        matches_applied=history.matches_applied,
        team1_history_matches=team1_matches,
        team2_history_matches=team2_matches,
        drivers=logistic_drivers(
            bundle,
            features,
            team1=team1,
            team2=team2,
            top_n=top_drivers,
        ),
        warnings=tuple(warnings),
    )
