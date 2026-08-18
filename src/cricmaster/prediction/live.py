"""Runtime live win-probability prediction for T20/T20I matches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cricmaster.data.formats import MatchFormat, normalize_competition
from cricmaster.data.models import MatchMetadata, MatchState
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.live import compute_live_metrics
from cricmaster.features.utils import innings_ball_limit
from cricmaster.models.live import CHASE_FEATURES, FIRST_INNINGS_FEATURES
from cricmaster.prediction.prematch import build_historical_state, prediction_edge


LIVE_FEATURE_LABELS = {
    "runs": "current score",
    "wickets": "wickets lost",
    "legal_balls_bowled": "balls bowled",
    "balls_remaining": "balls remaining",
    "current_run_rate": "current run rate",
    "wickets_in_hand": "wickets in hand",
    "target": "target",
    "runs_required": "runs required",
    "required_run_rate": "required run rate",
    "run_rate_difference": "current minus required run rate",
    "matches_before": "historical match experience",
    "win_rate_before": "historical win rate",
    "win_rate_last_5": "last-5 win rate",
    "win_rate_last_10": "last-10 win rate",
    "win_rate_last_20": "last-20 win rate",
    "h2h_team_win_rate": "head-to-head win rate",
    "team_win_rate_at_venue": "venue win rate",
    "elo_difference": "Elo advantage",
    "xi_batters_with_history": "XI batting-history coverage",
    "xi_mean_batting_average": "XI batting average",
    "xi_mean_recent_runs": "XI recent runs",
    "xi_bowlers_with_history": "XI bowling-history coverage",
    "xi_mean_bowling_economy": "XI bowling economy",
    "xi_mean_recent_wickets": "XI recent wickets",
    "toss_bat_signed": "toss/bat context",
    "toss_field_signed": "toss/field context",
    "is_t20i": "T20I format indicator",
    "is_male": "male-match indicator",
}


@dataclass(frozen=True)
class LivePredictionRequest:
    team1: str
    team2: str
    batting_team: str
    match_format: MatchFormat
    match_date: date
    gender: str
    innings_number: int
    runs: int
    wickets: int
    legal_balls: int
    target: int | None = None
    venue: str | None = None
    competition: str | None = None
    toss_winner: str | None = None
    toss_decision: str | None = None
    team1_xi: tuple[str, ...] = ()
    team2_xi: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiveDriver:
    feature: str
    label: str
    raw_value: float | None
    contribution: float
    supports: str


@dataclass(frozen=True)
class LivePredictionResult:
    batting_team: str
    bowling_team: str
    batting_probability: float
    bowling_probability: float
    predicted_winner: str
    edge: str
    model_name: str
    innings_number: int
    model_kind: str
    state_summary: dict[str, Any]
    drivers: tuple[LiveDriver, ...]
    warnings: tuple[str, ...]
    terminal: bool


def parse_cricket_overs(value: str, *, balls_per_over: int = 6) -> int:
    """Convert cricket notation such as 15.3 to legal balls, strictly."""

    text = str(value).strip()
    if not text:
        raise ValueError("overs cannot be empty")

    if "." in text:
        whole_text, ball_text = text.split(".", 1)
    else:
        whole_text, ball_text = text, "0"

    try:
        whole = int(whole_text)
        ball = int(ball_text)
    except ValueError as exc:
        raise ValueError("overs must use cricket notation such as 15.3") from exc

    if whole < 0 or ball < 0:
        raise ValueError("overs cannot be negative")
    if ball >= balls_per_over:
        raise ValueError(
            f"Invalid cricket over notation '{value}': "
            f"ball component must be 0-{balls_per_over - 1}"
        )

    return whole * balls_per_over + ball


def _validate_request(request: LivePredictionRequest) -> tuple[str, str]:
    if request.match_format not in {MatchFormat.T20, MatchFormat.T20I}:
        raise ValueError("Current live models support only T20 and T20I")
    if request.innings_number not in {1, 2}:
        raise ValueError("innings_number must be 1 or 2")
    if request.runs < 0:
        raise ValueError("runs cannot be negative")
    if not 0 <= request.wickets <= 10:
        raise ValueError("wickets must be between 0 and 10")
    if request.legal_balls < 0:
        raise ValueError("legal_balls cannot be negative")
    if request.innings_number == 2 and (request.target is None or request.target <= 0):
        raise ValueError("A positive target is required for innings 2")
    if request.innings_number == 1 and request.target is not None:
        raise ValueError("target must be omitted for innings 1")

    team1 = canonicalize_team(request.team1)
    team2 = canonicalize_team(request.team2)
    batting = canonicalize_team(request.batting_team)

    if team1 == team2:
        raise ValueError("team1 and team2 must be different")
    if batting not in {team1, team2}:
        raise ValueError("batting_team must be team1 or team2")

    bowling = team2 if batting == team1 else team1
    return batting, bowling


def _metadata_match(
    request: LivePredictionRequest,
) -> MatchState:
    metadata = MatchMetadata(
        match_id=(
            f"live:{request.match_date.isoformat()}:"
            f"{canonicalize_team(request.team1)}:{canonicalize_team(request.team2)}"
        ),
        format=request.match_format,
        competition=normalize_competition(request.competition),
        date=request.match_date,
        venue=request.venue,
        city=None,
        team1=request.team1,
        team2=request.team2,
        toss_winner=request.toss_winner,
        toss_decision=request.toss_decision,
        winner=None,
        result_type=None,
        player_of_match=None,
        source="live-prediction-request",
        gender=request.gender,
        team_type="international" if request.match_format is MatchFormat.T20I else None,
        balls_per_over=6,
        scheduled_overs=20,
        team1_players=list(request.team1_xi) or None,
        team2_players=list(request.team2_xi) or None,
    )
    return MatchState(
        metadata=metadata,
        current_innings=None,
        innings_history=[],
        deliveries=[],
        current_players=None,
        source="live-prediction-request",
        retrieved_at=None,
    )


def _signed_toss(
    *,
    batting_team: str,
    toss_winner: str | None,
    toss_decision: str | None,
) -> tuple[float, float]:
    if not toss_winner or not toss_decision:
        return 0.0, 0.0

    winner = canonicalize_team(toss_winner)
    decision = toss_decision.strip().lower()
    if decision == "bowl":
        decision = "field"

    sign = 1.0 if winner == batting_team else -1.0
    return (
        sign if decision == "bat" else 0.0,
        sign if decision == "field" else 0.0,
    )


def build_live_feature_frame(
    request: LivePredictionRequest,
    *,
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], tuple[str, ...]]:
    batting, bowling = _validate_request(request)

    history = build_historical_state(raw_dir, cutoff=request.match_date)
    match = _metadata_match(request)
    context = history.state.features_for(match, batting, bowling)

    ball_limit = innings_ball_limit(
        request.match_format,
        scheduled_overs=20,
        balls_per_over=6,
        target_overs=None,
    )
    metrics = compute_live_metrics(
        runs=request.runs,
        wickets=request.wickets,
        legal_balls=request.legal_balls,
        target=request.target,
        ball_limit=ball_limit,
        balls_per_over=6,
    )

    toss_bat, toss_field = _signed_toss(
        batting_team=batting,
        toss_winner=request.toss_winner,
        toss_decision=request.toss_decision,
    )

    row: dict[str, Any] = {
        **metrics,
        "matches_before": context.get("matches_before"),
        "win_rate_before": context.get("win_rate_before"),
        "win_rate_last_5": context.get("win_rate_last_5"),
        "win_rate_last_10": context.get("win_rate_last_10"),
        "win_rate_last_20": context.get("win_rate_last_20"),
        "h2h_team_win_rate": context.get("h2h_team_win_rate"),
        "team_win_rate_at_venue": context.get("team_win_rate_at_venue"),
        "elo_difference": context.get("elo_difference"),
        "xi_batters_with_history": context.get("xi_batters_with_history"),
        "xi_mean_batting_average": context.get("xi_mean_batting_average"),
        "xi_mean_recent_runs": context.get("xi_mean_recent_runs"),
        "xi_bowlers_with_history": context.get("xi_bowlers_with_history"),
        "xi_mean_bowling_economy": context.get("xi_mean_bowling_economy"),
        "xi_mean_recent_wickets": context.get("xi_mean_recent_wickets"),
        "toss_bat_signed": toss_bat,
        "toss_field_signed": toss_field,
        "is_t20i": float(request.match_format is MatchFormat.T20I),
        "is_male": float(request.gender.strip().lower() == "male"),
    }

    features = (
        FIRST_INNINGS_FEATURES
        if request.innings_number == 1
        else CHASE_FEATURES
    )
    frame = pd.DataFrame([row], columns=features)

    warnings: list[str] = []
    if request.venue is None:
        warnings.append("No venue supplied; venue-specific history is unavailable.")
    if not request.toss_winner or not request.toss_decision:
        warnings.append("Toss context was not supplied; toss features are neutral.")
    if not (request.team1_xi and request.team2_xi):
        warnings.append(
            "Both playing XIs were not supplied; XI-strength features may be unavailable."
        )
    if history.parse_errors:
        warnings.append(
            f"{history.parse_errors} historical files could not be parsed and were skipped."
        )

    return frame, metrics, tuple(warnings)


def _load_bundle(model_path: str | Path, *, innings_number: int) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    if bundle.get("prediction_mode") != "LIVE_AFTER_LEGAL_BALL":
        raise ValueError("Expected a live model bundle")
    if int(bundle.get("innings_number", -1)) != innings_number:
        raise ValueError(
            f"Model is for innings {bundle.get('innings_number')}, "
            f"request is innings {innings_number}"
        )
    return bundle


def _sanitize_model_input(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Match the training pipeline's constant-zero missing-value policy."""

    x = frame.loc[:, features].copy()

    for column in features:
        x[column] = pd.to_numeric(x[column], errors="coerce")

    return (
        x.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )


def _probability(bundle: dict[str, Any], frame: pd.DataFrame) -> float:
    model = bundle.get("model")
    if model is None:
        raise ValueError("Live model bundle contains no fitted estimator")

    features = list(bundle["features"])
    x = _sanitize_model_input(frame, features)

    probability = float(model.predict_proba(x)[:, 1][0])
    return float(np.clip(probability, 0.0, 1.0))


def _terminal_chase_probability(
    request: LivePredictionRequest,
    metrics: dict[str, Any],
) -> tuple[float | None, str | None]:
    if request.innings_number != 2 or request.target is None:
        return None, None

    if request.runs >= request.target:
        return 1.0, "target reached"
    if request.wickets >= 10:
        return 0.0, "all out before target"

    balls_remaining = metrics.get("balls_remaining")
    if balls_remaining == 0:
        if request.runs == request.target - 1:
            return 0.5, "regulation scores tied; super-over outcome not modeled"
        return 0.0, "balls exhausted before target"

    return None, None


def logistic_live_drivers(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    batting_team: str,
    bowling_team: str,
    top_n: int = 6,
) -> tuple[LiveDriver, ...]:
    if bundle.get("model_name") != "logistic_regression":
        return ()

    pipeline = bundle.get("model")
    try:
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scale"]
        estimator = pipeline.named_steps["model"]
    except (AttributeError, KeyError):
        return ()

    features = list(bundle["features"])
    raw_x = frame.loc[:, features]
    x = _sanitize_model_input(frame, features)
    imputed = imputer.transform(x)
    scaled = scaler.transform(imputed)
    coefs = np.asarray(estimator.coef_[0], dtype=float)
    effects = np.asarray(scaled[0], dtype=float) * coefs

    drivers: list[LiveDriver] = []
    for index, feature in enumerate(features):
        raw = raw_x.iloc[0][feature]

        # Missing runtime context may be imputed for prediction, but it
        # should not be presented to users as an observed model driver.
        if pd.isna(raw):
            continue

        raw_value = float(raw)
        effect = float(effects[index])
        if abs(effect) < 1e-12:
            continue
        drivers.append(
            LiveDriver(
                feature=feature,
                label=LIVE_FEATURE_LABELS.get(feature, feature),
                raw_value=raw_value,
                contribution=effect,
                supports=batting_team if effect > 0 else bowling_team,
            )
        )

    drivers.sort(key=lambda item: abs(item.contribution), reverse=True)
    return tuple(drivers[:max(top_n, 0)])


def predict_live(
    request: LivePredictionRequest,
    *,
    raw_dir: str | Path,
    first_innings_model: str | Path,
    chase_model: str | Path,
    top_drivers: int = 6,
) -> LivePredictionResult:
    batting, bowling = _validate_request(request)
    frame, metrics, warnings = build_live_feature_frame(request, raw_dir=raw_dir)

    model_path = (
        first_innings_model if request.innings_number == 1 else chase_model
    )
    bundle = _load_bundle(model_path, innings_number=request.innings_number)

    terminal_probability, terminal_reason = _terminal_chase_probability(
        request,
        metrics,
    )
    terminal = terminal_probability is not None

    if terminal:
        p_batting = float(terminal_probability)
        warnings = tuple(
            [*warnings, f"Terminal chase state: {terminal_reason}."]
        )
        drivers: tuple[LiveDriver, ...] = ()
    else:
        p_batting = _probability(bundle, frame)
        drivers = logistic_live_drivers(
            bundle,
            frame,
            batting_team=batting,
            bowling_team=bowling,
            top_n=top_drivers,
        )

    p_bowling = 1.0 - p_batting
    winner = batting if p_batting >= p_bowling else bowling

    return LivePredictionResult(
        batting_team=batting,
        bowling_team=bowling,
        batting_probability=p_batting,
        bowling_probability=p_bowling,
        predicted_winner=winner,
        edge=prediction_edge(max(p_batting, p_bowling)),
        model_name=str(bundle["model_name"]),
        innings_number=request.innings_number,
        model_kind="first_innings" if request.innings_number == 1 else "chase",
        state_summary=metrics,
        drivers=drivers,
        warnings=warnings,
        terminal=terminal,
    )
