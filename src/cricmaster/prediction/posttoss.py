"""POST_TOSS runtime prediction using toss and optional playing XIs."""

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
from cricmaster.models.posttoss import POST_TOSS_FEATURES, XI_SOURCE_FEATURES
from cricmaster.prediction.prematch import (
    HistoryBuild,
    PredictionRequest,
    build_historical_state,
    prediction_edge,
)


@dataclass(frozen=True)
class PostTossRequest(PredictionRequest):
    toss_winner: str = ""
    toss_decision: str = ""
    team1_xi: tuple[str, ...] = ()
    team2_xi: tuple[str, ...] = ()


@dataclass(frozen=True)
class PostTossResult:
    team1: str
    team2: str
    team1_probability: float
    team2_probability: float
    winner: str
    edge: str
    model_name: str
    lineup_mode: str
    warnings: tuple[str, ...]


def _number(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _request_frame(
    history: HistoryBuild,
    request: PostTossRequest,
) -> pd.DataFrame:
    team1 = canonicalize_team(request.team1)
    team2 = canonicalize_team(request.team2)
    toss_winner = canonicalize_team(request.toss_winner)

    if toss_winner not in {team1, team2}:
        raise ValueError("toss_winner must be one of team1/team2")

    decision = request.toss_decision.strip().lower()
    if decision not in {"bat", "field", "bowl"}:
        raise ValueError("toss_decision must be bat, field, or bowl")

    metadata = MatchMetadata(
        match_id=f"posttoss:{request.match_date.isoformat()}:{team1}:{team2}",
        format=request.match_format,
        competition=normalize_competition(request.competition),
        date=request.match_date,
        venue=request.venue,
        city=None,
        team1=request.team1,
        team2=request.team2,
        toss_winner=request.toss_winner,
        toss_decision="field" if decision == "bowl" else decision,
        winner=None,
        result_type=None,
        player_of_match=None,
        source="prediction-request",
        gender=request.gender,
        team_type="international" if request.match_format is MatchFormat.T20I else None,
        balls_per_over=6,
        scheduled_overs=20,
        team1_players=list(request.team1_xi) or None,
        team2_players=list(request.team2_xi) or None,
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

    left = history.state.features_for(match, team1, team2)
    right = history.state.features_for(match, team2, team1)

    row: dict[str, float] = {}

    # Base + XI differences.
    sources = [
        name[:-5]
        for name in POST_TOSS_FEATURES
        if name.endswith("_diff")
    ]
    for source in sources:
        a = _number(left.get(source))
        b = _number(right.get(source))
        row[f"{source}_diff"] = (
            a - b if not (np.isnan(a) or np.isnan(b)) else float("nan")
        )

    sign = 1.0 if toss_winner == team1 else -1.0
    row["toss_bat_advantage"] = sign if decision == "bat" else 0.0
    row["toss_field_advantage"] = sign if decision in {"field", "bowl"} else 0.0

    return pd.DataFrame([row], columns=POST_TOSS_FEATURES)


def _probability(bundle: dict[str, Any], x: pd.DataFrame) -> float:
    if bundle.get("prediction_mode") != "POST_TOSS":
        raise ValueError("Expected a POST_TOSS model bundle")
    model = bundle.get("model")
    if model is None:
        raise ValueError("POST_TOSS bundle contains no fitted model")
    features = list(bundle["features"])
    x = x.loc[:, features]
    forward = float(model.predict_proba(x)[:, 1][0])
    reverse = float(model.predict_proba(-x)[:, 1][0])
    return float(np.clip(0.5 * (forward + (1.0 - reverse)), 0.0, 1.0))


def predict_post_toss(
    request: PostTossRequest,
    *,
    raw_dir: str | Path,
    model_path: str | Path,
) -> PostTossResult:
    history = build_historical_state(raw_dir, cutoff=request.match_date)
    frame = _request_frame(history, request)
    bundle = joblib.load(model_path)
    p1 = _probability(bundle, frame)
    p2 = 1.0 - p1

    team1 = canonicalize_team(request.team1)
    team2 = canonicalize_team(request.team2)
    winner = team1 if p1 >= p2 else team2

    both_xi = bool(request.team1_xi and request.team2_xi)
    one_xi = bool(request.team1_xi or request.team2_xi)
    lineup_mode = "both_xi_known" if both_xi else ("partial_xi" if one_xi else "xi_unknown")

    warnings: list[str] = []
    if not both_xi:
        warnings.append(
            "Both playing XIs were not supplied; XI-strength features may be partially or fully unavailable."
        )
    if request.venue is None:
        warnings.append("No venue supplied; venue features are unavailable.")
    if history.parse_errors:
        warnings.append(f"{history.parse_errors} historical files were skipped after parse errors.")

    return PostTossResult(
        team1=team1,
        team2=team2,
        team1_probability=p1,
        team2_probability=p2,
        winner=winner,
        edge=prediction_edge(max(p1, p2)),
        model_name=str(bundle["model_name"]),
        lineup_mode=lineup_mode,
        warnings=tuple(warnings),
    )
