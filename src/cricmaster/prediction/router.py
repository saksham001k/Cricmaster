"""Production prediction router for T20I, franchise T20, and live modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from cricmaster.data.formats import MatchFormat, normalize_competition, normalize_match_type
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.history import HistoricalState
from cricmaster.prediction.artifacts import (
    ProductionArtifacts,
    ProductionRoute,
    load_production_bundle,
    parse_prediction_mode,
    resolve_production_route,
)
from cricmaster.prediction.confidence import (
    ConfidenceInputs,
    applicable_confidence_warnings,
    assess_confidence,
)
from cricmaster.prediction.errors import UnsupportedPredictionError
from cricmaster.prediction.history_cache import (
    load_parsed_t20_matches,
    matches_strictly_before,
)
from cricmaster.prediction.live import (
    LivePredictionRequest,
    LivePredictionResult,
    predict_live,
)
from cricmaster.prediction.posttoss import PostTossRequest, posttoss_feature_frame
from cricmaster.prediction.prematch import (
    Driver,
    HistoryBuild,
    PredictionRequest,
    logistic_drivers,
    request_features,
    symmetric_probability,
)
from cricmaster.prediction.result import (
    ProductionPredictionResult,
    complementary_probabilities,
    predicted_team_from_probabilities,
    prediction_edge,
)
from cricmaster.prediction.roster_runtime import (
    RosterRuntime,
    append_previous_xi_core_features,
    features_for_bundle,
    previous_xi_core_differences,
)


@dataclass(frozen=True)
class ProductionRequest:
    team1: str
    team2: str
    match_format: MatchFormat
    prediction_mode: str
    match_date: date
    gender: str = "male"
    venue: str | None = None
    competition: str | None = None
    toss_winner: str | None = None
    toss_decision: str | None = None
    team1_xi: tuple[str, ...] = ()
    team2_xi: tuple[str, ...] = ()
    batting_team: str | None = None
    innings_number: int | None = None
    runs: int | None = None
    wickets: int | None = None
    legal_balls: int | None = None
    target: int | None = None


@dataclass(frozen=True)
class ProductionHistory:
    history: HistoryBuild
    roster: RosterRuntime


def parse_production_format(
    value: str,
    *,
    competition: str | None = None,
) -> MatchFormat:
    """Parse T20I/T20 and reject Hundred and other unsupported formats."""

    text = (value or "").strip()
    competition_name = normalize_competition(competition)
    if competition_name == "The Hundred":
        raise UnsupportedPredictionError(
            "The Hundred is not a T20 format and is not supported by the "
            "production prediction router."
        )

    match_format: MatchFormat | None = None
    try:
        match_format = MatchFormat(text.upper())
    except ValueError:
        mapped = normalize_match_type(text, competition=competition)
        if mapped is not MatchFormat.OTHER:
            match_format = mapped

    if match_format is None:
        raise UnsupportedPredictionError(
            f"Unsupported format {value!r}. Production routing supports T20I and T20 only."
        )

    if match_format is MatchFormat.HUNDRED:
        raise UnsupportedPredictionError(
            "The Hundred is not a T20 format and is not supported by the "
            "production prediction router."
        )
    if match_format not in {MatchFormat.T20I, MatchFormat.T20}:
        raise UnsupportedPredictionError(
            f"Unsupported format {match_format.value}. "
            "Production routing supports T20I and T20 only. "
            "Hundred, T10, ODI, and Test are not routed."
        )
    return match_format


def to_prematch_request(request: ProductionRequest) -> PredictionRequest:
    """PRE_TOSS request with no current XI fields."""

    return PredictionRequest(
        team1=request.team1,
        team2=request.team2,
        match_format=request.match_format,
        match_date=request.match_date,
        gender=request.gender,
        venue=request.venue,
        competition=request.competition,
    )


def to_posttoss_request(request: ProductionRequest) -> PostTossRequest:
    if not request.toss_winner or not request.toss_decision:
        raise UnsupportedPredictionError(
            "POST_TOSS predictions require toss_winner and toss_decision."
        )
    return PostTossRequest(
        team1=request.team1,
        team2=request.team2,
        match_format=request.match_format,
        match_date=request.match_date,
        gender=request.gender,
        venue=request.venue,
        competition=request.competition,
        toss_winner=request.toss_winner,
        toss_decision=request.toss_decision,
        team1_xi=request.team1_xi,
        team2_xi=request.team2_xi,
    )


def to_live_request(request: ProductionRequest) -> LivePredictionRequest:
    if request.batting_team is None:
        raise UnsupportedPredictionError("LIVE predictions require batting_team.")
    if request.innings_number not in {1, 2}:
        raise UnsupportedPredictionError("LIVE predictions require innings_number 1 or 2.")
    if request.runs is None or request.wickets is None or request.legal_balls is None:
        raise UnsupportedPredictionError(
            "LIVE predictions require runs, wickets, and legal_balls."
        )
    return LivePredictionRequest(
        team1=request.team1,
        team2=request.team2,
        batting_team=request.batting_team,
        match_format=request.match_format,
        match_date=request.match_date,
        gender=request.gender,
        innings_number=request.innings_number,
        runs=request.runs,
        wickets=request.wickets,
        legal_balls=request.legal_balls,
        target=request.target,
        venue=request.venue,
        competition=request.competition,
        toss_winner=request.toss_winner,
        toss_decision=request.toss_decision,
        team1_xi=request.team1_xi,
        team2_xi=request.team2_xi,
    )


def build_production_history(
    raw_dir: str | Path,
    *,
    cutoff: date,
) -> ProductionHistory:
    parsed = load_parsed_t20_matches(raw_dir)
    state = HistoricalState()
    roster = RosterRuntime()
    selected = matches_strictly_before(parsed, cutoff)
    for match in selected:
        state.update(match)
        roster.observe_completed_match(match)
    return ProductionHistory(
        history=HistoryBuild(
            state=state,
            matches_applied=len(selected),
            parse_errors=parsed.parse_errors,
        ),
        roster=roster,
    )


def build_pre_toss_feature_frame(
    production: ProductionHistory,
    request: ProductionRequest,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], bool, bool]:
    """PRE_TOSS features. Current XI is never consulted."""

    pre_request = to_prematch_request(request)
    frame, team1_history, team2_history = request_features(
        production.history.state,
        pre_request,
    )
    known1 = True
    known2 = True
    if request.match_format is MatchFormat.T20:
        roster_diffs, known1, known2 = previous_xi_core_differences(
            production.roster,
            match_format=request.match_format,
            gender=request.gender,
            team1=canonicalize_team(request.team1),
            team2=canonicalize_team(request.team2),
        )
        frame = append_previous_xi_core_features(frame, roster_diffs)
    return (
        features_for_bundle(frame, bundle),
        team1_history,
        team2_history,
        known1,
        known2,
    )


def build_post_toss_feature_frame(
    production: ProductionHistory,
    request: ProductionRequest,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, bool, bool]:
    post_request = to_posttoss_request(request)
    frame = posttoss_feature_frame(production.history, post_request)
    known1 = True
    known2 = True
    if request.match_format is MatchFormat.T20:
        roster_diffs, known1, known2 = previous_xi_core_differences(
            production.roster,
            match_format=request.match_format,
            gender=request.gender,
            team1=canonicalize_team(request.team1),
            team2=canonicalize_team(request.team2),
        )
        frame = append_previous_xi_core_features(frame, roster_diffs)
    return features_for_bundle(frame, bundle), known1, known2


def _history_counts(
    state: HistoricalState,
    request: ProductionRequest,
    team1: str,
    team2: str,
) -> tuple[int, int]:
    snap1 = state.form.snapshot(request.match_format, team1, gender=request.gender)
    snap2 = state.form.snapshot(request.match_format, team2, gender=request.gender)
    return int(snap1.get("matches_before") or 0), int(snap2.get("matches_before") or 0)


def _lineup_mode(request: ProductionRequest) -> str | None:
    mode = parse_prediction_mode(request.prediction_mode)
    if mode not in {"POST_TOSS", "LIVE"}:
        return None
    both = bool(request.team1_xi and request.team2_xi)
    one = bool(request.team1_xi or request.team2_xi)
    if both:
        return "both_xi_known"
    if one:
        return "partial_xi"
    return "xi_unknown"


def _assemble_result(
    *,
    request: ProductionRequest,
    route: ProductionRoute,
    bundle: dict[str, Any],
    team1: str,
    team2: str,
    p1: float,
    production: ProductionHistory,
    team1_matches: int,
    team2_matches: int,
    drivers: tuple[Any, ...],
    extra_warnings: tuple[str, ...],
    previous_xi_known: tuple[bool, bool] | None,
    ignored_current_xi: bool,
    innings_number: int | None = None,
    model_kind: str | None = None,
    terminal: bool = False,
    legal_balls: int | None = None,
) -> ProductionPredictionResult:
    p1, p2 = complementary_probabilities(p1)
    predicted = predicted_team_from_probabilities(team1, team2, p1, p2)
    winner_p = max(p1, p2)
    previous_complete: bool | None = None
    if previous_xi_known is not None:
        previous_complete = previous_xi_known[0] and previous_xi_known[1]
    lineup_mode = _lineup_mode(request)
    lineup_complete: bool | None = None
    if lineup_mode is not None:
        lineup_complete = lineup_mode == "both_xi_known"

    assessment = assess_confidence(
        ConfidenceInputs(
            winner_probability=winner_p,
            match_format=request.match_format,
            prediction_mode=route.prediction_mode,
            team1_history_matches=team1_matches,
            team2_history_matches=team2_matches,
            venue_known=request.venue is not None,
            previous_xi_complete=previous_complete,
            lineup_complete=lineup_complete,
            parse_errors=production.history.parse_errors,
            terminal=terminal,
            innings_number=innings_number,
            legal_balls=legal_balls,
        )
    )

    warnings: list[str] = []
    warnings.extend(
        applicable_confidence_warnings(
            assessment=assessment,
            match_format=request.match_format,
            prediction_mode=route.prediction_mode,
            winner_probability=winner_p,
            previous_xi_complete=previous_complete,
            ignored_current_xi=ignored_current_xi,
        )
    )
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
    if request.venue is None and route.prediction_mode != "LIVE":
        warnings.append("No venue supplied; venue-specific features are unavailable.")
    if production.history.parse_errors:
        warnings.append(
            f"{production.history.parse_errors} historical files could not be parsed "
            "and were skipped."
        )
    if lineup_mode in {"partial_xi", "xi_unknown"} and route.prediction_mode in {
        "POST_TOSS",
        "LIVE",
    }:
        warnings.append(
            "Both playing XIs were not supplied; XI-strength features may be "
            "partially or fully unavailable."
        )
    warnings.extend(extra_warnings)

    seen: set[str] = set()
    unique_warnings: list[str] = []
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            unique_warnings.append(warning)

    known1, known2 = (None, None)
    if previous_xi_known is not None:
        known1, known2 = previous_xi_known

    return ProductionPredictionResult(
        team1=team1,
        team2=team2,
        team1_probability=p1,
        team2_probability=p2,
        predicted_team=predicted,
        edge=prediction_edge(winner_p),
        confidence=str(assessment.label),
        prediction_mode=route.prediction_mode,
        match_format=request.match_format.value,
        model_name=str(bundle["model_name"]),
        model_family=route.model_family,
        warnings=tuple(unique_warnings),
        drivers=drivers,
        matches_applied=production.history.matches_applied,
        team1_history_matches=team1_matches,
        team2_history_matches=team2_matches,
        historical_sample={
            "matches_applied": production.history.matches_applied,
            "team1_matches_before": team1_matches,
            "team2_matches_before": team2_matches,
            "parse_errors": production.history.parse_errors,
            "min_team_matches_before": min(team1_matches, team2_matches),
        },
        competition=normalize_competition(request.competition),
        venue=request.venue,
        lineup_mode=lineup_mode,
        previous_xi_team1_known=known1,
        previous_xi_team2_known=known2,
        innings_number=innings_number,
        model_kind=model_kind or route.live_kind,
        terminal=terminal if route.prediction_mode == "LIVE" else None,
    )


def _predict_pre_or_post(
    request: ProductionRequest,
    *,
    raw_dir: str | Path,
    artifacts: ProductionArtifacts,
) -> ProductionPredictionResult:
    route = resolve_production_route(
        request.match_format,
        request.prediction_mode,
        artifacts=artifacts,
    )
    bundle = load_production_bundle(route)
    production = build_production_history(raw_dir, cutoff=request.match_date)
    team1 = canonicalize_team(request.team1)
    team2 = canonicalize_team(request.team2)
    ignored_current_xi = False
    previous_known: tuple[bool, bool] | None = None

    if route.prediction_mode == "PRE_TOSS":
        ignored_current_xi = bool(request.team1_xi or request.team2_xi)
        frame, team1_history, team2_history, known1, known2 = build_pre_toss_feature_frame(
            production,
            request,
            bundle,
        )
        team1_matches = int(team1_history.get("matches_before") or 0)
        team2_matches = int(team2_history.get("matches_before") or 0)
        if request.match_format is MatchFormat.T20:
            previous_known = (known1, known2)
    else:
        frame, known1, known2 = build_post_toss_feature_frame(
            production,
            request,
            bundle,
        )
        team1_matches, team2_matches = _history_counts(
            production.history.state, request, team1, team2
        )
        if request.match_format is MatchFormat.T20:
            previous_known = (known1, known2)

    p1 = symmetric_probability(bundle, frame)
    drivers = logistic_drivers(
        bundle,
        frame,
        team1=team1,
        team2=team2,
    )
    return _assemble_result(
        request=request,
        route=route,
        bundle=bundle,
        team1=team1,
        team2=team2,
        p1=p1,
        production=production,
        team1_matches=team1_matches,
        team2_matches=team2_matches,
        drivers=drivers,
        extra_warnings=(),
        previous_xi_known=previous_known,
        ignored_current_xi=ignored_current_xi,
    )


def _from_live_result(
    request: ProductionRequest,
    route: ProductionRoute,
    bundle: dict[str, Any],
    live: LivePredictionResult,
    production: ProductionHistory,
) -> ProductionPredictionResult:
    team1 = canonicalize_team(request.team1)
    team2 = canonicalize_team(request.team2)
    if live.batting_team == team1:
        p1 = live.batting_probability
    else:
        p1 = live.bowling_probability

    extra = list(live.warnings)
    if live.terminal:
        extra.append("Terminal live state used instead of the in-play model.")

    drivers = tuple(
        Driver(
            feature=item.feature,
            label=item.label,
            raw_difference=item.raw_value,
            contribution=item.contribution,
            supports=item.supports,
        )
        for item in live.drivers
    )
    team1_matches, team2_matches = _history_counts(
        production.history.state, request, team1, team2
    )
    return _assemble_result(
        request=request,
        route=route,
        bundle=bundle,
        team1=team1,
        team2=team2,
        p1=p1,
        production=production,
        team1_matches=team1_matches,
        team2_matches=team2_matches,
        drivers=drivers,
        extra_warnings=tuple(extra),
        previous_xi_known=None,
        ignored_current_xi=False,
        innings_number=live.innings_number,
        model_kind=live.model_kind,
        terminal=live.terminal,
        legal_balls=request.legal_balls,
    )


def predict_production(
    request: ProductionRequest,
    *,
    raw_dir: str | Path,
    artifacts: ProductionArtifacts | None = None,
) -> ProductionPredictionResult:
    """Route a match to the correct frozen production model and predict."""

    parse_production_format(
        request.match_format.value,
        competition=request.competition,
    )
    mode = parse_prediction_mode(request.prediction_mode)
    request = replace(request, prediction_mode=mode)
    catalog = artifacts or ProductionArtifacts()

    if mode == "LIVE":
        route = resolve_production_route(
            request.match_format,
            mode,
            artifacts=catalog,
            innings_number=request.innings_number,
        )
        bundle = load_production_bundle(route)
        live_request = to_live_request(request)
        production = build_production_history(raw_dir, cutoff=request.match_date)
        live = predict_live(
            live_request,
            raw_dir=raw_dir,
            first_innings_model=catalog.live_first_innings,
            chase_model=catalog.live_chase,
        )
        return _from_live_result(request, route, bundle, live, production)

    return _predict_pre_or_post(request, raw_dir=raw_dir, artifacts=catalog)
