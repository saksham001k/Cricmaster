"""Reusable Cricmaster services. HTTP handlers should stay thin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from cricmaster import __version__
from cricmaster.api.errors import (
    InsufficientLiveStateError,
    LiveProviderError,
    MatchNotFoundError,
)
from cricmaster.api.schemas import (
    DriverOut,
    LiveMatchOut,
    PredictRequest,
    PredictionResponse,
)
from cricmaster.config import PROJECT_ROOT, Settings, load_settings
from cricmaster.live.automatic import to_live_prediction_request
from cricmaster.live.cricketdata import CricketDataMatch, CricketDataProvider
from cricmaster.prediction.artifacts import (
    ProductionArtifacts,
    artifact_availability,
    resolved_production_artifacts,
)
from cricmaster.prediction.result import ProductionPredictionResult
from cricmaster.prediction.router import ProductionRequest, predict_production


TRAINED_THROUGH = "2024-12-31"

MODEL_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "T20I": {
        "PRE_TOSS": {
            "family": "international T20I",
            "trained_through": TRAINED_THROUGH,
            "limitations": [
                "Probabilities are statistical estimates, not guarantees.",
                "Requires historical T20I coverage in the configured corpus.",
            ],
        },
        "POST_TOSS": {
            "family": "international T20I",
            "trained_through": TRAINED_THROUGH,
            "limitations": [
                "Playing XI features are used only when both lineups are supplied.",
                "Probabilities are statistical estimates, not guarantees.",
            ],
        },
        "LIVE": {
            "family": "live first innings / chase",
            "trained_through": TRAINED_THROUGH,
            "limitations": [
                "Live models share a T20/T20I architecture with an is_t20i flag.",
                "Chase predictions are refused when the target cannot be derived.",
            ],
        },
    },
    "T20": {
        "PRE_TOSS": {
            "family": "roster-aware T20",
            "trained_through": TRAINED_THROUGH,
            "limitations": [
                "Franchise PRE_TOSS discriminatory power is modest.",
                "Current playing XI is never used before the toss.",
                "Previous-XI roster features are leakage-safe historical information only.",
            ],
        },
        "POST_TOSS": {
            "family": "roster-aware T20",
            "trained_through": TRAINED_THROUGH,
            "limitations": [
                "Franchise POST_TOSS discriminatory power is limited.",
                "Frozen roster family uses previous XI, not newly invented current-XI cores.",
            ],
        },
        "LIVE": {
            "family": "live first innings / chase",
            "trained_through": TRAINED_THROUGH,
            "limitations": [
                "Live scoreboard signal can dominate weak franchise history.",
                "The Hundred is not treated as T20.",
            ],
        },
    },
}


def prediction_to_response(result: ProductionPredictionResult) -> PredictionResponse:
    return PredictionResponse(
        team1=result.team1,
        team2=result.team2,
        team1_probability=result.team1_probability,
        team2_probability=result.team2_probability,
        predicted_team=result.predicted_team,
        edge=result.edge,
        confidence=result.confidence,
        prediction_mode=result.prediction_mode,
        format=result.match_format,
        model_name=result.model_name,
        model_family=result.model_family,
        warnings=list(result.warnings),
        drivers=[
            DriverOut(
                feature=item.feature,
                label=item.label,
                raw_difference=item.raw_difference,
                contribution=item.contribution,
                supports=item.supports,
            )
            for item in result.drivers
        ],
        matches_applied=result.matches_applied,
        team1_history_matches=result.team1_history_matches,
        team2_history_matches=result.team2_history_matches,
        historical_sample=dict(result.historical_sample),
        competition=result.competition,
        venue=result.venue,
        lineup_mode=result.lineup_mode,
        previous_xi_team1_known=result.previous_xi_team1_known,
        previous_xi_team2_known=result.previous_xi_team2_known,
        innings_number=result.innings_number,
        model_kind=result.model_kind,
        terminal=result.terminal,
    )


class PredictionService:
    """Adapter over the frozen production prediction router."""

    def __init__(
        self,
        *,
        raw_dir: str | Path,
        artifacts: ProductionArtifacts | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.artifacts = artifacts or resolved_production_artifacts(PROJECT_ROOT)

    def predict(self, request: PredictRequest) -> ProductionPredictionResult:
        production = request.to_production_request()
        return predict_production(
            production,
            raw_dir=self.raw_dir,
            artifacts=self.artifacts,
        )

    def predict_production(
        self,
        request: ProductionRequest,
    ) -> ProductionPredictionResult:
        return predict_production(
            request,
            raw_dir=self.raw_dir,
            artifacts=self.artifacts,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "cricmaster",
            "version": __version__,
            "models": artifact_availability(self.artifacts),
        }

    def models(self) -> dict[str, Any]:
        available = artifact_availability(self.artifacts)
        payload: dict[str, Any] = {
            "supported_formats": ["T20I", "T20"],
            "unsupported_formats": ["HUNDRED", "ODI", "TEST", "T10", "LIST_A", "FIRST_CLASS"],
            "supported_modes": ["PRE_TOSS", "POST_TOSS", "LIVE"],
            "availability": available,
            "domains": {},
        }
        for domain, modes in MODEL_CATALOG.items():
            payload["domains"][domain] = {}
            for mode, meta in modes.items():
                key = {
                    ("T20I", "PRE_TOSS"): "t20i_pretoss",
                    ("T20I", "POST_TOSS"): "t20i_posttoss",
                    ("T20", "PRE_TOSS"): "t20_roster_pretoss",
                    ("T20", "POST_TOSS"): "t20_roster_posttoss",
                    ("T20I", "LIVE"): "live_first_innings",
                    ("T20", "LIVE"): "live_first_innings",
                }.get((domain, mode))
                live_ok = available["live_first_innings"] and available["live_chase"]
                artifact_ok = (
                    live_ok if mode == "LIVE" else bool(key and available.get(key, False))
                )
                payload["domains"][domain][mode] = {
                    **meta,
                    "artifact_available": artifact_ok,
                }
        return payload


class LiveMatchService:
    """Normalized current-match feed. Does not guess missing score fields."""

    def __init__(self, provider: Any | None = None) -> None:
        self._provider = provider if provider is not None else CricketDataProvider()

    @property
    def source_name(self) -> str:
        return getattr(self._provider, "name", "cricketdata")

    def current_matches(self) -> list[CricketDataMatch]:
        try:
            return self._provider.current_matches()
        except RuntimeError as exc:
            text = str(exc)
            if "CRICKET_API_KEY" in text or "not configured" in text.lower():
                raise LiveProviderError(
                    "live_provider_unconfigured",
                    "Live cricket data is not configured on this server.",
                ) from None
            raise LiveProviderError(
                "live_provider_failure",
                "The live cricket provider could not be queried.",
            ) from None
        except requests.RequestException:
            raise LiveProviderError(
                "live_provider_failure",
                "The live cricket provider could not be queried.",
            ) from None

    def list_matches(self) -> list[LiveMatchOut]:
        matches = self.current_matches()
        return [
            LiveMatchOut(
                match_id=item.match_id,
                name=item.name,
                format=item.match_format.value,
                status=item.status,
                teams=list(item.teams),
                source=self.source_name,
                venue=item.venue,
                competition=item.competition,
                date=item.match_date,
                predictable_live=item.predictable_live,
                warnings=list(item.warnings),
            )
            for item in matches
        ]

    def get_match(self, match_id: str) -> CricketDataMatch:
        cleaned = match_id.strip()
        if not cleaned or len(cleaned) > 80:
            raise InsufficientLiveStateError("match_id is invalid")
        matches = self.current_matches()
        found = next((item for item in matches if item.match_id == cleaned), None)
        if found is None:
            raise MatchNotFoundError()
        return found

    def live_prediction_request(self, match_id: str) -> ProductionRequest:
        match = self.get_match(match_id)
        try:
            live = to_live_prediction_request(match)
        except ValueError as exc:
            raise InsufficientLiveStateError(str(exc)) from None
        return ProductionRequest(
            team1=live.team1,
            team2=live.team2,
            match_format=live.match_format,
            prediction_mode="LIVE",
            match_date=live.match_date,
            gender=live.gender,
            venue=live.venue,
            competition=live.competition,
            toss_winner=live.toss_winner,
            toss_decision=live.toss_decision,
            team1_xi=live.team1_xi,
            team2_xi=live.team2_xi,
            batting_team=live.batting_team,
            innings_number=live.innings_number,
            runs=live.runs,
            wickets=live.wickets,
            legal_balls=live.legal_balls,
            target=live.target,
        )


def build_services(
    settings: Settings | None = None,
    *,
    prediction_service: PredictionService | None = None,
    live_service: LiveMatchService | None = None,
) -> tuple[PredictionService, LiveMatchService]:
    loaded = settings or load_settings()
    prediction = prediction_service or PredictionService(
        raw_dir=loaded.t20_corpus_dir,
        artifacts=resolved_production_artifacts(PROJECT_ROOT),
    )
    live = live_service or LiveMatchService()
    return prediction, live
