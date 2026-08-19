"""FastAPI application for the Cricmaster production backend."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cricmaster.api.errors import ApiError
from cricmaster.api.middleware import REQUEST_ID_HEADER, RequestIdMiddleware, request_id_from
from cricmaster.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    LiveMatchOut,
    PredictRequest,
    PredictionResponse,
)
from cricmaster.api.service import (
    LiveMatchService,
    PredictionService,
    build_services,
    prediction_to_response,
)
from cricmaster.chatbot.service import ChatService
from cricmaster.config import Settings, load_settings
from cricmaster.prediction.errors import ArtifactValidationError, UnsupportedPredictionError


def _error_body(request: Request, code: str, message: str) -> dict[str, str]:
    return {
        "error": code,
        "message": message,
        "request_id": request_id_from(request),
    }


def _unsupported_code(exc: UnsupportedPredictionError) -> str:
    text = str(exc).lower()
    if "format" in text or "hundred" in text:
        return "unsupported_format"
    if "mode" in text:
        return "unsupported_mode"
    return "unsupported_prediction"


def _artifact_response(request: Request, exc: ArtifactValidationError) -> JSONResponse:
    text = str(exc).lower()
    if "not found" in text:
        return JSONResponse(
            status_code=503,
            content=_error_body(
                request,
                "model_unavailable",
                "A required production model artifact is not available.",
            ),
        )
    return JSONResponse(
        status_code=500,
        content=_error_body(
            request,
            "invalid_model_artifact",
            "The production model artifact is invalid or incompatible with this request.",
        ),
    )


def create_app(
    settings: Settings | None = None,
    *,
    prediction_service: PredictionService | None = None,
    live_service: LiveMatchService | None = None,
    chat_service: ChatService | None = None,
) -> FastAPI:
    loaded = settings or load_settings()
    prediction, live = build_services(
        loaded,
        prediction_service=prediction_service,
        live_service=live_service,
    )
    chat = chat_service or ChatService(prediction)

    application = FastAPI(
        title="Cricmaster API",
        version="0.1.0",
        summary="Production prediction and chatbot backend for Cricmaster.",
        description=(
            "Probabilities are statistical estimates, not guarantees. "
            "v1 supports T20I and franchise T20 only."
        ),
    )
    application.state.settings = loaded
    application.state.prediction_service = prediction
    application.state.live_service = live
    application.state.chat_service = chat

    origins = list(loaded.cors_origins)
    allow_wildcard = origins == ["*"]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_wildcard else origins,
        allow_credentials=not allow_wildcard,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
    )
    application.add_middleware(RequestIdMiddleware)

    @application.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, exc.code, exc.message),
        )

    @application.exception_handler(UnsupportedPredictionError)
    async def unsupported_handler(
        request: Request,
        exc: UnsupportedPredictionError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_body(request, _unsupported_code(exc), str(exc)),
        )

    @application.exception_handler(ArtifactValidationError)
    async def artifact_handler(
        request: Request,
        exc: ArtifactValidationError,
    ) -> JSONResponse:
        return _artifact_response(request, exc)

    @application.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        message = "Request validation failed."
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(part) for part in first.get("loc", []) if part != "body")
            detail = first.get("msg", message)
            message = f"{loc}: {detail}" if loc else str(detail)
        return JSONResponse(
            status_code=422,
            content=_error_body(request, "validation_error", message),
        )

    @application.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        if isinstance(exc, UnsupportedPredictionError):
            return JSONResponse(
                status_code=400,
                content=_error_body(request, _unsupported_code(exc), str(exc)),
            )
        if isinstance(exc, ArtifactValidationError):
            return _artifact_response(request, exc)
        if isinstance(exc, ApiError):
            return JSONResponse(
                status_code=exc.status_code,
                content=_error_body(request, exc.code, exc.message),
            )
        return JSONResponse(
            status_code=400,
            content=_error_body(request, "invalid_request", str(exc)),
        )

    @application.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        return prediction.health()

    @application.get("/models")
    def models() -> dict[str, Any]:
        return prediction.models()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictRequest) -> PredictionResponse:
        result = prediction.predict(payload)
        return prediction_to_response(result)

    @application.post("/chat", response_model=ChatResponse)
    def chat_endpoint(payload: ChatRequest) -> ChatResponse:
        return chat.handle(payload)

    @application.get("/live/matches", response_model=list[LiveMatchOut])
    def live_matches() -> list[LiveMatchOut]:
        return live.list_matches()

    @application.post("/live/{match_id}/predict", response_model=PredictionResponse)
    def predict_live_match(match_id: str) -> PredictionResponse:
        production_request = live.live_prediction_request(match_id)
        result = prediction.predict_production(production_request)
        return prediction_to_response(result)

    return application


app = create_app()
