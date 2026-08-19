"""Chat orchestration. Translates structured (or simple) requests into router calls."""

from __future__ import annotations

from pydantic import ValidationError

from cricmaster.api.schemas import ChatRequest, ChatResponse, PredictRequest
from cricmaster.api.service import PredictionService, prediction_to_response
from cricmaster.chatbot.formatter import (
    capabilities_message,
    explain_prediction,
    format_chat_prediction,
    help_message,
    suggestions_for,
)
from cricmaster.chatbot.intents import ChatIntent, parse_intent_name
from cricmaster.chatbot.parser import (
    extract_competition,
    extract_date,
    extract_format,
    extract_mode,
    extract_teams,
    infer_intent_from_message,
)
from cricmaster.prediction.errors import ArtifactValidationError, UnsupportedPredictionError


class ChatService:
    def __init__(self, prediction: PredictionService) -> None:
        self._prediction = prediction

    def handle(self, request: ChatRequest) -> ChatResponse:
        intent = parse_intent_name(request.intent)
        if intent is None and request.message:
            intent = infer_intent_from_message(request.message)
        if intent is None:
            intent = ChatIntent.UNKNOWN

        if intent is ChatIntent.HELP:
            return ChatResponse(
                intent=intent.value,
                message=help_message(),
                suggestions=suggestions_for(intent.value, None),
            )
        if intent is ChatIntent.MODEL_CAPABILITIES:
            return ChatResponse(
                intent=intent.value,
                message=capabilities_message(),
                suggestions=suggestions_for(intent.value, None),
            )
        if intent is ChatIntent.UNKNOWN:
            return ChatResponse(
                intent=intent.value,
                message=(
                    "I could not determine a Cricmaster intent. "
                    + help_message()
                ),
                suggestions=suggestions_for("help", None),
            )

        predict_request = self._build_predict_request(request, intent)
        if predict_request is None:
            return ChatResponse(
                intent=intent.value,
                message=(
                    "I need a structured match: team1, team2, format (T20I or T20), "
                    "mode, and date. I will not invent missing live scores or lineups."
                ),
                suggestions=["Send intent predict_match with team1, team2, format, mode, date"],
            )

        try:
            result = self._prediction.predict(predict_request)
        except UnsupportedPredictionError as exc:
            return ChatResponse(
                intent=intent.value,
                message=str(exc),
                warnings=[str(exc)],
            )
        except ArtifactValidationError:
            return ChatResponse(
                intent=intent.value,
                message="A required production model is not available.",
            )
        payload = prediction_to_response(result)
        if intent is ChatIntent.EXPLAIN_PREDICTION:
            message = explain_prediction(result)
        else:
            message = format_chat_prediction(result)
        return ChatResponse(
            intent=intent.value,
            message=message,
            prediction=payload,
            suggestions=suggestions_for(intent.value, result),
            warnings=list(result.warnings),
        )

    def _build_predict_request(
        self,
        request: ChatRequest,
        intent: ChatIntent,
    ) -> PredictRequest | None:
        team1 = request.team1
        team2 = request.team2
        match_format = request.format
        mode = request.mode
        match_date = request.date
        competition = request.competition

        if request.message:
            if team1 is None or team2 is None:
                extracted = extract_teams(request.message)
                if extracted:
                    team1 = team1 or extracted[0]
                    team2 = team2 or extracted[1]
            match_format = match_format or extract_format(request.message)
            mode = mode or extract_mode(request.message)
            match_date = match_date or extract_date(request.message)
            competition = competition or extract_competition(request.message)

        if intent is ChatIntent.LIVE_PREDICTION:
            mode = mode or "LIVE"
        elif intent in {ChatIntent.PREDICT_MATCH, ChatIntent.EXPLAIN_PREDICTION}:
            mode = mode or "PRE_TOSS"

        if team1 is None or team2 is None or match_format is None or match_date is None:
            return None
        if mode is None:
            return None

        try:
            return PredictRequest(
                team1=team1,
                team2=team2,
                format=match_format,
                mode=mode,
                date=match_date,
                gender=request.gender,
                venue=request.venue,
                competition=competition,
                toss_winner=request.toss_winner,
                toss_decision=request.toss_decision,
                team1_xi=request.team1_xi,
                team2_xi=request.team2_xi,
                batting_team=request.batting_team,
                innings=request.innings,
                runs=request.runs,
                wickets=request.wickets,
                overs=request.overs,
                legal_balls=request.legal_balls,
                target=request.target,
            )
        except ValidationError:
            return None
