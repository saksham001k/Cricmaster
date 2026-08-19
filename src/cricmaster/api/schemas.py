"""Pydantic request and response schemas for the Cricmaster API."""

from __future__ import annotations

from datetime import date as Date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cricmaster.prediction.artifacts import parse_prediction_mode
from cricmaster.prediction.live import parse_cricket_overs
from cricmaster.prediction.router import ProductionRequest, parse_production_format

MAX_NAME_LEN = 80
MAX_TEXT_LEN = 120
MAX_MESSAGE_LEN = 500
MAX_XI = 11


def _clean_name(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError("value cannot be empty")
    if len(cleaned) > MAX_NAME_LEN:
        raise ValueError(f"value must be at most {MAX_NAME_LEN} characters")
    return cleaned


class DriverOut(BaseModel):
    feature: str
    label: str
    raw_difference: float | None
    contribution: float
    supports: str


class HistoricalSampleOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    matches_applied: int | None = None
    team1_matches_before: int | None = None
    team2_matches_before: int | None = None
    parse_errors: int | None = None
    min_team_matches_before: int | None = None


class PredictRequest(BaseModel):
    team1: str
    team2: str
    format: str
    mode: str
    date: Date
    gender: Literal["male", "female"] = "male"
    venue: str | None = None
    competition: str | None = None
    toss_winner: str | None = None
    toss_decision: Literal["bat", "field", "bowl"] | None = None
    team1_xi: list[str] = Field(default_factory=list)
    team2_xi: list[str] = Field(default_factory=list)
    batting_team: str | None = None
    innings: int | None = Field(default=None, ge=1, le=2)
    runs: int | None = Field(default=None, ge=0, le=500)
    wickets: int | None = Field(default=None, ge=0, le=10)
    overs: str | None = Field(default=None, max_length=8)
    legal_balls: int | None = Field(default=None, ge=0, le=120)
    target: int | None = Field(default=None, ge=1, le=500)

    @field_validator("team1", "team2")
    @classmethod
    def _teams(cls, value: str) -> str:
        return _clean_name(value)

    @field_validator("venue", "competition", "toss_winner", "batting_team")
    @classmethod
    def _optional_names(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_name(value)

    @field_validator("team1_xi", "team2_xi")
    @classmethod
    def _xi(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_XI:
            raise ValueError(f"playing XI cannot contain more than {MAX_XI} players")
        return [_clean_name(item) for item in value]

    @field_validator("format")
    @classmethod
    def _format(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 20:
            raise ValueError("format is invalid")
        return cleaned

    @field_validator("mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        return parse_prediction_mode(value)

    @model_validator(mode="after")
    def _mode_specific(self) -> PredictRequest:
        if self.team1.casefold() == self.team2.casefold():
            raise ValueError("team1 and team2 must be different")

        if self.mode == "POST_TOSS":
            if not self.toss_winner or not self.toss_decision:
                raise ValueError(
                    "POST_TOSS requests require toss_winner and toss_decision"
                )

        if self.mode == "LIVE":
            if self.batting_team is None:
                raise ValueError("LIVE requests require batting_team")
            if self.innings is None:
                raise ValueError("LIVE requests require innings (1 or 2)")
            if self.runs is None or self.wickets is None:
                raise ValueError("LIVE requests require runs and wickets")
            if self.overs is None and self.legal_balls is None:
                raise ValueError("LIVE requests require overs or legal_balls")
            if self.innings == 2 and self.target is None:
                raise ValueError("LIVE innings 2 requests require target")
            if self.innings == 1 and self.target is not None:
                raise ValueError("target must be omitted for innings 1")
            if self.overs is not None:
                parse_cricket_overs(self.overs, balls_per_over=6)
        return self

    def legal_balls_value(self) -> int | None:
        if self.overs is not None:
            return parse_cricket_overs(self.overs, balls_per_over=6)
        return self.legal_balls

    def to_production_request(self) -> ProductionRequest:
        match_format = parse_production_format(
            self.format,
            competition=self.competition,
        )
        return ProductionRequest(
            team1=self.team1,
            team2=self.team2,
            match_format=match_format,
            prediction_mode=self.mode,
            match_date=self.date,
            gender=self.gender,
            venue=self.venue,
            competition=self.competition,
            toss_winner=self.toss_winner,
            toss_decision=self.toss_decision,
            team1_xi=tuple(self.team1_xi),
            team2_xi=tuple(self.team2_xi),
            batting_team=self.batting_team,
            innings_number=self.innings,
            runs=self.runs,
            wickets=self.wickets,
            legal_balls=self.legal_balls_value(),
            target=self.target if self.mode == "LIVE" and self.innings == 2 else None,
        )


class PredictionResponse(BaseModel):
    team1: str
    team2: str
    team1_probability: float
    team2_probability: float
    predicted_team: str
    edge: str
    confidence: str
    prediction_mode: str
    format: str
    model_name: str
    model_family: str
    warnings: list[str]
    drivers: list[DriverOut]
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


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models: dict[str, bool]


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None


class LiveMatchOut(BaseModel):
    match_id: str
    name: str
    format: str
    status: str
    teams: list[str]
    source: str
    venue: str | None = None
    competition: str | None = None
    date: Date | None = None
    predictable_live: bool = False
    warnings: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    intent: str | None = None
    message: str | None = Field(default=None, max_length=MAX_MESSAGE_LEN)
    team1: str | None = None
    team2: str | None = None
    format: str | None = None
    mode: str | None = None
    date: Date | None = None
    gender: Literal["male", "female"] = "male"
    venue: str | None = None
    competition: str | None = None
    toss_winner: str | None = None
    toss_decision: Literal["bat", "field", "bowl"] | None = None
    team1_xi: list[str] = Field(default_factory=list)
    team2_xi: list[str] = Field(default_factory=list)
    batting_team: str | None = None
    innings: int | None = Field(default=None, ge=1, le=2)
    runs: int | None = Field(default=None, ge=0, le=500)
    wickets: int | None = Field(default=None, ge=0, le=10)
    overs: str | None = Field(default=None, max_length=8)
    legal_balls: int | None = Field(default=None, ge=0, le=120)
    target: int | None = Field(default=None, ge=1, le=500)

    @field_validator("message")
    @classmethod
    def _message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _has_content(self) -> ChatRequest:
        if self.intent is None and self.message is None:
            raise ValueError("Provide intent or message")
        return self


class ChatResponse(BaseModel):
    intent: str
    message: str
    prediction: PredictionResponse | None = None
    suggestions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
