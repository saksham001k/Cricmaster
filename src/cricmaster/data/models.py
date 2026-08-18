"""Internal cricket data models used across Cricmaster."""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime as DateTime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cricmaster.data.formats import MatchFormat


class Reliability(StrEnum):
    """Relative trust in a sourced observation. Not a prediction score."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class SourcedValue(BaseModel):
    """A single observed value with provenance.

    Conflicting observations must not be silently merged. Callers should keep
    the preferred value and record rejected alternatives separately.
    """

    model_config = ConfigDict(extra="forbid")

    value: Any
    source: str
    timestamp: DateTime | None = None
    reliability: Reliability = Reliability.UNKNOWN
    field: str | None = None


class FieldConflict(BaseModel):
    """Two providers reported different values for the same field."""

    model_config = ConfigDict(extra="forbid")

    field: str
    chosen: SourcedValue
    rejected: SourcedValue


class MatchMetadata(BaseModel):
    """Competition-level facts about a match, independent of live score."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    format: MatchFormat
    competition: str | None = None
    season: str | None = None
    match_number: int | None = None
    date: Date | None = None
    venue: str | None = None
    city: str | None = None
    team1: str
    team2: str
    toss_winner: str | None = None
    toss_decision: str | None = None
    winner: str | None = None
    result_type: str | None = None
    player_of_match: str | None = None
    source: str
    gender: str | None = None
    team_type: str | None = None
    balls_per_over: int | None = None
    scheduled_overs: int | None = None
    team1_players: list[str] | None = None
    team2_players: list[str] | None = None


class InningsState(BaseModel):
    """Scorecard snapshot for one innings. Optional fields stay unset until known."""

    model_config = ConfigDict(extra="forbid")

    batting_team: str
    bowling_team: str | None = None
    innings_number: int
    runs: int = 0
    wickets: int = 0
    overs: float | None = None
    balls: int = 0
    target: int | None = None
    target_overs: float | None = None
    required_runs: int | None = None
    balls_remaining: int | None = None
    current_run_rate: float | None = None
    required_run_rate: float | None = None
    declared: bool = False
    forfeited: bool = False
    super_over: bool = False


class Delivery(BaseModel):
    """One recorded delivery. Extra Cricsheet fields are omitted on purpose."""

    model_config = ConfigDict(extra="forbid")

    innings: int
    over: int
    ball: int
    batting_team: str
    striker: str
    non_striker: str
    bowler: str
    runs_batter: int = 0
    runs_extras: int = 0
    runs_total: int = 0
    wicket: bool = False
    wicket_type: str | None = None
    player_out: str | None = None
    actual_delivery: str | None = None
    is_wide: bool = False
    is_noball: bool = False

    @property
    def is_legal(self) -> bool:
        """Wides and no-balls do not count as legal deliveries."""

        return not self.is_wide and not self.is_noball


class CurrentPlayers(BaseModel):
    """Players involved in the most recently observed delivery, if known."""

    model_config = ConfigDict(extra="forbid")

    striker: str | None = None
    non_striker: str | None = None
    bowler: str | None = None


class MatchState(BaseModel):
    """Combined match snapshot later consumed by prediction components."""

    model_config = ConfigDict(extra="forbid")

    metadata: MatchMetadata
    current_innings: InningsState | None = None
    innings_history: list[InningsState] = Field(default_factory=list)
    deliveries: list[Delivery] = Field(default_factory=list)
    current_players: CurrentPlayers | None = None
    source: str
    retrieved_at: DateTime | None = None


class LoadError(BaseModel):
    """A single file that could not be imported."""

    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str


class LoadReport(BaseModel):
    """Result of importing a directory of match files."""

    model_config = ConfigDict(extra="forbid")

    matches: list[MatchState] = Field(default_factory=list)
    errors: list[LoadError] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors
