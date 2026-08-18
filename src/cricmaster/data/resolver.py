"""Resolve current match state from ordered data sources.

Lookup order:
1. Preferred structured live provider
2. Secondary structured providers
3. Search fallback

Fields already populated by an earlier source are not overwritten. If a later
source disagrees, the conflict is recorded instead of being merged silently.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from cricmaster.data.models import (
    CurrentPlayers,
    FieldConflict,
    InningsState,
    MatchMetadata,
    MatchState,
    Reliability,
    SourcedValue,
)
from cricmaster.live.provider import LiveCricketProvider
from cricmaster.search.provider import CricketSearchProvider

METADATA_FIELDS = (
    "match_id",
    "format",
    "competition",
    "season",
    "match_number",
    "date",
    "venue",
    "city",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner",
    "result_type",
    "player_of_match",
    "gender",
    "team_type",
    "balls_per_over",
    "scheduled_overs",
)

INNINGS_FIELDS = (
    "batting_team",
    "bowling_team",
    "innings_number",
    "runs",
    "wickets",
    "overs",
    "balls",
    "target",
    "required_runs",
    "balls_remaining",
    "current_run_rate",
    "required_run_rate",
    "declared",
    "forfeited",
    "super_over",
)

PLAYER_FIELDS = ("striker", "non_striker", "bowler")


class ResolvedMatch(BaseModel):
    """Best available match snapshot plus per-field provenance."""

    model_config = ConfigDict(extra="forbid")

    match: MatchState | None = None
    fields: dict[str, SourcedValue] = Field(default_factory=dict)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    sources_tried: list[str] = Field(default_factory=list)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict, tuple, set)) and not value:
        return True
    return False


def _reliability_for(source_kind: str) -> Reliability:
    if source_kind == "live":
        return Reliability.HIGH
    if source_kind == "search":
        return Reliability.LOW
    return Reliability.UNKNOWN


class MatchStateResolver:
    """Try providers in order and keep the first non-empty value for each field."""

    def __init__(
        self,
        live_providers: list[LiveCricketProvider] | None = None,
        search_provider: CricketSearchProvider | None = None,
    ) -> None:
        self._live_providers = list(live_providers or [])
        self._search_provider = search_provider

    def resolve_match(self, match_id: str, *, search_query: str | None = None) -> ResolvedMatch:
        result = ResolvedMatch()
        for provider in self._live_providers:
            result.sources_tried.append(provider.name)
            snapshot = provider.get_match(match_id)
            if snapshot is None:
                continue
            self._merge_snapshot(result, snapshot, source=provider.name, kind="live")

        if self._search_provider is not None:
            result.sources_tried.append(self._search_provider.name)
            query = search_query or match_id
            snapshot = self._search_provider.find_match_status(query)
            if snapshot is not None:
                self._merge_snapshot(
                    result,
                    snapshot,
                    source=self._search_provider.name,
                    kind="search",
                )

        return result

    def _merge_snapshot(
        self,
        result: ResolvedMatch,
        snapshot: MatchState,
        *,
        source: str,
        kind: str,
    ) -> None:
        timestamp = snapshot.retrieved_at or datetime.now().astimezone()
        reliability = _reliability_for(kind)
        metadata_values = snapshot.metadata.model_dump()
        for field in METADATA_FIELDS:
            self._assign(
                result,
                field=f"metadata.{field}",
                value=metadata_values.get(field),
                source=source,
                timestamp=timestamp,
                reliability=reliability,
            )

        if snapshot.current_innings is not None:
            innings_values = snapshot.current_innings.model_dump()
            for field in INNINGS_FIELDS:
                self._assign(
                    result,
                    field=f"current_innings.{field}",
                    value=innings_values.get(field),
                    source=source,
                    timestamp=timestamp,
                    reliability=reliability,
                )

        if snapshot.current_players is not None:
            player_values = snapshot.current_players.model_dump()
            for field in PLAYER_FIELDS:
                self._assign(
                    result,
                    field=f"current_players.{field}",
                    value=player_values.get(field),
                    source=source,
                    timestamp=timestamp,
                    reliability=reliability,
                )

        if snapshot.deliveries:
            self._assign(
                result,
                field="deliveries",
                value=snapshot.deliveries,
                source=source,
                timestamp=timestamp,
                reliability=reliability,
            )
        if snapshot.innings_history:
            self._assign(
                result,
                field="innings_history",
                value=snapshot.innings_history,
                source=source,
                timestamp=timestamp,
                reliability=reliability,
            )

        result.match = self._rebuild_match(result, fallback=snapshot, source=source)

    def _assign(
        self,
        result: ResolvedMatch,
        *,
        field: str,
        value: object,
        source: str,
        timestamp: datetime,
        reliability: Reliability,
    ) -> None:
        if _is_missing(value):
            return
        incoming = SourcedValue(
            value=value,
            source=source,
            timestamp=timestamp,
            reliability=reliability,
            field=field,
        )
        existing = result.fields.get(field)
        if existing is None:
            result.fields[field] = incoming
            return
        if existing.value != incoming.value:
            result.conflicts.append(
                FieldConflict(field=field, chosen=existing, rejected=incoming)
            )

    def _rebuild_match(
        self,
        result: ResolvedMatch,
        *,
        fallback: MatchState,
        source: str,
    ) -> MatchState:
        metadata_data = fallback.metadata.model_dump()
        for field in METADATA_FIELDS:
            sourced = result.fields.get(f"metadata.{field}")
            if sourced is not None:
                metadata_data[field] = sourced.value
        metadata_data["source"] = result.fields.get("metadata.match_id", SourcedValue(value="", source=source)).source

        innings = None
        if any(key.startswith("current_innings.") for key in result.fields):
            innings_data = (
                fallback.current_innings.model_dump()
                if fallback.current_innings is not None
                else {
                    "batting_team": "",
                    "innings_number": 1,
                }
            )
            for field in INNINGS_FIELDS:
                sourced = result.fields.get(f"current_innings.{field}")
                if sourced is not None:
                    innings_data[field] = sourced.value
            if innings_data.get("batting_team"):
                innings = InningsState.model_validate(innings_data)

        players = None
        if any(key.startswith("current_players.") for key in result.fields):
            player_data = {"striker": None, "non_striker": None, "bowler": None}
            for field in PLAYER_FIELDS:
                sourced = result.fields.get(f"current_players.{field}")
                if sourced is not None:
                    player_data[field] = sourced.value
            players = CurrentPlayers.model_validate(player_data)

        deliveries = result.fields["deliveries"].value if "deliveries" in result.fields else []
        history = (
            result.fields["innings_history"].value
            if "innings_history" in result.fields
            else []
        )
        primary_source = next(iter(result.fields.values())).source if result.fields else source
        return MatchState(
            metadata=MatchMetadata.model_validate(metadata_data),
            current_innings=innings,
            innings_history=history,
            deliveries=deliveries,
            current_players=players,
            source=primary_source,
            retrieved_at=datetime.now().astimezone(),
        )
