from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import InningsState, MatchMetadata, MatchState
from cricmaster.data.resolver import MatchStateResolver
from cricmaster.live.mock import MOCK_MATCH_ID, MockLiveCricketProvider
from cricmaster.live.provider import LiveCricketProvider
from cricmaster.search.provider import CricketSearchProvider
from cricmaster.search.stub import StubSearchProvider


class EmptyLiveProvider(LiveCricketProvider):
    @property
    def name(self) -> str:
        return "empty-live"

    def get_live_matches(self) -> list[MatchMetadata]:
        return []

    def get_match(self, match_id: str) -> MatchState | None:
        return None

    def get_score(self, match_id: str) -> InningsState | None:
        return None


class ConflictingLiveProvider(LiveCricketProvider):
    @property
    def name(self) -> str:
        return "conflict-live"

    def get_live_matches(self) -> list[MatchMetadata]:
        return []

    def get_match(self, match_id: str) -> MatchState | None:
        base = MockLiveCricketProvider().get_match(match_id)
        if base is None:
            return None
        metadata = base.metadata.model_copy(update={"winner": "Sample Knights", "source": self.name})
        innings = base.current_innings.model_copy(update={"runs": 99}) if base.current_innings else None
        return base.model_copy(update={"metadata": metadata, "current_innings": innings, "source": self.name})

    def get_score(self, match_id: str) -> InningsState | None:
        match = self.get_match(match_id)
        return match.current_innings if match else None


class RecordingSearchProvider(CricketSearchProvider):
    def __init__(self, match: MatchState | None) -> None:
        self._match = match
        self.queries: list[str] = []

    @property
    def name(self) -> str:
        return "test-search"

    def find_match_status(self, query: str) -> MatchState | None:
        self.queries.append(query)
        return self._match

    def find_team_news(self, team: str) -> list[str]:
        return []

    def find_playing_xi(self, query: str) -> dict[str, list[str]] | None:
        return None


def test_resolver_uses_first_live_provider() -> None:
    resolver = MatchStateResolver(
        live_providers=[MockLiveCricketProvider(), ConflictingLiveProvider()],
        search_provider=StubSearchProvider(),
    )
    result = resolver.resolve_match(MOCK_MATCH_ID)
    assert result.match is not None
    assert result.fields["metadata.team1"].source == "mock-live"
    assert result.fields["metadata.team1"].value == "Demo Strikers"
    assert result.fields["metadata.team1"].reliability.value == "high"
    assert result.sources_tried[:2] == ["mock-live", "conflict-live"]


def test_resolver_records_conflicts_instead_of_overwriting() -> None:
    resolver = MatchStateResolver(
        live_providers=[MockLiveCricketProvider(), ConflictingLiveProvider()],
    )
    result = resolver.resolve_match(MOCK_MATCH_ID)
    conflict_fields = {item.field for item in result.conflicts}
    assert "current_innings.runs" in conflict_fields
    chosen = next(item for item in result.conflicts if item.field == "current_innings.runs")
    assert chosen.chosen.value == 45
    assert chosen.rejected.value == 99
    assert result.fields["current_innings.runs"].value == 45


def test_resolver_falls_back_to_search() -> None:
    mock_match = MockLiveCricketProvider().get_match(MOCK_MATCH_ID)
    search = RecordingSearchProvider(mock_match)
    resolver = MatchStateResolver(
        live_providers=[EmptyLiveProvider()],
        search_provider=search,
    )
    result = resolver.resolve_match("unknown-id", search_query="Demo Strikers")
    assert search.queries == ["Demo Strikers"]
    assert result.match is not None
    assert result.fields["metadata.team1"].source == "test-search"
    assert result.fields["metadata.team1"].reliability.value == "low"
    assert result.sources_tried == ["empty-live", "test-search"]


def test_resolver_returns_empty_when_nothing_matches() -> None:
    resolver = MatchStateResolver(
        live_providers=[EmptyLiveProvider()],
        search_provider=StubSearchProvider(),
    )
    result = resolver.resolve_match("missing")
    assert result.match is None
    assert result.fields == {}
    assert result.conflicts == []
