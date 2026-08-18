"""No-op search provider. Real retrieval is deferred to a later step."""

from __future__ import annotations

from cricmaster.data.models import MatchState
from cricmaster.search.provider import CricketSearchProvider


class StubSearchProvider(CricketSearchProvider):
    """Always returns empty results so the resolver can be tested offline."""

    @property
    def name(self) -> str:
        return "search-stub"

    def find_match_status(self, query: str) -> MatchState | None:
        return None

    def find_team_news(self, team: str) -> list[str]:
        return []

    def find_playing_xi(self, query: str) -> dict[str, list[str]] | None:
        return None
