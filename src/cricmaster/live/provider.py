"""Live cricket data provider contracts.

Cricmaster must not be coupled to a single commercial API. Concrete adapters
for CricketData, Sportmonks, EntitySport, Roanuz, or others belong in later
steps and should read credentials from the environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cricmaster.data.models import InningsState, MatchMetadata, MatchState


class LiveCricketProvider(ABC):
    """Structured live-data source. Implementations must not log API keys."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable source identifier stored on resolved values."""

    @abstractmethod
    def get_live_matches(self) -> list[MatchMetadata]:
        """Return currently known live or upcoming matches."""

    @abstractmethod
    def get_match(self, match_id: str) -> MatchState | None:
        """Return a full match snapshot, or None if this provider has no data."""

    @abstractmethod
    def get_score(self, match_id: str) -> InningsState | None:
        """Return the current innings score, or None if unavailable."""
