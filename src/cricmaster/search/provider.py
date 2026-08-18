"""Search fallback contracts for matches missing from structured APIs.

Implementations must use legitimate public APIs or search endpoints. Do not
bypass anti-bot protections or scrape sites that forbid it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cricmaster.data.models import MatchState


class CricketSearchProvider(ABC):
    """Optional fallback when structured live data is incomplete."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable source identifier stored on resolved values."""

    @abstractmethod
    def find_match_status(self, query: str) -> MatchState | None:
        """Look up a current or recent match by free-text query."""

    @abstractmethod
    def find_team_news(self, team: str) -> list[str]:
        """Return short, already-public notes about a team if available."""

    @abstractmethod
    def find_playing_xi(self, query: str) -> dict[str, list[str]] | None:
        """Return batting-order lists keyed by team name, if known."""
