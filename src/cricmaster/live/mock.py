"""In-memory live provider used when no API key is configured."""

from __future__ import annotations

from datetime import date, datetime

from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import (
    CurrentPlayers,
    InningsState,
    MatchMetadata,
    MatchState,
)
from cricmaster.live.provider import LiveCricketProvider

MOCK_MATCH_ID = "mock-demo-t20"


class MockLiveCricketProvider(LiveCricketProvider):
    """Deterministic placeholder. Values are labelled as mock, not live scores."""

    def __init__(self, *, available: bool = True) -> None:
        self._available = available
        self._match = self._build_match()

    @property
    def name(self) -> str:
        return "mock-live"

    def get_live_matches(self) -> list[MatchMetadata]:
        if not self._available:
            return []
        return [self._match.metadata]

    def get_match(self, match_id: str) -> MatchState | None:
        if not self._available or match_id != MOCK_MATCH_ID:
            return None
        return self._match

    def get_score(self, match_id: str) -> InningsState | None:
        match = self.get_match(match_id)
        return match.current_innings if match else None

    def _build_match(self) -> MatchState:
        metadata = MatchMetadata(
            match_id=MOCK_MATCH_ID,
            format=MatchFormat.T20,
            competition="Demo Cup",
            season="2026",
            match_number=1,
            date=date(2026, 8, 18),
            venue="Demo Ground",
            city="Chennai",
            team1="Demo Strikers",
            team2="Sample Knights",
            toss_winner="Demo Strikers",
            toss_decision="bat",
            winner=None,
            result_type=None,
            player_of_match=None,
            source=self.name,
            gender="male",
            team_type="club",
            balls_per_over=6,
            scheduled_overs=20,
        )
        innings = InningsState(
            batting_team="Demo Strikers",
            bowling_team="Sample Knights",
            innings_number=1,
            runs=45,
            wickets=1,
            overs=5.2,
            balls=32,
            current_run_rate=8.44,
        )
        return MatchState(
            metadata=metadata,
            current_innings=innings,
            innings_history=[innings],
            deliveries=[],
            current_players=CurrentPlayers(
                striker="A Batter",
                non_striker="B Partner",
                bowler="C Bowler",
            ),
            source=self.name,
            retrieved_at=datetime.now().astimezone(),
        )
