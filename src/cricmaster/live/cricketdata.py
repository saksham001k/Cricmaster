"""CricketData/CricAPI adapter for the currentMatches endpoint.

The adapter deliberately treats upstream fields as untrusted. CricketData's
currentMatches feed can contain recent results alongside live matches, omit
toss information, and use inconsistent innings labels. Normalization is kept
separate from Cricmaster's prediction models.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import requests

from cricmaster.config import Settings, load_settings
from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import (
    InningsState,
    MatchMetadata,
    MatchState,
)
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.live.provider import LiveCricketProvider
from cricmaster.prediction.live import parse_cricket_overs


CURRENT_MATCHES_URL = "https://api.cricapi.com/v1/currentMatches"

_TERMINAL_STATUS_MARKERS = (
    " won by ",
    " won the match",
    "awarded the match",
    "abandoned",
    "cancelled",
    "canceled",
    "no result",
    "refused to play",
    "forfeit",
    "forfeited",
    "match tied",
    "match drawn",
)

_DLS_MARKERS = (
    "dls",
    "duckworth",
    "revised target",
    "due to rain",
    "rain reduced",
    "reduced due to rain",
    "overs game due to rain",
)

_COMPETITION_MARKERS = (
    ("tamil nadu premier league", "TNPL"),
    ("indian premier league", "IPL"),
    ("caribbean premier league", "CPL"),
    ("pakistan super league", "PSL"),
    ("big bash league", "BBL"),
    ("women's big bash league", "WBBL"),
    ("womens big bash league", "WBBL"),
    ("women's premier league", "WPL"),
    ("womens premier league", "WPL"),
    ("bangladesh premier league", "BPL"),
    ("lanka premier league", "LPL"),
    ("major league cricket", "MLC"),
    ("t20 blast", "T20 Blast"),
    ("syed mushtaq ali trophy", "SMAT"),
    ("the hundred", "The Hundred"),
)


@dataclass(frozen=True)
class CricketDataScore:
    runs: int
    wickets: int
    overs_text: str
    legal_balls: int | None
    innings_label: str
    batting_team: str | None


@dataclass(frozen=True)
class CricketDataMatch:
    match_id: str
    name: str
    match_type_raw: str | None
    match_format: MatchFormat
    competition: str | None
    gender: str
    status: str
    venue: str | None
    match_date: date | None
    teams: tuple[str, str]
    toss_winner: str | None
    toss_decision: str | None
    match_started: bool
    match_ended: bool
    scores: tuple[CricketDataScore, ...]
    target: int | None
    warnings: tuple[str, ...]

    @property
    def current_score(self) -> CricketDataScore | None:
        return self.scores[-1] if self.scores else None

    @property
    def innings_number(self) -> int | None:
        if not self.scores:
            return None
        return min(len(self.scores), 2)

    @property
    def batting_team(self) -> str | None:
        score = self.current_score
        return score.batting_team if score else None

    @property
    def terminal_status(self) -> bool:
        text = f" {self.status.strip().lower()} "
        return self.match_ended or any(marker in text for marker in _TERMINAL_STATUS_MARKERS)

    @property
    def supported_format(self) -> bool:
        return self.match_format in {MatchFormat.T20, MatchFormat.T20I}

    @property
    def predictable_live(self) -> bool:
        if not self.match_started or self.terminal_status:
            return False
        if self.match_date is None:
            return False
        if not self.supported_format:
            return False
        if not self.current_score or not self.batting_team:
            return False
        if self.current_score.legal_balls is None:
            return False
        if self.innings_number == 2 and self.target is None:
            return False
        return True


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def infer_competition(name: str) -> str | None:
    cleaned = _clean(name)
    for marker, label in _COMPETITION_MARKERS:
        if marker in cleaned:
            return label
    return None


def infer_gender(name: str, teams: tuple[str, str]) -> str:
    joined = _clean(" ".join((name, *teams)))
    return "female" if "women" in joined else "male"


def infer_format(match_type: object, name: str) -> MatchFormat:
    """Infer only formats that can be represented safely from currentMatches."""

    cleaned_name = _clean(name)
    raw = _clean(match_type)

    # CricketData can label The Hundred as matchType=t20 while score.o is
    # actually balls (e.g. 100, 98). Never route that into a 120-ball T20 model.
    if "the hundred" in cleaned_name or "hundred" in cleaned_name:
        return MatchFormat.HUNDRED

    if "t20i" in cleaned_name or "twenty20 international" in cleaned_name:
        return MatchFormat.T20I

    # Some currentMatches records omit matchType even though the match name
    # explicitly identifies the format.
    if "odi" in cleaned_name or "one-day international" in cleaned_name:
        return MatchFormat.ODI

    if "test" in cleaned_name:
        return MatchFormat.TEST

    if raw == "t20i":
        return MatchFormat.T20I

    if raw == "t20":
        return MatchFormat.T20

    if raw == "odi":
        return MatchFormat.ODI

    if raw == "test":
        return MatchFormat.TEST

    return MatchFormat.OTHER


def _parse_date(raw: dict[str, Any]) -> date | None:
    for key in ("date", "dateTimeGMT"):
        value = raw.get(key)
        if not value:
            continue
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            continue
    return None


def _unique_team_from_label(
    label: str,
    teams: tuple[str, str],
) -> str | None:
    cleaned = _clean(label)
    hits = [team for team in teams if _clean(team) in cleaned]
    return hits[0] if len(hits) == 1 else None


def _score_overs_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _score_legal_balls(value: object, match_format: MatchFormat) -> int | None:
    if match_format not in {MatchFormat.T20, MatchFormat.T20I}:
        return None
    text = _score_overs_text(value)
    if not text:
        return None
    try:
        return parse_cricket_overs(text, balls_per_over=6)
    except ValueError:
        return None


def _assign_batting_teams(
    raw_scores: list[dict[str, Any]],
    teams: tuple[str, str],
) -> list[str | None]:
    assignments = [
        _unique_team_from_label(str(score.get("inning") or ""), teams)
        for score in raw_scores
    ]

    # The feed sometimes emits malformed second-innings strings containing
    # both teams. If one innings is known, cricket's alternating batting order
    # lets us infer the other without trusting that malformed label.
    if len(assignments) >= 2:
        if assignments[0] and not assignments[1]:
            assignments[1] = teams[1] if assignments[0] == teams[0] else teams[0]
        elif assignments[1] and not assignments[0]:
            assignments[0] = teams[1] if assignments[1] == teams[0] else teams[0]
        elif assignments[0] and assignments[1] == assignments[0]:
            assignments[1] = teams[1] if assignments[0] == teams[0] else teams[0]

    return assignments


def _derive_target(
    *,
    status: str,
    scores: tuple[CricketDataScore, ...],
) -> tuple[int | None, str | None]:
    if len(scores) < 2:
        return None, None

    current = scores[-1]
    status_clean = _clean(status)

    # Prefer an explicit live status because it can reflect a revised DLS target.
    target_match = re.search(r"\btarget(?:\s+of|\s+is|:)?\s+(\d+)\b", status_clean)
    if target_match:
        return int(target_match.group(1)), None

    need_match = re.search(r"\bneeds?\s+(\d+)\s+runs?\b", status_clean)
    if need_match:
        return current.runs + int(need_match.group(1)), None

    if any(marker in status_clean for marker in _DLS_MARKERS):
        return None, (
            "Possible rain/DLS-adjusted chase detected but currentMatches did not "
            "provide a reliable target."
        )

    first_runs = scores[0].runs
    return first_runs + 1, None


def normalize_current_match(raw: dict[str, Any]) -> CricketDataMatch | None:
    raw_teams = raw.get("teams")
    if not isinstance(raw_teams, list) or len(raw_teams) != 2:
        return None

    teams = (str(raw_teams[0]), str(raw_teams[1]))
    match_id = str(raw.get("id") or "").strip()
    if not match_id:
        return None

    name = str(raw.get("name") or "").strip()
    match_format = infer_format(raw.get("matchType"), name)
    competition = infer_competition(name)
    gender = infer_gender(name, teams)
    status = str(raw.get("status") or "").strip()

    raw_scores = [
        item
        for item in (raw.get("score") or [])
        if isinstance(item, dict)
    ][:2]
    assignments = _assign_batting_teams(raw_scores, teams)

    scores: list[CricketDataScore] = []
    warnings: list[str] = []

    for index, item in enumerate(raw_scores):
        try:
            runs = int(item.get("r") or 0)
            wickets = int(item.get("w") or 0)
        except (TypeError, ValueError):
            continue

        overs_text = _score_overs_text(item.get("o"))
        legal_balls = _score_legal_balls(item.get("o"), match_format)
        batting_team = assignments[index] if index < len(assignments) else None

        if batting_team is None:
            warnings.append(
                f"Could not identify batting team for score entry {index + 1}."
            )
        if match_format in {MatchFormat.T20, MatchFormat.T20I} and legal_balls is None:
            warnings.append(
                f"Could not parse overs for score entry {index + 1}: {overs_text!r}."
            )

        scores.append(
            CricketDataScore(
                runs=runs,
                wickets=wickets,
                overs_text=overs_text,
                legal_balls=legal_balls,
                innings_label=str(item.get("inning") or ""),
                batting_team=batting_team,
            )
        )

    score_tuple = tuple(scores)
    target, target_warning = _derive_target(status=status, scores=score_tuple)
    if target_warning:
        warnings.append(target_warning)

    toss_winner = str(raw.get("tossWinner") or "").strip() or None
    toss_decision = str(raw.get("tossChoice") or "").strip().lower() or None
    if toss_decision == "bowl":
        toss_decision = "field"

    if competition == "The Hundred":
        warnings.append(
            "The Hundred is excluded from the current T20 live model because "
            "CricketData reports score.o as balls for this competition."
        )

    if match_format is MatchFormat.T20 and competition not in {None, "IPL"}:
        warnings.append(
            "This is a domestic/franchise T20 outside Cricmaster's original "
            "IPL-specific T20 training corpus; live scoreboard signal can still "
            "be useful, but league/team historical coverage may be weak."
        )

    return CricketDataMatch(
        match_id=match_id,
        name=name,
        match_type_raw=(
            str(raw.get("matchType")) if raw.get("matchType") is not None else None
        ),
        match_format=match_format,
        competition=competition,
        gender=gender,
        status=status,
        venue=str(raw.get("venue") or "").strip() or None,
        match_date=_parse_date(raw),
        teams=teams,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        match_started=bool(raw.get("matchStarted")),
        match_ended=bool(raw.get("matchEnded")),
        scores=score_tuple,
        target=target,
        warnings=tuple(warnings),
    )


class CricketDataProvider(LiveCricketProvider):
    """Concrete currentMatches provider with a short in-process cache."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        timeout: float = 20.0,
        cache_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._timeout = timeout
        self._cache_seconds = max(cache_seconds, 0.0)
        self._session = session or requests.Session()
        self._cached_at = 0.0
        self._cached_matches: list[CricketDataMatch] | None = None
        self.last_info: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "cricketdata"

    def current_matches(self, *, force: bool = False) -> list[CricketDataMatch]:
        key = self._settings.cricket_api_key
        if not key:
            raise RuntimeError("CRICKET_API_KEY is not configured")

        now = time.monotonic()
        if (
            not force
            and self._cached_matches is not None
            and now - self._cached_at <= self._cache_seconds
        ):
            return list(self._cached_matches)

        response = self._session.get(
            CURRENT_MATCHES_URL,
            params={"apikey": key, "offset": 0},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "success":
            raise RuntimeError(
                f"CricketData currentMatches failed: {payload.get('status')!r}"
            )

        self.last_info = dict(payload.get("info") or {})
        normalized = [
            match
            for item in (payload.get("data") or [])
            if isinstance(item, dict)
            if (match := normalize_current_match(item)) is not None
        ]

        self._cached_matches = normalized
        self._cached_at = now
        return list(normalized)

    def get_live_matches(self) -> list[MatchMetadata]:
        return [
            self._to_metadata(match)
            for match in self.current_matches()
            if match.predictable_live
        ]

    def get_match(self, match_id: str) -> MatchState | None:
        match = next(
            (item for item in self.current_matches() if item.match_id == match_id),
            None,
        )
        if match is None:
            return None

        metadata = self._to_metadata(match)
        current = self._to_current_innings(match)
        history = self._to_innings_history(match)

        return MatchState(
            metadata=metadata,
            current_innings=current,
            innings_history=history,
            deliveries=[],
            current_players=None,
            source="cricketdata:currentMatches",
            retrieved_at=datetime.now(timezone.utc),
        )

    def get_score(self, match_id: str) -> InningsState | None:
        match = next(
            (item for item in self.current_matches() if item.match_id == match_id),
            None,
        )
        return self._to_current_innings(match) if match is not None else None

    @staticmethod
    def _to_metadata(match: CricketDataMatch) -> MatchMetadata:
        return MatchMetadata(
            match_id=match.match_id,
            format=match.match_format,
            competition=match.competition,
            season=None,
            match_number=None,
            date=match.match_date,
            venue=match.venue,
            city=None,
            team1=match.teams[0],
            team2=match.teams[1],
            toss_winner=match.toss_winner,
            toss_decision=match.toss_decision,
            winner=None,
            result_type=None,
            player_of_match=None,
            source="cricketdata:currentMatches",
            gender=match.gender,
            team_type="international" if match.match_format is MatchFormat.T20I else None,
            balls_per_over=6 if match.match_format in {MatchFormat.T20, MatchFormat.T20I} else None,
            scheduled_overs=20 if match.match_format in {MatchFormat.T20, MatchFormat.T20I} else None,
            team1_players=None,
            team2_players=None,
        )

    @staticmethod
    def _to_current_innings(match: CricketDataMatch) -> InningsState | None:
        score = match.current_score
        innings_number = match.innings_number
        if score is None or innings_number is None or score.batting_team is None:
            return None

        bowling = (
            canonicalize_team(match.teams[1])
            if canonicalize_team(score.batting_team) == canonicalize_team(match.teams[0])
            else canonicalize_team(match.teams[0])
        )
        balls = score.legal_balls or 0
        balls_remaining = max(120 - balls, 0) if score.legal_balls is not None else None
        current_rr = (score.runs / (balls / 6.0)) if balls else None
        required_runs = (
            match.target - score.runs
            if innings_number == 2 and match.target is not None
            else None
        )
        required_rr = (
            required_runs / (balls_remaining / 6.0)
            if required_runs is not None and balls_remaining
            else None
        )

        return InningsState(
            batting_team=canonicalize_team(score.batting_team),
            bowling_team=bowling,
            innings_number=innings_number,
            runs=score.runs,
            wickets=score.wickets,
            overs=float(score.overs_text) if score.overs_text else None,
            balls=balls,
            target=match.target if innings_number == 2 else None,
            target_overs=None,
            required_runs=required_runs,
            balls_remaining=balls_remaining,
            current_run_rate=current_rr,
            required_run_rate=required_rr,
            declared=False,
            forfeited=False,
            super_over=False,
        )

    @classmethod
    def _to_innings_history(cls, match: CricketDataMatch) -> list[InningsState]:
        history: list[InningsState] = []
        for index, score in enumerate(match.scores, 1):
            if score.batting_team is None:
                continue
            bowling = (
                canonicalize_team(match.teams[1])
                if canonicalize_team(score.batting_team) == canonicalize_team(match.teams[0])
                else canonicalize_team(match.teams[0])
            )
            history.append(
                InningsState(
                    batting_team=canonicalize_team(score.batting_team),
                    bowling_team=bowling,
                    innings_number=index,
                    runs=score.runs,
                    wickets=score.wickets,
                    overs=float(score.overs_text) if score.overs_text else None,
                    balls=score.legal_balls or 0,
                    target=match.target if index == 2 else None,
                )
            )
        return history
