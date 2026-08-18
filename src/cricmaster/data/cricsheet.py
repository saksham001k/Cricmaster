"""Parse official Cricsheet JSON match files into Cricmaster models.

The expected file layout follows Cricsheet JSON format 1.2.0:
https://cricsheet.org/format/json/

One malformed file must not abort an entire directory import.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from cricmaster.data.formats import normalize_competition, normalize_match_type
from cricmaster.data.models import (
    CurrentPlayers,
    Delivery,
    InningsState,
    LoadError,
    LoadReport,
    MatchMetadata,
    MatchState,
)

LOGGER = logging.getLogger(__name__)
SOURCE_NAME = "cricsheet"
SKIP_FILENAMES = {"readme.txt", "readme.md"}


class CricsheetParseError(ValueError):
    """Raised when a single match file cannot be interpreted."""


def _as_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CricsheetParseError(f"Expected object for {label}")
    return value


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_date(raw: Any) -> date | None:
    if not raw:
        return None
    text = str(raw)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _legal_delivery(extras: dict[str, Any] | None) -> bool:
    if not extras:
        return True
    return not extras.get("wides") and not extras.get("noballs")


def _overs_notation(legal_balls: int, balls_per_over: int) -> float:
    if balls_per_over <= 0:
        balls_per_over = 6
    completed, remainder = divmod(legal_balls, balls_per_over)
    return completed + (remainder / 10.0)


def _parse_result(outcome: dict[str, Any]) -> tuple[str | None, str | None]:
    winner = outcome.get("winner")
    if isinstance(winner, str) and winner.strip():
        return winner, "win"
    if "result" in outcome and outcome["result"]:
        return None, str(outcome["result"])
    if outcome.get("eliminator"):
        return str(outcome["eliminator"]), "eliminator"
    if "bowl_out" in outcome:
        winner = outcome["bowl_out"]
        return (str(winner) if winner else None), "bowl_out"
    return None, None


def _player_of_match(info: dict[str, Any]) -> str | None:
    values = _as_list(info.get("player_of_match"))
    return str(values[0]) if values else None


def parse_metadata(payload: dict[str, Any], *, match_id: str) -> MatchMetadata:
    info = _as_dict(payload.get("info"), "info")
    teams = [str(team) for team in _as_list(info.get("teams")) if team]
    if len(teams) < 2:
        raise CricsheetParseError("Match is missing two team names")

    event = info.get("event") if isinstance(info.get("event"), dict) else {}
    competition = normalize_competition(event.get("name") if event else None)
    toss = info.get("toss") if isinstance(info.get("toss"), dict) else {}
    outcome = info.get("outcome") if isinstance(info.get("outcome"), dict) else {}
    dates = _as_list(info.get("dates"))
    winner, result_type = _parse_result(outcome)
    balls_per_over = info.get("balls_per_over")
    scheduled_overs = info.get("overs")
    match_format = normalize_match_type(
        info.get("match_type"),
        team_type=info.get("team_type"),
        competition=competition or (event.get("name") if event else None),
        balls_per_over=int(balls_per_over) if isinstance(balls_per_over, int) else None,
        scheduled_overs=int(scheduled_overs) if isinstance(scheduled_overs, int) else None,
    )
    match_number = event.get("match_number") if event else info.get("match_type_number")
    if match_number is not None:
        try:
            match_number = int(match_number)
        except (TypeError, ValueError):
            match_number = None

    players = info.get("players") if isinstance(info.get("players"), dict) else {}
    team1_players = [str(name) for name in _as_list(players.get(teams[0]))] or None
    team2_players = [str(name) for name in _as_list(players.get(teams[1]))] or None

    return MatchMetadata(
        match_id=match_id,
        format=match_format,
        competition=competition,
        season=str(info["season"]) if info.get("season") is not None else None,
        match_number=match_number,
        date=_parse_date(dates[0] if dates else None),
        venue=str(info["venue"]) if info.get("venue") else None,
        city=str(info["city"]) if info.get("city") else None,
        team1=teams[0],
        team2=teams[1],
        toss_winner=str(toss["winner"]) if toss.get("winner") else None,
        toss_decision=str(toss["decision"]) if toss.get("decision") else None,
        winner=winner,
        result_type=result_type,
        player_of_match=_player_of_match(info),
        source=SOURCE_NAME,
        gender=str(info["gender"]) if info.get("gender") else None,
        team_type=str(info["team_type"]) if info.get("team_type") else None,
        balls_per_over=int(balls_per_over) if isinstance(balls_per_over, int) else None,
        scheduled_overs=int(scheduled_overs) if isinstance(scheduled_overs, int) else None,
        team1_players=team1_players,
        team2_players=team2_players,
    )


def _other_team(batting_team: str, team1: str, team2: str) -> str:
    if batting_team == team1:
        return team2
    if batting_team == team2:
        return team1
    return team2


def _parse_delivery(
    raw: dict[str, Any],
    *,
    innings_number: int,
    over_number: int,
    ball_number: int,
    batting_team: str,
) -> Delivery:
    wickets = _as_list(raw.get("wickets"))

    # Cricsheet records events such as "retired hurt" inside the wickets
    # array, but they do not reduce the batting side's wickets in hand.
    non_counting_dismissals = {"retired hurt"}
    counting_wickets = [
        item
        for item in wickets
        if isinstance(item, dict)
        and str(item.get("kind") or "").strip().lower()
        not in non_counting_dismissals
    ]

    first_wicket = counting_wickets[0] if counting_wickets else {}
    runs = raw.get("runs") if isinstance(raw.get("runs"), dict) else {}
    extras = raw.get("extras") if isinstance(raw.get("extras"), dict) else {}
    return Delivery(
        innings=innings_number,
        over=over_number,
        ball=ball_number,
        batting_team=batting_team,
        striker=str(raw.get("batter") or raw.get("batsman") or ""),
        non_striker=str(raw.get("non_striker") or ""),
        bowler=str(raw.get("bowler") or ""),
        runs_batter=int(runs.get("batter") or 0),
        runs_extras=int(runs.get("extras") or 0),
        runs_total=int(runs.get("total") or 0),
        wicket=bool(counting_wickets),
        wicket_type=str(first_wicket["kind"]) if first_wicket.get("kind") else None,
        player_out=str(first_wicket["player_out"]) if first_wicket.get("player_out") else None,
        actual_delivery=str(raw["actual_delivery"]) if raw.get("actual_delivery") else None,
        is_wide=bool(extras.get("wides")),
        is_noball=bool(extras.get("noballs")),
    )


def parse_innings(
    payload: dict[str, Any],
    metadata: MatchMetadata,
) -> tuple[list[InningsState], list[Delivery]]:
    innings_history: list[InningsState] = []
    deliveries: list[Delivery] = []
    balls_per_over = metadata.balls_per_over or 6

    for index, raw_innings in enumerate(_as_list(payload.get("innings")), start=1):
        if not isinstance(raw_innings, dict):
            continue
        batting_team = str(raw_innings.get("team") or "")
        if not batting_team:
            raise CricsheetParseError(f"Innings {index} is missing a batting team")

        bowling_team = _other_team(batting_team, metadata.team1, metadata.team2)
        runs = 0
        wickets = 0
        legal_balls = 0
        penalty = raw_innings.get("penalty_runs") if isinstance(raw_innings.get("penalty_runs"), dict) else {}
        runs += int(penalty.get("pre") or 0)

        for over in _as_list(raw_innings.get("overs")):
            if not isinstance(over, dict):
                continue
            over_number = int(over.get("over") if over.get("over") is not None else 0)
            for ball_number, raw_delivery in enumerate(_as_list(over.get("deliveries")), start=1):
                if not isinstance(raw_delivery, dict):
                    continue
                delivery = _parse_delivery(
                    raw_delivery,
                    innings_number=index,
                    over_number=over_number,
                    ball_number=ball_number,
                    batting_team=batting_team,
                )
                deliveries.append(delivery)
                runs += delivery.runs_total
                if delivery.wicket:
                    wickets += 1
                if delivery.is_legal:
                    legal_balls += 1

        runs += int(penalty.get("post") or 0)
        target_info = raw_innings.get("target") if isinstance(raw_innings.get("target"), dict) else {}
        target = int(target_info["runs"]) if target_info.get("runs") is not None else None
        target_overs = target_info.get("overs")
        if target_overs is not None:
            try:
                target_overs = float(target_overs)
            except (TypeError, ValueError):
                target_overs = None
        overs = _overs_notation(legal_balls, balls_per_over)
        required_runs = (target - runs) if target is not None else None
        run_rate = (runs / legal_balls * balls_per_over) if legal_balls else None

        innings_history.append(
            InningsState(
                batting_team=batting_team,
                bowling_team=bowling_team,
                innings_number=index,
                runs=runs,
                wickets=wickets,
                overs=overs,
                balls=legal_balls,
                target=target,
                target_overs=target_overs,
                required_runs=required_runs if required_runs is not None and required_runs > 0 else required_runs,
                current_run_rate=round(run_rate, 2) if run_rate is not None else None,
                declared=bool(raw_innings.get("declared")),
                forfeited=bool(raw_innings.get("forfeited")),
                super_over=bool(raw_innings.get("super_over")),
            )
        )

    return innings_history, deliveries


def parse_match_payload(payload: dict[str, Any], *, match_id: str) -> MatchState:
    metadata = parse_metadata(payload, match_id=match_id)
    innings_history, deliveries = parse_innings(payload, metadata)
    current = innings_history[-1] if innings_history else None
    current_players = None
    if deliveries:
        last = deliveries[-1]
        current_players = CurrentPlayers(
            striker=last.striker or None,
            non_striker=last.non_striker or None,
            bowler=last.bowler or None,
        )
    return MatchState(
        metadata=metadata,
        current_innings=current,
        innings_history=innings_history,
        deliveries=deliveries,
        current_players=current_players,
        source=SOURCE_NAME,
        retrieved_at=datetime.now().astimezone(),
    )


def load_match(path: str | Path) -> MatchState:
    """Load and normalize a single Cricsheet JSON match file."""

    match_path = Path(path)
    try:
        text = match_path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except OSError as exc:
        raise CricsheetParseError(f"Could not read {match_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CricsheetParseError(f"Malformed JSON in {match_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CricsheetParseError("Match file root must be a JSON object")
    return parse_match_payload(payload, match_id=match_path.stem)


def load_directory(path: str | Path) -> LoadReport:
    """Load every JSON file in a directory. Failures are collected, not raised."""

    directory = Path(path)
    matches: list[MatchState] = []
    errors: list[LoadError] = []

    if not directory.exists():
        return LoadReport(
            errors=[LoadError(path=str(directory), reason="Directory does not exist")]
        )
    if not directory.is_dir():
        return LoadReport(
            errors=[LoadError(path=str(directory), reason="Path is not a directory")]
        )

    files = sorted(
        candidate
        for candidate in directory.glob("*.json")
        if candidate.name.lower() not in SKIP_FILENAMES
    )
    if not files:
        return LoadReport(matches=[], errors=[])

    for file_path in files:
        try:
            matches.append(load_match(file_path))
        except CricsheetParseError as exc:
            LOGGER.warning("Skipping %s: %s", file_path, exc)
            errors.append(LoadError(path=str(file_path), reason=str(exc)))
        except Exception as exc:  # noqa: BLE001 - isolate unexpected parse bugs
            LOGGER.exception("Unexpected error while parsing %s", file_path)
            errors.append(LoadError(path=str(file_path), reason=f"Unexpected error: {exc}"))

    return LoadReport(matches=matches, errors=errors)


def discover_match_files(path: str | Path) -> list[Path]:
    """Return JSON match files under path, sorted by relative path for stability."""

    root = Path(path)
    if root.is_file() and root.suffix.lower() == ".json":
        return [root]
    if not root.is_dir():
        return []
    files = [
        candidate
        for candidate in root.rglob("*.json")
        if candidate.name.lower() not in SKIP_FILENAMES
    ]
    return sorted(files, key=lambda item: item.as_posix().lower())
