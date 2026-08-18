"""Build leakage-safe pre-match and live-state datasets from Cricsheet JSON."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from cricmaster.data.cricsheet import CricsheetParseError, discover_match_files, load_match
from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import MatchState
from cricmaster.features.history import HistoricalState
from cricmaster.features.live import iter_live_states
from cricmaster.features.prematch import build_prematch_rows
from cricmaster.features.toss import PredictionMode
from cricmaster.features.utils import supports_live_states
from cricmaster.features.validate import validate_live, validate_prematch


@dataclass
class BuildReport:
    matches_discovered: int = 0
    matches_parsed: int = 0
    matches_skipped: int = 0
    prematch_rows: int = 0
    live_state_rows: int = 0
    formats: dict[str, int] = field(default_factory=dict)
    competitions: dict[str, int] = field(default_factory=dict)
    excluded_results: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    live_formats_supported: list[str] = field(default_factory=list)
    live_formats_planned: list[str] = field(default_factory=list)
    validation_issues: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


def _peek_date(path: Path) -> tuple[date, str] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        info = payload.get("info") if isinstance(payload, dict) else None
        dates = info.get("dates") if isinstance(info, dict) else None
        if not dates:
            return None
        return date.fromisoformat(str(dates[0])[:10]), path.stem
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None


def _passes_filters(
    match: MatchState,
    *,
    formats: set[str] | None,
    competitions: set[str] | None,
) -> bool:
    if formats and str(match.metadata.format) not in formats:
        return False
    if competitions:
        name = (match.metadata.competition or "").lower()
        if not any(item.lower() == name or item.lower() in name for item in competitions):
            return False
    return True


def build_feature_datasets(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    formats: Iterable[str] | None = None,
    competitions: Iterable[str] | None = None,
    limit: int | None = None,
    modes: Iterable[PredictionMode] | None = None,
) -> BuildReport:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = BuildReport(
        live_formats_supported=sorted(str(item) for item in MatchFormat if supports_live_states(item)),
        live_formats_planned=["TEST", "FIRST_CLASS", "OTHER"],
    )
    files = discover_match_files(input_path)
    report.matches_discovered = len(files)
    ordered = []
    for path in files:
        peeked = _peek_date(path)
        if peeked is None:
            report.matches_skipped += 1
            report.skipped["missing_or_invalid_date"] = report.skipped.get("missing_or_invalid_date", 0) + 1
            report.errors.append({"path": str(path), "reason": "missing_or_invalid_date"})
            continue
        ordered.append((peeked[0], peeked[1], path))
    ordered.sort()
    if limit is not None:
        ordered = ordered[:limit]

    format_filter = {item.upper() for item in formats} if formats else None
    competition_filter = {item for item in competitions} if competitions else None
    prediction_modes = tuple(modes or (PredictionMode.PRE_TOSS, PredictionMode.POST_TOSS))

    state = HistoricalState()
    prematch_rows: list[dict[str, Any]] = []
    live_rows: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    competition_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()

    for _match_date, _match_id, path in ordered:
        try:
            match = load_match(path)
        except CricsheetParseError as exc:
            report.matches_skipped += 1
            report.skipped["parse_error"] = report.skipped.get("parse_error", 0) + 1
            report.errors.append({"path": str(path), "reason": str(exc)})
            continue
        report.matches_parsed += 1
        if not _passes_filters(match, formats=format_filter, competitions=competition_filter):
            report.matches_skipped += 1
            report.skipped["filtered"] = report.skipped.get("filtered", 0) + 1
            continue
        format_counts[str(match.metadata.format)] += 1
        if match.metadata.competition:
            competition_counts[match.metadata.competition] += 1

        rows, reason = build_prematch_rows(match, state, modes=prediction_modes)
        if reason:
            excluded[reason] += 1
        else:
            prematch_rows.extend(rows)

        if supports_live_states(match.metadata.format):
            live_rows.extend(iter_live_states(match))
        else:
            report.skipped["live_format_planned"] = report.skipped.get("live_format_planned", 0) + 1

        state.update(match)

    prematch = pd.DataFrame(prematch_rows)
    live = pd.DataFrame(live_rows)
    if not prematch.empty:
        prematch = prematch.sort_values(["date", "match_id", "team", "prediction_mode"]).reset_index(drop=True)
    if not live.empty:
        live = live.sort_values(
            ["date", "match_id", "innings_number", "legal_balls_bowled"]
        ).reset_index(drop=True)

    report.prematch_rows = int(len(prematch))
    report.live_state_rows = int(len(live))
    report.formats = dict(format_counts)
    report.competitions = dict(competition_counts)
    report.excluded_results = dict(excluded)
    report.validation_issues = validate_prematch(prematch) + validate_live(live)

    prematch_path = output_path / "prematch_features.parquet"
    live_path = output_path / "live_states.parquet"
    prematch.to_parquet(prematch_path, index=False)
    live.to_parquet(live_path, index=False)

    payload = asdict(report)
    payload["generated_at"] = datetime.now().astimezone().isoformat()
    payload["input"] = str(input_path)
    payload["output"] = str(output_path)
    (output_path / "build_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return report
