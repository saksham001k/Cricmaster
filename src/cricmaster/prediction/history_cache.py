"""Parsed-match cache for runtime prediction.

Cricsheet JSON parsing is expensive. Caching parsed matches keyed by a corpus
fingerprint avoids re-reading thousands of files for successive predictions.

HistoricalState is still rebuilt for each cutoff from matches strictly before
that date. Parsed matches dated on or after the prediction date may sit in the
cache, but they are never applied to prediction state.

Do not cache a mutable HistoricalState from a later cutoff and reuse it for an
earlier date — that would leak future results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from cricmaster.data.cricsheet import CricsheetParseError, discover_match_files, load_match
from cricmaster.data.formats import MatchFormat
from cricmaster.data.models import MatchState

SUPPORTED_HISTORY_FORMATS = {MatchFormat.T20I, MatchFormat.T20}

_PARSED_CACHE: dict["CorpusFingerprint", "ParsedCorpus"] = {}


@dataclass(frozen=True)
class CorpusFingerprint:
    resolved_path: str
    file_count: int
    newest_mtime_ns: int
    total_size: int


@dataclass(frozen=True)
class ParsedCorpus:
    fingerprint: CorpusFingerprint
    matches: tuple[MatchState, ...]
    parse_errors: int


def corpus_fingerprint(raw_dir: str | Path) -> CorpusFingerprint:
    root = Path(raw_dir)
    files = discover_match_files(root)
    newest = 0
    total = 0
    for path in files:
        stat = path.stat()
        newest = max(newest, int(stat.st_mtime_ns))
        total += int(stat.st_size)
    resolved = str(root.resolve()) if root.exists() else str(root)
    return CorpusFingerprint(
        resolved_path=resolved,
        file_count=len(files),
        newest_mtime_ns=newest,
        total_size=total,
    )


def clear_parsed_match_cache() -> None:
    _PARSED_CACHE.clear()


def load_parsed_t20_matches(raw_dir: str | Path) -> ParsedCorpus:
    """Load T20/T20I matches from disk, using the fingerprint cache when valid."""

    fingerprint = corpus_fingerprint(raw_dir)
    cached = _PARSED_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    matches: list[MatchState] = []
    parse_errors = 0
    for path in discover_match_files(raw_dir):
        try:
            match = load_match(path)
        except CricsheetParseError:
            parse_errors += 1
            continue
        if match.metadata.format not in SUPPORTED_HISTORY_FORMATS:
            continue
        matches.append(match)

    matches.sort(
        key=lambda item: (
            item.metadata.date or date.min,
            item.metadata.match_id,
        )
    )
    parsed = ParsedCorpus(
        fingerprint=fingerprint,
        matches=tuple(matches),
        parse_errors=parse_errors,
    )
    _PARSED_CACHE[fingerprint] = parsed
    return parsed


def matches_strictly_before(
    parsed: ParsedCorpus,
    cutoff: date,
    *,
    formats: set[MatchFormat] | None = None,
) -> tuple[MatchState, ...]:
    allowed = formats or SUPPORTED_HISTORY_FORMATS
    return tuple(
        match
        for match in parsed.matches
        if match.metadata.date is not None
        and match.metadata.date < cutoff
        and match.metadata.format in allowed
    )
