"""Normalized cricket formats versus named competitions."""

from __future__ import annotations

from enum import StrEnum

# Cricsheet currently documents these match_type values:
# Test, ODI, T20, IT20, ODM, MDM.
# Competitions (IPL, BBL, PSL, ...) live in event metadata, not format.


class MatchFormat(StrEnum):
    """Normalized cricket format. Competitions are stored separately."""

    TEST = "TEST"
    ODI = "ODI"
    T20I = "T20I"
    T20 = "T20"
    T10 = "T10"
    HUNDRED = "HUNDRED"
    FIRST_CLASS = "FIRST_CLASS"
    LIST_A = "LIST_A"
    OTHER = "OTHER"


_MATCH_TYPE_MAP = {
    "test": MatchFormat.TEST,
    "odi": MatchFormat.ODI,
    "t20i": MatchFormat.T20I,
    "it20": MatchFormat.T20I,
    "t20": MatchFormat.T20,
    "t10": MatchFormat.T10,
    "the hundred": MatchFormat.HUNDRED,
    "hundred": MatchFormat.HUNDRED,
    "100": MatchFormat.HUNDRED,
    "mdm": MatchFormat.FIRST_CLASS,
    "first class": MatchFormat.FIRST_CLASS,
    "first-class": MatchFormat.FIRST_CLASS,
    "fc": MatchFormat.FIRST_CLASS,
    "odm": MatchFormat.LIST_A,
    "list a": MatchFormat.LIST_A,
    "list-a": MatchFormat.LIST_A,
}

HUNDRED_MARKERS = ("the hundred", "hundred")
T10_MARKERS = ("t10", "t:10", "t-10")

COMPETITION_ALIASES = {
    "indian premier league": "IPL",
    "ipl": "IPL",
    "tamil nadu premier league": "TNPL",
    "tnpl": "TNPL",
    "big bash league": "BBL",
    "bbl": "BBL",
    "women's big bash league": "WBBL",
    "wbbl": "WBBL",
    "pakistan super league": "PSL",
    "psl": "PSL",
    "caribbean premier league": "CPL",
    "cpl": "CPL",
    "the hundred": "The Hundred",
    "hundred": "The Hundred",
    "bangladesh premier league": "BPL",
    "lanka premier league": "LPL",
    "women's premier league": "WPL",
    "major league cricket": "MLC",
    "international league t20": "ILT20",
    "ilt20": "ILT20",
    "sa20": "SA20",
    "super smash": "Super Smash",
    "t20 blast": "T20 Blast",
    "syed mushtaq ali trophy": "SMAT",
    "county championship": "County Championship",
}


def _clean(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def normalize_competition(name: str | None) -> str | None:
    """Return a stable competition label when one can be inferred."""

    cleaned = _clean(name)
    if not cleaned:
        return None
    return COMPETITION_ALIASES.get(cleaned, name.strip() if name else None)


def normalize_match_type(
    match_type: str | None,
    *,
    team_type: str | None = None,
    competition: str | None = None,
    balls_per_over: int | None = None,
    scheduled_overs: int | None = None,
) -> MatchFormat:
    """Map Cricsheet and other source labels onto :class:`MatchFormat`.

    IPL, TNPL, BBL, PSL, and similar names are competitions, not formats.
    A club T20 such as IPL becomes ``format=T20`` and ``competition=IPL``.
    """

    competition_name = _clean(competition)
    raw_type = _clean(match_type)
    team = _clean(team_type)

    if balls_per_over == 5 or any(marker in competition_name for marker in HUNDRED_MARKERS):
        return MatchFormat.HUNDRED

    if scheduled_overs == 10 or any(marker in competition_name for marker in T10_MARKERS):
        return MatchFormat.T10

    if raw_type in _MATCH_TYPE_MAP:
        mapped = _MATCH_TYPE_MAP[raw_type]
        if mapped is MatchFormat.T20 and team == "international":
            return MatchFormat.T20I
        return mapped

    if "test" in raw_type:
        return MatchFormat.TEST
    if raw_type in {"t20i", "twenty20 international", "twenty20 internationals"}:
        return MatchFormat.T20I
    if "odi" in raw_type:
        return MatchFormat.ODI
    if "t20" in raw_type or "twenty20" in raw_type:
        if team == "international":
            return MatchFormat.T20I
        return MatchFormat.T20

    return MatchFormat.OTHER


LIMITED_OVERS_FORMATS = {
    MatchFormat.ODI,
    MatchFormat.T20I,
    MatchFormat.T20,
    MatchFormat.T10,
    MatchFormat.HUNDRED,
    MatchFormat.LIST_A,
}

UNLIMITED_OVERS_FORMATS = {
    MatchFormat.TEST,
    MatchFormat.FIRST_CLASS,
}
