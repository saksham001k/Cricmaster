"""Conservative cricket team aliases.

Only clearly documented renames are mapped. Speculative merges such as
Deccan Chargers → Sunrisers Hyderabad or Gujarat Lions → Gujarat Titans
are intentionally omitted.
"""

from __future__ import annotations

# Keys must be lowercase collapsed whitespace. Values are canonical display names.
TEAM_ALIASES: dict[str, str] = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "kings xi punjab": "Punjab Kings",
    "delhi daredevils": "Delhi Capitals",
    "rising pune supergiant": "Rising Pune Supergiants",
    "st lucia zouks": "St Lucia Kings",
    "st lucia stars": "St Lucia Kings",
    "barbados tridents": "Barbados Royals",
}


def normalize_team_key(name: str | None) -> str:
    """Collapse whitespace for alias lookup."""

    return " ".join((name or "").strip().split()).lower()


def canonicalize_team(name: str | None) -> str:
    """Return a stable team name. Unknown names are returned unchanged (trimmed)."""

    cleaned = " ".join((name or "").strip().split())
    if not cleaned:
        return ""
    return TEAM_ALIASES.get(normalize_team_key(cleaned), cleaned)
