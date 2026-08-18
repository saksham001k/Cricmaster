"""Competition-aware hierarchical routing for Cricmaster T20 models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cricmaster.models.routed import route_bundle, symmetric_probability


_COMPETITION_ALIASES = {
    "indian premier league": "IPL",
    "ipl": "IPL",
    "pakistan super league": "PSL",
    "psl": "PSL",
    "caribbean premier league": "CPL",
    "cpl": "CPL",
    "big bash league": "BBL",
    "bbl": "BBL",
    "women's big bash league": "WBBL",
    "womens big bash league": "WBBL",
    "wbbl": "WBBL",
    "women's premier league": "WPL",
    "womens premier league": "WPL",
    "wpl": "WPL",
    "bangladesh premier league": "BPL",
    "bpl": "BPL",
    "lanka premier league": "LPL",
    "lpl": "LPL",
    "major league cricket": "MLC",
    "mlc": "MLC",
    "international league t20": "ILT20",
    "ilt20": "ILT20",
    "sa20": "SA20",
    "vitality blast men": "T20 Blast",
    "t20 blast": "T20 Blast",
    "super smash": "Super Smash",
    "women's super smash": "Women's Super Smash",
    "syed mushtaq ali trophy": "SMAT",
    "smat": "SMAT",
}


def competition_key(value: object) -> str | None:
    """Normalize a competition label for specialist routing."""

    if value is None or pd.isna(value):
        return None

    raw = " ".join(str(value).strip().split())
    if not raw:
        return None

    return _COMPETITION_ALIASES.get(raw.casefold(), raw)


def specialist_bundle(
    router: dict[str, Any],
    *,
    match_format: str,
    competition: object,
) -> tuple[dict[str, Any], str]:
    """Return the selected hierarchical bundle and a human-readable route."""

    domain = str(match_format).upper()
    base_router = router.get("base_router")
    if not isinstance(base_router, dict):
        raise ValueError("Specialist router contains no base_router")

    if domain == "T20I":
        return route_bundle(base_router, "T20I"), "T20I"

    if domain != "T20":
        raise ValueError("Competition specialists support only T20I and T20")

    key = competition_key(competition)
    specialists = router.get("specialists") or {}

    if key and key in specialists:
        return specialists[key], f"T20/{key}"

    return route_bundle(base_router, "T20"), "T20/fallback"


def specialist_probability(
    router: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    """Predict a mixed T20I/T20 frame with approved competition specialists."""

    required = {"format", "competition"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Specialist prediction frame missing columns: {missing}")

    output = np.empty(len(frame), dtype=float)

    for position, (_, row) in enumerate(frame.iterrows()):
        bundle, _route = specialist_bundle(
            router,
            match_format=str(row["format"]),
            competition=row["competition"],
        )
        output[position] = symmetric_probability(
            bundle,
            pd.DataFrame([row]),
        )[0]

    return output
