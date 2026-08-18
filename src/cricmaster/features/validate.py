"""Validate processed datasets without deleting suspicious rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cricmaster.data.formats import MatchFormat


def _issue(table: str, check: str, count: int, examples: list[Any]) -> dict[str, Any]:
    return {
        "table": table,
        "check": check,
        "count": int(count),
        "examples": examples[:5],
    }


def validate_prematch(frame: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if frame.empty:
        return issues
    known_formats = {item.value for item in MatchFormat}
    if frame["match_id"].isna().any():
        issues.append(_issue("prematch", "missing_match_id", int(frame["match_id"].isna().sum()), []))
    dup = frame.duplicated(subset=["match_id", "team", "prediction_mode"])
    if dup.any():
        issues.append(
            _issue(
                "prematch",
                "duplicate_prediction_rows",
                int(dup.sum()),
                frame.loc[dup, "match_id"].astype(str).head().tolist(),
            )
        )
    unknown = ~frame["format"].isin(known_formats)
    if unknown.any():
        issues.append(
            _issue(
                "prematch",
                "unknown_formats",
                int(unknown.sum()),
                frame.loc[unknown, "format"].astype(str).head().tolist(),
            )
        )
    if "team_win" in frame:
        bad_target = ~frame["team_win"].isin([0, 1])
        if bad_target.any():
            issues.append(_issue("prematch", "invalid_team_win", int(bad_target.sum()), []))
    if "prediction_mode" in frame.columns:
        for mode, group in frame.groupby("prediction_mode"):
            mode_winners = group.groupby("match_id")["team_win"].sum()
            bad = mode_winners[mode_winners != 1]
            if len(bad):
                issues.append(
                    _issue(
                        "prematch",
                        f"inconsistent_winners_{mode}",
                        int(len(bad)),
                        bad.index.astype(str).tolist()[:5],
                    )
                )
    return issues


def validate_live(frame: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if frame.empty:
        return issues
    if frame["match_id"].isna().any():
        issues.append(_issue("live", "missing_match_id", int(frame["match_id"].isna().sum()), []))
    if (frame["wickets"] > 10).any():
        issues.append(
            _issue(
                "live",
                "wickets_gt_10",
                int((frame["wickets"] > 10).sum()),
                frame.loc[frame["wickets"] > 10, "match_id"].astype(str).head().tolist(),
            )
        )
    if (frame["wickets_in_hand"] < 0).any():
        issues.append(
            _issue("live", "negative_wickets_in_hand", int((frame["wickets_in_hand"] < 0).sum()), [])
        )
    remaining = frame["balls_remaining"].dropna()
    if (remaining < 0).any():
        issues.append(_issue("live", "negative_balls_remaining", int((remaining < 0).sum()), []))
    if (frame["runs"] < 0).any():
        issues.append(_issue("live", "negative_runs", int((frame["runs"] < 0).sum()), []))
    invalid_target = frame["target"].notna() & (frame["target"] <= 0)
    if invalid_target.any():
        issues.append(_issue("live", "invalid_targets", int(invalid_target.sum()), []))
    first_innings_target = (frame["innings_number"] == 1) & frame["target"].notna()
    if first_innings_target.any():
        issues.append(
            _issue("live", "first_innings_has_target", int(first_innings_target.sum()), [])
        )
    return issues
