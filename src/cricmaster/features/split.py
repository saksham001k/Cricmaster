"""Temporal train/validation/test split that keeps match rows together."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def _as_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def temporal_split(
    frame: pd.DataFrame,
    *,
    train_end: date | str,
    valid_end: date | str | None = None,
    date_column: str = "date",
    match_id_column: str = "match_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by match date.

    All rows sharing a match_id stay in the same split. Random splitting is
    not used because cricket features are time-ordered.
    """

    if match_id_column not in frame.columns or date_column not in frame.columns:
        raise ValueError("temporal_split requires date and match_id columns")
    train_cutoff = _as_date(train_end)
    valid_cutoff = _as_date(valid_end) if valid_end is not None else None
    if train_cutoff is None:
        raise ValueError("train_end must be a valid date")
    if valid_cutoff is not None and valid_cutoff < train_cutoff:
        raise ValueError("valid_end must be on or after train_end")

    match_dates = (
        frame[[match_id_column, date_column]]
        .assign(_date=lambda df: df[date_column].map(_as_date))
        .dropna(subset=["_date"])
        .groupby(match_id_column, sort=False)["_date"]
        .min()
    )
    train_ids = set(match_dates[match_dates <= train_cutoff].index)
    if valid_cutoff is None:
        valid_ids: set[Any] = set()
        test_ids = set(match_dates[match_dates > train_cutoff].index)
    else:
        valid_ids = set(
            match_dates[(match_dates > train_cutoff) & (match_dates <= valid_cutoff)].index
        )
        test_ids = set(match_dates[match_dates > valid_cutoff].index)

    train = frame[frame[match_id_column].isin(train_ids)].copy()
    valid = frame[frame[match_id_column].isin(valid_ids)].copy()
    test = frame[frame[match_id_column].isin(test_ids)].copy()
    return train, valid, test
