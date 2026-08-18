"""Live win-probability dataset preparation and match-balanced metrics."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from cricmaster.models.prematch import elo_probability

CONTEXT_FEATURES: Final[tuple[str, ...]] = (
    "matches_before",
    "win_rate_before",
    "win_rate_last_5",
    "win_rate_last_10",
    "win_rate_last_20",
    "h2h_team_win_rate",
    "team_win_rate_at_venue",
    "elo_difference",
    "xi_batters_with_history",
    "xi_mean_batting_average",
    "xi_mean_recent_runs",
    "xi_bowlers_with_history",
    "xi_mean_bowling_economy",
    "xi_mean_recent_wickets",
    "toss_bat_signed",
    "toss_field_signed",
    "is_t20i",
    "is_male",
)

COMMON_LIVE_FEATURES: Final[tuple[str, ...]] = (
    "runs",
    "wickets",
    "legal_balls_bowled",
    "balls_remaining",
    "current_run_rate",
    "wickets_in_hand",
)

FIRST_INNINGS_FEATURES: Final[tuple[str, ...]] = (
    *COMMON_LIVE_FEATURES,
    *CONTEXT_FEATURES,
)

CHASE_FEATURES: Final[tuple[str, ...]] = (
    *COMMON_LIVE_FEATURES,
    "target",
    "runs_required",
    "required_run_rate",
    "run_rate_difference",
    *CONTEXT_FEATURES,
)


def prepare_live_training_frame(
    live: pd.DataFrame,
    prematch: pd.DataFrame,
) -> pd.DataFrame:
    """Join legal-ball live states to leakage-safe POST_TOSS team context."""

    live_required = {
        "match_id",
        "date",
        "format",
        "innings_number",
        "batting_team",
        "runs",
        "wickets",
        "legal_balls_bowled",
        "balls_remaining",
        "current_run_rate",
        "target",
        "runs_required",
        "required_run_rate",
        "run_rate_difference",
        "wickets_in_hand",
        "is_legal",
        "batting_team_eventual_win",
    }
    context_required = {
        "match_id",
        "team",
        "prediction_mode",
        "gender",
        "toss_decision",
        "team_won_toss",
        "matches_before",
        "win_rate_before",
        "win_rate_last_5",
        "win_rate_last_10",
        "win_rate_last_20",
        "h2h_team_win_rate",
        "team_win_rate_at_venue",
        "elo_difference",
        "xi_batters_with_history",
        "xi_mean_batting_average",
        "xi_mean_recent_runs",
        "xi_bowlers_with_history",
        "xi_mean_bowling_economy",
        "xi_mean_recent_wickets",
    }

    missing_live = sorted(live_required - set(live.columns))
    missing_context = sorted(context_required - set(prematch.columns))
    if missing_live:
        raise ValueError(f"Missing live columns: {missing_live}")
    if missing_context:
        raise ValueError(f"Missing POST_TOSS context columns: {missing_context}")

    states = live.loc[
        live["is_legal"].astype(bool)
        & live["innings_number"].isin([1, 2])
        & live["batting_team_eventual_win"].notna()
    ].copy()

    context = prematch.loc[
        prematch["prediction_mode"] == "POST_TOSS",
        list(context_required),
    ].copy()

    context["team_won_toss"] = context["team_won_toss"].astype(bool)
    sign = np.where(context["team_won_toss"], 1.0, -1.0)
    decision = context["toss_decision"].fillna("").astype(str).str.lower()

    context["toss_bat_signed"] = np.where(decision.eq("bat"), sign, 0.0)
    context["toss_field_signed"] = np.where(
        decision.isin(["field", "bowl"]),
        sign,
        0.0,
    )
    context = context.drop(columns=["toss_decision", "team_won_toss"])
    context = context.rename(columns={"team": "batting_team"})

    merged = states.merge(
        context,
        on=["match_id", "batting_team"],
        how="inner",
        validate="many_to_one",
    )

    merged["date"] = pd.to_datetime(merged["date"], errors="raise")
    merged["batting_team_eventual_win"] = (
        merged["batting_team_eventual_win"].astype(int)
    )
    merged["is_t20i"] = (merged["format"] == "T20I").astype(float)
    merged["is_male"] = (
        merged["gender"].fillna("").astype(str).str.lower() == "male"
    ).astype(float)

    return merged.sort_values(
        ["date", "match_id", "innings_number", "legal_balls_bowled"],
        kind="stable",
    ).reset_index(drop=True)


def equal_match_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each match equal total weight within an innings model."""

    if frame.empty:
        return np.asarray([], dtype=float)
    counts = frame.groupby("match_id")["match_id"].transform("size").astype(float)
    weights = 1.0 / counts.to_numpy()
    # Normalize to mean 1 so optimizer regularization scales remain ordinary.
    return weights / weights.mean()


def weighted_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p, edges[1:-1], right=False)

    total_weight = float(w.sum())
    if total_weight <= 0:
        return float("nan")

    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if not mask.any():
            continue
        bucket_weight = float(w[mask].sum())
        observed = float(np.average(y[mask], weights=w[mask]))
        confidence = float(np.average(p[mask], weights=w[mask]))
        error += (bucket_weight / total_weight) * abs(observed - confidence)
    return float(error)


def live_probability_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    y = frame["batting_team_eventual_win"].astype(int).to_numpy()
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    w = equal_match_weights(frame)
    predicted = (p >= 0.5).astype(int)

    return {
        "states": int(len(frame)),
        "matches": int(frame["match_id"].nunique()),
        "accuracy": float(accuracy_score(y, predicted, sample_weight=w)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
        "brier_score": float(brier_score_loss(y, p, sample_weight=w)),
        "ece_10": weighted_ece(y, p, w, bins=10),
    }


def elo_live_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Static batting-team Elo probability for comparison."""

    diff = frame["elo_difference"].fillna(0.0).to_numpy(dtype=float)
    return elo_probability(diff)
