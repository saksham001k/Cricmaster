"""Format-aware model routing for expanded Cricmaster T20 data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_ROUTED_FORMATS = ("T20I", "T20")


def sanitize_features(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Apply Cricmaster's zero-imputation runtime convention."""

    x = frame.loc[:, list(features)].copy()
    for column in features:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    return (
        x.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )


def symmetric_probability(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    """Return Team-A probability while enforcing team-swap complementarity."""

    model = bundle.get("model")
    if model is None:
        raise ValueError("Model bundle contains no fitted estimator")

    features = list(bundle["features"])
    x = sanitize_features(frame, features)

    forward = model.predict_proba(x)[:, 1]
    reverse = model.predict_proba(-x)[:, 1]
    return np.asarray(
        0.5 * (forward + (1.0 - reverse)),
        dtype=float,
    )


def route_bundle(
    router: dict[str, Any],
    match_format: str,
) -> dict[str, Any]:
    """Select the independently trained model for T20I or domestic/franchise T20."""

    domain = str(match_format).upper()
    if domain not in SUPPORTED_ROUTED_FORMATS:
        raise ValueError(
            f"Routed model supports only {SUPPORTED_ROUTED_FORMATS}; got {match_format!r}"
        )

    bundles = router.get("bundles")
    if not isinstance(bundles, dict) or domain not in bundles:
        raise ValueError(f"Router contains no bundle for {domain}")
    return bundles[domain]


def routed_probability(
    router: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    """Predict a mixed T20I/T20 frame by dispatching each domain independently."""

    if "format" not in frame.columns:
        raise ValueError("Routed prediction frame requires a format column")

    output = np.empty(len(frame), dtype=float)

    for domain, indices in frame.groupby("format", sort=False).groups.items():
        bundle = route_bundle(router, str(domain))
        positions = frame.index.get_indexer(indices)
        if (positions < 0).any():
            raise ValueError("Could not align routed prediction indices")
        subset = frame.loc[indices]
        output[positions] = symmetric_probability(bundle, subset)

    return output
