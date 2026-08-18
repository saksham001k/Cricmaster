import numpy as np
import pandas as pd
import pytest

from cricmaster.models.routed import route_bundle, sanitize_features


def test_route_bundle_dispatches_by_format() -> None:
    router = {
        "bundles": {
            "T20I": {"name": "international"},
            "T20": {"name": "domestic"},
        }
    }

    assert route_bundle(router, "T20I")["name"] == "international"
    assert route_bundle(router, "t20")["name"] == "domestic"


def test_route_bundle_rejects_unsupported_format() -> None:
    router = {"bundles": {"T20I": {}, "T20": {}}}
    with pytest.raises(ValueError):
        route_bundle(router, "HUNDRED")


def test_sanitize_features_removes_non_finite_values() -> None:
    frame = pd.DataFrame(
        {
            "a": [np.nan],
            "b": [np.inf],
            "c": [5.0],
        }
    )
    cleaned = sanitize_features(frame, ["a", "b", "c"])

    assert cleaned.loc[0, "a"] == 0.0
    assert cleaned.loc[0, "b"] == 0.0
    assert cleaned.loc[0, "c"] == 5.0
    assert np.isfinite(cleaned.to_numpy()).all()
