import pandas as pd

from cricmaster.models.specialist import competition_key, specialist_bundle


def test_competition_key_normalizes_known_aliases() -> None:
    assert competition_key("Indian Premier League") == "IPL"
    assert competition_key("IPL") == "IPL"
    assert competition_key("Vitality Blast Men") == "T20 Blast"
    assert competition_key("International League T20") == "ILT20"


def test_specialist_bundle_uses_approved_t20_specialist() -> None:
    router = {
        "base_router": {
            "bundles": {
                "T20I": {"name": "international"},
                "T20": {"name": "fallback"},
            }
        },
        "specialists": {
            "IPL": {"name": "ipl-specialist"},
        },
    }

    bundle, route = specialist_bundle(
        router,
        match_format="T20",
        competition="Indian Premier League",
    )

    assert bundle["name"] == "ipl-specialist"
    assert route == "T20/IPL"


def test_specialist_bundle_falls_back_for_unknown_t20_competition() -> None:
    router = {
        "base_router": {
            "bundles": {
                "T20I": {"name": "international"},
                "T20": {"name": "fallback"},
            }
        },
        "specialists": {},
    }

    bundle, route = specialist_bundle(
        router,
        match_format="T20",
        competition="Small New League",
    )

    assert bundle["name"] == "fallback"
    assert route == "T20/fallback"


def test_specialist_bundle_keeps_t20i_on_international_model() -> None:
    router = {
        "base_router": {
            "bundles": {
                "T20I": {"name": "international"},
                "T20": {"name": "fallback"},
            }
        },
        "specialists": {"IPL": {"name": "ipl-specialist"}},
    }

    bundle, route = specialist_bundle(
        router,
        match_format="T20I",
        competition="ICC Men's T20 World Cup",
    )

    assert bundle["name"] == "international"
    assert route == "T20I"
