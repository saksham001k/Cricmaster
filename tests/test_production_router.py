from __future__ import annotations

import inspect
import shutil
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from cricmaster.data.formats import MatchFormat
from cricmaster.features.history import HistoricalState
from cricmaster.models.live import CHASE_FEATURES, FIRST_INNINGS_FEATURES
from cricmaster.models.posttoss import POST_TOSS_FEATURES
from cricmaster.models.prematch import MODEL_FEATURES, SIGNED_SOURCE_FEATURES
from cricmaster.models.roster_features import PREVIOUS_CORE_DIFFS
from cricmaster.prediction.artifacts import (
    DEFAULT_T20_PRE_TOSS,
    DEFAULT_T20I_PRE_TOSS,
    ProductionArtifacts,
    extract_leaf_bundle,
    load_production_bundle,
    resolve_production_route,
    validate_model_bundle,
)
from cricmaster.prediction.confidence import (
    Confidence,
    ConfidenceInputs,
    applicable_confidence_warnings,
    assess_confidence,
)
from cricmaster.prediction.errors import ArtifactValidationError, UnsupportedPredictionError
from cricmaster.prediction.history_cache import (
    clear_parsed_match_cache,
    load_parsed_t20_matches,
    matches_strictly_before,
)
from cricmaster.prediction.prematch import (
    PredictionRequest,
    build_historical_state,
    request_features,
)
from cricmaster.prediction.result import complementary_probabilities, format_prediction_report
from cricmaster.prediction.roster_runtime import (
    features_for_bundle,
    previous_xi_core_differences,
)
from cricmaster.prediction.router import (
    ProductionRequest,
    parse_production_format,
    predict_production,
    to_prematch_request,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clear_prediction_cache() -> None:
    clear_parsed_match_cache()
    yield
    clear_parsed_match_cache()


class DummyWinModel:
    """Odd function of the feature sum so team-swap complementarity holds."""

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        score = values.sum(axis=1)
        probability = 1.0 / (1.0 + np.exp(-score / 50.0))
        return np.column_stack([1.0 - probability, probability])


def _leaf_bundle(
    *,
    mode: str,
    domain: str,
    features: tuple[str, ...] | list[str],
    feature_family: str | None = None,
    innings_number: int | None = None,
) -> dict:
    bundle: dict = {
        "model": DummyWinModel(),
        "model_name": "logistic_regression",
        "features": list(features),
        "prediction_mode": mode,
        "domain": domain,
    }
    if domain in {"T20", "T20I"}:
        bundle["format"] = domain
    if feature_family is not None:
        bundle["feature_family"] = feature_family
    if innings_number is not None:
        bundle["innings_number"] = innings_number
    return bundle


def _write_catalog(tmp_path: Path) -> ProductionArtifacts:
    t20_pre_features = (*MODEL_FEATURES, *PREVIOUS_CORE_DIFFS)
    t20_post_features = (*POST_TOSS_FEATURES, *PREVIOUS_CORE_DIFFS)
    paths = {
        "t20i_pre": tmp_path / "t20i_pre.joblib",
        "t20i_post": tmp_path / "t20i_post.joblib",
        "t20_pre": tmp_path / "t20_pre.joblib",
        "t20_post": tmp_path / "t20_post.joblib",
        "live_first": tmp_path / "live_first.joblib",
        "live_chase": tmp_path / "live_chase.joblib",
    }
    joblib.dump(_leaf_bundle(mode="PRE_TOSS", domain="T20I", features=MODEL_FEATURES), paths["t20i_pre"])
    joblib.dump(
        _leaf_bundle(mode="POST_TOSS", domain="T20I", features=POST_TOSS_FEATURES),
        paths["t20i_post"],
    )
    joblib.dump(
        _leaf_bundle(
            mode="PRE_TOSS",
            domain="T20",
            features=t20_pre_features,
            feature_family="previous_xi_core_strength",
        ),
        paths["t20_pre"],
    )
    joblib.dump(
        _leaf_bundle(
            mode="POST_TOSS",
            domain="T20",
            features=t20_post_features,
            feature_family="previous_xi_core_strength",
        ),
        paths["t20_post"],
    )
    joblib.dump(
        _leaf_bundle(
            mode="LIVE_AFTER_LEGAL_BALL",
            domain="LIVE",
            features=FIRST_INNINGS_FEATURES,
            innings_number=1,
        ),
        paths["live_first"],
    )
    joblib.dump(
        _leaf_bundle(
            mode="LIVE_AFTER_LEGAL_BALL",
            domain="LIVE",
            features=CHASE_FEATURES,
            innings_number=2,
        ),
        paths["live_chase"],
    )
    return ProductionArtifacts(
        t20i_pre_toss=paths["t20i_pre"],
        t20i_post_toss=paths["t20i_post"],
        t20_pre_toss=paths["t20_pre"],
        t20_post_toss=paths["t20_post"],
        live_first_innings=paths["live_first"],
        live_chase=paths["live_chase"],
    )


def _corpus(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(FIXTURES / "sample_ipl_match.json", raw / "sample_ipl_match.json")
    return raw


def test_t20i_pre_toss_routes_to_international_router() -> None:
    route = resolve_production_route(MatchFormat.T20I, "PRE_TOSS")
    assert route.expected_domain == "T20I"
    assert route.expected_mode == "PRE_TOSS"
    assert route.model_family == "international T20I"
    assert route.artifact_path.as_posix().endswith("routed_expanded/prematch_router.joblib")
    assert route.expected_feature_family is None


def test_t20_pre_toss_routes_to_roster_model() -> None:
    route = resolve_production_route(MatchFormat.T20, "pre_toss")
    assert route.expected_domain == "T20"
    assert route.model_family == "roster-aware T20"
    assert route.expected_feature_family == "previous_xi_core_strength"
    assert "roster_candidate/prematch_t20_roster.joblib" in route.artifact_path.as_posix()
    assert "routed_expanded" not in route.artifact_path.as_posix()


def test_t20_post_toss_routes_to_roster_model() -> None:
    route = resolve_production_route(MatchFormat.T20, "POST_TOSS")
    assert route.expected_mode == "POST_TOSS"
    assert route.model_family == "roster-aware T20"
    assert "roster_candidate/posttoss_t20_roster.joblib" in route.artifact_path.as_posix()


def test_live_routing_is_preserved() -> None:
    first = resolve_production_route(MatchFormat.T20I, "LIVE", innings_number=1)
    chase = resolve_production_route(MatchFormat.T20, "LIVE", innings_number=2)
    assert first.live_kind == "first_innings"
    assert first.expected_mode == "LIVE_AFTER_LEGAL_BALL"
    assert first.artifact_path.as_posix().endswith("live/first_innings_model.joblib")
    assert chase.live_kind == "chase"
    assert chase.artifact_path.as_posix().endswith("live/chase_model.joblib")


def test_unsupported_format_rejected() -> None:
    with pytest.raises(UnsupportedPredictionError, match="ODI"):
        parse_production_format("ODI")
    with pytest.raises(UnsupportedPredictionError):
        resolve_production_route(MatchFormat.ODI, "PRE_TOSS")


def test_hundred_is_not_treated_as_t20() -> None:
    with pytest.raises(UnsupportedPredictionError, match="Hundred"):
        parse_production_format("HUNDRED")
    with pytest.raises(UnsupportedPredictionError, match="Hundred"):
        parse_production_format("T20", competition="The Hundred")
    with pytest.raises(UnsupportedPredictionError):
        resolve_production_route(MatchFormat.HUNDRED, "PRE_TOSS")


def test_complementary_probabilities() -> None:
    p1, p2 = complementary_probabilities(0.527)
    assert p1 + p2 == pytest.approx(1.0, abs=1e-12)


def test_pre_toss_request_features_never_attach_current_xi(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_features_for(self: HistoricalState, match: object, team: str, opponent: str) -> dict:
        metadata = match.metadata  # type: ignore[attr-defined]
        seen["team1_players"] = metadata.team1_players
        seen["team2_players"] = metadata.team2_players
        return {name: 0.0 for name in SIGNED_SOURCE_FEATURES}

    monkeypatch.setattr(HistoricalState, "features_for", fake_features_for)
    request = PredictionRequest(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        match_format=MatchFormat.T20,
        match_date=date(2026, 8, 20),
        gender="male",
    )
    request_features(HistoricalState(), request)
    assert seen["team1_players"] is None
    assert seen["team2_players"] is None


def test_pre_toss_helper_drops_supplied_current_xi() -> None:
    request = ProductionRequest(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        match_format=MatchFormat.T20,
        prediction_mode="PRE_TOSS",
        match_date=date(2026, 8, 20),
        team1_xi=("Rohit Sharma",),
        team2_xi=("MS Dhoni",),
    )
    pre = to_prematch_request(request)
    assert not hasattr(pre, "team1_xi")
    sig = inspect.signature(previous_xi_core_differences)
    assert "current" not in sig.parameters
    assert "xi" not in sig.parameters


def test_artifact_mode_mismatch_rejected() -> None:
    bundle = _leaf_bundle(mode="POST_TOSS", domain="T20I", features=MODEL_FEATURES)
    with pytest.raises(ArtifactValidationError, match="prediction_mode"):
        validate_model_bundle(
            bundle,
            expected_mode="PRE_TOSS",
            expected_domain="T20I",
        )


def test_artifact_format_domain_mismatch_rejected() -> None:
    roster = _leaf_bundle(
        mode="PRE_TOSS",
        domain="T20",
        features=MODEL_FEATURES,
        feature_family="previous_xi_core_strength",
    )
    with pytest.raises(ArtifactValidationError, match="domain/format"):
        validate_model_bundle(
            roster,
            expected_mode="PRE_TOSS",
            expected_domain="T20I",
        )

    router = {
        "prediction_mode": "PRE_TOSS",
        "bundles": {
            "T20I": _leaf_bundle(mode="PRE_TOSS", domain="T20I", features=MODEL_FEATURES),
            "T20": _leaf_bundle(mode="PRE_TOSS", domain="T20", features=MODEL_FEATURES),
        },
    }
    leaf = extract_leaf_bundle(router, expected_mode="PRE_TOSS", expected_domain="T20I")
    assert leaf["domain"] == "T20I"


def test_low_confidence_close_t20_prediction() -> None:
    assessment = assess_confidence(
        ConfidenceInputs(
            winner_probability=0.53,
            match_format=MatchFormat.T20,
            prediction_mode="PRE_TOSS",
            team1_history_matches=80,
            team2_history_matches=90,
            venue_known=True,
            previous_xi_complete=True,
        )
    )
    assert assessment.label is Confidence.LOW
    warnings = applicable_confidence_warnings(
        assessment=assessment,
        match_format=MatchFormat.T20,
        prediction_mode="PRE_TOSS",
        winner_probability=0.53,
        previous_xi_complete=True,
        ignored_current_xi=False,
    )
    assert "prediction is close" in warnings
    assert "franchise PRE_TOSS model has limited discriminatory power" in warnings


def test_stronger_supported_t20i_prediction_can_be_higher_confidence() -> None:
    assessment = assess_confidence(
        ConfidenceInputs(
            winner_probability=0.74,
            match_format=MatchFormat.T20I,
            prediction_mode="PRE_TOSS",
            team1_history_matches=80,
            team2_history_matches=90,
            venue_known=True,
        )
    )
    assert assessment.label is Confidence.HIGH


def test_missing_historical_roster_information_warns(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    raw = _corpus(tmp_path)
    result = predict_production(
        ProductionRequest(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            match_format=MatchFormat.T20,
            prediction_mode="PRE_TOSS",
            match_date=date(2024, 3, 1),
            gender="male",
            venue="Wankhede Stadium",
        ),
        raw_dir=raw,
        artifacts=catalog,
    )
    assert result.confidence == "LOW"
    assert result.previous_xi_team1_known is False
    assert result.previous_xi_team2_known is False
    assert any("previous-XI" in warning for warning in result.warnings)


def test_production_predict_t20i_pre_toss(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    raw = _corpus(tmp_path)
    result = predict_production(
        ProductionRequest(
            team1="India",
            team2="Australia",
            match_format=MatchFormat.T20I,
            prediction_mode="PRE_TOSS",
            match_date=date(2026, 8, 20),
            gender="male",
            venue="Melbourne Cricket Ground",
        ),
        raw_dir=raw,
        artifacts=catalog,
    )
    assert result.model_family == "international T20I"
    assert result.prediction_mode == "PRE_TOSS"
    assert result.match_format == "T20I"
    assert result.team1_probability + result.team2_probability == pytest.approx(1.0, abs=1e-12)
    assert "franchise PRE_TOSS" not in " ".join(result.warnings)


def test_production_predict_t20_and_live(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    raw = _corpus(tmp_path)

    pre = predict_production(
        ProductionRequest(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            match_format=MatchFormat.T20,
            prediction_mode="PRE_TOSS",
            match_date=date(2026, 8, 20),
            gender="male",
            competition="IPL",
            venue="Wankhede Stadium",
            team1_xi=("Fake Batter",),
            team2_xi=("Another Fake",),
        ),
        raw_dir=raw,
        artifacts=catalog,
    )
    assert pre.model_family == "roster-aware T20"
    assert pre.prediction_mode == "PRE_TOSS"
    assert any("current playing XI was ignored" in warning for warning in pre.warnings)
    assert pre.team1_probability + pre.team2_probability == pytest.approx(1.0, abs=1e-12)

    without_xi = predict_production(
        ProductionRequest(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            match_format=MatchFormat.T20,
            prediction_mode="PRE_TOSS",
            match_date=date(2026, 8, 20),
            gender="male",
            competition="IPL",
            venue="Wankhede Stadium",
        ),
        raw_dir=raw,
        artifacts=catalog,
    )
    assert pre.team1_probability == pytest.approx(without_xi.team1_probability)

    post = predict_production(
        ProductionRequest(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            match_format=MatchFormat.T20,
            prediction_mode="POST_TOSS",
            match_date=date(2026, 8, 20),
            gender="male",
            competition="IPL",
            venue="Wankhede Stadium",
            toss_winner="Mumbai Indians",
            toss_decision="field",
        ),
        raw_dir=raw,
        artifacts=catalog,
    )
    assert post.model_family == "roster-aware T20"
    assert post.prediction_mode == "POST_TOSS"

    live = predict_production(
        ProductionRequest(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            match_format=MatchFormat.T20,
            prediction_mode="LIVE",
            match_date=date(2026, 8, 20),
            gender="male",
            batting_team="Mumbai Indians",
            innings_number=1,
            runs=45,
            wickets=1,
            legal_balls=36,
            venue="Wankhede Stadium",
        ),
        raw_dir=raw,
        artifacts=catalog,
    )
    assert live.prediction_mode == "LIVE"
    assert live.model_kind == "first_innings"
    assert live.team1_probability + live.team2_probability == pytest.approx(1.0, abs=1e-12)
    report = format_prediction_report(pre)
    assert "Confidence:" in report
    assert "roster-aware T20" in report


def test_cache_does_not_apply_matches_on_or_after_cutoff(tmp_path: Path) -> None:
    raw = _corpus(tmp_path)
    parsed = load_parsed_t20_matches(raw)
    assert len(parsed.matches) == 1
    assert matches_strictly_before(parsed, date(2024, 4, 1)) == ()
    applied = build_historical_state(raw, cutoff=date(2024, 4, 1))
    assert applied.matches_applied == 0
    later = build_historical_state(raw, cutoff=date(2024, 4, 2))
    assert later.matches_applied == 1


def test_features_for_bundle_uses_artifact_order() -> None:
    frame = pd.DataFrame([{"b": 2.0, "a": 1.0, "extra": 9.0}])
    aligned = features_for_bundle(frame, {"features": ["a", "missing", "b"]})
    assert list(aligned.columns) == ["a", "missing", "b"]
    assert np.isnan(aligned.loc[0, "missing"])


@pytest.mark.skipif(not DEFAULT_T20I_PRE_TOSS.is_file(), reason="T20I router artifact missing")
def test_real_t20i_router_leaf_validates() -> None:
    route = resolve_production_route(MatchFormat.T20I, "PRE_TOSS")
    bundle = load_production_bundle(route)
    assert bundle["prediction_mode"] == "PRE_TOSS"
    assert bundle.get("domain") == "T20I"
    assert bundle.get("format") != "T20" or bundle.get("domain") == "T20I"


@pytest.mark.skipif(not DEFAULT_T20_PRE_TOSS.is_file(), reason="roster artifact missing")
def test_real_t20_roster_bundle_validates() -> None:
    route = resolve_production_route(MatchFormat.T20, "PRE_TOSS")
    bundle = load_production_bundle(route)
    assert bundle["format"] == "T20"
    assert bundle["feature_family"] == "previous_xi_core_strength"
    assert bundle["features"][-4:] == list(PREVIOUS_CORE_DIFFS)
