from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cricmaster.api.app import create_app
from cricmaster.api.service import LiveMatchService, PredictionService
from cricmaster.live.cricketdata import normalize_current_match
from cricmaster.models.live import CHASE_FEATURES, FIRST_INNINGS_FEATURES
from cricmaster.models.posttoss import POST_TOSS_FEATURES
from cricmaster.models.prematch import MODEL_FEATURES
from cricmaster.models.roster_features import PREVIOUS_CORE_DIFFS
from cricmaster.prediction.artifacts import ProductionArtifacts
from cricmaster.prediction.history_cache import clear_parsed_match_cache

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clear_prediction_cache() -> None:
    clear_parsed_match_cache()
    yield
    clear_parsed_match_cache()


class DummyWinModel:
    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        score = values.sum(axis=1)
        probability = 1.0 / (1.0 + np.exp(-score / 50.0))
        return np.column_stack([1.0 - probability, probability])


def _leaf(
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
    if feature_family:
        bundle["feature_family"] = feature_family
    if innings_number is not None:
        bundle["innings_number"] = innings_number
    return bundle


def _catalog(tmp_path: Path) -> ProductionArtifacts:
    t20_pre = (*MODEL_FEATURES, *PREVIOUS_CORE_DIFFS)
    t20_post = (*POST_TOSS_FEATURES, *PREVIOUS_CORE_DIFFS)
    paths = {name: tmp_path / f"{name}.joblib" for name in (
        "t20i_pre", "t20i_post", "t20_pre", "t20_post", "live_first", "live_chase"
    )}
    joblib.dump(_leaf(mode="PRE_TOSS", domain="T20I", features=MODEL_FEATURES), paths["t20i_pre"])
    joblib.dump(_leaf(mode="POST_TOSS", domain="T20I", features=POST_TOSS_FEATURES), paths["t20i_post"])
    joblib.dump(
        _leaf(
            mode="PRE_TOSS",
            domain="T20",
            features=t20_pre,
            feature_family="previous_xi_core_strength",
        ),
        paths["t20_pre"],
    )
    joblib.dump(
        _leaf(
            mode="POST_TOSS",
            domain="T20",
            features=t20_post,
            feature_family="previous_xi_core_strength",
        ),
        paths["t20_post"],
    )
    joblib.dump(
        _leaf(
            mode="LIVE_AFTER_LEGAL_BALL",
            domain="LIVE",
            features=FIRST_INNINGS_FEATURES,
            innings_number=1,
        ),
        paths["live_first"],
    )
    joblib.dump(
        _leaf(
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
    shutil.copy(FIXTURES / "sample_ipl_match.json", raw / "sample.json")
    return raw


class FakeLiveProvider:
    name = "cricketdata"

    def __init__(self, matches=None, error: Exception | None = None) -> None:
        self._matches = list(matches or [])
        self._error = error
        self.last_info: dict = {}

    def current_matches(self, *, force: bool = False):
        if self._error is not None:
            raise self._error
        return list(self._matches)


def _client(tmp_path: Path, *, live_provider=None) -> TestClient:
    catalog = _catalog(tmp_path)
    raw = _corpus(tmp_path)
    prediction = PredictionService(raw_dir=raw, artifacts=catalog)
    live = LiveMatchService(provider=live_provider or FakeLiveProvider([]))
    app = create_app(prediction_service=prediction, live_service=live)
    return TestClient(app)


def test_health_returns_200_and_model_availability(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "cricmaster"
    assert body["models"]["t20i_pretoss"] is True
    assert body["models"]["t20_roster_pretoss"] is True
    assert body["models"]["live_chase"] is True


def test_models_returns_safe_metadata(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    dumped = str(body)
    assert "C:\\" not in dumped
    assert "/Users/" not in dumped
    assert body["domains"]["T20"]["PRE_TOSS"]["family"] == "roster-aware T20"
    assert any(
        "modest" in item.lower() or "limited" in item.lower()
        for item in body["domains"]["T20"]["PRE_TOSS"]["limitations"]
    )


def test_predict_t20i_pre_toss(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/predict",
        json={
            "team1": "India",
            "team2": "Australia",
            "format": "T20I",
            "mode": "PRE_TOSS",
            "date": "2026-08-20",
            "venue": "MCG",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "T20I"
    assert body["prediction_mode"] == "PRE_TOSS"
    assert body["model_family"] == "international T20I"
    assert body["team1_probability"] + body["team2_probability"] == pytest.approx(1.0)
    assert body["confidence"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["warnings"], list)


def test_predict_t20_pre_toss_and_xi_does_not_leak(tmp_path: Path) -> None:
    client = _client(tmp_path)
    base = {
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "format": "T20",
        "mode": "PRE_TOSS",
        "competition": "IPL",
        "date": "2026-08-20",
        "venue": "Wankhede Stadium",
    }
    plain = client.post("/predict", json=base)
    with_xi = client.post(
        "/predict",
        json={
            **base,
            "team1_xi": ["Rohit Sharma"] * 11,
            "team2_xi": ["MS Dhoni"] * 11,
        },
    )
    assert plain.status_code == 200
    assert with_xi.status_code == 200
    assert plain.json()["team1_probability"] == pytest.approx(
        with_xi.json()["team1_probability"]
    )
    assert with_xi.json()["model_family"] == "roster-aware T20"
    assert with_xi.json()["confidence"] == "LOW"
    assert any("current playing XI was ignored" in item for item in with_xi.json()["warnings"])


def test_post_toss_validation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    missing = client.post(
        "/predict",
        json={
            "team1": "India",
            "team2": "Australia",
            "format": "T20I",
            "mode": "POST_TOSS",
            "date": "2026-08-20",
        },
    )
    assert missing.status_code == 422
    assert missing.json()["error"] == "validation_error"
    assert missing.headers.get("X-Request-ID")

    ok = client.post(
        "/predict",
        json={
            "team1": "India",
            "team2": "Australia",
            "format": "T20I",
            "mode": "POST_TOSS",
            "date": "2026-08-20",
            "toss_winner": "India",
            "toss_decision": "field",
        },
    )
    assert ok.status_code == 200
    assert ok.json()["prediction_mode"] == "POST_TOSS"


def test_live_validation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    missing = client.post(
        "/predict",
        json={
            "team1": "India",
            "team2": "Australia",
            "format": "T20I",
            "mode": "LIVE",
            "date": "2026-08-20",
        },
    )
    assert missing.status_code == 422

    ok = client.post(
        "/predict",
        json={
            "team1": "India",
            "team2": "Australia",
            "format": "T20I",
            "mode": "LIVE",
            "date": "2026-08-20",
            "batting_team": "India",
            "innings": 2,
            "runs": 132,
            "wickets": 4,
            "overs": "15.3",
            "target": 181,
        },
    )
    assert ok.status_code == 200
    assert ok.json()["prediction_mode"] == "LIVE"
    assert ok.json()["team1_probability"] + ok.json()["team2_probability"] == pytest.approx(1.0)


def test_hundred_and_odi_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path)
    hundred = client.post(
        "/predict",
        json={
            "team1": "Oval Invincibles",
            "team2": "London Spirit",
            "format": "HUNDRED",
            "mode": "PRE_TOSS",
            "date": "2026-08-20",
        },
    )
    assert hundred.status_code == 400
    assert hundred.json()["error"] == "unsupported_format"

    odi = client.post(
        "/predict",
        json={
            "team1": "India",
            "team2": "Australia",
            "format": "ODI",
            "mode": "PRE_TOSS",
            "date": "2026-08-20",
        },
    )
    assert odi.status_code == 400
    assert odi.json()["error"] == "unsupported_format"


def test_missing_artifact_is_503(tmp_path: Path) -> None:
    missing = tmp_path / "missing.joblib"
    artifacts = ProductionArtifacts(
        t20i_pre_toss=missing,
        t20i_post_toss=missing,
        t20_pre_toss=missing,
        t20_post_toss=missing,
        live_first_innings=missing,
        live_chase=missing,
    )
    app = create_app(
        prediction_service=PredictionService(raw_dir=_corpus(tmp_path), artifacts=artifacts),
        live_service=LiveMatchService(provider=FakeLiveProvider([])),
    )
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "team1": "India",
            "team2": "Australia",
            "format": "T20I",
            "mode": "PRE_TOSS",
            "date": "2026-08-20",
        },
    )
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"
    assert "joblib" not in response.json()["message"].lower()


def test_malformed_input_returns_422_with_request_id(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post("/predict", json={})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert response.json()["request_id"]
    assert response.headers.get("X-Request-ID")


def test_chat_structured_prediction(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/chat",
        json={
            "intent": "predict_match",
            "team1": "India",
            "team2": "Australia",
            "format": "T20I",
            "mode": "PRE_TOSS",
            "date": "2026-08-20",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "predict_match"
    assert "estimated win probability" in body["message"]
    assert "definitely" not in body["message"].lower()
    assert body["prediction"]["team1"] == "India"
    assert body["suggestions"]


def test_live_matches_without_key(tmp_path: Path) -> None:
    app = create_app(
        prediction_service=PredictionService(
            raw_dir=_corpus(tmp_path),
            artifacts=_catalog(tmp_path),
        ),
        live_service=LiveMatchService(
            provider=FakeLiveProvider(error=RuntimeError("CRICKET_API_KEY is not configured"))
        ),
    )
    client = TestClient(app)
    response = client.get("/live/matches")
    assert response.status_code == 503
    assert response.json()["error"] == "live_provider_unconfigured"
    assert "apikey" not in response.json()["message"].lower()


def test_live_matches_and_predict_by_id(tmp_path: Path) -> None:
    raw = {
        "id": "m-live-1",
        "name": "Alpha vs Beta, Example T20 2026",
        "matchType": "t20",
        "status": "Beta need 25 runs in 18 balls",
        "venue": "Example Ground",
        "date": "2026-08-19",
        "teams": ["Alpha", "Beta"],
        "matchStarted": True,
        "matchEnded": False,
        "score": [
            {"r": 180, "w": 6, "o": 20, "inning": "Alpha Inning 1"},
            {"r": 156, "w": 5, "o": 17, "inning": "Beta Inning 1"},
        ],
    }
    match = normalize_current_match(raw)
    assert match is not None
    client = _client(tmp_path, live_provider=FakeLiveProvider([match]))
    listed = client.get("/live/matches")
    assert listed.status_code == 200
    assert listed.json()[0]["match_id"] == "m-live-1"
    assert listed.json()[0]["source"] == "cricketdata"

    predicted = client.post("/live/m-live-1/predict")
    assert predicted.status_code == 200
    body = predicted.json()
    assert body["prediction_mode"] == "LIVE"
    assert body["team1_probability"] + body["team2_probability"] == pytest.approx(1.0)

    missing = client.post("/live/no-such-match/predict")
    assert missing.status_code == 404


def test_insufficient_live_state_is_not_guessed(tmp_path: Path) -> None:
    raw = {
        "id": "m-upcoming",
        "name": "Alpha vs Beta, Example T20 2026",
        "matchType": "t20",
        "status": "Match starts at 19:00",
        "date": "2026-08-19",
        "teams": ["Alpha", "Beta"],
        "matchStarted": False,
        "matchEnded": False,
        "score": [],
    }
    match = normalize_current_match(raw)
    client = _client(tmp_path, live_provider=FakeLiveProvider([match]))
    response = client.post("/live/m-upcoming/predict")
    assert response.status_code == 422
    assert response.json()["error"] == "insufficient_live_state"
