"""Load and validate production model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from cricmaster.data.formats import MatchFormat
from cricmaster.models.routed import route_bundle
from cricmaster.prediction.errors import ArtifactValidationError, UnsupportedPredictionError


FROZEN_T20_FEATURE_FAMILY = "previous_xi_core_strength"
LIVE_PREDICTION_MODE = "LIVE_AFTER_LEGAL_BALL"

DEFAULT_T20I_PRE_TOSS = Path("models/routed_expanded/prematch_router.joblib")
DEFAULT_T20I_POST_TOSS = Path("models/routed_expanded/posttoss_router.joblib")
DEFAULT_T20_PRE_TOSS = Path("models/roster_candidate/prematch_t20_roster.joblib")
DEFAULT_T20_POST_TOSS = Path("models/roster_candidate/posttoss_t20_roster.joblib")
DEFAULT_LIVE_FIRST = Path("models/live/first_innings_model.joblib")
DEFAULT_LIVE_CHASE = Path("models/live/chase_model.joblib")


@dataclass(frozen=True)
class ProductionArtifacts:
    t20i_pre_toss: Path = DEFAULT_T20I_PRE_TOSS
    t20i_post_toss: Path = DEFAULT_T20I_POST_TOSS
    t20_pre_toss: Path = DEFAULT_T20_PRE_TOSS
    t20_post_toss: Path = DEFAULT_T20_POST_TOSS
    live_first_innings: Path = DEFAULT_LIVE_FIRST
    live_chase: Path = DEFAULT_LIVE_CHASE


@dataclass(frozen=True)
class ProductionRoute:
    match_format: MatchFormat
    prediction_mode: str
    model_family: str
    artifact_path: Path
    expected_mode: str
    expected_domain: str
    expected_feature_family: str | None = None
    innings_number: int | None = None
    live_kind: str | None = None


def parse_prediction_mode(value: str) -> str:
    normalized = value.strip().upper().replace("-", "_")
    aliases = {
        "PRE_TOSS": "PRE_TOSS",
        "PRETOSS": "PRE_TOSS",
        "POST_TOSS": "POST_TOSS",
        "POSTTOSS": "POST_TOSS",
        "LIVE": "LIVE",
    }
    if normalized not in aliases:
        raise UnsupportedPredictionError(
            f"Unsupported prediction mode {value!r}. "
            "Allowed: PRE_TOSS, POST_TOSS, LIVE."
        )
    return aliases[normalized]


def resolve_production_route(
    match_format: MatchFormat,
    prediction_mode: str,
    *,
    artifacts: ProductionArtifacts | None = None,
    innings_number: int | None = None,
) -> ProductionRoute:
    """Choose the production artifact for format + mode. Does not load it."""

    mode = parse_prediction_mode(prediction_mode)
    catalog = artifacts or ProductionArtifacts()

    if match_format is MatchFormat.T20I and mode == "PRE_TOSS":
        return ProductionRoute(
            match_format=match_format,
            prediction_mode=mode,
            model_family="international T20I",
            artifact_path=catalog.t20i_pre_toss,
            expected_mode="PRE_TOSS",
            expected_domain="T20I",
        )
    if match_format is MatchFormat.T20I and mode == "POST_TOSS":
        return ProductionRoute(
            match_format=match_format,
            prediction_mode=mode,
            model_family="international T20I",
            artifact_path=catalog.t20i_post_toss,
            expected_mode="POST_TOSS",
            expected_domain="T20I",
        )
    if match_format is MatchFormat.T20 and mode == "PRE_TOSS":
        return ProductionRoute(
            match_format=match_format,
            prediction_mode=mode,
            model_family="roster-aware T20",
            artifact_path=catalog.t20_pre_toss,
            expected_mode="PRE_TOSS",
            expected_domain="T20",
            expected_feature_family=FROZEN_T20_FEATURE_FAMILY,
        )
    if match_format is MatchFormat.T20 and mode == "POST_TOSS":
        return ProductionRoute(
            match_format=match_format,
            prediction_mode=mode,
            model_family="roster-aware T20",
            artifact_path=catalog.t20_post_toss,
            expected_mode="POST_TOSS",
            expected_domain="T20",
            expected_feature_family=FROZEN_T20_FEATURE_FAMILY,
        )
    if mode == "LIVE":
        if innings_number not in {1, 2}:
            raise UnsupportedPredictionError(
                "LIVE routing requires innings_number 1 (first innings) or 2 (chase)."
            )
        live_kind = "first_innings" if innings_number == 1 else "chase"
        path = (
            catalog.live_first_innings
            if innings_number == 1
            else catalog.live_chase
        )
        family = (
            "live first innings"
            if innings_number == 1
            else "live chase"
        )
        return ProductionRoute(
            match_format=match_format,
            prediction_mode=mode,
            model_family=family,
            artifact_path=path,
            expected_mode=LIVE_PREDICTION_MODE,
            expected_domain="LIVE",
            innings_number=innings_number,
            live_kind=live_kind,
        )

    raise UnsupportedPredictionError(
        f"No production route for format={match_format.value} mode={mode}."
    )


def recorded_domain(bundle: dict[str, Any]) -> str | None:
    for key in ("format", "domain"):
        value = bundle.get(key)
        if value:
            return str(value).upper()
    return None


def validate_model_bundle(
    bundle: dict[str, Any],
    *,
    expected_mode: str,
    expected_domain: str,
    expected_feature_family: str | None = None,
    innings_number: int | None = None,
) -> dict[str, Any]:
    """Reject mode / domain / feature-family mismatches before prediction."""

    if not isinstance(bundle, dict):
        raise ArtifactValidationError("Model artifact must be a mapping")

    required = {"model_name", "features", "prediction_mode"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ArtifactValidationError(
            f"Invalid model bundle; missing fields: {missing}"
        )

    recorded_mode = str(bundle["prediction_mode"])
    if recorded_mode != expected_mode:
        raise ArtifactValidationError(
            f"Artifact prediction_mode is {recorded_mode!r}, "
            f"expected {expected_mode!r}"
        )

    if not bundle.get("features"):
        raise ArtifactValidationError("Model bundle has an empty feature list")

    model_name = str(bundle["model_name"])
    if model_name != "elo_baseline" and bundle.get("model") is None:
        raise ArtifactValidationError(
            f"Model bundle {model_name!r} contains no fitted estimator"
        )

    domain = recorded_domain(bundle)
    if expected_domain == "LIVE":
        if innings_number is not None and int(bundle.get("innings_number", -1)) != innings_number:
            raise ArtifactValidationError(
                f"Live artifact is for innings {bundle.get('innings_number')}, "
                f"request is innings {innings_number}"
            )
        return bundle

    if domain is not None and domain != expected_domain:
        raise ArtifactValidationError(
            f"Artifact domain/format is {domain!r}, expected {expected_domain!r}. "
            "T20 roster models cannot be used for T20I, and T20I models cannot "
            "be used for franchise T20."
        )

    recorded_family = bundle.get("feature_family")
    if expected_feature_family:
        if recorded_family != expected_feature_family:
            raise ArtifactValidationError(
                f"Artifact feature_family is {recorded_family!r}, "
                f"expected {expected_feature_family!r}"
            )
    elif recorded_family == FROZEN_T20_FEATURE_FAMILY and expected_domain == "T20I":
        raise ArtifactValidationError(
            "Roster-aware T20 artifacts cannot be used for T20I predictions"
        )

    return bundle


def extract_leaf_bundle(
    artifact: dict[str, Any],
    *,
    expected_mode: str,
    expected_domain: str,
) -> dict[str, Any]:
    """Unwrap a format router, or return a leaf bundle unchanged."""

    if "bundles" in artifact:
        router_mode = artifact.get("prediction_mode")
        if router_mode is not None and str(router_mode) != expected_mode:
            raise ArtifactValidationError(
                f"Router prediction_mode is {router_mode!r}, expected {expected_mode!r}"
            )
        if expected_domain == "LIVE":
            raise ArtifactValidationError("Live models are not format routers")
        try:
            return route_bundle(artifact, expected_domain)
        except ValueError as exc:
            raise ArtifactValidationError(str(exc)) from exc
    return artifact


_BUNDLE_CACHE: dict[tuple[str, int, int, str, str, str | None, int | None], dict[str, Any]] = {}


def clear_bundle_cache() -> None:
    _BUNDLE_CACHE.clear()


def resolved_production_artifacts(root: str | Path) -> ProductionArtifacts:
    """Resolve default artifact locations against a project root."""

    base = Path(root)
    return ProductionArtifacts(
        t20i_pre_toss=base / DEFAULT_T20I_PRE_TOSS,
        t20i_post_toss=base / DEFAULT_T20I_POST_TOSS,
        t20_pre_toss=base / DEFAULT_T20_PRE_TOSS,
        t20_post_toss=base / DEFAULT_T20_POST_TOSS,
        live_first_innings=base / DEFAULT_LIVE_FIRST,
        live_chase=base / DEFAULT_LIVE_CHASE,
    )


def artifact_availability(artifacts: ProductionArtifacts | None = None) -> dict[str, bool]:
    """Cheap existence checks. Does not load estimators or historical data."""

    catalog = artifacts or ProductionArtifacts()
    return {
        "t20i_pretoss": Path(catalog.t20i_pre_toss).is_file(),
        "t20i_posttoss": Path(catalog.t20i_post_toss).is_file(),
        "t20_roster_pretoss": Path(catalog.t20_pre_toss).is_file(),
        "t20_roster_posttoss": Path(catalog.t20_post_toss).is_file(),
        "live_first_innings": Path(catalog.live_first_innings).is_file(),
        "live_chase": Path(catalog.live_chase).is_file(),
    }


def load_production_bundle(route: ProductionRoute) -> dict[str, Any]:
    path = Path(route.artifact_path)
    if not path.is_file():
        raise ArtifactValidationError("Model artifact not found")

    stat = path.stat()
    cache_key = (
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        route.expected_mode,
        route.expected_domain,
        route.expected_feature_family,
        route.innings_number,
    )
    cached = _BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ArtifactValidationError("Invalid model artifact")

    leaf = extract_leaf_bundle(
        artifact,
        expected_mode=route.expected_mode,
        expected_domain=route.expected_domain,
    )
    validated = validate_model_bundle(
        leaf,
        expected_mode=route.expected_mode,
        expected_domain=route.expected_domain,
        expected_feature_family=route.expected_feature_family,
        innings_number=route.innings_number,
    )
    _BUNDLE_CACHE[cache_key] = validated
    return validated
