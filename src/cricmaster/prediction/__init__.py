"""Prediction runtime interfaces for Cricmaster."""

from cricmaster.prediction.artifacts import (
    ProductionArtifacts,
    ProductionRoute,
    resolve_production_route,
)
from cricmaster.prediction.errors import ArtifactValidationError, UnsupportedPredictionError
from cricmaster.prediction.result import ProductionPredictionResult, format_prediction_report
from cricmaster.prediction.router import ProductionRequest, predict_production

__all__ = [
    "ArtifactValidationError",
    "ProductionArtifacts",
    "ProductionPredictionResult",
    "ProductionRequest",
    "ProductionRoute",
    "UnsupportedPredictionError",
    "format_prediction_report",
    "predict_production",
    "resolve_production_route",
]
