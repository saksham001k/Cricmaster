"""Errors raised by the production prediction router."""

from __future__ import annotations


class UnsupportedPredictionError(ValueError):
    """Raised when format, mode, or competition cannot be routed."""


class ArtifactValidationError(ValueError):
    """Raised when a model bundle does not match the requested route."""
