"""HTTP-facing errors for the Cricmaster API. Messages must not leak secrets."""

from __future__ import annotations


class ApiError(Exception):
    """Public API error with a stable code and HTTP status."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class LiveProviderError(ApiError):
    def __init__(self, code: str, message: str, *, status_code: int = 503) -> None:
        super().__init__(code, message, status_code=status_code)


class InsufficientLiveStateError(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "insufficient_live_state",
            message,
            status_code=422,
        )


class MatchNotFoundError(ApiError):
    def __init__(self, message: str = "No current match was found for that id.") -> None:
        super().__init__("match_not_found", message, status_code=404)
