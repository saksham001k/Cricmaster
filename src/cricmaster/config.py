"""Environment-backed configuration for Cricmaster."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

CRICSHEET_DOWNLOADS_URL = "https://cricsheet.org/downloads/"
CRICSHEET_JSON_FORMAT_URL = "https://cricsheet.org/format/json/"


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables.

    API keys are never required for the historical data layer. Live providers
    should read these later without embedding secrets in source control.
    """

    cricket_api_key: str | None
    secondary_cricket_api_key: str | None
    openai_api_key: str | None
    data_raw_dir: Path
    data_processed_dir: Path
    data_external_dir: Path
    cricsheet_downloads_url: str


def load_settings() -> Settings:
    """Load settings from the process environment."""

    return Settings(
        cricket_api_key=_optional_env("CRICKET_API_KEY"),
        secondary_cricket_api_key=_optional_env("SECONDARY_CRICKET_API_KEY"),
        openai_api_key=_optional_env("OPENAI_API_KEY"),
        data_raw_dir=Path(os.environ.get("CRICMASTER_RAW_DIR", DEFAULT_RAW_DIR)),
        data_processed_dir=Path(
            os.environ.get("CRICMASTER_PROCESSED_DIR", DEFAULT_PROCESSED_DIR)
        ),
        data_external_dir=Path(
            os.environ.get("CRICMASTER_EXTERNAL_DIR", DEFAULT_EXTERNAL_DIR)
        ),
        cricsheet_downloads_url=os.environ.get(
            "CRICSHEET_DOWNLOADS_URL", CRICSHEET_DOWNLOADS_URL
        ).rstrip("/")
        + "/",
    )
