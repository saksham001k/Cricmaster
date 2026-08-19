"""Run the Cricmaster API locally.

    python scripts/run_api.py

Or:

    uvicorn cricmaster.api.app:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import uvicorn

from cricmaster.config import load_settings


def main() -> int:
    settings = load_settings()
    uvicorn.run(
        "cricmaster.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
