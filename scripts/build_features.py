"""Build leakage-safe historical feature datasets from Cricsheet JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.config import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR
from cricmaster.features.pipeline import build_feature_datasets
from cricmaster.features.toss import PredictionMode


def _csv_set(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build pre-match and live-state feature datasets."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_RAW_DIR / "cricsheet"),
        help="Directory of Cricsheet JSON files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Output directory for parquet files and build_report.json.",
    )
    parser.add_argument("--format", dest="formats", default=None, help="Comma-separated formats.")
    parser.add_argument(
        "--competition",
        dest="competitions",
        default=None,
        help="Comma-separated competition names.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N dated matches.")
    parser.add_argument(
        "--mode",
        choices=["both", "pre_toss", "post_toss"],
        default="both",
        help="Which toss-availability rows to emit.",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help=(
            "Build only pre/post-toss historical features. "
            "Do not generate or overwrite live_states.parquet."
        ),
    )
    args = parser.parse_args(argv)
    modes = {
        "both": (PredictionMode.PRE_TOSS, PredictionMode.POST_TOSS),
        "pre_toss": (PredictionMode.PRE_TOSS,),
        "post_toss": (PredictionMode.POST_TOSS,),
    }[args.mode]
    report = build_feature_datasets(
        args.input,
        args.output,
        formats=_csv_set(args.formats),
        competitions=_csv_set(args.competitions),
        limit=args.limit,
        modes=modes,
        include_live=not args.skip_live,
    )
    print(f"parsed={report.matches_parsed} skipped={report.matches_skipped}")
    print(f"prematch_rows={report.prematch_rows} live_state_rows={report.live_state_rows}")
    print(
        "live_generation="
        + ("enabled" if report.live_generation_enabled else "skipped")
    )
    print(f"formats={report.formats}")
    print(f"excluded={report.excluded_results}")
    print(f"validation_issues={len(report.validation_issues)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
