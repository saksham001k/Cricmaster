"""Download Cricmaster's curated expanded T20 historical corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.data.download import ARCHIVES, CricsheetDownloadError, download_archive

DEFAULT_T20_ARCHIVES = (
    "t20s",
    "ipl",
    "bbl",
    "wbb",
    "psl",
    "cpl",
    "wpl",
    "bpl",
    "lpl",
    "mlc",
    "ilt",
    "sat",
    "ssm",
    "ntb",
    "sma",
)


def _archives(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_T20_ARCHIVES
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("No archive keys were supplied")
    unknown = [item for item in items if item not in ARCHIVES]
    if unknown:
        raise ValueError(f"Unknown archive keys: {unknown}")
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a curated T20/T20I corpus into a separate expansion directory."
        )
    )
    parser.add_argument(
        "--output",
        default="data/raw/cricsheet/t20_expanded",
        help="Corpus root; each archive extracts into its own subdirectory.",
    )
    parser.add_argument(
        "--archives",
        default=None,
        help="Optional comma-separated archive keys.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite archive contents.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the default expansion set and exit.",
    )
    args = parser.parse_args(argv)

    selected = _archives(args.archives)

    if args.list:
        for key in selected:
            filename, description = ARCHIVES[key]
            print(f"{key:8} {filename:20} {description}")
        return 0

    output = Path(args.output)
    print(f"Expanded corpus root: {output}")
    print(f"Archives: {', '.join(selected)}")

    failures: list[tuple[str, str]] = []

    for index, key in enumerate(selected, 1):
        _filename, description = ARCHIVES[key]
        print(f"\n[{index}/{len(selected)}] {key}: {description}")
        try:
            result = download_archive(
                archive=key,
                output_dir=output,
                force=args.force,
                extract=True,
            )
        except CricsheetDownloadError as exc:
            failures.append((key, str(exc)))
            print(f"FAILED {key}: {exc}", file=sys.stderr)
            continue

        print(
            f"ready {key}: downloaded={result.downloaded} "
            f"extracted_files={result.extracted_files}"
        )

    if failures:
        print("\nExpansion completed with failures:", file=sys.stderr)
        for key, reason in failures:
            print(f"  {key}: {reason}", file=sys.stderr)
        return 1

    json_count = sum(1 for _ in output.rglob("*.json"))
    print(f"\nExpanded corpus ready. JSON files={json_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
