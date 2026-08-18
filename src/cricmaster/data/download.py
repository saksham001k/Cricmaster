"""Download official Cricsheet JSON archives.

Source of truth: https://cricsheet.org/downloads/
Format documentation: https://cricsheet.org/format/json/
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from cricmaster.config import CRICSHEET_DOWNLOADS_URL, DEFAULT_RAW_DIR, load_settings

ARCHIVES: dict[str, tuple[str, str]] = {
    "t20s": ("t20s_json.zip", "T20 internationals"),
    "it20s": ("it20s_json.zip", "Non-official T20 internationals"),
    "odis": ("odis_json.zip", "One-day internationals"),
    "tests": ("tests_json.zip", "Test matches"),
    "mdms": ("mdms_json.zip", "Other multi-day matches"),
    "odms": ("odms_json.zip", "Other one-day matches"),
    "ipl": ("ipl_json.zip", "Indian Premier League"),
    "bbl": ("bbl_json.zip", "Big Bash League"),
    "wbb": ("wbb_json.zip", "Women's Big Bash League"),
    "psl": ("psl_json.zip", "Pakistan Super League"),
    "cpl": ("cpl_json.zip", "Caribbean Premier League"),
    "hnd": ("hnd_json.zip", "The Hundred"),
    "wpl": ("wpl_json.zip", "Women's Premier League"),
    "bpl": ("bpl_json.zip", "Bangladesh Premier League"),
    "lpl": ("lpl_json.zip", "Lanka Premier League"),
    "mlc": ("mlc_json.zip", "Major League Cricket"),
    "ilt": ("ilt_json.zip", "International League T20"),
    "sat": ("sat_json.zip", "SA20"),
    "ssm": ("ssm_json.zip", "Super Smash"),
    "ntb": ("ntb_json.zip", "T20 Blast"),
    "sma": ("sma_json.zip", "Syed Mushtaq Ali Trophy"),
    "cch": ("cch_json.zip", "County Championship"),
    "recently_played_2": ("recently_played_2_json.zip", "2 most recently played matches"),
    "all": ("all_json.zip", "Every published match (very large)"),
}

DEFAULT_ARCHIVE = "t20s"
ALLOWED_SUFFIXES = {".json", ".txt", ".md"}


@dataclass(frozen=True, slots=True)
class DownloadResult:
    archive: str
    zip_path: Path
    extract_dir: Path
    downloaded: bool
    extracted_files: int


class CricsheetDownloadError(RuntimeError):
    """Raised when an official archive cannot be fetched or unpacked."""


def list_archives() -> dict[str, tuple[str, str]]:
    return dict(ARCHIVES)


def archive_url(archive: str, base_url: str = CRICSHEET_DOWNLOADS_URL) -> str:
    key = archive.strip().lower()
    if key not in ARCHIVES:
        raise CricsheetDownloadError(
            f"Unknown archive '{archive}'. Use --list to see supported names."
        )
    filename, _description = ARCHIVES[key]
    return f"{base_url.rstrip('/')}/{filename}"


def _is_within(directory: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(directory)
        return True
    except ValueError:
        return False


def _safe_destination(extract_dir: Path, member_name: str) -> Path | None:
    raw = Path(member_name)
    if raw.is_absolute() or ".." in raw.parts:
        return None
    filename = raw.name
    if not filename or filename.startswith("."):
        return None
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        return None
    destination = (extract_dir / filename).resolve()
    if not _is_within(extract_dir.resolve(), destination):
        return None
    return destination


def extract_archive(zip_path: Path, extract_dir: Path, *, overwrite: bool = False) -> int:
    """Extract JSON/text files into extract_dir without leaving that directory."""

    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                destination = _safe_destination(extract_dir, info.filename)
                if destination is None:
                    continue
                if destination.exists() and not overwrite:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as target:
                    target.write(source.read())
                extracted += 1
    except zipfile.BadZipFile as exc:
        raise CricsheetDownloadError(f"Invalid zip archive: {zip_path}") from exc
    return extracted


def _download_to_path(url: str, destination: Path, *, timeout: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as tmp:
                tmp_path = Path(tmp.name)
                progress = tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                )
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    tmp.write(chunk)
                    progress.update(len(chunk))
                progress.close()
        if tmp_path is not None:
            tmp_path.replace(destination)
    except requests.RequestException as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise CricsheetDownloadError(f"Failed to download {url}: {exc}") from exc
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def download_archive(
    archive: str = DEFAULT_ARCHIVE,
    output_dir: str | Path | None = None,
    *,
    force: bool = False,
    extract: bool = True,
    base_url: str | None = None,
) -> DownloadResult:
    settings = load_settings()
    key = archive.strip().lower()
    if key not in ARCHIVES:
        raise CricsheetDownloadError(
            f"Unknown archive '{archive}'. Use --list to see supported names."
        )
    filename, _description = ARCHIVES[key]
    root = Path(output_dir) if output_dir is not None else settings.data_raw_dir / "cricsheet"
    zip_path = root / filename
    extract_dir = root / key
    url = archive_url(key, base_url or settings.cricsheet_downloads_url)

    downloaded = False
    if force or not zip_path.exists():
        print(f"Downloading {url}")
        _download_to_path(url, zip_path)
        downloaded = True
    else:
        print(f"Already present, skipping download: {zip_path}")

    extracted_files = 0
    if extract:
        extracted_files = extract_archive(zip_path, extract_dir, overwrite=force)
        print(f"Extracted {extracted_files} files into {extract_dir}")

    return DownloadResult(
        archive=key,
        zip_path=zip_path,
        extract_dir=extract_dir,
        downloaded=downloaded,
        extracted_files=extracted_files,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download official Cricsheet JSON match archives."
    )
    parser.add_argument(
        "--archive",
        default=DEFAULT_ARCHIVE,
        help=f"Archive key to download (default: {DEFAULT_ARCHIVE}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to data/raw/cricsheet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite extracted files.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download the zip without extracting it.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List curated official archives and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for key, (filename, description) in ARCHIVES.items():
            print(f"{key:20} {filename:28} {description}")
        return 0
    try:
        download_archive(
            archive=args.archive,
            output_dir=args.output or (DEFAULT_RAW_DIR / "cricsheet"),
            force=args.force,
            extract=not args.no_extract,
        )
    except CricsheetDownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
