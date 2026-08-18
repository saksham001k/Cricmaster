import zipfile
from pathlib import Path

import pytest

from cricmaster.data.download import (
    ARCHIVES,
    CricsheetDownloadError,
    archive_url,
    extract_archive,
    main,
)


def test_archive_catalog_uses_official_json_zips() -> None:
    assert ARCHIVES["t20s"][0] == "t20s_json.zip"
    assert archive_url("ipl") == "https://cricsheet.org/downloads/ipl_json.zip"
    assert archive_url("recently_played_2").endswith("recently_played_2_json.zip")


def test_unknown_archive_is_rejected() -> None:
    with pytest.raises(CricsheetDownloadError):
        archive_url("not-a-real-archive")


def test_extract_archive_ignores_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "sample.zip"
    extract_dir = tmp_path / "out"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("12345.json", '{"ok": true}')
        archive.writestr("../escape.json", '{"bad": true}')
        archive.writestr("nested/ok.json", '{"nested": true}')
        archive.writestr("notes.txt", "readme")
        archive.writestr("binary.bin", b"nope")
    extracted = extract_archive(zip_path, extract_dir)
    names = {path.name for path in extract_dir.iterdir()}
    assert extracted == 3
    assert names == {"12345.json", "ok.json", "notes.txt"}
    assert not (tmp_path / "escape.json").exists()


def test_extract_does_not_overwrite_unless_forced(tmp_path: Path) -> None:
    zip_path = tmp_path / "sample.zip"
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()
    existing = extract_dir / "12345.json"
    existing.write_text("keep-me", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("12345.json", "new-content")
    assert extract_archive(zip_path, extract_dir, overwrite=False) == 0
    assert existing.read_text(encoding="utf-8") == "keep-me"
    assert extract_archive(zip_path, extract_dir, overwrite=True) == 1
    assert existing.read_text(encoding="utf-8") == "new-content"


def test_cli_list_exits_without_network(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "t20s" in output
    assert "ipl_json.zip" in output

def test_expanded_t20_archive_catalog() -> None:
    assert ARCHIVES["wbb"][0] == "wbb_json.zip"
    assert ARCHIVES["ilt"][0] == "ilt_json.zip"
    assert ARCHIVES["sat"][0] == "sat_json.zip"
    assert ARCHIVES["ssm"][0] == "ssm_json.zip"
