from pathlib import Path

import pytest

from cricmaster.data.cricsheet import CricsheetParseError, load_directory, load_match
from cricmaster.data.formats import MatchFormat

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_sample_ipl_match() -> None:
    match = load_match(FIXTURES / "sample_ipl_match.json")
    assert match.metadata.match_id == "sample_ipl_match"
    assert match.metadata.format is MatchFormat.T20
    assert match.metadata.competition == "IPL"
    assert match.metadata.team1 == "Mumbai Indians"
    assert match.metadata.team2 == "Chennai Super Kings"
    assert match.metadata.toss_winner == "Chennai Super Kings"
    assert match.metadata.winner == "Mumbai Indians"
    assert match.metadata.player_of_match == "Rohit Sharma"
    assert match.metadata.source == "cricsheet"
    assert len(match.innings_history) == 2
    first = match.innings_history[0]
    assert first.batting_team == "Chennai Super Kings"
    assert first.runs == 5
    assert first.wickets == 1
    assert first.balls == 2
    assert first.overs == 0.2
    second = match.innings_history[1]
    assert second.target == 6
    assert second.runs == 6
    assert match.current_players is not None
    assert match.current_players.striker == "Rohit Sharma"
    wicket = next(delivery for delivery in match.deliveries if delivery.wicket)
    assert wicket.wicket_type == "bowled"
    assert wicket.player_out == "Ruturaj Gaikwad"
    wide = next(delivery for delivery in match.deliveries if delivery.is_wide)
    assert wide.is_legal is False
    assert match.metadata.team1_players is not None
    assert "Rohit Sharma" in match.metadata.team1_players


def test_malformed_json_raises_parse_error() -> None:
    with pytest.raises(CricsheetParseError, match="Malformed JSON"):
        load_match(FIXTURES / "malformed.json")


def test_missing_teams_raises_parse_error() -> None:
    with pytest.raises(CricsheetParseError, match="two team names"):
        load_match(FIXTURES / "missing_teams.json")


def test_load_directory_skips_bad_files(tmp_path: Path) -> None:
    good = (FIXTURES / "sample_ipl_match.json").read_text(encoding="utf-8")
    (tmp_path / "ok.json").write_text(good, encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not-json", encoding="utf-8")
    report = load_directory(tmp_path)
    assert len(report.matches) == 1
    assert len(report.errors) == 1
    assert "Malformed JSON" in report.errors[0].reason


def test_empty_directory_returns_no_errors(tmp_path: Path) -> None:
    report = load_directory(tmp_path)
    assert report.matches == []
    assert report.errors == []
    assert report.ok


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    report = load_directory(tmp_path / "does-not-exist")
    assert report.matches == []
    assert len(report.errors) == 1
