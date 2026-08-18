from datetime import date
from pathlib import Path

from cricmaster.features.history import HistoricalState
from cricmaster.features.pipeline import build_feature_datasets
from cricmaster.features.prematch import build_prematch_rows
from cricmaster.features.toss import PredictionMode
from tests.helpers import completed_t20

FIXTURES = Path(__file__).parent / "fixtures"


def test_no_result_matches_are_excluded_not_silently_dropped_from_counts() -> None:
    match = completed_t20(
        "nr1",
        date(2024, 1, 1),
        "India",
        "Australia",
        "India",
        result_type="no result",
    )
    rows, reason = build_prematch_rows(match, HistoricalState(), modes=[PredictionMode.PRE_TOSS])
    assert rows == []
    assert reason == "no result"


def test_pipeline_writes_parquet_and_report(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    (source / "sample_ipl_match.json").write_text(
        (FIXTURES / "sample_ipl_match.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    output = tmp_path / "processed"
    report = build_feature_datasets(source, output)
    assert report.matches_parsed == 1
    assert report.prematch_rows == 4  # 2 teams x 2 toss modes
    assert report.live_state_rows > 0
    assert (output / "prematch_features.parquet").exists()
    assert (output / "live_states.parquet").exists()
    assert (output / "build_report.json").exists()
    assert "T20" in report.formats
    assert "IPL" in report.competitions
