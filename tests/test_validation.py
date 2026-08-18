import pandas as pd

from cricmaster.features.validate import validate_live, validate_prematch


def test_validation_records_impossible_live_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "wickets": 11,
                "wickets_in_hand": -1,
                "balls_remaining": -3,
                "runs": -1,
                "target": 0,
                "innings_number": 1,
                "runs_required": 1,
            }
        ]
    )
    checks = {item["check"] for item in validate_live(frame)}
    assert "wickets_gt_10" in checks
    assert "negative_balls_remaining" in checks
    assert "invalid_targets" in checks
    assert "first_innings_has_target" in checks


def test_validation_records_duplicate_prematch_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team": "India",
                "prediction_mode": "PRE_TOSS",
                "format": "T20I",
                "team_win": 1,
            },
            {
                "match_id": "m1",
                "team": "India",
                "prediction_mode": "PRE_TOSS",
                "format": "T20I",
                "team_win": 1,
            },
            {
                "match_id": "m1",
                "team": "Australia",
                "prediction_mode": "PRE_TOSS",
                "format": "T20I",
                "team_win": 0,
            },
        ]
    )
    checks = {item["check"] for item in validate_prematch(frame)}
    assert "duplicate_prediction_rows" in checks
