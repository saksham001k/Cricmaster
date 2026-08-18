from cricmaster.data.formats import MatchFormat
from cricmaster.live.automatic import to_live_prediction_request
from cricmaster.live.cricketdata import (
    infer_format,
    normalize_current_match,
)


def _base() -> dict:
    return {
        "id": "m1",
        "name": "Alpha vs Beta, Example T20 2026",
        "matchType": "t20",
        "status": "Beta need 25 runs in 18 balls",
        "venue": "Example Ground",
        "date": "2026-08-19",
        "teams": ["Alpha", "Beta"],
        "tossWinner": None,
        "tossChoice": None,
        "matchStarted": True,
        "matchEnded": False,
        "score": [
            {"r": 180, "w": 6, "o": 20, "inning": "Alpha Inning 1"},
            {"r": 156, "w": 5, "o": 17, "inning": "Beta Inning 1"},
        ],
    }


def test_t20i_is_inferred_from_match_name() -> None:
    assert infer_format(
        "t20",
        "Tanzania Women vs Uganda Women, Womens T20I Quadrangular Series",
    ) is MatchFormat.T20I


def test_hundred_is_not_routed_to_t20_model() -> None:
    assert infer_format(
        "t20",
        "Trent Rockets vs Manchester Super Giants, The Hundred Mens Competition",
    ) is MatchFormat.HUNDRED


def test_malformed_second_innings_label_uses_other_team() -> None:
    raw = _base()
    raw["score"][1]["inning"] = "Alpha,Beta Inning 1"

    match = normalize_current_match(raw)

    assert match is not None
    assert match.scores[0].batting_team == "Alpha"
    assert match.scores[1].batting_team == "Beta"


def test_live_need_status_derives_revised_target() -> None:
    match = normalize_current_match(_base())

    assert match is not None
    assert match.target == 181
    assert match.predictable_live is True

    request = to_live_prediction_request(match)
    assert request.innings_number == 2
    assert request.batting_team == "Beta"
    assert request.runs == 156
    assert request.wickets == 5
    assert request.legal_balls == 102
    assert request.target == 181


def test_rain_chase_without_explicit_target_is_refused() -> None:
    raw = _base()
    raw["status"] = "18 overs game due to rain"

    match = normalize_current_match(raw)

    assert match is not None
    assert match.target is None
    assert match.predictable_live is False


def test_terminal_status_overrides_matchended_false() -> None:
    raw = _base()
    raw["status"] = "Beta won by 5 wkts"
    raw["matchEnded"] = False

    match = normalize_current_match(raw)

    assert match is not None
    assert match.terminal_status is True
    assert match.predictable_live is False


def test_missing_date_is_not_automatically_predictable() -> None:
    raw = _base()
    raw["date"] = None
    raw["dateTimeGMT"] = None

    match = normalize_current_match(raw)

    assert match is not None
    assert match.predictable_live is False

def test_odi_is_inferred_from_name_when_match_type_missing() -> None:
    assert infer_format(
        None,
        "Papua New Guinea Women vs Thailand Women, 3rd ODI",
    ) is MatchFormat.ODI
