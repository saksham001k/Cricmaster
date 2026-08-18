from cricmaster.data.formats import MatchFormat, normalize_competition, normalize_match_type


def test_ipl_is_competition_not_format() -> None:
    match_format = normalize_match_type(
        "T20",
        team_type="club",
        competition="Indian Premier League",
    )
    assert match_format is MatchFormat.T20
    assert normalize_competition("Indian Premier League") == "IPL"


def test_international_t20_becomes_t20i() -> None:
    assert (
        normalize_match_type("T20", team_type="international") is MatchFormat.T20I
    )
    assert normalize_match_type("IT20") is MatchFormat.T20I


def test_test_odi_and_limited_overs_maps() -> None:
    assert normalize_match_type("Test") is MatchFormat.TEST
    assert normalize_match_type("ODI") is MatchFormat.ODI
    assert normalize_match_type("ODM") is MatchFormat.LIST_A
    assert normalize_match_type("MDM") is MatchFormat.FIRST_CLASS


def test_hundred_detected_from_competition_or_balls() -> None:
    assert (
        normalize_match_type("T20", competition="The Hundred", balls_per_over=5)
        is MatchFormat.HUNDRED
    )
    assert normalize_match_type("T20", balls_per_over=5) is MatchFormat.HUNDRED


def test_t10_detected_from_overs() -> None:
    assert normalize_match_type("T20", scheduled_overs=10) is MatchFormat.T10


def test_unknown_type_is_other() -> None:
    assert normalize_match_type("exhibition") is MatchFormat.OTHER
