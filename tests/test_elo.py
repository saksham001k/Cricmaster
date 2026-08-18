from cricmaster.data.formats import MatchFormat
from cricmaster.features.elo import FormatElo


def test_elo_uses_only_past_results_and_is_format_specific() -> None:
    book = FormatElo(initial=1500, k_factor=20)
    before_first = book.snapshot(MatchFormat.T20I, "India", "Australia")
    assert before_first["team_elo_before"] == 1500
    assert before_first["elo_difference"] == 0

    book.update(MatchFormat.T20I, "India", "Australia", team_score=1)
    after_t20 = book.snapshot(MatchFormat.T20I, "India", "Australia")
    assert after_t20["team_elo_before"] > 1500
    assert after_t20["opponent_elo_before"] < 1500

    test_snapshot = book.snapshot(MatchFormat.TEST, "India", "Australia")
    assert test_snapshot["team_elo_before"] == 1500


def test_elo_draw_moves_ratings_toward_each_other() -> None:
    book = FormatElo(initial=1500, k_factor=20)
    book.update(MatchFormat.ODI, "India", "Australia", team_score=1)
    india_before_draw = book.rating(MatchFormat.ODI, "India")
    book.update(MatchFormat.ODI, "India", "Australia", team_score=0.5)
    india_after_draw = book.rating(MatchFormat.ODI, "India")
    assert india_after_draw < india_before_draw


def test_elo_is_gender_specific() -> None:
    book = FormatElo(initial=1500, k_factor=20)
    book.update(MatchFormat.HUNDRED, "Trent Rockets", "Oval Invincibles", team_score=1, gender="female")
    men = book.snapshot(MatchFormat.HUNDRED, "Trent Rockets", "Oval Invincibles", gender="male")
    women = book.snapshot(MatchFormat.HUNDRED, "Trent Rockets", "Oval Invincibles", gender="female")
    assert men["team_elo_before"] == 1500
    assert women["team_elo_before"] > 1500
