from cricmaster.data.formats import MatchFormat
from cricmaster.features.utils import (
    balls_to_over_parts,
    cricket_overs_to_balls,
    innings_ball_limit,
    is_legal_delivery,
    overs_notation,
)


def test_wide_and_noball_are_not_legal() -> None:
    assert is_legal_delivery(is_wide=True, is_noball=False) is False
    assert is_legal_delivery(is_wide=False, is_noball=True) is False
    assert is_legal_delivery(is_wide=False, is_noball=False) is True


def test_cricket_overs_are_not_decimal() -> None:
    assert cricket_overs_to_balls(17.4, 6) == 17 * 6 + 4
    assert cricket_overs_to_balls(17.4, 6) != int(17.4 * 6)
    assert balls_to_over_parts(106, 6) == (17, 4)
    assert overs_notation(106, 6) == 17.4


def test_format_ball_limits() -> None:
    assert innings_ball_limit(MatchFormat.T20, scheduled_overs=20, balls_per_over=6) == 120
    assert innings_ball_limit(MatchFormat.ODI, scheduled_overs=50, balls_per_over=6) == 300
    assert innings_ball_limit(MatchFormat.HUNDRED) == 100
    assert innings_ball_limit(MatchFormat.TEST) is None
    assert innings_ball_limit(MatchFormat.FIRST_CLASS) is None
    assert innings_ball_limit(MatchFormat.T20, target_overs=17.4, balls_per_over=6) == 106
