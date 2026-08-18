"""Chronological Elo ratings. This is a strength feature, not a prediction model."""

from __future__ import annotations

from cricmaster.data.formats import MatchFormat
from cricmaster.data.team_aliases import canonicalize_team

DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K = 20.0


class FormatElo:
    """Format- and gender-specific Elo. Men's Hundred never updates women's Hundred."""

    def __init__(
        self,
        *,
        initial: float = DEFAULT_INITIAL_RATING,
        k_factor: float = DEFAULT_K,
    ) -> None:
        self.initial = initial
        self.k_factor = k_factor
        self._ratings: dict[tuple[str, str, str], float] = {}

    def rating(
        self,
        match_format: MatchFormat | str,
        team: str,
        gender: str | None = None,
    ) -> float:
        key = (str(match_format), gender or "", canonicalize_team(team))
        return self._ratings.get(key, self.initial)

    def expected(self, team_rating: float, opponent_rating: float) -> float:
        return 1.0 / (1.0 + 10 ** ((opponent_rating - team_rating) / 400.0))

    def snapshot(
        self,
        match_format: MatchFormat | str,
        team: str,
        opponent: str,
        gender: str | None = None,
    ) -> dict[str, float]:
        team_elo = self.rating(match_format, team, gender)
        opponent_elo = self.rating(match_format, opponent, gender)
        return {
            "team_elo_before": team_elo,
            "opponent_elo_before": opponent_elo,
            "elo_difference": team_elo - opponent_elo,
        }

    def update(
        self,
        match_format: MatchFormat | str,
        team: str,
        opponent: str,
        *,
        team_score: float,
        gender: str | None = None,
    ) -> None:
        """team_score is 1 for a win, 0 for a loss, 0.5 for a draw/tie."""

        fmt = str(match_format)
        sex = gender or ""
        team_key = (fmt, sex, canonicalize_team(team))
        opp_key = (fmt, sex, canonicalize_team(opponent))
        team_elo = self._ratings.get(team_key, self.initial)
        opp_elo = self._ratings.get(opp_key, self.initial)
        team_expected = self.expected(team_elo, opp_elo)
        opp_expected = 1.0 - team_expected
        self._ratings[team_key] = team_elo + self.k_factor * (team_score - team_expected)
        self._ratings[opp_key] = opp_elo + self.k_factor * ((1.0 - team_score) - opp_expected)
