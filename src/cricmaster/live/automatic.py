"""Translate normalized CricketData matches into Cricmaster live requests."""

from __future__ import annotations

from cricmaster.live.cricketdata import CricketDataMatch
from cricmaster.prediction.live import LivePredictionRequest


def to_live_prediction_request(match: CricketDataMatch) -> LivePredictionRequest:
    if not match.predictable_live:
        reasons: list[str] = []
        if not match.match_started:
            reasons.append("match has not started")
        if match.terminal_status:
            reasons.append("match is already terminal")
        if not match.supported_format:
            reasons.append(f"unsupported format {match.match_format.value}")
        if match.current_score is None:
            reasons.append("no score available")
        elif match.current_score.legal_balls is None:
            reasons.append("overs could not be parsed")
        if not match.batting_team:
            reasons.append("batting team is ambiguous")
        if match.innings_number == 2 and match.target is None:
            reasons.append("chase target is unavailable")
        raise ValueError(
            "Cannot create automatic live prediction: "
            + ", ".join(reasons or ["state is not predictable"])
        )

    score = match.current_score
    assert score is not None
    assert score.legal_balls is not None
    assert match.batting_team is not None
    assert match.match_date is not None
    assert match.innings_number is not None

    return LivePredictionRequest(
        team1=match.teams[0],
        team2=match.teams[1],
        batting_team=match.batting_team,
        match_format=match.match_format,
        match_date=match.match_date,
        gender=match.gender,
        innings_number=match.innings_number,
        runs=score.runs,
        wickets=score.wickets,
        legal_balls=score.legal_balls,
        target=match.target if match.innings_number == 2 else None,
        venue=match.venue,
        competition=match.competition,
        toss_winner=match.toss_winner,
        toss_decision=match.toss_decision,
        team1_xi=(),
        team2_xi=(),
    )
