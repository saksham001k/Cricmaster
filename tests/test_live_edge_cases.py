from cricmaster.data.cricsheet import parse_match_payload
from cricmaster.features.live import compute_live_metrics


def test_retired_hurt_does_not_count_as_scoreboard_wicket() -> None:
    payload = {
        "info": {
            "dates": ["2025-01-01"],
            "teams": ["Team A", "Team B"],
            "match_type": "T20",
            "team_type": "international",
            "gender": "male",
            "balls_per_over": 6,
            "overs": 20,
            "toss": {
                "winner": "Team A",
                "decision": "bat",
            },
            "outcome": {
                "winner": "Team A",
                "by": {"runs": 1},
            },
        },
        "innings": [
            {
                "team": "Team A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Player A",
                                "non_striker": "Player B",
                                "bowler": "Bowler A",
                                "runs": {
                                    "batter": 0,
                                    "extras": 0,
                                    "total": 0,
                                },
                                "wickets": [
                                    {
                                        "kind": "retired hurt",
                                        "player_out": "Player A",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    match = parse_match_payload(payload, match_id="retired-hurt-test")

    assert len(match.deliveries) == 1
    assert match.deliveries[0].wicket is False
    assert match.innings_history[0].wickets == 0


def test_balls_remaining_never_becomes_negative() -> None:
    metrics = compute_live_metrics(
        runs=150,
        wickets=5,
        legal_balls=121,
        target=None,
        ball_limit=120,
        balls_per_over=6,
    )

    assert metrics["balls_remaining"] == 0
    assert metrics["wickets_in_hand"] == 5
