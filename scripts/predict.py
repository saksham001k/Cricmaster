"""Unified production prediction CLI for Cricmaster.

Existing scripts remain:
- scripts/predict_match.py
- scripts/predict_posttoss.py
- scripts/predict_live.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.prediction.artifacts import parse_prediction_mode
from cricmaster.prediction.errors import ArtifactValidationError, UnsupportedPredictionError
from cricmaster.prediction.live import parse_cricket_overs
from cricmaster.prediction.result import format_prediction_report
from cricmaster.prediction.router import (
    ProductionRequest,
    parse_production_format,
    predict_production,
)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _xi(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Production Cricmaster prediction router for T20I and franchise T20. "
            "Probabilities are estimates, not guarantees."
        )
    )
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument(
        "--format",
        required=True,
        help="T20I or T20. The Hundred is not treated as T20.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="pre_toss, post_toss, or live",
    )
    parser.add_argument("--date", required=True, type=_date)
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--venue", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--toss-winner", default=None)
    parser.add_argument(
        "--toss-decision",
        choices=["bat", "field", "bowl"],
        default=None,
    )
    parser.add_argument("--team1-xi", default=None)
    parser.add_argument("--team2-xi", default=None)
    parser.add_argument("--batting-team", default=None)
    parser.add_argument("--innings", type=int, choices=[1, 2], default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--wickets", type=int, default=None)
    parser.add_argument(
        "--overs",
        default=None,
        help="LIVE only. Cricket over notation such as 15.3.",
    )
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument(
        "--raw",
        default="data/raw/cricsheet/t20_corpus",
        help="Cricsheet raw corpus directory.",
    )
    args = parser.parse_args(argv)

    match_format = parse_production_format(args.format, competition=args.competition)
    mode = parse_prediction_mode(args.mode)

    legal_balls = None
    if args.overs is not None:
        legal_balls = parse_cricket_overs(args.overs, balls_per_over=6)

    request = ProductionRequest(
        team1=args.team1,
        team2=args.team2,
        match_format=match_format,
        prediction_mode=mode,
        match_date=args.date,
        gender=args.gender,
        venue=args.venue,
        competition=args.competition,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
        team1_xi=_xi(args.team1_xi),
        team2_xi=_xi(args.team2_xi),
        batting_team=args.batting_team,
        innings_number=args.innings,
        runs=args.runs,
        wickets=args.wickets,
        legal_balls=legal_balls,
        target=args.target,
    )

    try:
        result = predict_production(request, raw_dir=args.raw)
    except (UnsupportedPredictionError, ArtifactValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_prediction_report(result))
    print(
        "Historical matches applied: "
        f"{result.matches_applied} | "
        f"{result.team1}={result.team1_history_matches}, "
        f"{result.team2}={result.team2_history_matches}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
