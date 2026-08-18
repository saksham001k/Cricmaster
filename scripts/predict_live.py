"""Command-line interface for Cricmaster live win probability."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.data.formats import MatchFormat
from cricmaster.prediction.live import (
    LivePredictionRequest,
    parse_cricket_overs,
    predict_live,
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
        description="Predict T20/T20I win probability from a live score state."
    )
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--batting-team", required=True)
    parser.add_argument("--format", required=True, choices=["T20I", "T20"])
    parser.add_argument("--date", required=True, type=_date)
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--innings", required=True, type=int, choices=[1, 2])
    parser.add_argument("--runs", required=True, type=int)
    parser.add_argument("--wickets", required=True, type=int)
    parser.add_argument(
        "--overs",
        required=True,
        help="Cricket over notation, e.g. 15.3 means 15 overs + 3 balls.",
    )
    parser.add_argument("--target", type=int, default=None)
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
    parser.add_argument("--raw", default="data/raw/cricsheet/t20_corpus")
    parser.add_argument(
        "--first-innings-model",
        default="models/live/first_innings_model.joblib",
    )
    parser.add_argument(
        "--chase-model",
        default="models/live/chase_model.joblib",
    )
    parser.add_argument("--top-drivers", type=int, default=6)
    args = parser.parse_args(argv)

    match_format = MatchFormat(args.format)
    legal_balls = parse_cricket_overs(args.overs, balls_per_over=6)

    request = LivePredictionRequest(
        team1=args.team1,
        team2=args.team2,
        batting_team=args.batting_team,
        match_format=match_format,
        match_date=args.date,
        gender=args.gender,
        innings_number=args.innings,
        runs=args.runs,
        wickets=args.wickets,
        legal_balls=legal_balls,
        target=args.target,
        venue=args.venue,
        competition=args.competition,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
        team1_xi=_xi(args.team1_xi),
        team2_xi=_xi(args.team2_xi),
    )

    result = predict_live(
        request,
        raw_dir=args.raw,
        first_innings_model=args.first_innings_model,
        chase_model=args.chase_model,
        top_drivers=args.top_drivers,
    )

    state = result.state_summary
    print("\n=== CRICMASTER LIVE ===")
    print(
        f"Innings {result.innings_number} | "
        f"{result.batting_team} {state['runs']}/{state['wickets']} "
        f"after {args.overs} overs"
    )
    if result.innings_number == 2:
        rrr = state["required_run_rate"]
        rrr_text = "N/A" if rrr is None else f"{rrr:.2f}"
        print(
            f"Target {state['target']} | "
            f"need {state['runs_required']} from {state['balls_remaining']} balls | "
            f"RRR {rrr_text}"
        )

    print(f"\n{result.batting_team:30} {result.batting_probability * 100:6.2f}%")
    print(f"{result.bowling_team:30} {result.bowling_probability * 100:6.2f}%")

    winner_p = max(result.batting_probability, result.bowling_probability)
    print(
        f"\nPrediction: {result.predicted_winner} "
        f"({winner_p * 100:.2f}%, {result.edge} edge)"
    )
    print(
        f"Model: {result.model_name} | "
        f"{result.model_kind} | terminal={result.terminal}"
    )

    if result.drivers:
        print("\nStrongest model effects:")
        for driver in result.drivers:
            raw = "missing" if driver.raw_value is None else f"{driver.raw_value:.3f}"
            print(
                f"  {'+' if driver.contribution > 0 else '-'} "
                f"{driver.label}: value={raw}; "
                f"supports {driver.supports}; "
                f"log-odds effect={driver.contribution:+.3f}"
            )

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print(
        "\nNote: probabilities are statistical estimates. "
        "Model effects are associations learned from historical data, not causal claims."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
