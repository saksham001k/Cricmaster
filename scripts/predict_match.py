"""CLI for leakage-safe PRE_TOSS Cricmaster predictions."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.prediction.prematch import (
    PredictionRequest,
    parse_match_format,
    predict_prematch,
)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _format_difference(value: float | None) -> str:
    if value is None:
        return "history unavailable"
    if abs(value) >= 10:
        return f"{value:+.1f}"
    return f"{value:+.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict a T20/T20I match before the toss using Cricmaster."
    )
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--format", required=True, choices=["T20I", "T20"])
    parser.add_argument("--date", required=True, type=_date)
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--venue", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument(
        "--raw",
        default="data/raw/cricsheet/t20_corpus",
        help="Cricsheet raw corpus directory.",
    )
    parser.add_argument(
        "--model",
        default="models/prematch/prematch_model.joblib",
        help="Trained PRE_TOSS model bundle.",
    )
    parser.add_argument("--top-drivers", type=int, default=5)
    args = parser.parse_args(argv)

    request = PredictionRequest(
        team1=args.team1,
        team2=args.team2,
        match_format=parse_match_format(args.format),
        match_date=args.date,
        gender=args.gender,
        venue=args.venue,
        competition=args.competition,
    )

    print(
        f"Building historical state from matches strictly before "
        f"{request.match_date.isoformat()} ..."
    )
    result = predict_prematch(
        request,
        raw_dir=args.raw,
        model_path=args.model,
        top_drivers=args.top_drivers,
    )

    print("\n=== CRICMASTER PRE-TOSS ===")
    print(f"{result.team1:30} {result.team1_probability * 100:6.2f}%")
    print(f"{result.team2:30} {result.team2_probability * 100:6.2f}%")

    winner = (
        result.team1
        if result.team1_probability >= result.team2_probability
        else result.team2
    )
    winner_probability = max(
        result.team1_probability,
        result.team2_probability,
    )

    print(
        f"\nPrediction: {winner} "
        f"({winner_probability * 100:.2f}%, {result.edge} edge)"
    )
    print(f"Model: {result.model_name} | mode: {result.prediction_mode}")
    print(f"Historical matches applied: {result.matches_applied}")
    print(
        f"Format-specific history: "
        f"{result.team1}={result.team1_history_matches}, "
        f"{result.team2}={result.team2_history_matches}"
    )

    if result.drivers:
        print("\nStrongest model drivers:")
        for driver in result.drivers:
            raw = _format_difference(driver.raw_difference)
            print(
                f"  {'+' if driver.contribution > 0 else '-'} "
                f"{driver.label}: diff={raw}; "
                f"supports {driver.supports}; "
                f"log-odds contribution={driver.contribution:+.3f}"
            )

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print(
        "\nNote: this is a statistical probability estimate, not a guaranteed result. "
        "Drivers describe model contributions, not causal effects."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
