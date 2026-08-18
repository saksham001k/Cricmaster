"""CLI for Cricmaster POST_TOSS predictions."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.prediction.posttoss import PostTossRequest, predict_post_toss
from cricmaster.prediction.prematch import parse_match_format


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
        description="Predict a T20/T20I match after the toss."
    )
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--format", required=True, choices=["T20I", "T20"])
    parser.add_argument("--date", required=True, type=_date)
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--venue", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--toss-winner", required=True)
    parser.add_argument("--toss-decision", required=True, choices=["bat", "field", "bowl"])
    parser.add_argument("--team1-xi", default=None, help="Comma-separated playing XI")
    parser.add_argument("--team2-xi", default=None, help="Comma-separated playing XI")
    parser.add_argument("--raw", default="data/raw/cricsheet/t20_corpus")
    parser.add_argument(
        "--model",
        default="models/posttoss/posttoss_model.joblib",
    )
    args = parser.parse_args(argv)

    request = PostTossRequest(
        team1=args.team1,
        team2=args.team2,
        match_format=parse_match_format(args.format),
        match_date=args.date,
        gender=args.gender,
        venue=args.venue,
        competition=args.competition,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
        team1_xi=_xi(args.team1_xi),
        team2_xi=_xi(args.team2_xi),
    )

    print(
        f"Building historical state from matches strictly before "
        f"{request.match_date.isoformat()} ..."
    )
    result = predict_post_toss(
        request,
        raw_dir=args.raw,
        model_path=args.model,
    )

    print("\n=== CRICMASTER POST-TOSS ===")
    print(f"{result.team1:30} {result.team1_probability * 100:6.2f}%")
    print(f"{result.team2:30} {result.team2_probability * 100:6.2f}%")
    winner_p = max(result.team1_probability, result.team2_probability)
    print(
        f"\nPrediction: {result.winner} "
        f"({winner_p * 100:.2f}%, {result.edge} edge)"
    )
    print(f"Model: {result.model_name} | lineup mode: {result.lineup_mode}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print(
        "\nNote: this is a statistical probability estimate, not a guaranteed result."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
