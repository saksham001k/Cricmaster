"""Fetch current CricketData matches and run Cricmaster automatically."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricmaster.live.automatic import to_live_prediction_request
from cricmaster.live.cricketdata import CricketDataMatch, CricketDataProvider
from cricmaster.prediction.live import predict_live


def _state_label(match: CricketDataMatch) -> str:
    if match.predictable_live:
        return "LIVE/PREDICTABLE"
    if match.terminal_status:
        return "ENDED/TERMINAL"
    if not match.match_started:
        return "UPCOMING"
    if not match.supported_format:
        return f"UNSUPPORTED/{match.match_format.value}"
    return "LIVE/AMBIGUOUS"


def _score_text(match: CricketDataMatch) -> str:
    if not match.scores:
        return "no score"
    parts = []
    for score in match.scores:
        team = score.batting_team or "?"
        parts.append(
            f"{team} {score.runs}/{score.wickets} ({score.overs_text or '?'})"
        )
    return " | ".join(parts)


def _print_quota(provider: CricketDataProvider) -> None:
    info = provider.last_info
    if not info:
        return
    used = info.get("hitsToday", info.get("hitsUsed"))
    limit = info.get("hitsLimit")
    if used is not None and limit is not None:
        print(f"API quota today: {used}/{limit}")


def _choose_match(
    matches: list[CricketDataMatch],
    *,
    match_id: str | None,
    search: str | None,
) -> CricketDataMatch:
    if match_id:
        found = [m for m in matches if m.match_id == match_id]
        if not found:
            raise SystemExit(f"No currentMatches record with id={match_id}")
        return found[0]

    if search:
        needle = search.casefold()
        found = [
            m
            for m in matches
            if needle in m.name.casefold()
            or any(needle in team.casefold() for team in m.teams)
        ]
        if len(found) == 1:
            return found[0]
        if not found:
            raise SystemExit(f"No match found for search={search!r}")
        print(f"Search matched {len(found)} records; refine --search or use --match-id.")
        for item in found:
            print(f"  {item.match_id} | {item.name}")
        raise SystemExit(2)

    live = [m for m in matches if m.predictable_live]
    if len(live) == 1:
        return live[0]
    if not live:
        raise SystemExit(
            "No predictable live T20/T20I match is present in this currentMatches response."
        )
    print("Multiple predictable live matches are available; use --search or --match-id.")
    for item in live:
        print(f"  {item.match_id} | {item.name}")
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch currentMatches and run Cricmaster live automatically."
    )
    parser.add_argument("--list", action="store_true", help="List all returned records.")
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--search", default=None, help="Team or match-name substring.")
    parser.add_argument("--raw", default="data/raw/cricsheet/t20_corpus")
    parser.add_argument(
        "--first-innings-model",
        default="models/live/first_innings_model.joblib",
    )
    parser.add_argument(
        "--chase-model",
        default="models/live/chase_model.joblib",
    )
    args = parser.parse_args(argv)

    provider = CricketDataProvider()
    matches = provider.current_matches()

    print(f"CricketData returned {len(matches)} normalized records.")
    _print_quota(provider)

    if args.list:
        for index, match in enumerate(matches, 1):
            print("\n" + "=" * 78)
            print(f"{index}. [{_state_label(match)}] {match.name}")
            print(f"id: {match.match_id}")
            print(
                f"format={match.match_format.value} "
                f"competition={match.competition or '-'} "
                f"gender={match.gender}"
            )
            print(f"status: {match.status}")
            print(f"score: {_score_text(match)}")
            if match.target is not None:
                print(f"derived target: {match.target}")
            for warning in match.warnings:
                print(f"warning: {warning}")
        return 0

    match = _choose_match(
        matches,
        match_id=args.match_id,
        search=args.search,
    )

    print(f"\nSelected: {match.name}")
    print(f"Status: {match.status}")
    print(f"Score: {_score_text(match)}")
    for warning in match.warnings:
        print(f"Provider warning: {warning}")

    try:
        request = to_live_prediction_request(match)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    result = predict_live(
        request,
        raw_dir=args.raw,
        first_innings_model=args.first_innings_model,
        chase_model=args.chase_model,
    )

    state = result.state_summary
    print("\n=== CRICMASTER AUTO LIVE ===")
    print(
        f"Innings {result.innings_number} | "
        f"{result.batting_team} {state['runs']}/{state['wickets']}"
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
    print(f"Model: {result.model_name} | {result.model_kind}")

    if result.warnings:
        print("\nPrediction warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print(
        "\nNote: upstream live data can be incomplete or delayed. "
        "Cricmaster refuses ambiguous chase targets rather than inventing them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
