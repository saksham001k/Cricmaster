# Cricmaster feature schema

This document describes the historical datasets produced by:

```powershell
python scripts/build_features.py --input data/raw/cricsheet --output data/processed
```

No trained prediction model exists yet. `FEATURE` columns are inputs. `TARGET` columns are labels. `IDENTIFIER` columns group rows. `METADATA` columns describe context.

Leakage rule: every `FEATURE` for a row at time T uses only information available at T. `TARGET` values may use the final match result.

## Prediction modes

| Mode | Toss fields | Use |
| --- | --- | --- |
| `PRE_TOSS` | null | Pre-match model before the toss |
| `POST_TOSS` | populated when known | Pre-match model after the toss |

Do not train a `PRE_TOSS` model on toss columns.

Playing XI:

| Status | Meaning |
| --- | --- |
| `LINEUP_KNOWN` | Cricsheet (or another source) provided a team list |
| `LINEUP_UNKNOWN` | XI features are null; players are not invented |

## Live-state format support

Supported now: `T20`, `T20I`, `ODI`, `LIST_A`, `HUNDRED`, `T10`.

Planned: `TEST`, `FIRST_CLASS` live states. Pre-match team features still cover those formats when historical results exist.

## `prematch_features.parquet`

Two teams × two toss modes are emitted for each eligible completed win/loss match. Both sides share `match_id` so they stay together in `temporal_split`.

Ties, draws, no-results, and abandoned matches are excluded from this table. Counts are recorded in `build_report.json`.

| name | type | role | nullable | prediction availability | source | description |
| --- | --- | --- | --- | --- | --- | --- |
| match_id | string | IDENTIFIER | no | always | Cricsheet file stem | Groups both team rows |
| date | string | IDENTIFIER | no | always | match info.dates | ISO date used for chronological order |
| format | string | METADATA | no | always | normalized match type | TEST/ODI/T20I/T20/... |
| competition | string | METADATA | yes | always | event name | IPL, BBL, ... not a format |
| season | string | METADATA | yes | always | match info | Season label |
| venue | string | FEATURE | yes | always | match info | Venue name |
| city | string | METADATA | yes | always | match info | City if provided |
| gender | string | METADATA | yes | always | match info | male/female/other |
| home_away | string | FEATURE | yes | always | derived | Currently null; not guessed |
| raw_team_name | string | METADATA | no | always | match info | Original team string |
| team | string | FEATURE | no | always | alias map | Canonical team |
| raw_opponent_name | string | METADATA | no | always | match info | Original opponent string |
| opponent | string | FEATURE | no | always | alias map | Canonical opponent |
| prediction_mode | string | METADATA | no | always | pipeline | PRE_TOSS or POST_TOSS |
| toss_winner | string | FEATURE | yes | POST_TOSS | match info | Null in PRE_TOSS rows |
| toss_decision | string | FEATURE | yes | POST_TOSS | match info | bat/field |
| team_won_toss | bool | FEATURE | yes | POST_TOSS | derived | Null in PRE_TOSS rows |
| matches_before | int | FEATURE | no | always | historical state | Prior matches in this format |
| wins_before | int | FEATURE | no | always | historical state | Prior wins in this format |
| win_rate_before | float | FEATURE | yes | always | historical state | Null if no history |
| matches_last_5/10/20 | int | FEATURE | no | always | historical state | Recent sample size |
| wins_last_5/10/20 | int | FEATURE | no | always | historical state | Recent wins |
| win_rate_last_5/10/20 | float | FEATURE | yes | always | historical state | Null if sample size is 0 |
| h2h_matches_before | int | FEATURE | no | always | historical state | Prior H2H in this format |
| h2h_team_wins | int | FEATURE | no | always | historical state | Prior H2H wins |
| h2h_opponent_wins | int | FEATURE | no | always | historical state | Prior H2H losses |
| h2h_team_win_rate | float | FEATURE | yes | always | historical state | Null if no H2H |
| h2h_last_5_matches | int | FEATURE | no | always | historical state | Recent H2H sample size |
| h2h_last_5_win_rate | float | FEATURE | yes | always | historical state | Null if none |
| team_matches_at_venue | int | FEATURE | no | always | historical state | Sample size |
| team_wins_at_venue | int | FEATURE | no | always | historical state | Wins at venue |
| team_win_rate_at_venue | float | FEATURE | yes | always | historical state | Null if no venue history |
| opponent_matches_at_venue | int | FEATURE | no | always | historical state | Opponent sample size |
| opponent_wins_at_venue | int | FEATURE | no | always | historical state | Opponent wins at venue |
| opponent_win_rate_at_venue | float | FEATURE | yes | always | historical state | Null if none |
| venue_batting_first_win_rate | float | FEATURE | yes | always | historical state | Limited-overs venue tendency |
| venue_chasing_win_rate | float | FEATURE | yes | always | historical state | Limited-overs venue tendency |
| venue_decided_matches | int | FEATURE | no | always | historical state | Sample size for venue rates |
| historical_first_innings_average | float | FEATURE | yes | always | historical state | Prior first-innings totals |
| historical_first_innings_matches | int | FEATURE | no | always | historical state | Sample size |
| team_elo_before | float | FEATURE | no | always | format Elo | Rating before this match |
| opponent_elo_before | float | FEATURE | no | always | format Elo | Opponent rating before this match |
| elo_difference | float | FEATURE | no | always | format Elo | team minus opponent |
| lineup_status | string | METADATA | no | always | players metadata | LINEUP_KNOWN or LINEUP_UNKNOWN |
| xi_batters_with_history | int | FEATURE | yes | LINEUP_KNOWN | player history | Null if lineup unknown |
| xi_mean_batting_average | float | FEATURE | yes | LINEUP_KNOWN | player history | Null if no prior dismissals |
| xi_mean_recent_runs | float | FEATURE | yes | LINEUP_KNOWN | player history | Null if no prior innings |
| xi_bowlers_with_history | int | FEATURE | yes | LINEUP_KNOWN | player history | Null if lineup unknown |
| xi_mean_bowling_economy | float | FEATURE | yes | LINEUP_KNOWN | player history | Null if no prior overs |
| xi_mean_recent_wickets | float | FEATURE | yes | LINEUP_KNOWN | player history | Null if no prior spells |
| team_win | int | TARGET | no | label only | match result | 1 if this team won |

## `live_states.parquet`

One row per recorded delivery in supported limited-overs formats.

| name | type | role | nullable | prediction availability | source | description |
| --- | --- | --- | --- | --- | --- | --- |
| match_id | string | IDENTIFIER | no | always | Cricsheet | Match key |
| date | string | IDENTIFIER | yes | always | match info | ISO date |
| format | string | METADATA | no | always | normalized type | Limited-overs format |
| competition | string | METADATA | yes | always | event name | Competition |
| innings_number | int | FEATURE | no | always | delivery | 1-based innings |
| batting_team | string | FEATURE | no | always | delivery | Canonical batting side |
| bowling_team | string | FEATURE | no | always | derived | Canonical bowling side |
| runs | int | FEATURE | no | always | cumulative | Innings runs after this ball |
| wickets | int | FEATURE | no | always | cumulative | Wickets down |
| legal_balls_bowled | int | FEATURE | no | always | cumulative | Wides/no-balls excluded |
| overs | float | FEATURE | no | always | derived | Cricket notation from legal balls |
| balls_remaining | int | FEATURE | yes | limited-overs | schedule/target | Null for unlimited formats |
| current_run_rate | float | FEATURE | yes | always | derived | Null before a legal ball |
| target | int | FEATURE | yes | 2nd innings | Cricsheet target.runs | Null in first innings |
| runs_required | int | FEATURE | yes | 2nd innings | target - runs | 181 vs 150 → 31 |
| required_run_rate | float | FEATURE | yes | 2nd innings | derived | Null if no balls remaining |
| run_rate_difference | float | FEATURE | yes | 2nd innings | derived | Current minus required |
| wickets_in_hand | int | FEATURE | no | always | 10 - wickets | |
| is_wide | bool | FEATURE | no | always | extras | This delivery |
| is_noball | bool | FEATURE | no | always | extras | This delivery |
| is_wicket | bool | FEATURE | no | always | wickets | This delivery |
| is_legal | bool | FEATURE | no | always | derived | Not wide/no-ball |
| runs_this_ball | int | FEATURE | no | always | runs.total | |
| striker | string | FEATURE | yes | always | delivery | |
| non_striker | string | FEATURE | yes | always | delivery | |
| bowler | string | FEATURE | yes | always | delivery | |
| eventual_winner | string | TARGET | yes | label only | match result | Canonical winner |
| batting_team_eventual_win | int | TARGET | yes | label only | match result | 1 if batting side won |

## Temporal split

Use `cricmaster.features.temporal_split` with match dates. All rows sharing `match_id` stay in one split. Do not randomly split cricket time-series rows.
