# Cricmaster

Cricmaster is a cricket intelligence and match prediction system designed to estimate match outcomes using historical data, live match state, player/team statistics, venue conditions, and other relevant cricket information.

This repository currently provides the **data foundation** only. It does not yet produce match predictions, live win probabilities, or chatbot answers.

## Project objective

Build a production-quality pipeline that can:

- ingest historical ball-by-ball cricket data
- represent live match state in a provider-agnostic way
- fall back to additional sources when a preferred source is incomplete
- later support pre-match and in-play prediction across many competitions

## Current development status

| Area | Status |
| --- | --- |
| Project layout and private GitHub repository | Supported now |
| Cricsheet JSON ingestion | Supported now |
| Format vs competition normalization | Supported now |
| Historical archive downloader | Supported now |
| Leakage-safe historical feature pipeline | Supported now |
| Pre-match feature dataset | Supported now |
| Limited-overs live-state dataset | Supported now |
| Live API adapters (CricketData, Sportmonks, EntitySport, Roanuz, ...) | Planned |
| Search fallback retrieval | Planned (interface only) |
| Supervised prediction models | Planned |
| Ball-by-ball live win probability model | Planned |
| Conversational cricket assistant | Planned |

## Supported / target cricket formats

**Formats** describe how a match is played. **Competitions** are named tournaments.

| Format | Meaning | Status |
| --- | --- | --- |
| `TEST` | Test cricket | Historical ingestion + pre-match features. Live states planned |
| `ODI` | One-day international | Historical ingestion, pre-match, live states |
| `T20I` | Twenty20 international | Historical ingestion, pre-match, live states |
| `T20` | Franchise / domestic T20 | Historical ingestion, pre-match, live states |
| `T10` | Ten-over cricket | Pre-match + live states when data appears |
| `HUNDRED` | 100-ball cricket | Historical ingestion, pre-match, live states (100-ball semantics) |
| `FIRST_CLASS` | Multi-day domestic cricket | Historical ingestion + pre-match features. Live states planned |
| `LIST_A` | One-day domestic cricket | Historical ingestion, pre-match, live states |
| `OTHER` | Unrecognized types | Captured without failing the import |

Example:

```text
competition = IPL
format = T20
```

IPL, TNPL, BBL, PSL, CPL, The Hundred, and similar names are competitions, not formats.

## Target competitions

Planned coverage includes Test cricket, ODI, T20I, IPL, TNPL, BBL, PSL, CPL, The Hundred, domestic cricket, franchise cricket, and women's competitions where data exists.

Cricsheet currently publishes ball-by-ball JSON for many of those competitions, including IPL, BBL, PSL, CPL, The Hundred, WPL, WBBL, T20 Blast, and international cricket. **TNPL was not present in the verified Cricsheet catalog** and will need a later API or search fallback.

Do not treat a competition as fully supported for live prediction until a live provider and trained model exist. Today Cricmaster can ingest history and build training tables. It cannot yet predict match winners.

## Historical feature pipeline

Step 3 builds two datasets from parsed matches, in chronological order:

1. `data/processed/prematch_features.parquet` — team-perspective pre-match rows
2. `data/processed/live_states.parquet` — one row after each limited-overs delivery

A `build_report.json` records discovered/parsed/skipped matches, exclusion counts, formats, competitions, and validation issues. Generated files are gitignored.

### Leakage prevention

Matches are sorted by date, then `match_id`. For match T the pipeline:

1. computes features from historical state
2. writes pre-match and live-state rows
3. only then updates Elo, form, venue, H2H, and player summaries with match T

Features never include that match's result, later matches, future player career totals, or season aggregates computed from the full file. Labels (`team_win`, `eventual_winner`) may use the final result.

Unknown rates are null, not zero. Sample sizes are stored beside rates (`matches_last_5`, `team_matches_at_venue`, ...).

### Pre-match dataset

Each completed win/loss match produces four rows: two teams × `PRE_TOSS` / `POST_TOSS`. Both sides share `match_id` so `temporal_split` can keep them together. Ties, draws, no-results, and abandoned games are excluded and counted, not deleted silently.

Features include format- and gender-specific recent/long-term form, Elo, head-to-head, venue records, optional toss fields, and conservative XI summaries when a lineup is known. If the XI is unknown, player features stay null. Men's and women's sides that share a franchise name (for example The Hundred) are tracked separately.

### Live-state dataset

Limited-overs formats emit a row after every delivery using `legal_balls`, not decimal overs. Wides and no-balls do not increment legal balls. First innings leave `target` and `required_run_rate` null. Chases use `runs_required = target - current_runs` (150 vs 181 → 31). Test and first-class live states are not emitted yet.

Schema details: `docs/feature_schema.md`.

### Feature-building command

```powershell
python scripts/build_features.py --input data/raw/cricsheet --output data/processed
```

Optional filters:

```powershell
python scripts/build_features.py --format T20,T20I --competition IPL --limit 200 --mode both
```

Future model training should split with `temporal_split` (older matches train, newer validate/test). Random row splits are not the default.

## Current limitations

- No prediction model is trained. There is no accuracy claim.
- TNPL is not in the verified Cricsheet catalog.
- Live states omit Test/First-Class innings.
- Home/away is not inferred.
- Playing XI features require source lineups; they are not invented.
- Toss-aware models must use `POST_TOSS` rows only.


## Project structure

```text
Cricmaster/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── data/                  # raw / processed / external datasets (not committed)
├── models/                # trained artifacts later (not committed)
├── notebooks/
├── scripts/               # CLI utilities
├── src/cricmaster/
│   ├── config.py
│   ├── data/              # models, Cricsheet parser, resolver
│   ├── live/              # live provider interface + mock
│   ├── search/            # search fallback interface + stub
│   ├── features/          # leakage-safe historical features
│   ├── models/            # planned
│   ├── prediction/        # planned
│   └── chatbot/           # planned
├── tests/
└── docs/
```

## Data architecture

Cricmaster uses three data tiers:

1. **Historical structured data** from [Cricsheet](https://cricsheet.org/), preferring official JSON archives.
2. **Live structured APIs**, behind a `LiveCricketProvider` interface so the project is not tied to one vendor. No API key is required yet. A mock provider exists for local development.
3. **Search fallback**, behind a `CricketSearchProvider` interface, for matches missing from structured sources. The current implementation is a stub and does not scrape websites.

A `MatchStateResolver` tries preferred live providers, then secondary providers, then search. Each populated field stores `source`, optional `timestamp`, and `reliability`. Conflicting values are recorded, not silently overwritten.

Raw downloads belong in `data/raw/`. Normalized datasets will later belong in `data/processed/`. Auxiliary files belong in `data/external/`. Large files are gitignored.

## Python environment

Requires Python 3.11+.

```powershell
cd C:\Users\Admin\Cricmaster
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` only when you later add API keys. Never commit `.env`.

## Tests

```powershell
python -m pytest -v
```

Unit tests do not require internet access.

## Download initial historical data

Cricsheet JSON is the official format ([JSON documentation](https://cricsheet.org/format/json/), currently version 1.2.0). Archives come from [https://cricsheet.org/downloads/](https://cricsheet.org/downloads/).

List curated archives:

```powershell
python scripts/download_cricsheet.py --list
```

Download the default T20 international archive:

```powershell
python scripts/download_cricsheet.py
```

Or choose an output directory and a smaller official sample:

```powershell
python scripts/download_cricsheet.py --archive recently_played_2 --output data/raw/cricsheet
```

The `all` archive is very large. Do not download it unless you intend to.

## Configuration

Environment placeholders in `.env.example`:

```text
CRICKET_API_KEY=
SECONDARY_CRICKET_API_KEY=
OPENAI_API_KEY=
```

The data layer reads these through `cricmaster.config.load_settings()` and never requires them for historical imports.

## Planned capabilities

- Pre-match prediction
- Live win probability
- Ball-by-ball match-state analysis
- Team and player form analysis
- Venue and toss effects
- Playing XI and chase/target analysis
- Head-to-head history
- Live-data API integration
- Fallback information retrieval when structured data is unavailable
- Conversational cricket assistant
