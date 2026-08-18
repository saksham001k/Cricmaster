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
| Live API adapters (CricketData, Sportmonks, EntitySport, Roanuz, ...) | Planned |
| Search fallback retrieval | Planned (interface only) |
| Feature engineering and prediction models | Planned |
| Ball-by-ball live win probability | Planned |
| Conversational cricket assistant | Planned |

## Supported / target cricket formats

**Formats** describe how a match is played. **Competitions** are named tournaments.

| Format | Meaning | Status |
| --- | --- | --- |
| `TEST` | Test cricket | Historical ingestion ready |
| `ODI` | One-day international | Historical ingestion ready |
| `T20I` | Twenty20 international | Historical ingestion ready |
| `T20` | Franchise / domestic T20 | Historical ingestion ready |
| `T10` | Ten-over cricket | Normalized if source data appears |
| `HUNDRED` | 100-ball cricket | Historical ingestion ready when Cricsheet event data is present |
| `FIRST_CLASS` | Multi-day domestic cricket | Historical ingestion ready |
| `LIST_A` | One-day domestic cricket | Historical ingestion ready |
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

Do not treat a competition as fully supported until a live provider and prediction path exist for it. Today only historical JSON parsing is implemented.

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
│   ├── features/          # planned
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
