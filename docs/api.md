# Cricmaster API

The HTTP layer is a thin adapter over the frozen production prediction router.
It does not retrain models, change features, or invent confidence.

```text
CLIENT
  → FastAPI (validation, CORS, request IDs, errors)
    → PredictionService / LiveMatchService / ChatService
      → production prediction router
```

Probabilities are statistical estimates, not guarantees. Confidence is not
`max(probability)`.

## Run locally

From the repository root, with `src` on `PYTHONPATH` or the package installed:

```powershell
pip install -r requirements.txt
python scripts/run_api.py
```

Or:

```powershell
uvicorn cricmaster.api.app:app --reload --host 127.0.0.1 --port 8000
```

OpenAPI / Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

ReDoc: `/redoc`

## Configuration

| Variable | Purpose |
| --- | --- |
| `CRICMASTER_T20_CORPUS_DIR` | Historical Cricsheet corpus used at prediction time |
| `CRICMASTER_CORS_ORIGINS` | Comma-separated allowed browser origins |
| `CRICMASTER_API_HOST` | Bind host for `scripts/run_api.py` (default `127.0.0.1`) |
| `CRICMASTER_API_PORT` | Bind port (default `8000`) |
| `CRICKET_API_KEY` | CricketData/CricAPI key for `/live/*` only |

Do not send API keys or model filesystem paths in HTTP requests. Keys stay
server-side. `CRICKET_API_KEY`, `SECONDARY_CRICKET_API_KEY`, and
`OPENAI_API_KEY` are never logged.

Default CORS origins when `CRICMASTER_CORS_ORIGINS` is unset:

- `http://localhost:3000`
- `http://localhost:5173`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

Do not set production CORS to `*` together with credentials. If you explicitly
set `CRICMASTER_CORS_ORIGINS=*`, the API allows that origin list and disables
credentials.

## Endpoints

### `GET /health`

Cheap liveness check. Confirms process health and whether model files exist.
It does not parse historical JSON or load sklearn estimators.

### `GET /models`

Safe metadata: supported formats/modes, model families, limitations,
`trained_through` (`2024-12-31` for v1 artifacts), and artifact availability.
No absolute filesystem paths and no estimator internals.

### `POST /predict`

Unified PRE_TOSS / POST_TOSS / LIVE prediction. The body is mode-aware:

PRE_TOSS:

```json
{
  "team1": "India",
  "team2": "Australia",
  "format": "T20I",
  "mode": "PRE_TOSS",
  "date": "2026-08-20",
  "venue": "Melbourne Cricket Ground"
}
```

Franchise T20:

```json
{
  "team1": "Mumbai Indians",
  "team2": "Chennai Super Kings",
  "format": "T20",
  "mode": "PRE_TOSS",
  "competition": "IPL",
  "date": "2026-08-20"
}
```

POST_TOSS requires `toss_winner` and `toss_decision` (`bat`, `field`, or `bowl`).
Optional `team1_xi` / `team2_xi` are used only after the toss.

LIVE requires `batting_team`, `innings`, `runs`, `wickets`, and `overs` (or
`legal_balls`). Innings 2 also requires `target`. Overs use cricket notation
such as `15.3`.

PRE_TOSS current XI fields are accepted but ignored by the router, which emits
a warning. Clients cannot supply model artifact paths.

Example response fields: `team1_probability`, `team2_probability`,
`predicted_team`, `edge`, `confidence`, `prediction_mode`, `format`,
`model_name`, `model_family`, `warnings`, `drivers`.

`P(team1) + P(team2) = 1`.

### `POST /chat`

Deterministic chatbot orchestration. See `docs/chatbot_backend.md`.

### `GET /live/matches`

Normalized CricketData `currentMatches` records. Returns `[]` when the feed is
empty. If `CRICKET_API_KEY` is missing, responds **503**
`live_provider_unconfigured` without exposing secrets.

The provider already caches the feed for 30 seconds in-process. This endpoint
does not invent scores.

### `POST /live/{match_id}/predict`

Fetches the normalized current match, converts it only when the state is
predictable (started T20/T20I, parseable overs, known batting team, and a
reliable chase target when required), then calls the production LIVE router.

If conversion is unsafe, the API returns **422** `insufficient_live_state`
rather than guessing runs, wickets, overs, or a DLS target. Unknown ids are
**404** `match_not_found`.

## Routing (unchanged)

| Format | Mode | Family |
| --- | --- | --- |
| T20I | PRE_TOSS / POST_TOSS | international T20I |
| T20 | PRE_TOSS / POST_TOSS | roster-aware T20 |
| T20I or T20 | LIVE | live first innings / chase |

Unsupported: Hundred, ODI, Test, T10, unknown formats. These fail clearly.

## Confidence

`LOW` / `MEDIUM` / `HIGH` describe estimate reliability, not a percentage chance
the prediction is correct. Franchise PRE_TOSS is treated conservatively.

## Errors

| Situation | HTTP | `error` |
| --- | --- | --- |
| Unsupported format / Hundred | 400 | `unsupported_format` |
| Invalid mode-specific fields | 422 | `validation_error` |
| Missing model artifact | 503 | `model_unavailable` |
| Live provider has no key / failed | 503 | `live_provider_unconfigured` / `live_provider_failure` |
| Live state cannot be converted | 422 | `insufficient_live_state` |
| Unexpected server error | 500 | FastAPI/Starlette default or mapped artifact error |

Every response includes `X-Request-ID`. Error JSON also includes `request_id`.
Bodies never include stack traces, `.env` contents, or API keys.

## Security

- Team names, venues, and XI entries have length limits.
- Wickets must be 0–10; runs are non-negative and capped.
- Overs are parsed with existing cricket notation rules.
- Clients cannot choose artifact paths or pickle payloads.
- Joblib artifacts are loaded only from server configuration.
- No authentication in v1.

TODO (deployment): add rate limiting, TLS termination, and an allowlist for
production CORS origins.

## Performance

- `/health` and `/models` only `stat()` artifact files.
- Loaded model bundles are cached in-process by file fingerprint.
- Historical JSON parsing uses the existing cutoff-safe cache. Mutable
  `HistoricalState` is still rebuilt per prediction date so future matches
  cannot leak into earlier cutoffs.
