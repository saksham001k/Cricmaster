# Production prediction router

Cricmaster's production layer routes a match to a frozen model, then returns
probabilities, an edge label, a conservative confidence label, warnings, and
model provenance.

**Probabilities are statistical estimates, not guarantees.** A 60% win
probability does not mean the predicted team will win, and confidence is not a
percentage chance that the prediction is correct.

This router does not retrain models and does not tune against 2025+ data.

## Routing table

```text
                    MATCH
                      |
             +--------+--------+
             |                 |
           T20I               T20
             |                 |
      international       roster-aware
         models             models
             |                 |
             +--------+--------+
                      |
                 prediction mode
                      |
          +-----------+-----------+
          |           |           |
      PRE_TOSS    POST_TOSS      LIVE
```

| Format | Mode | Production artifact | Model family |
| --- | --- | --- | --- |
| T20I | PRE_TOSS | `models/routed_expanded/prematch_router.joblib` → T20I leaf | international T20I |
| T20I | POST_TOSS | `models/routed_expanded/posttoss_router.joblib` → T20I leaf | international T20I |
| T20 | PRE_TOSS | `models/roster_candidate/prematch_t20_roster.joblib` | roster-aware T20 |
| T20 | POST_TOSS | `models/roster_candidate/posttoss_t20_roster.joblib` | roster-aware T20 |
| T20I or T20 | LIVE innings 1 | `models/live/first_innings_model.joblib` | live first innings |
| T20I or T20 | LIVE innings 2 | `models/live/chase_model.joblib` | live chase |

T20I and franchise/domestic T20 stay separate. The generic T20 leaf inside
`prematch_router.joblib` is **not** used for production T20: that weaker model
was rejected as a full solution. Production T20 uses the accepted roster-aware
candidate.

Unsupported combinations raise `UnsupportedPredictionError`. That includes:

- ODI, Test, T10, List A, First Class
- **The Hundred** (format `HUNDRED` or competition `The Hundred`)
- any attempt to treat Hundred as T20

Existing CLIs (`scripts/predict_match.py`, `scripts/predict_posttoss.py`,
`scripts/predict_live.py`) are unchanged. The unified CLI is
`scripts/predict.py`.

## Frozen roster feature family

Feature research is frozen. The accepted T20 family is
`previous_xi_core_strength`:

```text
previous_xi_core_batting_recent_runs_diff
previous_xi_core_batting_recent_strike_rate_diff
previous_xi_core_bowling_recent_wickets_diff
previous_xi_core_bowling_recent_economy_diff
```

These are appended to the existing PRE_TOSS or POST_TOSS base features. Runtime
construction uses the artifact's `features` list in trained order. New feature
families are not added just because they exist in the experimental roster code.

Locked 2025+ sanity check (not for further tuning):

| Mode | Model | Accuracy | AUC | Brier |
| --- | --- | --- | --- | --- |
| PRE_TOSS | baseline | 0.5482 | 0.5610 | 0.2474 |
| PRE_TOSS | roster candidate | 0.5587 | 0.5630 | 0.2468 |
| POST_TOSS | baseline | 0.5587 | 0.5772 | 0.2453 |
| POST_TOSS | roster candidate | 0.5521 | 0.5731 | 0.2451 |

PRE_TOSS was accepted on improvement. POST_TOSS was accepted under the
predeclared non-inferiority gate. Franchise T20 discrimination remains weak
compared with T20I (T20I PRE_TOSS test accuracy around 0.72 / AUC around 0.80
on the routed international model).

## Leakage rules

1. Historical state uses matches **strictly before** the prediction date.
2. PRE_TOSS never uses the current playing XI. If a caller supplies an XI, it
   is ignored and a warning is emitted.
3. PRE_TOSS roster features use only the last previously known XI for each
   team, rebuilt from completed historical T20 matches.
4. POST_TOSS may use the current XI for the existing XI-strength features when
   lineups are known. It still uses **previous** XI for the frozen roster
   family. Current-XI core-strength features are not added unless the loaded
   artifact lists them (the accepted candidate does not).
5. Players are never invented. Missing previous-XI history yields NaN features
   which the trained imputer maps to 0, plus a warning.
6. A PRE_TOSS artifact cannot be used for POST_TOSS. A T20 roster artifact
   cannot be used for T20I.

## Probability, edge, and confidence

These are three different things:

| Field | Meaning |
| --- | --- |
| Probability | Estimated P(team wins). `P(team1) + P(team2) = 1`. Not a guarantee. |
| Edge | How far the favourite's probability is from a coin flip: `very close`, `slight`, `moderate`, `strong`. |
| Confidence | Conservative reliability of *this estimate*: `LOW`, `MEDIUM`, `HIGH`. |

Confidence is **not** `max(probability)`. Example: CSK 53% vs MI 47% is a
`very close` edge and `LOW` confidence, especially on franchise PRE_TOSS.

Confidence combines:

- distance from 50%
- domain reliability (T20I models are substantially stronger than franchise T20)
- historical sample depth (`matches_before` for both teams)
- missing venue
- missing or sparse previous-XI information on T20 roster models
- missing current XI on POST_TOSS / LIVE
- parse errors
- for LIVE: terminal states vs very early first innings

Hard domain caps:

- Franchise T20 PRE_TOSS never receives `HIGH` confidence.
- Close estimates (`favourite < 55%`) are `LOW`.
- Sparse history (`min matches_before < 5`) is capped at `LOW`.

Do not report claims such as "85% confidence of prediction correctness."

## Fallback behavior

| Situation | Behavior |
| --- | --- |
| Missing previous XI | Predict with imputed roster features; emit a warning; do not invent players |
| Missing venue | Predict without venue differentials; warning; confidence cap |
| PRE_TOSS current XI supplied | Ignored; warning |
| POST_TOSS / LIVE XI missing | Existing XI features unavailable; warning |
| Unsupported format / Hundred | Hard error, no silent T20 mapping |
| Artifact mode or domain mismatch | `ArtifactValidationError` |
| Missing artifact file | `ArtifactValidationError` |
| LIVE terminal chase (target reached, all out, overs finished) | Existing live terminal handling is preserved |

## Cache

JSON parsing is cached by a corpus fingerprint (resolved path, file count,
newest mtime, total size). When the corpus changes, the cache is dropped.

**Correctness over speed:** parsed matches after the prediction date may live
in the cache, but `HistoricalState` and `RosterRuntime` are rebuilt for each
request from matches with `date < cutoff` only. Mutable state is not reused
across cutoffs.

TODO: a cutoff-keyed snapshot of `HistoricalState` could make repeated
predictions faster, but only if clones are taken per cutoff. Do not reuse a
later cutoff's state for an earlier date.

## CLI

```powershell
python scripts/predict.py `
  --team1 "India" `
  --team2 "Australia" `
  --format T20I `
  --mode pre_toss `
  --date 2026-08-20

python scripts/predict.py `
  --team1 "Mumbai Indians" `
  --team2 "Chennai Super Kings" `
  --format T20 `
  --competition IPL `
  --mode pre_toss `
  --date 2026-08-20
```

The franchise models were trained on the expanded T20 corpus. For production
T20 calls prefer `--raw data/raw/cricsheet/t20_expanded` when that corpus is
present.

## Model limitations

- Franchise/domestic T20 PRE_TOSS remains close to coin-flip discrimination.
  Treat those probabilities as weakly informative.
- Roster features help only modestly and only through previous known XIs.
- LIVE models share one T20/T20I architecture with an `is_t20i` flag; they are
  not format-specialists.
- Calibration is not a claim of future accuracy. Do not retune thresholds
  against 2025+ just to make reports look sharper.
