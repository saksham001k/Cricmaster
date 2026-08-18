# Step 10 — Expand the historical T20 corpus

Step 10 broadens Cricmaster's historical T20 coverage while leaving the current
production corpus and live models untouched.

## Separate evaluation directories

New raw data:

```text
data/raw/cricsheet/t20_expanded/
```

New processed pre/post-toss data:

```text
data/processed/t20_expanded/
```

Existing `t20_corpus` files are not modified by the Step 10 commands.

## Curated archive set

The bulk downloader includes:

- official T20 internationals
- IPL
- Big Bash League
- Women's Big Bash League
- PSL
- CPL
- WPL
- BPL
- LPL
- Major League Cricket
- International League T20
- SA20
- Super Smash
- T20 Blast
- Syed Mushtaq Ali Trophy

The Hundred is intentionally excluded because Cricmaster currently uses a
separate HUNDRED format and the first live models are T20/T20I only.

## Download

```powershell
python scripts/download_t20_corpus.py
```

Re-running is safe: existing zip files and extracted files are skipped unless
`--force` is used.

## Build only historical pre/post-toss features

```powershell
python scripts/build_features.py `
  --input data/raw/cricsheet/t20_expanded `
  --output data/processed/t20_expanded `
  --mode both `
  --skip-live
```

`--skip-live` still parses deliveries as needed to update historical player and
team state, but it does not materialize delivery-by-delivery live rows and does
not write `live_states.parquet`.

Expected invariant:

```text
live_state_rows=0
live_generation=skipped
validation_issues=0
```

## Train comparison models without replacing production models

```powershell
python scripts/train_prematch.py `
  --input data/processed/t20_expanded/prematch_features.parquet `
  --output models/prematch_expanded

python scripts/train_posttoss.py `
  --input data/processed/t20_expanded/prematch_features.parquet `
  --output models/posttoss_expanded
```

Do not promote these bundles yet. Compare validation/test metrics and corpus
coverage first.
