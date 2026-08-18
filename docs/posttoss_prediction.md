# Step 6 — POST_TOSS + playing-XI model

Rebuild the feature table with both prediction modes:

```powershell
python scripts/build_features.py `
  --input data/raw/cricsheet/t20_corpus `
  --output data/processed/t20_corpus `
  --mode both
```

Train:

```powershell
python scripts/train_posttoss.py
```

Predict without XI:

```powershell
python scripts/predict_posttoss.py `
  --team1 "India" `
  --team2 "Australia" `
  --format T20I `
  --date 2026-08-19 `
  --gender male `
  --venue "Melbourne Cricket Ground" `
  --toss-winner "Australia" `
  --toss-decision field
```

Add `--team1-xi` and `--team2-xi` with comma-separated player names when
the actual lineups are available. The model never invents lineups.
