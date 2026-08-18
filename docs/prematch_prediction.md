# Step 5 — PRE_TOSS prediction CLI

After training `models/prematch/prematch_model.joblib`, run:

```powershell
python scripts/predict_match.py `
  --team1 "India" `
  --team2 "Australia" `
  --format T20I `
  --date 2026-08-19 `
  --gender male `
  --venue "Melbourne Cricket Ground"
```

For IPL/franchise matches use `--format T20` and optionally provide
`--competition IPL`.

The runtime reconstructs historical state using only Cricsheet matches strictly
before the requested date. It currently supports only T20I and T20 because those
are the formats used to train the first PRE_TOSS model.
