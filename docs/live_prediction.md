# Step 8 — Live prediction CLI

After Step 7 has trained:

- `models/live/first_innings_model.joblib`
- `models/live/chase_model.joblib`

you can query a live state manually.

## Chase example

```powershell
python scripts/predict_live.py `
  --team1 "India" `
  --team2 "Australia" `
  --batting-team "India" `
  --format T20I `
  --date 2026-08-19 `
  --gender male `
  --innings 2 `
  --runs 132 `
  --wickets 4 `
  --overs 15.3 `
  --target 181 `
  --venue "Melbourne Cricket Ground" `
  --toss-winner "Australia" `
  --toss-decision field
```

`15.3` is cricket notation: 15 overs and 3 legal balls, not 15.3 decimal
overs.

## First innings example

```powershell
python scripts/predict_live.py `
  --team1 "India" `
  --team2 "Australia" `
  --batting-team "India" `
  --format T20I `
  --date 2026-08-19 `
  --gender male `
  --innings 1 `
  --runs 96 `
  --wickets 3 `
  --overs 11.2 `
  --venue "Melbourne Cricket Ground" `
  --toss-winner "India" `
  --toss-decision bat
```

Do not pass `--target` in the first innings.

Playing XIs are optional through `--team1-xi` and `--team2-xi`. Unknown
lineups are left missing rather than invented.
