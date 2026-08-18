# Step 15 — Rolling T20 feature engineering

Step 14 showed weak model signal rather than missing raw coverage.

Step 15 tests two conservative improvements:

1. Player efficiency features already computed internally:
   - batting strike rate;
   - recent batting strike rate;
   - recent bowling economy.

2. Bayesian-style shrinkage of noisy historical rates:
   - overall team win rate;
   - H2H win rate;
   - venue win rate.

Shrinkage formula:

```text
(wins + 10 * 0.5) / (matches + 10)
```

## Evaluation policy

Do not use 2025+ to select these features.

Rolling validation:

```text
train <= 2021 -> validate 2022
train <= 2022 -> validate 2023
train <= 2023 -> validate 2024
```

Candidates survive only if they improve at least 2 of 3 folds and their mean
Brier delta is negative.

## Rebuild

```powershell
python scripts/build_features.py `
  --input data/raw/cricsheet/t20_expanded `
  --output data/processed/t20_expanded_step15 `
  --mode both `
  --skip-live
```

Then:

```powershell
python scripts/evaluate_rolling_t20_features.py
```

This evaluation intentionally does not inspect 2025+.
