# Step 16 — Independent rolling feature-group ablation

Step 15's combined enhanced feature set failed its rolling gate. Step 16 avoids
bundling ideas together.

It evaluates each candidate independently on the existing expanded T20 dataset:

- competition-season Elo with 50% offseason regression toward 1500;
- shrunk overall team win rate;
- shrunk H2H win rate;
- shrunk venue win rate;
- log-scaled prior-match experience;
- POST_TOSS venue/toss alignment.

No production feature code is changed.

## Evaluation windows

```text
train <= 2021 -> validate 2022
train <= 2022 -> validate 2023
train <= 2023 -> validate 2024
```

The script explicitly drops 2025+ before feature construction/evaluation.

A feature group passes only if:

- it improves at least 2 of 3 folds;
- mean Brier delta is <= -0.001;
- no fold degrades by more than +0.003.

Run:

```powershell
python scripts/evaluate_t20_feature_groups.py
```

Output:

```text
data/processed/model_comparison/step16_feature_groups.json
```

Only groups that pass this gate should be considered for production
implementation.
