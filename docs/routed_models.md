# Step 12 — Format-aware model routing

Step 11 showed that expanded historical coverage is valuable, but one universal
T20I + franchise/domestic T20 model performs unevenly on the new competitions.

Step 12 trains separate models for:

- PRE_TOSS / T20I
- PRE_TOSS / T20
- POST_TOSS / T20I
- POST_TOSS / T20

Each domain independently compares Logistic Regression and
HistGradientBoosting on the 2024 validation partition. The final 2025+ test
partition is not used for selection.

## Train

```powershell
python scripts/train_routed_models.py
```

Artifacts:

```text
models/routed_expanded/
├── prematch_router.joblib
├── posttoss_router.joblib
└── metrics.json
```

## Evaluate against the single expanded models

```powershell
python scripts/evaluate_routed_models.py
```

This compares the single expanded model and the routed model on exactly the
same expanded 2025+ population, both globally and by format/competition.

Do not promote the routers until the evaluation is reviewed.
