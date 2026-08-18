# Step 7 — Live win-probability training

This stage trains two separate models:

- `first_innings_model.joblib`
- `chase_model.joblib`

The training frame uses only legal-ball states and completed binary-result
matches. Live states are joined to the leakage-safe `POST_TOSS` team row, so
early-innings predictions retain team-strength, venue, toss, and available XI
context.

Every match receives equal total sample weight within each innings model. This
prevents long innings from dominating training merely because they contain more
delivery states.

Train:

```powershell
python scripts/train_live.py
```

The temporal policy remains:

- training: through 2023-12-31
- validation/model selection: 2024
- final untouched test: 2025 onward

Do not select a model based on the final test metrics.
