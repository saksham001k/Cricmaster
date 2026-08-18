# Step 13 — Validation-gated competition specialists

Format routing did not materially solve Cricmaster's domestic/franchise T20
problem. The generic T20 branch remained near random on many leagues.

Step 13 adds a hierarchical router:

```text
T20I
  -> T20I model

T20
  -> approved competition specialist
  -> otherwise generic T20 fallback
```

A competition specialist is activated only when all of the following are true:

- at least 200 training matches through 2023;
- at least 25 validation matches in 2024;
- its selected model beats the generic T20 fallback on those same validation
  matches by at least 0.005 Brier score.

The fallback comparison is leakage-safe: the fallback development model used
for gating is fit only on data through 2023.

## Train

```powershell
python scripts/train_competition_specialists.py
```

## Evaluate

```powershell
python scripts/evaluate_competition_specialists.py
```

Artifacts are written under:

```text
models/specialist_expanded/
```

Do not promote them until the frozen 2025+ evaluation is reviewed.
