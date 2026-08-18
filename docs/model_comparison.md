# Step 11 — Apples-to-apples model comparison

The expanded corpus has a different and harder 2025+ test population than the
original T20I + IPL corpus. Aggregate test metrics therefore cannot be compared
directly.

Step 11 evaluates both systems on the exact same match IDs.

Run:

```powershell
python scripts/compare_models.py
```

The report includes:

- old 2025+ test match count;
- expanded 2025+ test match count;
- exact common match count;
- old model metrics on common matches;
- expanded model metrics on those same common matches;
- expanded-model metrics on all new coverage;
- common-match comparisons by format and gender;
- expanded-model metrics by competition for sufficiently large groups.

Output:

```text
data/processed/model_comparison/step11_metrics.json
```

Do not promote expanded models until the common-match comparison is reviewed.
A broader model can be useful even with lower aggregate accuracy if it preserves
performance on the original domain while adding acceptable calibration on new
competitions.
