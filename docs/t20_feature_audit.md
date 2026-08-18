# Step 14 — T20 feature audit

Steps 12 and 13 showed that model splitting does not solve Cricmaster's weak
domestic/franchise T20 performance. Step 14 is diagnostic: it measures whether
the underlying historical features have enough signal and coverage.

It audits:

- prior-match history depth by competition;
- H2H coverage;
- team-at-venue coverage;
- playing-XI availability and prior player-history coverage;
- cold-start teams;
- univariate feature discrimination in 2024 versus 2025+;
- currently available dataset fields that are not used by the model.

Run:

```powershell
python scripts/audit_t20_features.py
```

Outputs:

```text
data/processed/feature_audit/
├── summary.json
├── competition_coverage.csv
├── team_cold_start.csv
├── prematch_feature_signal.csv
├── posttoss_feature_signal.csv
└── unused_feature_availability.csv
```

Interpretation:

- orientation-free AUC near 0.50 = little standalone signal;
- a large drop from 2024 to 2025+ = unstable feature / season shift;
- high no-H2H or no-venue percentages = sparse historical context;
- low lineup coverage = POST_TOSS player-strength features cannot help often.

This step intentionally does not train another model.
