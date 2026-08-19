# Step 17 — Roster continuity and player-weighted T20 strength

Step 17 tests a new source of signal after summary-stat feature groups failed.

PRE_TOSS uses only the last previously known XI, player form known before the
current match, continuity between the last two known XIs, and rest days.
It never uses the current playing XI.

POST_TOSS may additionally use the actual current XI and its overlap with the
previous XI.

Role-weighted player cores are historical:
- batting core: seven XI players with the most prior batting innings;
- bowling core: five XI players with the most prior legal balls bowled.

Recent runs, strike rate, wickets, and economy are measured before the current
match. Current-match deliveries update player history only after the feature
snapshot is recorded.

Evaluation remains rolling and pre-2025 only:
- train <= 2021 -> validate 2022
- train <= 2022 -> validate 2023
- train <= 2023 -> validate 2024

Run:
```powershell
python scripts/evaluate_t20_roster_features.py
```
The first run builds a reusable pre-2025 roster cache under
`data/processed/model_comparison/`.
