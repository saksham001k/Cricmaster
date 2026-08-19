# Step 18 — Locked roster candidate sanity check

Step 17 selected exactly one production candidate feature family:

```text
previous_xi_core_strength
```

No new feature selection is allowed in Step 18.

The candidate adds four differences:

```text
previous_xi_core_batting_recent_runs_diff
previous_xi_core_batting_recent_strike_rate_diff
previous_xi_core_bowling_recent_wickets_diff
previous_xi_core_bowling_recent_economy_diff
```

Model architecture is selected using 2024 validation only. The model is then fit
through 2024 and evaluated on 2025+ once.

Locked acceptance rules, declared before viewing Step 18 output:

```text
candidate Brier may not worsen by > 0.003
candidate AUC may not worsen by > 0.010
```

If accepted, the candidate artifact is saved under:

```text
models/roster_candidate/
```

This test is final. Do not tune features or thresholds based on the result.
