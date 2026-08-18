# Step 9 — CricketData automatic live integration

Cricmaster now has a concrete adapter for CricketData/CricAPI's
`currentMatches` endpoint.

The API key is read only from `CRICKET_API_KEY`. It is never written to logs or
source control.

## List the current feed

```powershell
python scripts/live_now.py --list
```

This costs one `currentMatches` API hit.

The listing labels each record as one of:

- `LIVE/PREDICTABLE`
- `LIVE/AMBIGUOUS`
- `ENDED/TERMINAL`
- `UPCOMING`
- `UNSUPPORTED/<format>`

## Automatically predict one live match

If exactly one predictable live T20/T20I match exists:

```powershell
python scripts/live_now.py
```

Choose by team/name:

```powershell
python scripts/live_now.py --search "India"
```

Or use the CricketData match id:

```powershell
python scripts/live_now.py --match-id "<uuid>"
```

Each invocation fetches `currentMatches` once. The provider caches the result
within the process for 30 seconds.

## Defensive normalization rules

`currentMatches` is treated as an imperfect upstream source:

- completed results can appear in the feed;
- `matchEnded=False` can still accompany terminal status text;
- toss fields can be absent;
- innings labels can contain both team names;
- The Hundred can appear as `matchType=t20` while `score.o` is balls;
- rain/DLS matches may require a revised target not derivable from first-innings
  score alone.

Cricmaster refuses an automatic chase prediction when a reliable target cannot
be established. It never invents toss data, playing XIs, or a DLS target.
