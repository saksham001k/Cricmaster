# Cricmaster chatbot backend

The chatbot is a **deterministic orchestrator**, not a free-form cricket expert
and not an LLM. It translates structured (or very simple) user requests into
safe calls to the frozen production prediction layer.

Do not expect it to answer arbitrary cricket trivia.

## Intents

| Intent | Purpose |
| --- | --- |
| `predict_match` | PRE_TOSS / POST_TOSS prediction |
| `live_prediction` | LIVE prediction when score fields are present |
| `explain_prediction` | Re-run the match and format actual drivers/warnings |
| `model_capabilities` | Supported formats and limitations |
| `help` | How to call the chatbot |

Unknown intents return help text instead of invented answers.

## Structured request

`POST /chat`

```json
{
  "intent": "predict_match",
  "team1": "India",
  "team2": "Australia",
  "format": "T20I",
  "mode": "PRE_TOSS",
  "date": "2026-08-20"
}
```

Response:

```json
{
  "intent": "predict_match",
  "message": "Cricmaster gives India a 61% estimated win probability against Australia. Edge is moderate. Confidence is MEDIUM. ...",
  "prediction": { },
  "suggestions": [
    "Ask for the main prediction drivers",
    "Switch to post-toss once the XI is known"
  ]
}
```

The `message` always includes probability, edge, and confidence language.
It never says a team will definitely win.

## Explanation rules

Explanations use the prediction result's real `drivers` and `warnings`.

- If drivers exist, they are described as **statistical associations**, not
  causes.
- If drivers are missing (for example tree models without local coefficients):
  "The selected model does not expose reliable local feature contributions."
- Player names are not invented.
- Franchise PRE_TOSS limitations are repeated when the backend already warned.

## Optional plain text

A `message` field may be supplied for a few safe patterns:

- "What formats can you predict?"
- "Predict MI vs CSK before toss"
- "Who will win India vs Australia tomorrow in T20I?"

Team shortcuts such as MI/CSK are expanded only in this chat parser. Live
scorelines without structured `runs` / `wickets` / `overs` / `target` are
**not** guessed; the chatbot asks for a structured LIVE request.

## What this layer will not do

- Call OpenAI or any LLM
- Fetch undocumented web sources
- Treat Hundred/ODI/Test as T20
- Override frozen model probabilities to sound more confident
