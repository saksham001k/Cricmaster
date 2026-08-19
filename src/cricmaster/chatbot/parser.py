"""Conservative text hints for chat. Full NLP is out of scope for v1."""

from __future__ import annotations

import re
from datetime import date, timedelta

from cricmaster.chatbot.intents import ChatIntent

TEAM_SHORTCUTS = {
    "mi": "Mumbai Indians",
    "csk": "Chennai Super Kings",
    "rcb": "Royal Challengers Bengaluru",
    "kkr": "Kolkata Knight Riders",
    "dc": "Delhi Capitals",
    "rr": "Rajasthan Royals",
    "srh": "Sunrisers Hyderabad",
    "pbks": "Punjab Kings",
    "gt": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
    "ind": "India",
    "india": "India",
    "aus": "Australia",
    "australia": "Australia",
    "eng": "England",
    "england": "England",
}

_LEAD_WORDS = {
    "predict",
    "who",
    "will",
    "win",
    "wins",
    "please",
    "can",
    "you",
    "for",
}
_STOP_WORDS = {
    "tomorrow",
    "today",
    "before",
    "after",
    "in",
    "on",
    "t20i",
    "t20",
    "ipl",
    "odi",
}


def expand_team(value: str) -> str:
    key = " ".join(value.strip().lower().split())
    return TEAM_SHORTCUTS.get(key, " ".join(value.strip().split()))


def infer_intent_from_message(message: str) -> ChatIntent:
    text = message.strip().lower()
    if re.search(r"\b(what formats|which formats|capabilities|what can you predict)\b", text):
        return ChatIntent.MODEL_CAPABILITIES
    if re.search(r"^\s*(help|how do i|what can you)\b", text):
        return ChatIntent.HELP
    if re.search(r"\b(why|explain|drivers?)\b", text):
        return ChatIntent.EXPLAIN_PREDICTION
    if re.search(r"\b(need \d+|from \d+ balls|wickets left|live)\b", text):
        return ChatIntent.LIVE_PREDICTION
    if re.search(r"\b(predict|who will win|who wins|before toss|pre[- ]toss)\b", text):
        return ChatIntent.PREDICT_MATCH
    if re.search(r"\bvs\b|\bversus\b", text):
        return ChatIntent.PREDICT_MATCH
    return ChatIntent.UNKNOWN


def extract_teams(message: str) -> tuple[str, str] | None:
    parts = re.split(r"\s+(?:vs\.?|versus)\s+", message, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None

    left_words = [word.strip(" .,?!") for word in parts[0].split() if word.strip(" .,?!")]
    while left_words and left_words[0].lower() in _LEAD_WORDS:
        left_words.pop(0)
    right_words: list[str] = []
    for word in parts[1].split():
        cleaned = word.strip(" .,?!")
        if cleaned.lower() in _STOP_WORDS:
            break
        if cleaned:
            right_words.append(cleaned)

    left = expand_team(" ".join(left_words[-4:]))
    right = expand_team(" ".join(right_words[:4]))
    if not left or not right or left.lower() == right.lower():
        return None
    return left, right


def extract_format(message: str) -> str | None:
    text = message.lower()
    if "t20i" in text or "twenty20 international" in text:
        return "T20I"
    if "hundred" in text:
        return "HUNDRED"
    if re.search(r"\bodi\b", text):
        return "ODI"
    if re.search(r"\bt20\b", text) or "ipl" in text:
        return "T20"
    return None


def extract_mode(message: str) -> str | None:
    text = message.lower()
    if "after toss" in text or "post toss" in text or "post-toss" in text:
        return "POST_TOSS"
    if "before toss" in text or "pre toss" in text or "pre-toss" in text:
        return "PRE_TOSS"
    if "live" in text or "need " in text:
        return "LIVE"
    return None


def extract_competition(message: str) -> str | None:
    if re.search(r"\bipl\b", message, flags=re.IGNORECASE):
        return "IPL"
    return None


def extract_date(message: str, *, today: date | None = None) -> date | None:
    now = today or date.today()
    text = message.lower()
    if "tomorrow" in text:
        return now + timedelta(days=1)
    if "today" in text:
        return now
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
    if match:
        return date.fromisoformat(match.group(1))
    return None
