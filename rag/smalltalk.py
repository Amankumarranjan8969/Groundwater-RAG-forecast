"""rag/smalltalk.py — fast-path replies for greetings and small talk.

These bypass the retriever entirely so a simple "hi" doesn't waste a TF-IDF
query or an LLM call. The replies are deliberately short and conversational —
save the structured, numbered breakdowns for when the person actually asks a
substantive question (that's what rag/assistant.py is for).
"""
from __future__ import annotations

import random
import re


GREETING = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|hola|namaste|good\s+(morning|afternoon|evening)|"
    r"how\s+are\s+you|how'?s\s+it\s+going)\s*[!.,?]*\s*$",
    re.IGNORECASE,
)
THANKS = re.compile(
    r"^\s*(thanks?|thank\s+you|thx|ty|cheers|much\s+appreciated)\s*[!.,?]*\s*$",
    re.IGNORECASE,
)
BYE = re.compile(
    r"^\s*(bye+|goodbye|see\s+you|cya|later|good\s+night)\s*[!.,?]*\s*$",
    re.IGNORECASE,
)
HELP = re.compile(
    r"^\s*(help|what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
    r"who\s+are\s+you|what\s+are\s+you)\s*[!.,?]*\s*$",
    re.IGNORECASE,
)

GREETING_REPLIES = [
    "Hey! I'm sitting on top of this groundwater run — ask me about a metric, "
    "a season's winner, the Taylor diagram, or anything else on the page.",

    "Hi there. Happy to dig into the analysis with you — a model, a season, "
    "a plot, whatever's on your mind.",

    "Hello! I know this dashboard's methodology and, once you've run an "
    "analysis, the actual numbers too. What are you curious about?",

    "Hey, good to see you. Ask away — a quick metric definition or a deep "
    "dive into one season, either works.",
]

MORNING_REPLY = "Morning! What would you like to look at first — a metric, a season, or one of the plots?"
EVENING_REPLY = "Evening! I'm here if you want to go through the results or talk through any of the plots."
HOW_ARE_YOU_REPLY = (
    "Doing well, thanks for asking. I'm grounded in the methodology docs and, once you've "
    "run something, the live numbers from your session — so I'll give you a straight answer "
    "rather than a guess. What do you want to know?"
)

THANKS_REPLIES = [
    "Anytime — happy to keep going if you've got more questions.",
    "Sure thing. Shout if you want me to walk through any of the figures too.",
    "Glad that helped. I'm around if anything else comes up.",
    "No problem at all.",
]

BYE_REPLIES = [
    "Take care — your analysis stays in this session, so come back anytime without re-running it.",
    "See you! Everything you've run is still here whenever you're ready to pick it back up.",
]

HELP_REPLY = (
    "I'm the assistant built into this groundwater dashboard, so I can help with a few "
    "different things depending on what you need:\n\n"
    "- **Metrics** — what R², RMSE, KGE, NSE, WI, LMI, or Bias actually mean, in plain terms.\n"
    "- **This run** — which model won each season, and its exact numbers, once you've run an analysis.\n"
    "- **The plots** — how to read the Taylor diagram, the wavelet scalogram, or anything else on the page.\n"
    "- **The pipeline** — how features, lags, selection, training, and bias-correction fit together.\n"
    "- **The model line-up** — the six headline models (Cubist, Random Forest, XGBoost, CatBoost, "
    "AdaBoost+CART, Extra Trees) plus the separate PINN / reservoir-computing / SARIMA / LSTM tab.\n\n"
    "No need to phrase things a special way — just ask like you would a colleague."
)


def reply_for(question: str) -> str | None:
    """Return a canned reply if the message is pure small talk, else None."""
    text = (question or "").strip()
    if not text:
        return None

    if GREETING.match(text):
        lowered = text.lower()
        if "morning" in lowered:
            return MORNING_REPLY
        if "evening" in lowered or "night" in lowered:
            return EVENING_REPLY
        if "how are you" in lowered or "how's it going" in lowered:
            return HOW_ARE_YOU_REPLY
        return random.choice(GREETING_REPLIES)

    if THANKS.match(text):
        return random.choice(THANKS_REPLIES)
    if BYE.match(text):
        return random.choice(BYE_REPLIES)
    if HELP.match(text):
        return HELP_REPLY
    return None
