"""rag/assistant.py — grounded, conversational assistant for the dashboard.

Two answer paths:
  1. Small talk → instant canned reply (rag.smalltalk), no retrieval, no LLM.
  2. Everything else → retrieval + (LLM if a key is set, otherwise offline).

Default LLM: Gemini (GEMINI_API_KEY). Falls back to Anthropic if configured,
or to a composer that mixes live numbers with knowledge-base background.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from .context import build_job_context
from .llm_provider import active_provider, complete
from .retriever import KnowledgeBase
from .smalltalk import reply_for as smalltalk_reply


# --------------------------------------------------------------------------- #
# System prompt for the LLM path
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """You are the assistant embedded in a seasonal groundwater
machine-learning dashboard. The user just ran an analysis and is asking you
about it. Talk like a knowledgeable colleague sitting next to them, not a
manual — this should read like a real conversation, not a templated report.

HOW TO ANSWER
- Lead with the direct answer in the first sentence. No preamble.
- If the CONTEXT contains numbers that answer the question, use them.
  Quote exact values, exact model names, exact seasons.
- If the question is conceptual (e.g. "what is KGE"), give one clear sentence
  of definition, then weave in 1-3 short points of how to interpret it —
  prose or light bullets, whichever reads more naturally for that answer.
- If the user asks "which model won / performed best", read the answer straight
  from CONTEXT. Do not pick a model yourself. If they don't specify which
  model family, default to the six headline models unless they mention PINN,
  reservoir computing / ESN, SARIMA, or LSTM — that's a separate model family
  shown in its own "12 · Deep learning" tab, scored the same way.
- If the user asks "explain like I'm new", drop all jargon. Use a plain analogy.
- If the question has two parts, answer both — but let it read as one
  connected answer, not a rigid enumerated checklist.

STYLE
- Vary your openings. Do not start every answer with "Sure!" or "Here's…".
- Avoid mechanical scaffolding like "Point by point:" or restating the
  question back before answering — just answer it.
- Use markdown sparingly and purposefully: **bold** for metric names or model
  names, `code` for column names. Reach for a numbered or bulleted list only
  when the content is genuinely a sequence or a set of parallel items — not
  as a default structure for every answer.
- One emoji maximum, and only for greetings or thanks.
- Answers under 180 words unless the user asks for depth.
- Never say "based on the context provided", "as an AI", or "the knowledge base".

GROUND RULES
- Never invent numbers, model names, or seasons. If CONTEXT lacks the answer,
  say so in one sentence and suggest a related question.
- Never mention retrieval, chunks, or documents.
- If a prior assistant turn answers part of the question, build on it — don't
  repeat it verbatim.
"""


# --------------------------------------------------------------------------- #
# Variation banks for the offline composer
# --------------------------------------------------------------------------- #
_OPENERS = [
    "Quick answer:",
    "Here's the short version:",
    "Short version:",
    "Let me unpack that:",
    "Here's what's going on:",
    "Okay — here's the picture:",
]

_CLOSERS = [
    "Want me to explain any term in more detail?",
    "Happy to go deeper on any of these points.",
    "If you tell me which season you care about, I can pull its exact numbers.",
    "Ask me about a specific model and I'll compare it.",
    "Want the same thing broken down season by season?",
]

_NO_ANSWER = (
    "I don't have anything on that one. A few things I can definitely help with: "
    "a metric like R² or KGE, which model won a given season, how to read the "
    "Taylor diagram, what a pipeline stage does, or a summary of the uploaded workbook. "
    "What sounds closest to what you're after?"
)

_SHORT_CONFIRMATIONS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
    "go on", "please do", "tell me more", "more", "continue",
}


# --------------------------------------------------------------------------- #
# Lazy knowledge-base singleton
# --------------------------------------------------------------------------- #
_KB = None


def _kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _pick(options: list[str], seed_text: str) -> str:
    """Stable-but-not-fixed choice: same question → same phrasing."""
    h = int(hashlib.md5(seed_text.encode()).hexdigest(), 16)
    return options[h % len(options)]


def _format_history(history: list[dict] | None, max_turns: int = 4) -> str:
    """Return the last N turns as a compact transcript for the LLM."""
    if not history:
        return ""
    recent = history[-max_turns * 2:]
    lines = []
    for turn in recent:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip().replace("\n", " ")
        if len(content) > 240:
            content = content[:240].rsplit(" ", 1)[0] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #
_METRIC_ALIASES = {
    "r²": "Bias-corrected test R2",
    "r2": "Bias-corrected test R2",
    "rmse": "Bias-corrected test RMSE",
    "mae": "Bias-corrected test MAE",
    "kge": "Bias-corrected test KGE",
    "nse": "Bias-corrected test NSE",
    "bias": "Bias-corrected test Bias",
    "pearson": "Bias-corrected test Pearson r",
}

_SEASON_ALIASES = {
    "premonsoon": "Pre-Monsoon",
    "monsoon": "Monsoon",
    "rainy": "Monsoon",
    "postmonsoon": "Post-Monsoon",
    "nonmonsoon": "Non-Monsoon",
    "dry": "Non-Monsoon",
    "winter": "Non-Monsoon",
}


def _detect_season(question_lower: str, seasons: list[str]) -> str | None:
    """Return the season named in the question.

    Longest-first matching prevents 'Monsoon' from winning over 'Non-Monsoon'
    or 'Post-Monsoon' when the user types 'non monsoon'.
    """
    q_compact = question_lower.replace("-", "").replace(" ", "")

    ranked = sorted(
        seasons,
        key=lambda s: len(s.replace("-", "").replace(" ", "")),
        reverse=True,
    )
    for season in ranked:
        s_compact = season.lower().replace("-", "").replace(" ", "")
        if s_compact in q_compact:
            return season

    # Alias fallback (rainy / dry / winter)
    for token, canonical in _SEASON_ALIASES.items():
        if token in q_compact and canonical in seasons:
            return canonical
    return None


def _detect_metric(question_lower: str) -> tuple[str, str] | None:
    """Return (alias, column_name) for the first metric named, else None."""
    for alias, column in _METRIC_ALIASES.items():
        if alias in question_lower:
            return alias, column
    return None


def _detect_model(question_lower: str, model_names: list[str]) -> str | None:
    """Return the model named in the question, if any."""
    q_norm = question_lower.replace("_", " ")
    for name in model_names:
        n_lower = name.lower().strip()
        if len(n_lower) < 4:
            continue
        if n_lower in q_norm:
            return name
    return None


_DEEP_MODEL_ALIASES = {
    "physics-informed neural network": "PINN (Physics-Informed NN)",
    "physics informed neural network": "PINN (Physics-Informed NN)",
    "physics-informed": "PINN (Physics-Informed NN)",
    "physics informed": "PINN (Physics-Informed NN)",
    "pinn": "PINN (Physics-Informed NN)",
    "reservoir computing": "Reservoir Computing (ESN)",
    "echo state network": "Reservoir Computing (ESN)",
    "echo state": "Reservoir Computing (ESN)",
    "reservoir": "Reservoir Computing (ESN)",
    "esn": "Reservoir Computing (ESN)",
    "sarimax": "SARIMA (SARIMAX)",
    "sarima": "SARIMA (SARIMAX)",
    "lstm": "LSTM (Deep Learning)",
}
# Keywords that mean "the deep-learning tab" even with no specific model named.
_DEEP_FAMILY_KEYWORDS = ["deep learning", "deep-learning"] + list(_DEEP_MODEL_ALIASES)


def _detect_deep_model(question_lower: str) -> str | None:
    """Return the canonical deep-learning-tab model named in the question, if any."""
    q_norm = question_lower.replace("_", " ")
    for alias in sorted(_DEEP_MODEL_ALIASES, key=len, reverse=True):
        if alias in q_norm:
            return _DEEP_MODEL_ALIASES[alias]
    return None


def _mentions_deep_family(question_lower: str) -> bool:
    q_norm = question_lower.replace("_", " ")
    return any(keyword in q_norm for keyword in _DEEP_FAMILY_KEYWORDS)


def _is_numeric_question(question_lower: str) -> bool:
    """True when the user clearly wants numbers from the current run,
    not methodology prose."""
    signals = [
        "compare", "top", "best", "winner", "won", "leading",
        "score", "scored", "performance",
        "r²", "r2", "rmse", "mae", "kge", "nse", "bias",
        "this run", "current run", "my run",
    ]
    return any(sig in question_lower for sig in signals)


# --------------------------------------------------------------------------- #
# Live-number extraction
# --------------------------------------------------------------------------- #
def _extract_live_numbers(question: str, payload: dict | None) -> list[str]:
    """Return a short list of live facts relevant to the question."""
    if not payload:
        return []
    results = payload.get("results") or {}
    if not results:
        return []

    q = question.lower()
    seasons = list(results.keys())
    facts: list[str] = []

    detected_season = _detect_season(q, seasons)
    detected_metric = _detect_metric(q)

    all_models: list[str] = []
    for res in results.values():
        for name in res["metrics"]["Model"].astype(str).tolist():
            if name not in all_models:
                all_models.append(name)
    detected_model = _detect_model(q, all_models)

    # ---- Case -1: was data imputed / rescued for a small dataset? ----
    imputation_keywords = ["imput", "missing data", "missing value", "low data",
                           "small dataset", "rescue", "dropped row", " nan "]
    if any(kw in f" {q} " for kw in imputation_keywords):
        seasons_to_check = [detected_season] if detected_season else seasons
        for season in seasons_to_check:
            rescued = results[season].get("low_data_rescue_used")
            if rescued:
                facts.append(
                    f"Yes — **{season}** had too few rows for the split after the strict "
                    "pass, so incomplete lag/rolling/trend values were imputed (forward-fill "
                    "within the village, then median) instead of dropped."
                )
            else:
                facts.append(f"No — **{season}** had enough data; rows were dropped as usual, nothing was imputed.")
        return facts[:8]

    # ---- Case 0: comparison intent ("compare top two", "compare X and Y") ----
    compare_intent = any(w in q for w in
                         ["compare", "versus", " vs ", "top two", "top 2"])
    if compare_intent:
        season = detected_season or next(iter(results))
        metrics_table = results[season]["metrics"]

        named = [m for m in all_models if m.lower() in q]

        if len(named) >= 2:
            rows = metrics_table[metrics_table["Model"].isin(named)]
            # Order by R² descending
            rows = rows.sort_values("Bias-corrected test R2", ascending=False)
        else:
            rows = metrics_table.sort_values(
                "Bias-corrected test R2", ascending=False
            ).head(2)

        for _, row in rows.iterrows():
            try:
                r2 = float(row["Bias-corrected test R2"])
                rmse = float(row["Bias-corrected test RMSE"])
                mae = float(row["Bias-corrected test MAE"])
                facts.append(
                    f"**{row['Model']}** on **{season}**: "
                    f"R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f} mbgl"
                )
            except (KeyError, TypeError, ValueError):
                facts.append(f"**{row['Model']}** on **{season}** — scored")

        if not rows.empty:
            facts.append(
                f"Winner on R² for **{season}**: "
                f"**{rows.iloc[0]['Model']}**."
            )
        return facts[:8]

    # ---- Case 0b: the separate deep-learning / physics-informed tab ----
    if _mentions_deep_family(q):
        detected_deep_model = _detect_deep_model(q)
        if detected_season:
            deep_metrics = results[detected_season].get("deep_metrics")
            if deep_metrics is None or deep_metrics.empty:
                facts.append(
                    f"The deep-learning tab has no results for **{detected_season}** — "
                    "either the sidebar toggle was off for this run, or every model in "
                    "that family failed to fit."
                )
                return facts
            rows = deep_metrics
            if detected_deep_model:
                matched = rows[rows["Model"] == detected_deep_model]
                if not matched.empty:
                    rows = matched
            for _, row in rows.iterrows():
                try:
                    r2 = float(row["Bias-corrected test R2"])
                    rmse = float(row["Bias-corrected test RMSE"])
                    facts.append(
                        f"**{row['Model']}** on **{detected_season}** (deep-learning tab): "
                        f"R² = {r2:.3f}, RMSE = {rmse:.3f} mbgl"
                    )
                except (KeyError, TypeError, ValueError):
                    facts.append(f"**{row['Model']}** on **{detected_season}** — scored")
            best_deep = results[detected_season].get("deep_best_model")
            if best_deep and not detected_deep_model:
                facts.append(f"Best in that tab for **{detected_season}**: **{best_deep}**.")
            return facts
        # No season named — one line per season.
        for season, res in results.items():
            deep_metrics = res.get("deep_metrics")
            if deep_metrics is None or deep_metrics.empty:
                continue
            if detected_deep_model:
                matched = deep_metrics[deep_metrics["Model"] == detected_deep_model]
                if matched.empty:
                    continue
                row = matched.iloc[0]
                try:
                    r2 = float(row["Bias-corrected test R2"])
                    facts.append(f"{season}: **{detected_deep_model}** scored R² = {r2:.3f}")
                except (KeyError, TypeError, ValueError):
                    facts.append(f"{season}: **{detected_deep_model}** scored")
            else:
                top_row = deep_metrics.iloc[0]
                try:
                    r2 = float(top_row["Bias-corrected test R2"])
                    facts.append(
                        f"{season}: best in the deep-learning tab is "
                        f"**{res.get('deep_best_model')}** (R² = {r2:.3f})"
                    )
                except (KeyError, TypeError, ValueError):
                    facts.append(f"{season}: best in the deep-learning tab is "
                                f"**{res.get('deep_best_model')}**")
        if not facts:
            facts.append(
                "No deep-learning results are available in this session — the sidebar "
                "toggle for PINN / reservoir computing / SARIMA / LSTM may have been off."
            )
        return facts[:8]

    # ---- Case 1: specific season named ----
    if detected_season:
        top = results[detected_season]["metrics"].iloc[0]
        winner = results[detected_season]["best_model"]
        if detected_metric:
            alias, column = detected_metric
            try:
                value = float(top.get(column))
                facts.append(
                    f"for **{detected_season}**, the winner **{winner}** scored "
                    f"{alias.upper()} = {value:.3f}"
                )
            except (TypeError, ValueError):
                facts.append(f"for **{detected_season}**, the winner is **{winner}**")
        else:
            try:
                r2 = float(top["Bias-corrected test R2"])
                rmse = float(top["Bias-corrected test RMSE"])
                facts.append(
                    f"for **{detected_season}**, **{winner}** leads with "
                    f"R² = {r2:.3f} and RMSE = {rmse:.3f} mbgl"
                )
            except (KeyError, TypeError, ValueError):
                facts.append(f"for **{detected_season}**, the winner is **{winner}**")
        return facts

    # ---- Case 2: specific model named ----
    if detected_model:
        model_rows = []
        for season, res in results.items():
            matches = res["metrics"][
                res["metrics"]["Model"].astype(str) == detected_model
            ]
            if matches.empty:
                continue
            row = matches.iloc[0]
            try:
                r2 = float(row["Bias-corrected test R2"])
                rmse = float(row["Bias-corrected test RMSE"])
                model_rows.append(
                    f"**{season}** — R² = {r2:.3f}, RMSE = {rmse:.3f} mbgl"
                )
            except (KeyError, TypeError, ValueError):
                model_rows.append(f"**{season}** — scored")
        if model_rows:
            facts.append(f"{detected_model} across seasons:")
            facts.extend(f"  - {line}" for line in model_rows)
        return facts

    # ---- Case 3: "best"/"winner"/"top" without a specific season ----
    if any(w in q for w in ["best", "winner", "won", "top", "leads", "leading"]):
        for season, res in results.items():
            top = res["metrics"].iloc[0]
            try:
                r2 = float(top["Bias-corrected test R2"])
                facts.append(f"{season}: **{res['best_model']}** (R² = {r2:.3f})")
            except (KeyError, TypeError, ValueError):
                facts.append(f"{season}: **{res['best_model']}**")
        return facts[:6]

    return facts


# --------------------------------------------------------------------------- #
# Offline answer — composer
# --------------------------------------------------------------------------- #
_LIVE_LEAD_INS = [
    "Here's what this run shows:",
    "Pulling from your current results:",
    "From this session's numbers:",
    "Here's how that breaks down:",
]


def _offline_answer(question: str, hits: list[dict], job_context: str,
                    payload: dict | None) -> str:
    """Compose an answer from KB background + live numbers, not a copy-paste."""
    opener = _pick(_OPENERS, question)
    closer = _pick(_CLOSERS, question + "::closer")

    live = _extract_live_numbers(question, payload)

    # Live numbers from the current run are already a complete, specific
    # answer — layering generic methodology text on top of them is exactly
    # the "point by point" over-explaining this assistant should avoid.
    # Background only comes in when there's nothing live to say.
    use_background = bool(hits) and not live

    background = ""
    if use_background:
        raw = hits[0]["text"].strip()
        heading = hits[0].get("heading") or ""
        if heading and raw.startswith(f"{heading}\n\n"):
            raw = raw[len(heading) + 2:].strip()
        raw = re.sub(r"^Q:.*?\n", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^A:\s*", "", raw).strip()
        # Strip leftover "Conversation guide" wrappers.
        raw = re.sub(r"^\s*Conversation guide\s*\n+", "", raw).strip()
        if re.search(r"(?m)^\s*1\.\s", raw):
            background = raw
            if len(background) > 650:
                # Trim to whole list items instead of an arbitrary character
                # cut, so we never end on a dangling "3." with nothing after it.
                lines, kept, total = background.split("\n"), [], 0
                for line in lines:
                    if total + len(line) > 650 and kept:
                        break
                    kept.append(line)
                    total += len(line) + 1
                background = "\n".join(kept).rstrip() + "  …"
        else:
            sentences = re.split(r"(?<=[.!?])\s+", raw)
            background = " ".join(sentences[:3]).strip()
            if len(background) > 650:
                background = background[:650].rsplit(" ", 1)[0] + "…"

    parts: list[str] = []
    if live:
        if len(live) == 1 and not live[0].startswith("  - "):
            # A single fact reads better as one sentence than a list of one.
            fact = live[0]
            parts.append(fact[0].upper() + fact[1:] if fact else fact)
        else:
            parts.append(_pick(_LIVE_LEAD_INS, question + "::live"))
            for fact in live:
                parts.append(fact if fact.startswith("  - ") else f"- {fact}")
    if background:
        parts.append(background if live else f"{opener}\n\n{background}")

    if not parts:
        return _NO_ANSWER

    if len(hits) > 1 and use_background:
        related = ", ".join(h["title"] for h in hits[1:3])
        parts.append(f"_Related: {related}._")

    parts.append(closer)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Online answer — provider-agnostic (Gemini by default)
# --------------------------------------------------------------------------- #
def _online_answer(question: str, hits: list[dict], job_context: str,
                   history: list[dict] | None,
                   payload: dict | None) -> str:
    """Try the configured LLM. Fall back to offline composer on any issue."""
    if active_provider() == "none":
        return _offline_answer(question, hits, job_context, payload)

    knowledge = "\n\n---\n\n".join(
        f"[{h['title']}]\n{h['text']}" for h in hits
    ) or "(no relevant knowledge-base chunks found)"

    transcript = _format_history(history)

    user_message = (
        f"CONTEXT (current analysis run — read numbers from here):\n"
        f"{job_context}\n\n"
        f"KNOWLEDGE (methodology excerpts, cite freely but do not name the source):\n"
        f"{knowledge}\n\n"
        + (f"RECENT CONVERSATION:\n{transcript}\n\n" if transcript else "")
        + f"USER QUESTION:\n{question}"
    )

    result = complete(system=_SYSTEM_PROMPT, user=user_message, max_tokens=700)

    if result is None:
        return _offline_answer(question, hits, job_context, payload)

    if result.startswith("__ERROR__"):
        error_name = result.replace("__ERROR__", "", 1)
        fallback = _offline_answer(question, hits, job_context, payload)
        return (
            f"_(LLM call failed: {error_name}; answering offline.)_\n\n"
            f"{fallback}"
        )

    return result.strip()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def answer(question: str,
           payload: dict[str, Any] | None,
           history: list[dict] | None = None,
           prefer_llm: bool = True) -> dict[str, Any]:
    """Main entry point.

    Returns
    -------
    {'answer': str, 'sources': [filenames], 'provider': str}
        where provider is 'gemini', 'anthropic', or 'offline'.
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "Please type a question.", "sources": [], "provider": "offline"}

    # Fast path: greetings, thanks, help, goodbyes.
    canned = smalltalk_reply(question)
    if canned is not None:
        return {"answer": canned, "sources": [], "provider": "offline"}

    # Short confirmations ("yes", "ok") — don't dump a random chunk.
    if question.lower().strip(" .!?") in _SHORT_CONFIRMATIONS:
        return {
            "answer": (
                "Sure — which term should I explain?\n\n"
                "For example:\n"
                "- **R²**\n"
                "- **KGE**\n"
                "- **Taylor diagram**\n"
                "- **Cone of influence**\n"
                "- **Bias correction**\n\n"
                "Or say *\"summarise the run\"* for a per-season overview."
            ),
            "sources": [],
            "provider": "offline",
        }

    # Grounded path.
    hits = _kb().search(question, top_k=4)
    job_context = build_job_context(payload)

    if prefer_llm:
        text = _online_answer(question, hits, job_context, history, payload)
        provider = active_provider() if not text.startswith("_(") else "offline"
    else:
        text = _offline_answer(question, hits, job_context, payload)
        provider = "offline"

    return {
        "answer": text,
        "sources": [h["source"] for h in hits],
        "provider": provider,
    }