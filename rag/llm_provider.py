"""rag/llm_provider.py — one place that talks to whichever LLM is configured.

Priority order (first key that is set wins):
    1. GEMINI_API_KEY    → Google Gemini   (default for this project)
    2. ANTHROPIC_API_KEY → Anthropic Claude
    3. (no key)          → returns None; caller falls back to offline mode

Usage:
    from .llm_provider import complete, active_provider
    text = complete(system=..., user=..., max_tokens=700)
"""
from __future__ import annotations

import os


# --------------------------------------------------------------------------- #
# Key discovery — checks env vars first, then Streamlit secrets if available.
# --------------------------------------------------------------------------- #
def _read_key(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip() or None
    try:
        import streamlit as st  # noqa: WPS433
        try:
            secret = st.secrets.get(name)  # type: ignore[attr-defined]
        except Exception:
            secret = None
        if secret:
            return str(secret).strip() or None
    except Exception:
        pass
    return None


def active_provider() -> str:
    """Return 'gemini', 'anthropic', or 'none'."""
    if _read_key("GEMINI_API_KEY"):
        return "gemini"
    if _read_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
_GEMINI_MODEL = "gemini-2.0-flash"


def _call_gemini(system: str, user: str, max_tokens: int) -> str | None:
    key = _read_key("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai  # pip install google-generativeai
    except ImportError:
        return "__ERROR__ImportError"

    genai.configure(api_key=key)

    model = genai.GenerativeModel(
        model_name=_GEMINI_MODEL,
        system_instruction=system,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": 0.4,
            "top_p": 0.9,
        },
    )
    try:
        response = model.generate_content(user)
        text = getattr(response, "text", "") or ""
        return text.strip() or None
    except Exception as exc:
        return f"__ERROR__{type(exc).__name__}"


# --------------------------------------------------------------------------- #
# Anthropic (kept in case both keys are present)
# --------------------------------------------------------------------------- #
_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"


def _call_anthropic(system: str, user: str, max_tokens: int) -> str | None:
    key = _read_key("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic  # pip install anthropic
    except ImportError:
        return "__ERROR__ImportError"

    client = anthropic.Anthropic(api_key=key)
    try:
        response = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()
        return text or None
    except Exception as exc:
        return f"__ERROR__{type(exc).__name__}"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def complete(system: str, user: str, max_tokens: int = 700) -> str | None:
    """Send a single-turn completion to whichever provider is configured.

    Returns the assistant's text, or None if no provider is configured,
    or a string starting with '__ERROR__' if the provider call raised.
    """
    provider = active_provider()
    if provider == "gemini":
        return _call_gemini(system, user, max_tokens)
    if provider == "anthropic":
        return _call_anthropic(system, user, max_tokens)
    return None