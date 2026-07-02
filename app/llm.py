"""Simple Gemini LLM wrapper.

This module uses the supported `google.genai` package to call Gemini.
Set `GEN_API_KEY` environment variable to your Google Cloud API key.
"""
import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # loads .env if present


def _get_api_key() -> str | None:
    load_dotenv()
    return os.getenv("GEN_API_KEY")


def _get_client():
    api_key = _get_api_key()
    if genai is None or not api_key:
        return None
    return genai.Client(api_key=api_key)


def _extract_generated_text(response) -> str | None:
    if response is None:
        return None
    text = getattr(response, "text", None)
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if content is None:
            return None
        content_text = getattr(content, "text", None)
        if content_text:
            return content_text
        parts = getattr(content, "parts", None) or []
        if isinstance(parts, list) and parts:
            return "".join(getattr(part, "text", "") or "" for part in parts)
    return None


def generate_response(prompt: str, context: List[Dict] = None, model: str = "gemini-2.5-flash-lite") -> Dict:
    """Call Gemini with a prompt and optional context. Returns dict with 'text' and 'metadata'."""
    client = _get_client()
    if client is None:
        return {"text": None, "error": "google-genai package not available or API key missing", "source": "fallback"}
    if not _get_api_key():
        return {"text": None, "error": "GEN_API_KEY is not configured", "source": "fallback"}
    contents = [prompt]
    if context:
        context_text = "\n\n".join([c.get("text", "") for c in context])
        contents = [context_text, prompt]

    fallback_models = [model]
    if model != "gemini-2.5-flash-lite":
        fallback_models.append("gemini-2.5-flash-lite")
    if model != "gemini-2.5-flash":
        fallback_models.append("gemini-2.5-flash")

    max_retries = 2
    delay = 1.0
    for current_model in fallback_models:
        for attempt in range(max_retries + 1):
            try:
                resp = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.1),
                )
                text = _extract_generated_text(resp)
                return {"text": text, "raw": resp, "error": None, "source": current_model}
            except Exception as exc:
                error_text = str(exc)
                upper_error = error_text.upper()
                if attempt < max_retries and "UNAVAILABLE" in upper_error:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if current_model != fallback_models[-1] and ("RESOURCE_EXHAUSTED" in upper_error or "UNAVAILABLE" in upper_error):
                    break
                return {"text": None, "error": error_text, "source": "fallback"}
    return {"text": None, "error": error_text, "source": "fallback"}
