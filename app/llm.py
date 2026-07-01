"""Simple Gemini LLM wrapper.

This module uses the `google-generativeai` package to call Gemini.
Set `GEN_API_KEY` environment variable to your Google Cloud API key.
"""
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


API_KEY = os.environ.get("GEN_API_KEY")
if genai and API_KEY:
    genai.configure(api_key=API_KEY)


def generate_response(prompt: str, context: List[Dict] = None, model: str = "gemini-1.0") -> Dict:
    """Call Gemini with a prompt and optional context. Returns dict with 'text' and 'metadata'."""
    if genai is None:
        raise RuntimeError("google-generativeai package not installed")
    messages = prompt
    if context:
        # simple concatenation; RAG pipeline should prepare the prompt
        context_text = "\n\n".join([c.get("text", "") for c in context])
        messages = context_text + "\n\n" + prompt

    resp = genai.chat.create(model=model, messages=[{"role": "user", "content": messages}])
    text = resp.last
    return {"text": text, "raw": resp}
