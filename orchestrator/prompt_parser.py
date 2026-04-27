"""
Prompt Parser  (Harshit's LLM layer)

Converts a free-text user prompt into a structured ParsedPrompt dict by calling
the local Ollama model.

BUGS FIXED vs original:
  1. Missing  "stream": False  → Ollama defaults to streaming; json() on a
     streaming response raises JSONDecodeError.
  2. System prompt said only "Return ONLY JSON" — model had no idea what fields
     to produce.  Now includes explicit schema + example.
  3. extract_json used match.group() without checking if match is None first.
  4. Response key was r.json()["message"]["content"] but the Ollama /api/chat
     endpoint nests inside "message" only when stream=False; guard added.
  5. Bare `except:` swallowed all errors silently.  Now logs and re-raises so
     the retry logic in workflow.py can count the failure.
"""
import json
import re
import logging
import requests

from shared.config import settings

logger = logging.getLogger(__name__)

# ── System prompt with explicit JSON schema ───────────────────────────────────

_SYSTEM_PROMPT = """You are PRANAG-AI's scientific prompt parser.

Your ONLY job is to convert the user's natural language prompt into a single
JSON object.  Output NOTHING except the JSON — no explanation, no markdown
fences, no extra text.

The JSON must follow this exact schema:
{
  "crop":              string,               // crop name, e.g. "wheat"
  "location":          string,               // place name, e.g. "Jodhpur, Rajasthan"
  "temperature":       number | null,        // target temperature in °C
  "humidity":          number | null,        // relative humidity in %
  "rainfall":          number | null,        // annual rainfall in mm
  "soil_type":         string | null,        // e.g. "sandy loam"
  "stress_conditions": [string],             // e.g. ["heat stress", "drought"]
  "target_traits":     [string],             // desired traits, e.g. ["heat tolerance"]
  "constraints":       { string: any }       // any other key-value constraints
}

EXAMPLE
Input:  "I want wheat that can grow in Jodhpur at 48°C with low rainfall"
Output:
{
  "crop": "wheat",
  "location": "Jodhpur, Rajasthan",
  "temperature": 48.0,
  "humidity": null,
  "rainfall": 300,
  "soil_type": "sandy loam",
  "stress_conditions": ["extreme heat stress", "drought stress", "low rainfall"],
  "target_traits": ["heat tolerance", "drought resistance", "deep root system"],
  "constraints": {}
}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Robustly extract the first JSON object from *text*.
    Handles models that wrap output in markdown code fences.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to regex extraction
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text[:300]}")
    return json.loads(match.group(0))


_FALLBACK: dict = {
    "crop": "unknown",
    "location": "unknown",
    "temperature": None,
    "humidity": None,
    "rainfall": None,
    "soil_type": None,
    "stress_conditions": [],
    "target_traits": [],
    "constraints": {},
}

# ── Main function ─────────────────────────────────────────────────────────────

def parse_prompt(user_prompt: str) -> dict:
    """
    Call the Ollama model and return a validated ParsedPrompt dict.
    Raises RuntimeError if the model is unreachable or returns unparseable output.
    """
    payload = {
        "model": settings.ollama_model,
        "stream": False,                       # ← BUG FIX: must be False
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Parse this prompt into the JSON schema above:\n\n{user_prompt}"
                ),
            },
        ],
    }

    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=settings.ollama_timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.ollama_base_url}. "
            "Is the server running?  `ollama serve`"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

    body = resp.json()

    # /api/chat wraps the reply in body["message"]["content"] when stream=False
    raw_text: str = body.get("message", {}).get("content", "")
    if not raw_text:
        raise RuntimeError(f"Empty content from Ollama.  Full response: {body}")

    try:
        parsed = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM output could not be parsed as JSON: {exc}") from exc

    # Merge with fallback to ensure all keys exist
    result = {**_FALLBACK, **parsed}
    logger.info("[prompt_parser] Parsed: %s", result)
    return result
