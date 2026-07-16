"""
Domain Router — Layer 0
ONE LLM call to:
  1. Decide which domains are relevant to the prompt
  2. Assign a weight (0.0-1.0) to each active domain
  3. Return only domains above a threshold (default 0.15)

Reads domain definitions from domain.txt at the project root.
"""
import configparser
import logging
import json
import requests
from pathlib import Path
from shared.config import settings

logger = logging.getLogger(__name__)

DOMAIN_FILE = Path(__file__).parent.parent / "domain.txt"
WEIGHT_THRESHOLD = 0.30  # domains below this are ignored


def load_domains() -> dict:
    """Read domain.txt and return a dict of domain_name → {description, keywords}."""
    parser = configparser.ConfigParser()
    parser.read(DOMAIN_FILE)
    domains = {}
    for section in parser.sections():
        domains[section] = {
            "description": parser.get(section, "description", fallback=""),
            "keywords":    parser.get(section, "keywords", fallback=""),
        }
    return domains


def _build_weighting_prompt(domains: dict) -> str:
    lines = ["Available domains:\n"]
    for name, info in domains.items():
        lines.append(f"- {name}: {info['description']}")
        lines.append(f"  Keywords: {info['keywords']}\n")
    return "\n".join(lines)


def score_domains(prompt: str) -> dict:
    """
    Returns dict of domain_name → weight (float 0.0-1.0)
    Only includes domains above WEIGHT_THRESHOLD.
    Falls back to keyword scoring if LLM fails.
    """
    domains = load_domains()

    system = f"""You are a scientific domain classifier for PRANA-G AI.

Given a user prompt, assign a relevance weight (0.0 to 1.0) to EACH domain below.
Weight rules:
- 0.0   = domain is completely irrelevant to the prompt
- 0.1-0.3 = slight relevance (mentioned briefly or implied)
- 0.4-0.6 = moderate relevance (one significant aspect)  
- 0.7-0.9 = high relevance (major focus of the prompt)
- 1.0   = the prompt is entirely about this domain

Weights do NOT need to sum to 1.0. A prompt can be 0.8 biology AND 0.7 chemistry.
Only assign weight > 0 if the domain is genuinely relevant.

{_build_weighting_prompt(domains)}

Output ONLY a JSON object mapping domain names to weights. Example:
{{"biology": 0.8, "chemistry": 0.6, "earth_environment": 0.3}}
No explanation. No extra text. Just the JSON."""

    try:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.openrouter_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Classify this prompt:\n\n{prompt}"},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
        }
        resp = requests.post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=settings.llm_timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        weights = json.loads(raw)

        # Filter to known domains + above threshold
        active = {
            k: round(float(v), 2)
            for k, v in weights.items()
            if k in domains and float(v) >= WEIGHT_THRESHOLD
        }
        logger.info("[domain_router] Active domains: %s", active)
        return active

    except Exception as exc:
        logger.warning("[domain_router] LLM failed (%s) — keyword fallback.", exc)
        return _keyword_fallback(prompt, domains)


def _keyword_fallback(prompt: str, domains: dict) -> dict:
    p = prompt.lower()
    scores = {}
    for name, info in domains.items():
        kws = [k.strip().lower() for k in info["keywords"].split(",")]
        hits = sum(1 for kw in kws if kw and kw in p)
        if hits > 0:
            scores[name] = round(min(hits * 0.15, 1.0), 2)
    if not scores:
        scores["general"] = 0.3   # safe default
    return scores