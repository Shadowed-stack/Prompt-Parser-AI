"""
Prompt Parser — Layer 1 (domain-aware)

Takes: user prompt + active domain weights (from domain_router)
Does:  ONE LLM call per active domain → extracts domain-specific parameters
Returns: { "domain_weights": {...}, "domain_parameters": { domain: {params} } }

Changes over previous version:
  • Added _regex_fallback() — when LLM fails OR returns empty dict,
    extracts crop, location, temperature, rainfall directly from the prompt
    using regex + keyword lists. No LLM needed.
  • Added parameters to domain.txt hint for biology domain.
  • parse_prompt() now always returns non-empty domain_parameters
    even when OpenRouter is rate-limiting or down.
"""
import json
import re
import logging
import requests
from pathlib import Path
import configparser
from shared.config import settings

logger = logging.getLogger(__name__)

DOMAIN_FILE = Path(__file__).parent.parent / "domain.txt"

# ── Known crop names for regex fallback ──────────────────────────────────────
_CROPS = [
    "wheat", "rice", "maize", "corn", "sugarcane", "cotton", "bajra",
    "pearl millet", "mustard", "soybean", "sorghum", "barley", "chickpea",
    "lentil", "groundnut", "peanut", "tomato", "potato", "onion", "mango",
    "banana", "tea", "coffee", "jute", "sunflower", "canola",
]

# ── Indian cities + common location words ─────────────────────────────────────
_LOCATIONS = [
    "jamshedpur", "jodhpur", "delhi", "mumbai", "bangalore", "bengaluru",
    "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur",
    "lucknow", "kanpur", "nagpur", "patna", "bhopal", "indore", "vadodara",
    "rajasthan", "uttar pradesh", "punjab", "haryana", "bihar", "maharashtra",
    "gujarat", "madhya pradesh", "andhra pradesh", "telangana", "karnataka",
    "tamil nadu", "west bengal", "odisha", "assam", "kerala",
    "north india", "south india", "east india", "west india",
    "gangetic plains", "deccan plateau", "coastal", "arid", "semi-arid",
]


def _fallback_rainfall_value(prompt: str):
    """Convert rainfall wording into a numeric estimate in mm."""
    rain_mm = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:of\s+)?rain", prompt.lower())
    if rain_mm:
        return float(rain_mm.group(1))
    if "low rainfall" in prompt.lower() or "less rainfall" in prompt.lower() or "drought" in prompt.lower():
        return 300.0
    if "high rainfall" in prompt.lower() or "heavy rain" in prompt.lower() or "flood" in prompt.lower():
        return 1200.0
    return None


def _regex_fallback(prompt: str) -> dict:
    """
    Extract key parameters from the prompt using regex + keyword lists.
    Used when LLM fails (rate limit, timeout) or returns empty.

    Returns a flat dict with whatever could be extracted.
    """
    p = prompt.lower()
    result = {}

    # Crop
    for crop in _CROPS:
        if crop in p:
            result["crop"] = crop
            break

    # Location
    for loc in _LOCATIONS:
        if loc in p:
            result["location"] = loc.title()
            break

    # Temperature — look for patterns like "45°C", "45 degrees", "45C"
    temp_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:°|degrees?|deg)?\s*[cC](?:\b|elsius)", prompt
    )
    if not temp_match:
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*°[cC]", prompt)
    if temp_match:
        result["temperature"] = float(temp_match.group(1))

    # Rainfall — "low rainfall", "high rainfall", "500mm rainfall"
    rainfall = _fallback_rainfall_value(prompt)
    if rainfall is not None:
        result["rainfall"] = rainfall

    # Stress conditions
    stresses = []
    stress_map = {
        "heat":      "heat stress",
        "drought":   "drought stress",
        "flood":     "flood tolerance",
        "waterlog":  "waterlogging tolerance",
        "salinity":  "salinity tolerance",
        "frost":     "frost tolerance",
        "cold":      "cold stress",
        "pest":      "pest resistance",
        "disease":   "disease resistance",
    }
    for kw, label in stress_map.items():
        if kw in p:
            stresses.append(label)
    if stresses:
        result["stress_conditions"] = stresses

    if result:
        logger.info("[prompt_parser] Regex fallback extracted: %s", result)
    else:
        logger.warning("[prompt_parser] Regex fallback found nothing for: '%s'", prompt[:60])

    return result


def _load_domain_parameters(domain_name: str) -> str:
    """Read the parameters section for a domain from domain.txt."""
    parser = configparser.ConfigParser()
    parser.read(DOMAIN_FILE)
    if parser.has_option(domain_name, "parameters"):
        return parser.get(domain_name, "parameters")

    # Built-in hints for domains not yet in domain.txt
    _HINTS = {
        "biology":           "crop, location, temperature, humidity, rainfall, soil_type, stress_conditions, target_traits",
        "computer_science_ai": "crop, location, temperature, humidity, rainfall, soil_type, stress_conditions, target_traits, algorithm, model_type, dataset",
        "physics":           "temperature, pressure, energy, force, wavelength, radiation",
        "chemistry":         "compound, reaction, catalyst, concentration, pH, temperature",
        "earth_environment": "location, climate, CO2, temperature, rainfall, soil_type, ecosystem",
        "materials_science": "material, hardness, conductivity, tensile_strength, temperature",
        "medicine":          "disease, drug, dosage, patient_condition, treatment",
        "engineering":       "load, stress, voltage, current, fluid_flow, temperature",
        "mathematics":       "equation, theorem, variables, domain, constraints",
        "quant_finance":     "asset, risk, return, volatility, time_horizon",
        "astronomy_space":   "celestial_body, distance, mass, luminosity, orbital_period",
        "human_social":      "population, behavior, culture, language, cognitive_factors",
        "economics":         "GDP, inflation, market, policy, trade_volume",
    }
    return _HINTS.get(domain_name, "Extract all relevant scientific parameters mentioned in the prompt.")


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            return {}
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except:
                        return {}
        return {}


def _extract_for_domain(prompt: str, domain_name: str, weight: float) -> dict:
    """One LLM call: extract parameters for a single domain."""
    parameters_hint = _load_domain_parameters(domain_name)

    system = f"""You are PRANA-G AI's scientific parameter extractor.

Domain: {domain_name} (relevance weight: {weight})
Parameters to extract: {parameters_hint}

Rules:
- Extract ONLY values explicitly stated or clearly implied in the prompt.
- Use null for anything not mentioned.
- Output ONLY a flat JSON object with the extracted parameters.
- No explanation, no markdown, just JSON.
"""
    try:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": settings.openrouter_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Extract parameters from:\n\n{prompt}"},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 500,
        }
        resp = requests.post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=settings.llm_timeout,
        )
        resp.raise_for_status()
        raw    = resp.json()["choices"][0]["message"]["content"]
        result = _extract_json(raw)

        # If LLM returned empty, use regex fallback
        if not result:
            logger.warning(
                "[prompt_parser] LLM returned empty for domain %s — using regex fallback.",
                domain_name,
            )
            result = _regex_fallback(prompt)

        return result

    except Exception as exc:
        logger.warning(
            "[prompt_parser] Domain %s LLM failed (%s) — using regex fallback.",
            domain_name, exc,
        )
        return _regex_fallback(prompt)


def parse_prompt(user_prompt: str, domain_weights: dict) -> dict:
    """
    For each active domain, extract its parameters.
    Falls back to regex extraction if LLM fails or returns empty.
    Always returns non-empty domain_parameters for agriculture prompts.
    """
    domain_parameters = {}

    for domain, weight in domain_weights.items():
        logger.info("[prompt_parser] Extracting for domain: %s (weight=%.2f)", domain, weight)
        params = _extract_for_domain(user_prompt, domain, weight)
        if params:
            domain_parameters[domain] = params

    # If everything failed (no domain weights or all LLM calls failed),
    # run regex fallback and put results under "biology" as best guess
    if not domain_parameters:
        logger.warning("[prompt_parser] All extractions failed — running standalone regex fallback.")
        fallback = _regex_fallback(user_prompt)
        if fallback:
            domain_parameters["biology"] = fallback

    return {
        "domain_weights":    domain_weights,
        "domain_parameters": domain_parameters,
    }