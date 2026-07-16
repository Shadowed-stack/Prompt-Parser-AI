"""
Spec Builder — multi-domain version

Changes over previous version:
  • Extracts crop, location, temperature, humidity, rainfall, soil_type,
    stress_conditions from domain_parameters so they appear as top-level
    fields in the spec (previously always null).
  • Looks across ALL active domains for these fields — whichever domain
    the LLM put them in (biology, computer_science_ai, physics, etc.)
  • All other fields unchanged.
"""
import logging
from datetime import datetime, timezone
from shared.config import settings

logger = logging.getLogger(__name__)

# Field name variants the LLM might use for each canonical field
# (LLM sometimes uses "crop_type" instead of "crop", "city" instead of "location" etc.)
_CROP_KEYS        = {"crop", "crop_type", "crop_name", "plant", "variety"}
_LOCATION_KEYS    = {"location", "target_location", "city", "region", "place", "area", "site"}
_TEMPERATURE_KEYS = {"temperature", "temp", "temperature_c", "max_temp", "heat_tolerance"}
_HUMIDITY_KEYS    = {"humidity", "humidity_percent", "relative_humidity"}
_RAINFALL_KEYS    = {"rainfall", "rainfall_mm", "precipitation", "water_requirement", "rainfall_requirement"}
_SOIL_KEYS        = {"soil_type", "soil", "soil_condition", "soil_preference"}
_STRESS_KEYS      = {"stress_conditions", "stress", "stresses", "environmental_stress", "abiotic_stress"}
_TRAIT_KEYS       = {"target_traits", "traits", "desired_traits", "key_traits"}


def _find(params: dict, keys: set):
    """Return first non-null value whose key matches any name in keys."""
    for k, v in params.items():
        if k.lower() in keys and v is not None:
            return v
    return None


def _coerce_rainfall(value):
    """Normalize rainfall values to a numeric estimate in mm when possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        try:
            return float(lowered)
        except ValueError:
            if lowered in {"low", "low rainfall", "drought"}:
                return 300.0
            if lowered in {"high", "high rainfall", "heavy rain", "flood"}:
                return 1200.0
    return None


def _extract_agriculture_fields(domain_parameters: dict) -> dict:
    """
    Scan all domain_parameters dicts and extract agriculture/common fields.
    Returns a flat dict of {field: value} for any field found.
    """
    fields = {
        "crop":              None,
        "location":          None,
        "temperature":       None,
        "humidity":          None,
        "rainfall":          None,
        "soil_type":         None,
        "stress_conditions": [],
        "target_traits":     [],
    }

    for domain, params in domain_parameters.items():
        if not isinstance(params, dict):
            continue

        # Lowercase all keys for matching
        lc = {k.lower(): v for k, v in params.items()}

        if fields["crop"]        is None: fields["crop"]        = _find(lc, _CROP_KEYS)
        if fields["location"]    is None: fields["location"]    = _find(lc, _LOCATION_KEYS)
        if fields["temperature"] is None:
            t = _find(lc, _TEMPERATURE_KEYS)
            if t is not None:
                try:    fields["temperature"] = float(t)
                except: pass
        if fields["humidity"]    is None: fields["humidity"]    = _find(lc, _HUMIDITY_KEYS)
        if fields["rainfall"]    is None: fields["rainfall"]    = _coerce_rainfall(_find(lc, _RAINFALL_KEYS) or _find(lc, {"rainfall_condition"}))
        if fields["soil_type"]   is None: fields["soil_type"]   = _find(lc, _SOIL_KEYS)

        # stress_conditions and target_traits — collect from all domains
        stress = _find(lc, _STRESS_KEYS)
        if stress:
            if isinstance(stress, list):
                fields["stress_conditions"].extend(stress)
            elif isinstance(stress, str):
                fields["stress_conditions"].append(stress)

        traits = _find(lc, _TRAIT_KEYS)
        if traits:
            if isinstance(traits, list):
                fields["target_traits"].extend(traits)
            elif isinstance(traits, str):
                fields["target_traits"].append(traits)

    # Clean up strings
    if isinstance(fields["crop"],     str): fields["crop"]     = fields["crop"].strip()
    if isinstance(fields["location"], str): fields["location"] = fields["location"].strip()

    return fields


def build_spec(parsed: dict, traits: list, research: list) -> dict:
    domain_weights    = parsed.get("domain_weights", {})
    domain_parameters = parsed.get("domain_parameters", {})

    # Confidence: based on how many domains returned parameters
    active     = len(domain_weights)
    filled     = sum(1 for p in domain_parameters.values() if p)
    confidence = round(
        min(0.5 + (filled / max(active, 1)) * 0.4 + min(len(research) * 0.02, 0.10), 1.0), 2
    ) if active else 0.3

    # Extract top-level fields from wherever the LLM put them
    ag = _extract_agriculture_fields(domain_parameters)

    spec = {
        # ── Multi-domain fields ───────────────────────────────────────────────
        "domain_weights":    domain_weights,
        "domain_parameters": domain_parameters,

        # ── Agriculture / common fields (extracted from domain_parameters) ────
        "crop":              ag["crop"],
        "location":          ag["location"],
        "temperature":       ag["temperature"],
        "humidity":          ag["humidity"],
        "rainfall":          ag["rainfall"],
        "soil_type":         ag["soil_type"],
        "stress_conditions": list(dict.fromkeys(ag["stress_conditions"])),  # dedup
        "target_traits":     list(dict.fromkeys(ag["target_traits"])),      # dedup
        "constraints":       {},

        # ── Search + research results ─────────────────────────────────────────
        "retrieved_traits":  traits,
        "scientific_basis":  [r.get("key_finding", "") for r in research],
        "research_titles":   [r.get("title",       "") for r in research],
        "research_sources":  [r.get("url",         "") for r in research],
        "research_years":    [r.get("year")            for r in research],

        # ── Metadata ─────────────────────────────────────────────────────────
        "confidence":        confidence,
        "pipeline_version":  settings.pipeline_version,
        "generated_at":      datetime.now(timezone.utc).isoformat(),
    }

    if ag["crop"] or ag["location"]:
        logger.info(
            "[spec_builder] Extracted: crop=%s, location=%s, temp=%s",
            ag["crop"], ag["location"], ag["temperature"]
        )
    else:
        logger.info("[spec_builder] No crop/location found — multi-domain non-agriculture spec.")

    return spec