def build_spec(parsed, traits, research):
    return {
        "crop": parsed["crop"],
        "location": parsed["location"],
        "temperature": parsed.get("temperature", 0),
        "stress": parsed.get("stress", []),
        "traits": traits,
        "scientific_basis": [r["key_finding"] for r in research],
        "confidence": 0.9
    }
