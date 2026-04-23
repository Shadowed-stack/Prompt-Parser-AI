import requests
import json
import re
from shared.config import settings

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group())

def parse_prompt(user_prompt: str):
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": "Return ONLY JSON"},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        r = requests.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        text = r.json()["message"]["content"]
        return extract_json(text)
    except:
        return {
            "crop": "unknown",
            "location": "unknown",
            "temperature": 0,
            "stress": []
        }
