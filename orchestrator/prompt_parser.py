import requests
import json
from shared.config import settings

def parse_prompt(prompt):
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "user", "content": f"Convert to JSON: {prompt}"}
        ]
    }

    try:
        r = requests.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        text = r.json()["message"]["content"]
        return json.loads(text)
    except:
        return {
            "crop": "unknown",
            "location": "unknown",
            "temperature": 0,
            "stress": []
        }
