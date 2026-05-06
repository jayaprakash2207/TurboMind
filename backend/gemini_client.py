import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_model = None

def _resolve_model_name(preferred: str) -> str:
    try:
        models = genai.list_models()
    except Exception:
        # If listing fails, fall back to preferred name.
        return preferred

    candidates = []
    for m in models:
        if "generateContent" in getattr(m, "supported_generation_methods", []):
            name = getattr(m, "name", "")
            if name.startswith("models/"):
                name = name[len("models/"):]
            if name:
                candidates.append(name)

    if not candidates:
        return preferred

    # Normalize preferred name
    pref = preferred
    if pref.startswith("models/"):
        pref = pref[len("models/"):]

    if pref in candidates:
        return pref

    # Heuristic: prefer a flash model if available, else first candidate
    for name in candidates:
        if "flash" in name:
            return name
    return candidates[0]


def _get_model():
    global _model
    if _model is None:
        preferred = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        resolved = _resolve_model_name(preferred)
        _model = genai.GenerativeModel(resolved)
    return _model


def generate_response(prompt: str):
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
