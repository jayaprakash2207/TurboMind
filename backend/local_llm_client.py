import os
from urllib.parse import urlparse
import requests


def _models_url_from_chat_url(chat_url: str) -> str:
    # Replace /v1/chat/completions (or /v1/responses) with /v1/models
    parsed = urlparse(chat_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return f"{base}/v1/models"


def _resolve_model_id(chat_url: str, configured: str, timeout: float) -> str:
    if configured and configured not in {"local-model", "auto"}:
        return configured
    try:
        resp = requests.get(_models_url_from_chat_url(chat_url), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if models:
            return models[0].get("id", configured or "local-model")
    except Exception:
        pass
    return configured or "local-model"


def generate_response(prompt: str) -> str:
    # OpenAI-compatible local server endpoint (llama.cpp, LM Studio, etc.)
    url = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1/chat/completions")
    configured_model = os.getenv("LOCAL_LLM_MODEL", "auto")
    timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "120"))
    model = _resolve_model_id(url, configured_model, timeout)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a medical AI assistant. Answer clearly and simply."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"
