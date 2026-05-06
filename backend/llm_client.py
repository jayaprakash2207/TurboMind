import os
from gemini_client import generate_response as gemini_generate
from local_llm_client import generate_response as local_generate


def generate_response(prompt: str) -> str:
    use_local = os.getenv("USE_LOCAL_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
    if use_local:
        return local_generate(prompt)
    return gemini_generate(prompt)
