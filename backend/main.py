from fastapi import FastAPI
import traceback
from memory import MemoryManager
from rag import retrieve, initialize_rag
from compression import turboquant_compress
from llm_client import generate_response
import psutil
import time

app = FastAPI()
memory = MemoryManager()

# Load dataset once
try:
    initialize_rag()
except Exception as e:
    print("RAG init error:", e)


@app.get("/")
def home():
    return {"message": "TurboMind API running 🚀"}


@app.post("/chat")
def chat(query: str, bits: int = 4):
    try:
        start = time.time()

        # 🔍 RAG
        docs = retrieve(query)
        context = " ".join(docs)

        # 🔥 BEFORE
        original_length = len(context)

        # 🔥 TurboQuant compression
        compressed_context = turboquant_compress(context, bits=bits)
        compressed_length = len(compressed_context)

        compression_ratio = (
            round(compressed_length / original_length, 2)
            if original_length > 0 else 1
        )

        # 🧠 Memory
        memory.add_message(query)
        memory_context = memory.get_context()

        # Prompt
        final_prompt = f"""
        You are a medical AI assistant.
        Answer clearly and simply.

        Context: {compressed_context}
        Conversation: {memory_context}
        Question: {query}

        ⚠️ This is not medical advice.
        """

        # 🤖 LLM
        response = generate_response(final_prompt)

        latency = time.time() - start

        return {
            "response": response,
            "memory_usage": psutil.virtual_memory().used,
            "latency": round(latency, 3),
            "context_length": len(final_prompt),

            # 🔥 TurboQuant Metrics
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": compression_ratio,
            "bits": bits
        }
    except Exception as e:
        print("Chat error:", e)
        traceback.print_exc()
        return {
            "response": f"Error: {str(e)}",
            "memory_usage": psutil.virtual_memory().used,
            "latency": 0,
            "context_length": 0,
            "original_length": 0,
            "compressed_length": 0,
            "compression_ratio": 1,
            "bits": bits
        }
