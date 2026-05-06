# TurboMind

TurboMind is a small medical Q&A prototype that combines:

- A FastAPI backend with retrieval-augmented generation (RAG)
- A Streamlit frontend
- A "TurboQuant" compression simulation to shrink retrieved context
- A simple short-term/long-term memory buffer
- Pluggable LLM backends (Gemini or a local OpenAI-compatible server)

This project is meant for experimentation and learning, not production medical use.

## Architecture

High-level flow for a `/chat` request:

1. User question arrives at the FastAPI `/chat` endpoint.
2. The RAG pipeline retrieves top documents from a Chroma vector store.
3. Retrieved context is "compressed" using a quantization-inspired method.
4. The memory manager adds the new message, and provides conversation context.
5. A final prompt is assembled and sent to the LLM.
6. The response + metrics are returned to the frontend.

Key modules:

- `backend/main.py`: FastAPI app, main `/chat` endpoint, metrics.
- `backend/rag.py`: Loads Kaggle medical text, builds embeddings, stores in Chroma, retrieves top docs.
- `backend/compression.py`: Memory compression and TurboQuant-like context compression.
- `backend/memory.py`: Short-term + long-term conversation buffer.
- `backend/llm_client.py`: Router for Gemini vs local LLM.
- `backend/gemini_client.py`: Gemini model selection and request logic.
- `backend/local_llm_client.py`: OpenAI-compatible local model client.
- `frontend/app.py`: Streamlit UI.

## Project Layout

- `backend/` FastAPI backend and core logic
- `frontend/` Streamlit app
- `requirements.txt` Python dependencies

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root (or export environment variables).

Example:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
USE_LOCAL_LLM=0
LOCAL_LLM_URL=http://127.0.0.1:1234/v1/chat/completions
LOCAL_LLM_MODEL=auto
LOCAL_LLM_TIMEOUT=120
```

## Running the Backend

From the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

## Running the Frontend

In a second terminal:

```bash
streamlit run frontend/app.py
```

The UI will open in your browser and call the backend at `http://127.0.0.1:8000/chat`.

## API

### `POST /chat`

Query parameters:

- `query` (string, required): user question
- `bits` (int, optional, default 4): compression level

Response JSON:

- `response`: model output
- `memory_usage`: RAM used by the process
- `latency`: end-to-end request time in seconds
- `context_length`: characters in final prompt
- `original_length`: original RAG context size
- `compressed_length`: compressed context size
- `compression_ratio`: compressed / original ratio
- `bits`: compression bits used

Example:

```bash
curl -X POST "http://127.0.0.1:8000/chat?query=What%20is%20asthma%3F&bits=4"
```

## RAG Dataset

TurboMind downloads a Kaggle dataset at runtime:

- `chaitanyakck/medical-text`

The dataset is fetched using `kagglehub` inside `backend/rag.py`. On first run, the download can take a while.

Notes:

- The current implementation limits indexing to the first 500 chunks for speed.
- Chunks are simple fixed-length slices (default 300 chars).

## TurboQuant Compression

`backend/compression.py` simulates quantization-based compression:

- Text is embedded using `all-MiniLM-L6-v2`.
- Embedding values are quantized into `2 ** bits` levels.
- A compression factor of `bits / 8` is applied to the original text length.

This is a simulation (not actual semantic compression).

## Memory Manager

`backend/memory.py` stores:

- `short_term`: last 5 messages
- `long_term`: older messages trimmed to 200 chars

The final prompt merges long-term and short-term memory with the current query.

## LLM Backends

### Gemini (default)

Uses `google-generativeai`. Model selection:

- Preferred: `GEMINI_MODEL` (default `gemini-1.5-flash`)
- If model listing fails, it uses the preferred name
- If listing succeeds, it tries to match the preferred name, otherwise picks a `flash` model if available

### Local LLM

Set `USE_LOCAL_LLM=1` to route requests through a local OpenAI-compatible server:

- Default URL: `http://127.0.0.1:1234/v1/chat/completions`
- The client attempts to resolve a model id via `/v1/models` if `LOCAL_LLM_MODEL=auto`.

## Troubleshooting

- If the RAG dataset download fails, ensure you are logged in for KaggleHub or have access configured.
- If Gemini requests fail, check `GEMINI_API_KEY`.
- If using a local server, confirm it exposes `/v1/chat/completions` and `/v1/models`.
- Streamlit frontend assumes the backend runs on `127.0.0.1:8000`.

## Limitations

- This is a prototype. RAG indexing is very small (first 500 chunks).
- Compression is a simulation and may remove important context.
- Medical responses are not safe for real-world use.

## Next Ideas

- Add persistent vector store (disk-backed Chroma).
- Improve chunking and filtering.
- Add citations or source tracing.
- Store conversation history per user/session.
