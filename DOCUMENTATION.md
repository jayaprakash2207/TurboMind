# TurboMind — Project Documentation

**Version:** 1.0  
**Type:** Research Prototype  
**Domain:** Medical Question-Answering (AI-Assisted)  
**Stack:** Python · FastAPI · Streamlit · ChromaDB · Google Gemini / Local LLM

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Goals and Objectives](#2-goals-and-objectives)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [Module Reference](#6-module-reference)
   - 6.1 [Backend — `main.py`](#61-backend--mainpy)
   - 6.2 [RAG Pipeline — `rag.py`](#62-rag-pipeline--ragpy)
   - 6.3 [Compression Engine — `compression.py`](#63-compression-engine--compressionpy)
   - 6.4 [Memory Manager — `memory.py`](#64-memory-manager--memorypy)
   - 6.5 [LLM Router — `llm_client.py`](#65-llm-router--llm_clientpy)
   - 6.6 [Gemini Client — `gemini_client.py`](#66-gemini-client--gemini_clientpy)
   - 6.7 [Local LLM Client — `local_llm_client.py`](#67-local-llm-client--local_llm_clientpy)
   - 6.8 [Frontend — `app.py`](#68-frontend--apppy)
7. [Request Lifecycle](#7-request-lifecycle)
8. [API Reference](#8-api-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Setup and Deployment Guide](#10-setup-and-deployment-guide)
11. [TurboQuant Compression — Deep Dive](#11-turboquant-compression--deep-dive)
12. [RAG Dataset and Indexing](#12-rag-dataset-and-indexing)
13. [Memory System Design](#13-memory-system-design)
14. [LLM Backend Options](#14-llm-backend-options)
15. [Frontend UI Guide](#15-frontend-ui-guide)
16. [Performance Metrics Explained](#16-performance-metrics-explained)
17. [Known Limitations](#17-known-limitations)
18. [Future Roadmap](#18-future-roadmap)
19. [Disclaimer](#19-disclaimer)

---

## 1. Project Overview

**TurboMind** is a medical question-answering (Q&A) prototype that demonstrates the integration of several advanced AI techniques into a single, cohesive system. It is designed as a research and learning platform to explore how large language models (LLMs) can be augmented with external knowledge, conversation memory, and quantization-inspired context compression.

The system allows a user to ask natural-language medical questions and receive AI-generated answers enriched by relevant medical documents retrieved at query time.

Key highlights:
- **Retrieval-Augmented Generation (RAG):** Answers are grounded in a real medical text corpus from Kaggle, not purely from the LLM's pre-trained knowledge.
- **TurboQuant Compression:** A novel simulation of quantization-based context compression that reduces the size of retrieved context before it is fed to the LLM.
- **Dual-Memory Architecture:** A short-term + long-term memory buffer maintains conversation continuity across multiple turns.
- **Pluggable LLM Backends:** Supports Google Gemini (cloud) and any OpenAI-compatible local server (e.g., LM Studio, llama.cpp).

---

## 2. Goals and Objectives

| Goal | Description |
|------|-------------|
| **Knowledge Grounding** | Reduce LLM hallucination by retrieving relevant medical text and providing it as context. |
| **Context Efficiency** | Compress retrieved context to reduce token consumption and latency via TurboQuant simulation. |
| **Conversation Continuity** | Maintain a memory of prior questions to allow follow-up queries without losing context. |
| **LLM Flexibility** | Allow the same application to run with either a cloud-hosted Gemini model or a locally-running open-source LLM. |
| **Observability** | Return rich performance metrics (latency, memory usage, compression ratio) alongside every response. |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User (Browser)                        │
│                   Streamlit Frontend UI                       │
│           (compression slider + question input box)          │
└────────────────────────┬────────────────────────────────────┘
                         │  HTTP POST /chat?query=...&bits=4
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Port 8000)                 │
│                                                               │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌────────┐  │
│  │  RAG     │──▶│ TurboQuant │──▶│ Memory   │──▶│  LLM   │  │
│  │ Retrieve │   │ Compress   │   │ Manager  │   │ Router │  │
│  └──────────┘   └────────────┘   └──────────┘   └────────┘  │
│        │                                              │       │
│        ▼                                              ▼       │
│  ┌──────────────────────┐               ┌────────────────┐   │
│  │  ChromaDB Vector DB  │               │ Gemini / Local │   │
│  │  (in-memory)         │               │ LLM API        │   │
│  └──────────────────────┘               └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Component roles:**

| Component | Role |
|-----------|------|
| **Streamlit Frontend** | Collects user input; displays LLM response and performance metrics. |
| **FastAPI Backend** | Orchestrates all pipeline steps; exposes the `/chat` REST endpoint. |
| **RAG Module** | Downloads and indexes a Kaggle medical corpus; performs semantic retrieval via ChromaDB. |
| **TurboQuant Compression** | Embeds retrieved context and simulates quantization-based compression. |
| **Memory Manager** | Tracks conversation history; provides short-term and compressed long-term context. |
| **LLM Router** | Selects between Google Gemini and a local OpenAI-compatible LLM based on environment configuration. |
| **ChromaDB** | Stores sentence embeddings for the medical corpus; performs nearest-neighbor lookup at query time. |

---

## 4. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.9+ | All backend and frontend code |
| **Web Framework** | FastAPI | REST API server |
| **ASGI Server** | Uvicorn | Serves the FastAPI application |
| **Frontend** | Streamlit | Interactive web UI |
| **Embedding Model** | `all-MiniLM-L6-v2` (sentence-transformers) | Text embeddings for RAG and TurboQuant |
| **Vector Database** | ChromaDB (in-memory) | Stores and queries document embeddings |
| **Dataset Source** | KaggleHub (`chaitanyakck/medical-text`) | Medical text corpus for RAG |
| **LLM — Cloud** | Google Generative AI (Gemini 1.5 Flash) | Primary AI response generation |
| **LLM — Local** | Any OpenAI-compatible server (llama.cpp, LM Studio) | Optional offline LLM backend |
| **Numerical Processing** | NumPy | Quantization math in TurboQuant |
| **Data Handling** | Pandas | Dataset loading utilities |
| **Environment Config** | python-dotenv | Loads `.env` configuration file |
| **HTTP Client** | Requests | Frontend → Backend; Local LLM API calls |
| **Process Monitoring** | psutil | Measures RAM consumption per request |

---

## 5. Project Structure

```
TurboMind/
├── README.md                  # Quick-start reference
├── DOCUMENTATION.md           # This document
├── requirements.txt           # Python package dependencies
│
├── backend/
│   ├── .env                   # Environment variables (gitignored in production)
│   ├── main.py                # FastAPI app entry point; /chat endpoint
│   ├── rag.py                 # RAG pipeline: dataset loading, chunking, indexing, retrieval
│   ├── compression.py         # TurboQuant compression + text truncation utility
│   ├── memory.py              # Short-term and long-term conversation memory
│   ├── llm_client.py          # LLM backend router (Gemini vs. local)
│   ├── gemini_client.py       # Google Gemini client
│   └── local_llm_client.py    # OpenAI-compatible local LLM client
│
└── frontend/
    └── app.py                 # Streamlit web UI
```

---

## 6. Module Reference

### 6.1 Backend — `main.py`

**Purpose:** Entry point for the FastAPI application. Initializes the RAG index on startup and handles the single `/chat` POST endpoint.

**Key responsibilities:**
- Instantiates the global `MemoryManager`.
- Calls `initialize_rag()` once at application startup to download and index the medical corpus.
- Exposes `GET /` (health check) and `POST /chat` (main chat endpoint).
- Orchestrates the full pipeline: RAG → TurboQuant → Memory → LLM.
- Measures and returns latency and memory usage with every response.
- Catches all exceptions gracefully and returns a structured error response.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check; returns `{"message": "TurboMind API running 🚀"}` |
| `POST` | `/chat` | Main Q&A endpoint (see [API Reference](#8-api-reference)) |

---

### 6.2 RAG Pipeline — `rag.py`

**Purpose:** Implements Retrieval-Augmented Generation by building a searchable vector index over a real medical text dataset.

**Functions:**

| Function | Description |
|----------|-------------|
| `load_medical_docs()` | Downloads the Kaggle `chaitanyakck/medical-text` dataset via `kagglehub` and reads all `.dat` files, returning lines longer than 50 characters. |
| `chunk_text(text, chunk_size=300)` | Splits a text string into fixed-size chunks of `chunk_size` characters. |
| `prepare_documents()` | Calls `load_medical_docs()` and chunks every line, returning a flat list of text chunks. |
| `add_documents(docs)` | Encodes documents using `all-MiniLM-L6-v2` and inserts them into ChromaDB with auto-generated integer IDs. |
| `initialize_rag()` | Calls `prepare_documents()` and adds the **first 500 chunks** to ChromaDB. Called once at application startup. |
| `retrieve(query)` | Encodes the user query, performs a nearest-neighbor search in ChromaDB, and returns the top 3 matching text chunks. |

**Design notes:**
- ChromaDB is used in **in-memory** mode (`chromadb.Client()`), so the index is rebuilt every time the server restarts.
- Indexing is capped at 500 chunks for performance on development hardware.
- The same `SentenceTransformer` model is shared between `rag.py` and `compression.py`.

---

### 6.3 Compression Engine — `compression.py`

**Purpose:** Implements two compression utilities — a simple text truncation for memory storage and the "TurboQuant" quantization-inspired context compression.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `compress_text` | `(text, max_length=200) → str` | Truncates text to `max_length` characters. Used by `MemoryManager` to compress old messages into long-term memory. |
| `turboquant_compress` | `(text, bits=4) → str` | Simulates quantization compression. See [Section 11](#11-turboquant-compression--deep-dive) for full details. |

**TurboQuant algorithm (summary):**
1. Embed the full context string using `all-MiniLM-L6-v2`.
2. Find the min/max of the embedding vector.
3. Quantize embedding values into `2^bits` discrete levels.
4. Dequantize (reconstruct) the embedding.
5. Apply a `bits / 8` compression factor to the **original text length** and return the truncated text.

---

### 6.4 Memory Manager — `memory.py`

**Purpose:** Maintains a two-tier conversation memory buffer to provide conversational continuity across multiple turns.

**Class:** `MemoryManager`

| Attribute | Type | Description |
|-----------|------|-------------|
| `short_term` | `list[str]` | Holds the last 5 user messages in full. |
| `long_term` | `list[str]` | Holds older messages compressed to 200 characters each. |

**Methods:**

| Method | Description |
|--------|-------------|
| `add_message(message)` | Appends `message` to `short_term`. If `short_term` exceeds 5 entries, the oldest message is popped, compressed with `compress_text(old, 200)`, and appended to `long_term`. |
| `get_context()` | Returns a single string concatenating all `long_term` and `short_term` messages, used in the final LLM prompt. |

**Memory lifecycle:**

```
Turn 1–5:   [Q1, Q2, Q3, Q4, Q5]                  → short_term (full text)
Turn 6:     [Q2, Q3, Q4, Q5, Q6] + [Q1_compressed] → short_term + long_term
Turn 7:     [Q3, Q4, Q5, Q6, Q7] + [Q1_c, Q2_c]   → short_term + long_term
```

---

### 6.5 LLM Router — `llm_client.py`

**Purpose:** A thin routing layer that decides whether to call the Gemini cloud API or a local OpenAI-compatible server, based on the `USE_LOCAL_LLM` environment variable.

**Function:** `generate_response(prompt: str) → str`

- Reads `USE_LOCAL_LLM` from the environment.
- If the value is `"1"`, `"true"`, `"yes"`, or `"on"` (case-insensitive), routes to `local_llm_client.generate_response`.
- Otherwise routes to `gemini_client.generate_response`.

---

### 6.6 Gemini Client — `gemini_client.py`

**Purpose:** Handles all communication with the Google Generative AI (Gemini) API.

**Key design — model resolution (`_resolve_model_name`):**
1. Tries to list all available models from the Gemini API.
2. Filters for models that support `generateContent`.
3. Attempts to match the preferred model name (from `GEMINI_MODEL` env var, default `gemini-1.5-flash`).
4. Falls back to any model with "flash" in its name.
5. If all else fails, falls back to the preferred name directly.

This ensures the client degrades gracefully when the model list API is unavailable or when a specific model is retired.

**Lazy initialization:** The model is only instantiated on the first call (`_get_model()`), and then cached in the module-level `_model` variable.

**Function:** `generate_response(prompt: str) → str`

Returns `response.text` from the Gemini model, or an `"Error: ..."` string on failure.

---

### 6.7 Local LLM Client — `local_llm_client.py`

**Purpose:** Provides a client for any locally running OpenAI-compatible LLM server (e.g., LM Studio, llama.cpp, Ollama with OpenAI compat layer).

**Functions:**

| Function | Description |
|----------|-------------|
| `_models_url_from_chat_url(chat_url)` | Derives the `/v1/models` endpoint URL from the configured chat completions URL. |
| `_resolve_model_id(chat_url, configured, timeout)` | If `LOCAL_LLM_MODEL` is `"auto"`, queries `/v1/models` to discover the first loaded model ID. Otherwise uses the configured value directly. |
| `generate_response(prompt)` | Sends a POST to the local chat completions endpoint and returns the assistant message content. |

**Request format:** Uses the standard OpenAI Chat Completions format with a system message and a user message. Temperature is set to `0.2` for deterministic, factual responses.

---

### 6.8 Frontend — `app.py`

**Purpose:** A single-page Streamlit web application that provides the user interface for TurboMind.

**UI elements:**

| Element | Type | Description |
|---------|------|-------------|
| **Compression Level (Bits)** | Slider (2–8, default 4) | Controls the TurboQuant compression level passed to the backend. |
| **Ask a medical question** | Text input | User's natural-language question. |
| **Send** | Button | Submits the question to the backend `/chat` endpoint. |
| **Response** | Text area | Displays the LLM's answer. |
| **TurboQuant Metrics** | Three-column metric display | Shows original context length, compressed context length, and compression ratio. |
| **Performance** | Text | Shows memory usage (bytes), latency (seconds), and bits used. |
| **Compression Comparison** | Bar chart | Visual before/after bar chart of context lengths. |

The frontend calls `http://127.0.0.1:8000/chat` with a 120-second timeout.

---

## 7. Request Lifecycle

The following sequence describes everything that happens when a user clicks **Send**:

```
1. [FRONTEND]
   User types question, sets compression bits slider, clicks Send.
   → HTTP POST http://127.0.0.1:8000/chat?query=<question>&bits=<bits>

2. [BACKEND — main.py /chat]
   Timer starts (for latency measurement).

3. [RAG — rag.py retrieve()]
   Query is embedded using all-MiniLM-L6-v2.
   ChromaDB performs nearest-neighbor search.
   Top 3 matching medical text chunks are returned and joined into `context`.

4. [COMPRESSION — compression.py turboquant_compress()]
   `context` is embedded using all-MiniLM-L6-v2.
   Embedding values are quantized into 2^bits levels.
   A compression_factor = bits / 8 is computed.
   The context is truncated to len(context) * compression_factor characters.
   → compressed_context

5. [MEMORY — memory.py]
   The user's query is added to short_term memory.
   If short_term > 5 entries, the oldest is compressed and moved to long_term.
   get_context() returns the combined long_term + short_term string.

6. [PROMPT ASSEMBLY — main.py]
   A structured prompt is assembled:
     - Role instruction: "You are a medical AI assistant."
     - Context: compressed_context
     - Conversation history: memory_context
     - Question: query
     - Disclaimer: "This is not medical advice."

7. [LLM — llm_client.py → gemini_client.py / local_llm_client.py]
   The prompt is sent to the configured LLM.
   The LLM's text response is returned.

8. [METRICS — main.py]
   Timer stops. psutil reads current RAM usage.
   Compression ratio = compressed_length / original_length.

9. [RESPONSE — main.py → frontend]
   JSON payload returned:
     response, memory_usage, latency, context_length,
     original_length, compressed_length, compression_ratio, bits.

10. [FRONTEND — app.py]
    Response text is rendered.
    Metrics are displayed in columns and bar chart.
```

---

## 8. API Reference

### `GET /`

**Description:** Health check endpoint.

**Response:**
```json
{
  "message": "TurboMind API running 🚀"
}
```

---

### `POST /chat`

**Description:** Main question-answering endpoint. Performs RAG retrieval, TurboQuant compression, memory management, and LLM generation in a single call.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ Yes | — | The user's medical question. |
| `bits` | integer | ❌ No | `4` | Compression bit-depth for TurboQuant (2–8). Lower = more compression. |

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/chat?query=What%20is%20asthma%3F&bits=4"
```

**Success Response (HTTP 200):**
```json
{
  "response": "Asthma is a chronic inflammatory disease of the airways...",
  "memory_usage": 524288000,
  "latency": 1.842,
  "context_length": 1024,
  "original_length": 856,
  "compressed_length": 428,
  "compression_ratio": 0.5,
  "bits": 4
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | LLM-generated answer to the question. |
| `memory_usage` | integer | RAM used by the process in bytes (from `psutil`). |
| `latency` | float | Total end-to-end time in seconds, rounded to 3 decimal places. |
| `context_length` | integer | Character count of the full assembled LLM prompt. |
| `original_length` | integer | Character count of the raw RAG-retrieved context (before compression). |
| `compressed_length` | integer | Character count of the context after TurboQuant compression. |
| `compression_ratio` | float | `compressed_length / original_length`, rounded to 2 decimal places. |
| `bits` | integer | Compression bit-depth actually used. |

**Error Response:** On exception, the same JSON shape is returned with `response` set to `"Error: <message>"` and numeric fields set to `0` or `1`.

---

## 9. Configuration Reference

Configuration is read from a `.env` file located in the `backend/` directory or from system environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required for Gemini)* | Google Generative AI API key. |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Preferred Gemini model name. The client resolves the best available match at startup. |
| `USE_LOCAL_LLM` | `0` | Set to `1` (or `true`/`yes`/`on`) to use the local LLM instead of Gemini. |
| `LOCAL_LLM_URL` | `http://127.0.0.1:1234/v1/chat/completions` | URL of the OpenAI-compatible local LLM chat completions endpoint. |
| `LOCAL_LLM_MODEL` | `auto` | Model ID to use. `auto` queries `/v1/models` to discover it automatically. |
| `LOCAL_LLM_TIMEOUT` | `120` | Seconds to wait for a response from the local LLM server before timing out. |

**Example `.env` file:**
```env
GEMINI_API_KEY=AIzaSy...your_key_here
GEMINI_MODEL=gemini-1.5-flash
USE_LOCAL_LLM=0
LOCAL_LLM_URL=http://127.0.0.1:1234/v1/chat/completions
LOCAL_LLM_MODEL=auto
LOCAL_LLM_TIMEOUT=120
```

---

## 10. Setup and Deployment Guide

### Prerequisites

- Python 3.9 or higher
- pip
- A Google Gemini API key **or** a locally running OpenAI-compatible LLM server
- Kaggle credentials configured for `kagglehub` (for the RAG dataset download)

### Step 1 — Clone the repository

```bash
git clone https://github.com/jayaprakash2207/TurboMind.git
cd TurboMind
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs:

```
fastapi
uvicorn
google-generativeai
chromadb
sentence-transformers
psutil
numpy
streamlit
requests
python-dotenv
kagglehub
pandas
```

### Step 4 — Configure environment variables

Create a `backend/.env` file:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
USE_LOCAL_LLM=0
```

### Step 5 — Start the backend

From the project root directory:

```bash
uvicorn backend.main:app --reload --port 8000
```

On first run, TurboMind automatically downloads the Kaggle `chaitanyakck/medical-text` dataset and builds the vector index. This may take a few minutes.

The API will be available at: `http://127.0.0.1:8000`

### Step 6 — Start the frontend

In a second terminal (with the virtual environment activated):

```bash
streamlit run frontend/app.py
```

Streamlit will open the application in your default web browser.

### Using a Local LLM (optional)

1. Start a local OpenAI-compatible server (e.g., LM Studio, llama.cpp with the OpenAI compat layer, or Ollama).
2. Note the endpoint URL (typically `http://127.0.0.1:1234/v1/chat/completions`).
3. Set `USE_LOCAL_LLM=1` and `LOCAL_LLM_URL=<your_endpoint>` in `.env`.
4. Restart the backend.

---

## 11. TurboQuant Compression — Deep Dive

TurboQuant is the signature feature of TurboMind. It simulates how quantization (a technique used in LLM model compression) might be applied to reduce the size of context passed to an LLM.

### Background

In model quantization, floating-point weight values are mapped to a smaller set of discrete levels (e.g., 4-bit = 16 levels). This reduces memory and computation at the cost of some precision. TurboQuant applies the same mathematical concept to *text context*, using the text's embedding vector as a proxy.

### Algorithm

```
Input: text (string), bits (integer, 2–8)

1. Embed text → vector of floats (384 dimensions, all-MiniLM-L6-v2)

2. Find min and max values of the embedding vector:
      min_val, max_val = embedding.min(), embedding.max()

3. Compute number of quantization levels:
      levels = 2 ^ bits
      scale = (max_val - min_val) / levels

4. Quantize:
      quantized = round((embedding - min_val) / scale)

5. Dequantize (reconstruct):
      reconstructed = quantized * scale + min_val

6. Compute compression factor:
      compression_factor = bits / 8

7. Truncate original text:
      max_chars = floor(len(text) * compression_factor)
      return text[:max_chars]
```

### Compression Factor by Bits

| Bits | Levels | Compression Factor | Context Retained |
|------|--------|-------------------|------------------|
| 2 | 4 | 25% | Heavy compression |
| 4 | 16 | 50% | Moderate compression |
| 6 | 64 | 75% | Light compression |
| 8 | 256 | 100% | No compression |

### Important Note

The quantization of the embedding (steps 1–5) is a conceptual simulation. The actual compression effect on the **text** is determined by the `bits / 8` ratio applied to the character count (step 6–7). The embedding quantization demonstrates the mathematical underpinning but does not perform true semantic compression or sentence selection.

---

## 12. RAG Dataset and Indexing

### Dataset

TurboMind uses the **`chaitanyakck/medical-text`** dataset from Kaggle, downloaded automatically via the `kagglehub` library on first run.

The dataset contains raw medical text stored in `.dat` files. Each file is read line-by-line; lines shorter than 50 characters are discarded.

### Indexing Pipeline

```
Raw .dat files
     │
     ▼ filter (len > 50)
Raw lines
     │
     ▼ chunk_text(chunk_size=300)
Text chunks
     │
     ▼ limit to first 500 chunks
500 chunks
     │
     ▼ SentenceTransformer.encode()
384-dim embeddings
     │
     ▼ chromadb.collection.add()
ChromaDB in-memory collection ("docs")
```

### Retrieval

When a user asks a question:
1. The question is embedded using the same `all-MiniLM-L6-v2` model.
2. ChromaDB performs a cosine similarity search.
3. The top 3 nearest document chunks are returned.
4. The chunks are joined with spaces to form the RAG context.

---

## 13. Memory System Design

The memory system provides conversational context across multiple turns within a single server session.

```
┌─────────────────────────────────────────────────────┐
│                  MemoryManager                       │
│                                                      │
│  short_term: [Q3, Q4, Q5]   ← last 5 full messages  │
│  long_term:  [Q1_c, Q2_c]   ← older, truncated      │
│                                                      │
│  get_context() = "Q1_c Q2_c Q3 Q4 Q5"               │
└─────────────────────────────────────────────────────┘
```

**Characteristics:**
- The `MemoryManager` instance is global — it persists for the lifetime of the backend server process.
- All users sharing the same backend instance share the same memory. In a multi-user production deployment, this would need to be replaced with per-session memory.
- Long-term memory entries are truncated to 200 characters, so very long prior questions lose detail but their topic is preserved.

---

## 14. LLM Backend Options

### Option A: Google Gemini (Default)

**When to use:** When you have a Gemini API key and internet access.

**Model:** `gemini-1.5-flash` (configurable via `GEMINI_MODEL`)

**Advantages:**
- No local GPU required.
- Access to a powerful frontier model.
- Model selection is automatic with graceful fallback.

**Setup:** Set `GEMINI_API_KEY` in `.env`. Ensure `USE_LOCAL_LLM=0`.

---

### Option B: Local OpenAI-Compatible LLM

**When to use:** When operating in an air-gapped environment, or when testing with open-source models like LLaMA, Mistral, etc.

**Compatible servers:**
- [LM Studio](https://lmstudio.ai/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) with `--api-server`
- [Ollama](https://ollama.com/) with the OpenAI compatibility layer

**Setup:**
1. Start your local server.
2. Set `USE_LOCAL_LLM=1` in `.env`.
3. Set `LOCAL_LLM_URL` to your server's chat completions endpoint.
4. Optionally set `LOCAL_LLM_MODEL` to your model's ID, or leave as `auto`.

**Request format:** Standard OpenAI Chat Completions with `temperature=0.2`.

---

## 15. Frontend UI Guide

The Streamlit UI is a single-page application with the following sections (from top to bottom):

1. **Title bar:** "🚀 TurboMind AI Assistant (TurboQuant Simulation)"
2. **Compression slider:** Adjusts the `bits` parameter (2–8). Dragging left increases compression; dragging right reduces it.
3. **Question input:** Free-text field for the user's medical question.
4. **Send button:** Submits the request.
5. **Response section:** The LLM's formatted answer.
6. **TurboQuant Metrics:** Three metric cards showing original context size, compressed size, and ratio.
7. **Performance section:** RAM usage, end-to-end latency, and bit-depth used.
8. **Compression chart:** A bar chart comparing context size before and after compression.

---

## 16. Performance Metrics Explained

Every `/chat` response includes the following metrics:

| Metric | How It Is Measured | What It Tells You |
|--------|-------------------|--------------------|
| `memory_usage` | `psutil.virtual_memory().used` (bytes) | Current RAM used by the entire system process. Useful for monitoring memory growth across many requests. |
| `latency` | `time.time()` delta from request start to before return (seconds) | Total pipeline time: RAG + compression + LLM call. The LLM call typically dominates. |
| `context_length` | `len(final_prompt)` (characters) | Size of the full prompt sent to the LLM, including system instructions, context, memory, and question. |
| `original_length` | `len(context)` before compression (characters) | How much raw text the RAG retrieval returned. |
| `compressed_length` | `len(compressed_context)` (characters) | How much text was passed to the LLM after TurboQuant compression. |
| `compression_ratio` | `compressed_length / original_length` | A ratio < 1 means context was reduced. `0.5` means half the original context was kept. |
| `bits` | Integer parameter from request | Which compression setting was used (2 = most compressed, 8 = no compression). |

---

## 17. Known Limitations

| Limitation | Details |
|-----------|---------|
| **Prototype-grade RAG** | Only the first 500 text chunks are indexed. The full dataset contains far more content. |
| **In-memory vector store** | ChromaDB runs in-memory. The index is lost on every server restart, requiring a fresh build. |
| **Naive chunking** | Text is split by fixed character count (300 chars), not by sentence or semantic boundary. This may cut mid-sentence, degrading retrieval quality. |
| **Simulated compression** | TurboQuant is a simulation — it truncates text by a ratio derived from quantization math, but does not actually select the most semantically relevant portion. |
| **Global shared memory** | All requests share the same `MemoryManager` instance. In a multi-user setting, memory from one user's session bleeds into another's. |
| **No authentication** | The API has no authentication or rate limiting. |
| **No source citations** | Responses do not indicate which specific document or passage supported the answer. |
| **Medical safety** | LLM responses are AI-generated and **not safe for real-world clinical use**. |

---

## 18. Future Roadmap

The following improvements have been identified for potential future development:

| Priority | Feature | Description |
|----------|---------|-------------|
| High | **Persistent vector store** | Use a disk-backed ChromaDB instance to avoid re-indexing on every restart. |
| High | **Per-session memory** | Replace the global `MemoryManager` with a session-scoped store (e.g., keyed by a session ID cookie). |
| Medium | **Semantic chunking** | Replace fixed-length chunking with sentence-boundary-aware or paragraph-level chunking. |
| Medium | **Source citations** | Return document IDs or excerpts with each response so users can verify sources. |
| Medium | **Full dataset indexing** | Remove or raise the 500-chunk limit and index the complete medical corpus. |
| Medium | **True semantic compression** | Replace text truncation with an extractive summarization step that preserves the most relevant sentences. |
| Low | **Authentication & rate limiting** | Add API key authentication and per-user rate limits. |
| Low | **Streaming responses** | Use SSE (Server-Sent Events) to stream tokens from the LLM to the frontend in real time. |
| Low | **Docker packaging** | Provide a `Dockerfile` and `docker-compose.yml` for one-command deployment. |

---

## 19. Disclaimer

> **TurboMind is a research prototype intended for educational and experimentation purposes only.**
>
> The medical information provided by TurboMind is generated by AI models and sourced from a publicly available text dataset. It has **not** been reviewed, validated, or approved by any medical professional or regulatory authority.
>
> **Do not use TurboMind responses for real medical decisions, diagnosis, treatment, or any clinical purpose.**
>
> Always consult a qualified healthcare professional for medical advice.

---

*End of TurboMind Project Documentation*
