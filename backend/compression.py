import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Simple length-based compression for memory storage
def compress_text(text, max_length=200):
    if not text:
        return text
    return text[:max_length]

# 🔥 TurboQuant-like compression
def turboquant_compress(text, bits=4):
    if not text or len(text) < 50:
        return text

    # Step 1: Convert text → embedding
    embedding = model.encode([text])[0]

    # Step 2: Quantization
    min_val, max_val = embedding.min(), embedding.max()

    if max_val - min_val == 0:
        return text

    levels = 2 ** bits
    scale = (max_val - min_val) / levels

    quantized = np.round((embedding - min_val) / scale)

    # Step 3: Dequantization (reconstruction)
    reconstructed = quantized * scale + min_val

    # Step 4: Simulate compression effect on text
    compression_factor = bits / 8  # 2-bit = heavy compression
    max_chars = int(len(text) * compression_factor)

    return text[:max_chars]
