from sentence_transformers import SentenceTransformer
import chromadb
import kagglehub
import os
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection("docs")


# 🔥 Load Kaggle dataset
def load_medical_docs():
    import kagglehub
    import os

    path = kagglehub.dataset_download("chaitanyakck/medical-text")

    files = os.listdir(path)
    print("Dataset files:", files)

    docs = []

    for file in files:
        file_path = os.path.join(path, file)

        # 🔥 Read .dat as text
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

            for line in lines:
                clean_line = line.strip()

                # skip empty or too short
                if len(clean_line) > 50:
                    docs.append(clean_line)

    print(f"Loaded {len(docs)} raw lines")

    return docs

# 🔥 Chunking
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# 🔥 Prepare docs
def prepare_documents():
    raw_docs = load_medical_docs()

    chunks = []
    for doc in raw_docs:
        chunks.extend(chunk_text(str(doc)))

    return chunks


# 🔥 Add to vector DB
def add_documents(docs):
    embeddings = model.encode(docs).tolist()
    ids = [str(i) for i in range(len(docs))]

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=ids
    )


# 🔥 Initialize RAG
def initialize_rag():
    docs = prepare_documents()

    print(f"Loaded {len(docs)} chunks")

    # IMPORTANT: limit for performance
    add_documents(docs[:500])


# 🔍 Retrieval
def retrieve(query):
    q_emb = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=3
    )

    return results["documents"][0] if results["documents"] else []