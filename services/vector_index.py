"""
for the FAISS index (vec-embed + store) - modular semantic search

the engine learns as it's fed memes.
"""
from importlib.metadata import metadata


# imports
import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

# configs -----
VECTOR_DIM = 1280 # 768 + 512
INDEX_PATH = "vector_index/index.faiss"
META_PATH = "vector_index/metadata.json"
os.makedirs("vector_index", exist_ok=True)
# model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# loading the index
def load_or_create_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata=json.load(f)
    else:
        index = faiss.IndexFlatL2(VECTOR_DIM)
        metadata = []

    return index, metadata

def add_to_index(embedding: np.ndarray, meta: dict):
    index, metadata = load_or_create_index()
    if embedding.shape != (1, VECTOR_DIM):
        raise ValueError(f"Expected shape (1, {VECTOR_DIM}, got {embedding.shape}")

    index.add(embedding) # add vec to the FAISS index

    metadata.append(meta)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
