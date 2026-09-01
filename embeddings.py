"""
embeddings.py
Step 4 in the flow: Create embeddings.
Uses a local, free HuggingFace sentence-transformers model (no API key, no cost).
"""
from langchain_huggingface import HuggingFaceEmbeddings
import config

_embedding_model = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a singleton HuggingFace embedding model (downloaded once, runs locally)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[embeddings] Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embedding_model


if __name__ == "__main__":
    emb = get_embeddings()
    vector = emb.embed_query("What is RAG?")
    print(f"Embedding dimension: {len(vector)}")