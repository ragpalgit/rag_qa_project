"""
retriever.py
Step 6 in the flow: Similarity search against Chroma to get top relevant chunks.
"""
from langchain_core.vectorstores import VectorStoreRetriever
from vector_store import load_vector_store


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    """Return a retriever that does top-k similarity search over the Chroma DB."""
    vectorstore = load_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    retriever = get_retriever()
    results = retriever.invoke("What is RAG?")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content[:200])
        print()