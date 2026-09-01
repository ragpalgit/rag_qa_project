"""
chunker.py
Step 3 in the flow: Split the loaded document into chunks.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from document_loader import load_document


def split_into_chunks(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[chunker] Split into {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    docs = load_document()
    chunks = split_into_chunks(docs)
    print(chunks[0].page_content[:300])
