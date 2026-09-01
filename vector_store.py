"""
vector_store.py
Step 5 in the flow: Store vectors + chunks in Chroma (local, free, persisted to disk).

Now supports:
  --file  : ingest any document path (docx, pdf, txt, md, csv), not just AI_Book.docx
  --reset : wipe the existing Chroma DB before ingesting, so old documents' chunks
            don't linger and pollute retrieval when you swap in a new source doc

Usage:
    python vector_store.py --file data/Kayak_Login_QA_Spec.docx --reset
    python vector_store.py                          # uses config.SOURCE_DOCX, appends
"""
import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
import config
from embeddings import get_embeddings
from document_loader import load_document
from chunker import split_into_chunks


def reset_vector_store(persist_directory: str = None):
    """Delete the on-disk Chroma DB so the next build starts clean."""
    persist_directory = persist_directory or config.CHROMA_DIR
    path = Path(persist_directory)
    if path.exists():
        shutil.rmtree(path)
        print(f"[vector_store] Cleared existing Chroma DB at '{persist_directory}'")
    else:
        print(f"[vector_store] No existing Chroma DB found at '{persist_directory}' (nothing to clear)")


def build_vector_store(chunks: list[Document], persist_directory: str = None) -> Chroma:
    """Create (or append to) a Chroma DB from document chunks.

    Note: this APPENDS to an existing collection if one is present at
    persist_directory. Call reset_vector_store() first if you want a clean
    index containing only the current document.
    """
    persist_directory = persist_directory or config.CHROMA_DIR
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"[vector_store] Stored {len(chunks)} chunks in Chroma at '{persist_directory}'")
    return vectorstore


def load_vector_store(persist_directory: str = None) -> Chroma:
    """Load an existing Chroma DB from disk without rebuilding it."""
    persist_directory = persist_directory or config.CHROMA_DIR
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vectorstore


def parse_args():
    parser = argparse.ArgumentParser(description="Build the Chroma vector store from a source document.")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to the source document. Defaults to SOURCE_DOCX in .env if omitted.",
    )
    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        help="Delete the existing Chroma DB before ingesting, so only this document's "
             "chunks are present (recommended when switching to a new source document).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.reset:
        reset_vector_store()

    docs = load_document(args.file)
    chunks = split_into_chunks(docs)
    build_vector_store(chunks)
