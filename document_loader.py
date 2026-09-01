"""
document_loader.py
Step 2 in the flow: Load a source document.

Now supports any file type (not just the original AI_Book.docx) and accepts
the path via CLI argument, so you don't have to rename files or edit config.py
every time you want to index a new document.

Usage:
    python document_loader.py --file data/Kayak_Login_QA_Spec.docx
    python document_loader.py                     # falls back to config.SOURCE_DOCX
"""
import argparse
from pathlib import Path
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_core.documents import Document
import config

LOADER_MAP = {
    ".docx": Docx2txtLoader,
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
}


def load_document(path: str = None) -> list[Document]:
    """Load a document (docx, pdf, txt, md, or csv) and return LangChain Document objects.

    The loader class is picked automatically based on the file extension.
    """
    path = path or config.SOURCE_DOCX
    ext = Path(path).suffix.lower()

    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type '{ext}' for {path}. "
            f"Supported types: {', '.join(LOADER_MAP.keys())}"
        )

    loader_cls = LOADER_MAP[ext]
    loader = loader_cls(path)
    docs = loader.load()
    print(f"[document_loader] Loaded {len(docs)} document(s) from {path} (type: {ext})")
    return docs


def parse_args():
    parser = argparse.ArgumentParser(description="Load a source document for the RAG pipeline.")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to the source document (.docx, .pdf, .txt, .md, .csv). "
             "Defaults to SOURCE_DOCX in .env if omitted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    docs = load_document(args.file)
    print(docs[0].page_content[:500])
