"""
config.py
Central place to load environment variables and shared settings.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3.5-lightning:free")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SOURCE_DOCX = os.environ.get("SOURCE_DOCX", "data/AI_Book.docx")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    print(
        "[config] WARNING: OPENROUTER_API_KEY is not set. "
        "Copy .env.example to .env and add your free key from https://openrouter.ai/keys"
    )