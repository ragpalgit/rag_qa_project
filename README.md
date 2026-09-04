# RAG QA Project — LangChain + LangGraph + RAGAS (Free APIs)

Implements the full flow from your diagram: Document → Chunking → Embeddings →
Chroma → LangGraph RAG → RAGAS Evaluation, using **only free APIs/models**:

- **LLM**: OpenRouter free-tier model (`nvidia/nemotron-3.5-lightning:free` by default)
- **Embeddings**: local HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (no API key, runs on CPU)
- **Vector store**: Chroma (local, persisted to disk)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your free OpenRouter key from https://openrouter.ai/keys
```

Place your source document at `data/AI_Book.docx` (or update `SOURCE_DOCX` in `.env`).

## File Map (matches diagram numbering)

| Step | File | Purpose |
|------|------|----------|
| 1-2  | `document_loader.py` | Load the .docx source document |
| 3    | `chunker.py` | Split into ~800-char chunks with overlap |
| 4    | `embeddings.py` | Local HuggingFace embedding model |
| 5    | `vector_store.py` | Build/load the Chroma DB |
| 6    | `retriever.py` | Top-k similarity search |
| 7    | `rag_graph.py` | LangGraph: Retrieve Node → Generate Node |
| 8    | `llm.py` | LLM call via OpenRouter (free models) |
| 9    | `data/rag_test_cases.json` | Test questions + expected answers |
| 10-11| `run_rag_tests.py` | Runs all test cases, builds eval dataset |
| 12-13| `evaluate_ragas.py` | RAGAS scoring: faithfulness, answer relevancy, context precision/recall |

## Run the full pipeline

```bash
# 1. Build the vector store (chunks + embeddings -> Chroma)
#    --reset clears any previously indexed document first (recommended when switching docs)
python vector_store.py --file data/AI_Book.docx --reset

# 2. Ask a single question through the LangGraph RAG pipeline
python rag_graph.py

# 3. Run all test cases and build the RAGAS evaluation dataset
python run_rag_tests.py

# 4. Score the results with RAGAS
python evaluate_ragas.py
```

## Run the web app

The web app accepts a specification document and a matching JSON test-case file,
then replaces the local vector index and runs every test case through the RAG
graph. The JSON must be a list of objects containing `user_input` and `reference`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Releases and public download counts

This repository now includes a GitHub Actions release workflow at
`.github/workflows/release.yml`.
When you push a tag such as `v0.1.0`, GitHub will create a release and attach
downloadable `.zip` and `.tar.gz` archives.

```bash
git tag v0.1.0
git push origin v0.1.0
```

After the workflow finishes:

- Open the repository's **Releases** page.
- Each attached asset shows its public **download count**.
- The count applies to the uploaded release assets, not to repository clones.