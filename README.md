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
|------|------|---------|
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

The resulting table can be downloaded as `eval_dataset.json` for optional RAGAS
scoring with `python evaluate_ragas.py`.

### Swapping in a new source document

You no longer need to rename files to `AI_Book.docx` or edit `.env`. Just point
`vector_store.py` at whatever file you want to index:

```bash
python vector_store.py --file data/Kayak_Login_QA_Spec.docx --reset
python vector_store.py --file data/some_report.pdf --reset
python vector_store.py --file data/notes.md --reset
```

Supported file types: `.docx`, `.pdf`, `.txt`, `.md`, `.csv` (see `LOADER_MAP` in `document_loader.py`).

**Always use `--reset` when switching to a different document.** Without it,
`Chroma.from_documents()` appends to the existing collection, so old chunks
from a previous document stick around and get mixed into retrieval results —
this is a common source of "why is it citing content that isn't in my doc"
confusion.

If you omit `--file`, it falls back to `SOURCE_DOCX` in `.env` (still useful
as a default for a recurring pipeline run, e.g. in CI).

Also swap the matching test cases before running evaluation:
```bash
cp data/kayak_login_test_cases.json data/rag_test_cases.json
python run_rag_tests.py
python evaluate_ragas.py
```

Outputs:
- `chroma_db/` — persisted vector store
- `data/eval_dataset.json` — question/response/context/reference rows
- `data/ragas_results.json` — per-question RAGAS scores

## Prompting notes

`rag_graph.py`'s generation prompt is written to:
- Force grounding strictly in retrieved context (reduces hallucination)
- Explicitly refuse to answer when context is insufficient, with a fixed fallback sentence
  (this fallback phrasing is what lets you detect "I don't know" answers programmatically)
- Keep answers concise (2–5 sentences) unless the question needs a list/steps
- Avoid meta-references like "according to the context" for a more natural answer

If you see retrieval quality issues (irrelevant chunks), the `retrieve_node` in
`rag_graph.py` has a comment showing where to add an LLM-based query rewrite step.

## Free-tier caveats

- OpenRouter free models are rate-limited (commonly ~20 req/min, daily caps). `run_rag_tests.py`
  sleeps 1s between calls — increase `sleep_between` if you hit 429 errors.
- RAGAS's metrics call the evaluator LLM multiple times per question (once per metric).
  With 5 test cases × 4 metrics, expect ~20 LLM calls — budget for free-tier limits accordingly.
- Free model availability changes regularly. If OpenRouter returns a model-unavailable error,
  choose a currently free text model from its catalog and update `LLM_MODEL` in `.env`.
