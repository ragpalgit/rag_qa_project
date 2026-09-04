# RAG QA Project Architecture 🏗️

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG QA SYSTEM ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                            INPUT LAYER
                                │
                    ┌───────────┴────────────┐
                    │                        │
            📄 Documents              🌐 Web Interface
         (DOCX/PDF/TXT/              (Streamlit)
          MD/CSV)                     │
                    │                 │
                    └────────┬────────┘
                             │
                      ┌──────▼──────┐
                      │  DOCUMENT   │
                      │   LOADER    │
                      └──────┬──────┘
                             │
                      ┌──────▼──────┐
                      │  CHUNKER    │ (~800 chars + overlap)
                      └──────┬──────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
         ┌───────▼────────┐    ┌────────▼────────┐
         │   LOCAL        │    │     CHROMA      │
         │  EMBEDDINGS    │───▶│   VECTOR STORE  │
         │ (HuggingFace)  │    │   (Persistent)  │
         └────────────────┘    └────────┬────────┘
                                        │
                    RETRIEVAL LAYER     │
                                        │
                        ┌───────────────▼─────────────────┐
                        │                                 │
                   Question                      Top-K Similarity
                        │                              │
                        │      ┌──────────────────────┘
                        │      │
                   ┌────▼──────▼──────┐
                   │   RETRIEVER      │
                   │  (Top-K Search)  │
                   └────┬─────────────┘
                        │
                 Retrieved Contexts
                        │
         ┌──────────────┴──────────────┐
         │                             │
    ┌────▼────────┐         ┌──────────▼────────┐
    │  LANGGRAPH  │◄────────│   OPENROUTER LLM  │
    │   PIPELINE  │         │ (Free-tier Model) │
    │             │         └───────────────────┘
    │ Retrieve → │
    │ Generate   │
    └────┬───────┘
         │
    Generated Answer
         │
    ┌────▼──────────────┐
    │   RAGAS EVAL      │
    │ • Faithfulness    │
    │ • Answer Relevancy│
    │ • Ctx Precision   │
    │ • Ctx Recall      │
    └───────────────────┘
         │
    Results & Metrics
         │
    ┌────▼──────────────┐
    │   JSON DATASET    │
    │  (eval_dataset)   │
    └───────────────────┘
```

---

## Detailed Component Architecture

### **Layer 1: Input Processing**

```
┌─────────────────────────────────┐
│   DOCUMENT LOADER               │
├─────────────────────────────────┤
│ Supported Formats:              │
│ • DOCX (Docx2txtLoader)         │
│ • PDF  (PyPDFLoader)            │
│ • TXT  (TextLoader)             │
│ • MD   (UnstructuredMarkdownL)  │
│ • CSV  (CSVLoader)              │
│                                 │
│ Output: LangChain Documents     │
│ Metadata: Source, page info     │
└─────────────────────────────────┘
         │
         │ documents
         ▼
┌─────────────────────────────────┐
│   CHUNKER                       │
├─────────────────────────────────┤
│ Algorithm:                      │
│ RecursiveCharacterTextSplitter  │
│                                 │
│ Config:                         │
│ • Chunk Size: 800 chars         │
│ • Overlap: 100 chars            │
│ • Separators: ["\\n\\n", "\\n"] │
│                                 │
│ Output: Chunked documents       │
│ Quality: No context loss        │
└─────────────────────────────────┘
```

### **Layer 2: Embeddings & Storage**

```
┌─────────────────────────────────┐
│   EMBEDDINGS                    │
├─────────────────────────────────┤
│ Model:                          │
│ sentence-transformers/          │
│ all-MiniLM-L6-v2                │
│                                 │
│ Dimension: 384                  │
│ Type: Dense Vector              │
│ Execution: Local (CPU/GPU)      │
│ Cost: FREE ✓                    │
│                                 │
│ Input: Text chunks              │
│ Output: 384-dim vectors         │
└─────────────────────────────────┘
         │
         │ embedded_chunks
         ▼
┌─────────────────────────────────┐
│   CHROMA VECTOR STORE           │
├─────────────────────────────────┤
│ Type: Local Persistent DB       │
│ Location: chroma_db/ directory  │
│                                 │
│ Operations:                     │
│ • Insert: Chunks → Vectors      │
│ • Query: Vector similarity      │
│ • Retrieve: Top-K results       │
│                                 │
│ Scalability: CPU-efficient      │
│ Durability: On-disk persistence │
└─────────────────────────────────┘
```

### **Layer 3: Retrieval & Ranking**

```
┌─────────────────────────────────┐
│   RETRIEVER                     │
├─────────────────────────────────┤
│ Algorithm: Cosine Similarity    │
│ Index: Chroma Vector DB         │
│                                 │
│ Process:                        │
│ 1. Embed incoming question      │
│ 2. Compute similarity scores    │
│ 3. Rank by relevance            │
│ 4. Return top-K results         │
│                                 │
│ Parameters:                     │
│ • k: 4 (configurable)           │
│ • Metric: Cosine distance       │
│                                 │
│ Input: Question (text)          │
│ Output: Top-K chunks (context)  │
└─────────────────────────────────┘
```

### **Layer 4: Generation (LangGraph RAG)**

```
┌──────────────────────────────────────────────┐
│           LANGGRAPH STATE MACHINE            │
├──────────────────────────────────────────────┤
│                                              │
│  State: { question, context[], answer }     │
│                                              │
│                 START                        │
│                   │                          │
│            ┌──────▼──────┐                   │
│            │ RETRIEVE    │                   │
│            │ NODE        │                   │
│            │ • Get chunks│                   │
│            │ • Format ctx│                   │
│            └──────┬──────┘                   │
│                   │                          │
│            ┌──────▼──────┐                   │
│            │ GENERATE    │                   │
│            │ NODE        │                   │
│            │ • LLM call  │                   │
│            │ • Grounded  │                   │
│            │   answer    │                   │
│            └──────┬──────┘                   │
│                   │                          │
│                  END                         │
│                                              │
└──────────────────────────────────────────────┘
```

### **Layer 5: LLM Integration**

```
┌─────────────────────────────────┐
│   LLM: OpenRouter               │
├─────────────────────────────────┤
│ Base URL:                       │
│ https://openrouter.ai/api/v1    │
│                                 │
│ Model (Free-tier):              │
│ nvidia/nemotron-3.5-lightning   │
│                                 │
│ Alternative Free Models:        │
│ • DeepSeek R1                   │
│ • Llama 2                       │
│ • Mixtral 8x7B                  │
│                                 │
│ Features:                       │
│ • Grounded responses            │
│ • Context-aware generation      │
│ • Rate limiting support         │
│ • Cost: FREE (free-tier) ✓      │
│                                 │
│ Input:                          │
│ • Question + Retrieved context  │
│ • System prompt                 │
│                                 │
│ Output:                         │
│ • Grounded answer               │
│ • Based on retrieved context    │
└─────────────────────────────────┘
```

### **Layer 6: Evaluation (RAGAS)**

```
┌──────────────────────────────────────────┐
│           RAGAS EVALUATION               │
├──────────────────────────────────────────┤
│                                          │
│  Input Dataset:                          │
│  {                                       │
│    user_input: "question",               │
│    response: "answer",                   │
│    retrieved_contexts: [...],            │
│    reference: "expected_answer"          │
│  }                                       │
│                                          │
│  Metrics Computed:                       │
│                                          │
│  ┌────────────────────────────────┐     │
│  │ FAITHFULNESS                   │     │
│  │ • Answer grounded in context?  │     │
│  │ • No hallucinations?           │     │
│  │ • Score: 0.0 - 1.0            │     │
│  └────────────────────────────────┘     │
│                                          │
│  ┌────────────────────────────────┐     │
│  │ ANSWER RELEVANCY               │     │
│  │ • Does answer address Q?       │     │
│  │ • Relevant to question?        │     │
│  │ • Score: 0.0 - 1.0            │     │
│  └────────────────────────────────┘     │
│                                          │
│  ┌────────────────────────────────┐     │
│  │ CONTEXT PRECISION              │     │
│  │ • Is all retrieved context OK? │     │
│  │ • No irrelevant chunks?        │     │
│  │ • Score: 0.0 - 1.0            │     │
│  └────────────────────────────────┘     │
│                                          │
│  ┌────────────────────────────────┐     │
│  │ CONTEXT RECALL                 │     │
│  │ • Did we retrieve all needed?  │     │
│  │ • Sufficient context retrieved?│     │
│  │ • Score: 0.0 - 1.0            │     │
│  └────────────────────────────────┘     │
│                                          │
│  LLM Used: OpenRouter (evaluator)       │
│  Embeddings: HuggingFace (for scoring)  │
│                                          │
│  Output: Metric scores per question     │
│                                          │
└──────────────────────────────────────────┘
```

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        TECH STACK                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FRAMEWORK LAYER                                                │
│  ├─ LangChain (0.3)         - Text processing & chain          │
│  ├─ LangGraph (0.2)         - State machine orchestration      │
│  └─ RAGAS (0.2)             - LLM evaluation framework         │
│                                                                 │
│  EMBEDDING & VECTOR LAYER                                       │
│  ├─ HuggingFace Transformers - Pre-trained embeddings         │
│  ├─ sentence-transformers   - all-MiniLM-L6-v2 model          │
│  └─ Chroma (0.5)            - Vector database                 │
│                                                                 │
│  LLM LAYER                                                       │
│  ├─ OpenRouter API          - LLM provider (free-tier)         │
│  ├─ NVIDIA Nemotron         - Free model                       │
│  └─ LangChain OpenAI        - LLM abstraction                  │
│                                                                 │
│  WEB FRAMEWORK                                                   │
│  ├─ Streamlit (1.x)         - UI framework                     │
│  └─ Python (3.9+)           - Runtime                          │
│                                                                 │
│  UTILITIES                                                       │
│  ├─ datasets (3.x)          - Data handling                    │
│  ├─ python-dotenv           - Config management                │
│  └─ docx2txt                - DOCX parsing                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
        📄 Input Document
              │
              ▼
        ┌──────────────┐
        │ Load Document│
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Split Chunks │
        └──────┬───────┘
               │
         ┌─────┴─────┐
         │ Batch 1   │ Batch N
         ▼           ▼
    ┌────────────────────────┐
    │   Embed (Local)        │
    │   384-dim vectors      │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  Store in Chroma       │
    │  Vector DB             │
    │  (Persistent)          │
    └────────┬───────────────┘
             │
        ❓ User Question
             │
             ▼
    ┌────────────────────────┐
    │ 1. Embed Question      │
    │ 2. Retrieve Top-4      │
    │ 3. Format Context      │
    └────────┬───────────────┘
             │
         (Context)
             │
             ▼
    ┌────────────────────────┐
    │   LangGraph Pipeline   │
    │   ┌────────────────┐   │
    │   │ Retrieve Node  │   │
    │   └────────┬───────┘   │
    │            │           │
    │   ┌────────▼───────┐   │
    │   │ Generate Node  │   │
    │   └────────┬───────┘   │
    │            │           │
    └────────────┼───────────┘
                 │
            💬 Answer
                 │
        ┌────────┴────────┐
        │                 │
    (Development)   (Production)
        │                 │
        ▼                 ▼
    ┌────────────┐  ┌──────────────┐
    │ Evaluation │  │   Web UI     │
    │  (RAGAS)   │  │  (Streamlit) │
    └────┬───────┘  └──────┬───────┘
         │                 │
         ▼                 ▼
    📊 Metrics         📥 Upload
    📈 Scores          📤 Download
    🔍 Analysis        ✅ Results
```

---

## Scalability & Performance

```
┌─────────────────────────────────────────────────────────────────┐
│              PERFORMANCE CHARACTERISTICS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EMBEDDING GENERATION                                           │
│  • Speed: ~50-100 docs/sec (local)                             │
│  • Memory: ~2GB for all models                                 │
│  • Scalability: Linear with document count                     │
│                                                                 │
│  RETRIEVAL (Per Query)                                          │
│  • Speed: <100ms (Chroma search)                               │
│  • Scalability: O(log n) with indexing                         │
│  • Throughput: 1000+ qps                                       │
│                                                                 │
│  LLM GENERATION (Per Query)                                     │
│  • Speed: 2-5 seconds (OpenRouter)                             │
│  • Bottleneck: API latency                                     │
│  • Rate Limit: Free-tier caps (~20 req/min)                    │
│                                                                 │
│  RAGAS EVALUATION                                               │
│  • 1 Question: ~5-10 seconds (4 metrics)                       │
│  • 5 Questions: ~25-50 seconds                                 │
│  • Cost: ~20 LLM calls total                                   │
│                                                                 │
│  STORAGE                                                         │
│  • Vector DB: ~500MB per 10K documents                         │
│  • Metadata: Minimal overhead                                  │
│  • Persistence: Disk-based (durable)                           │
│                                                                 │
│  DEPLOYMENT OPTIONS                                             │
│  • Local: Single machine (tested)                              │
│  • Cloud: Streamlit Cloud, AWS Lambda, Docker                  │
│  • Scaling: Distributed Chroma, LB for API calls              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Free-Tier Cost Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│                     COST ANALYSIS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  COMPONENT          │  COST      │  NOTES                      │
│  ──────────────────┼────────────┼────────────────────────────  │
│  Embeddings        │  $0.00     │ ✓ Local (HuggingFace)       │
│  Vector DB         │  $0.00     │ ✓ Chroma (local)            │
│  LLM Inference     │  $0.00     │ ✓ OpenRouter free-tier      │
│  RAGAS Eval        │  $0.00     │ ✓ Uses free LLM             │
│  Web Hosting       │  $0.00     │ ✓ Streamlit Community       │
│  ──────────────────┼────────────┼────────────────────────────  │
│  TOTAL COST        │  $0.00     │ 100% FREE                   │
│                                                                 │
│  RATE LIMITS (Free-tier):                                      │
│  • OpenRouter: ~20 requests/minute                             │
│  • Daily cap: ~1500 requests/day                               │
│  • Handled by: Configurable delays                             │
│                                                                 │
│  RECOMMENDATIONS:                                               │
│  • 5-10 test questions: ✓ Comfortable                          │
│  • 50+ test questions: ⚠ Plan for delays                       │
│  • 1000+ documents: ✓ No additional cost                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

```
┌──────────────────────────────────────────────────────────────────┐
│                   INTEGRATION ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EXTERNAL SERVICES                                               │
│                                                                  │
│  ┌─────────────────┐         ┌──────────────────┐               │
│  │  OpenRouter API │         │  HuggingFace Hub │               │
│  │  (Free LLM)     │         │ (Model Hosting)  │               │
│  └────────┬────────┘         └────────┬─────────┘               │
│           │                           │                         │
│           │  Generate responses       │  Download embeddings    │
│           │  Rate limited             │  Cached locally         │
│           │                           │                         │
│           └──────────────┬────────────┘                         │
│                          │                                      │
│                    ┌─────▼──────────┐                          │
│                    │  RAG Pipeline  │                          │
│                    └─────┬──────────┘                          │
│                          │                                      │
│         ┌────────────────┼────────────────┐                    │
│         │                │                │                    │
│         ▼                ▼                ▼                    │
│  ┌────────────┐   ┌────────────┐   ┌─────────────┐            │
│  │ File System│   │Local DB    │   │ Web Browser │            │
│  │ (Docs)     │   │ (Chroma)   │   │(Streamlit)  │            │
│  └────────────┘   └────────────┘   └─────────────┘            │
│                                                                  │
│  DATA PERSISTENCE                                                │
│  • chroma_db/     - Vector embeddings & metadata               │
│  • data/          - Input documents & test cases               │
│  • .env           - Configuration & API keys                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT OPTIONS                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OPTION 1: LOCAL DEVELOPMENT                                    │
│  ┌───────────────────────────────────────┐                     │
│  │ Local Machine                         │                     │
│  │  ├─ Virtual Environment               │                     │
│  │  ├─ Chroma DB (disk)                  │                     │
│  │  ├─ Python Runtime                    │                     │
│  │  └─ Streamlit Server                  │                     │
│  │     └─ http://localhost:8501          │                     │
│  └───────────────────────────────────────┘                     │
│                                                                  │
│  OPTION 2: CLOUD DEPLOYMENT                                     │
│  ┌──────────────────┐      ┌────────────┐                      │
│  │ Streamlit Cloud  │      │ AWS Lambda │                      │
│  │ • Free tier      │      │ • Serverless                      │
│  │ • GitHub sync    │      │ • Auto-scale                      │
│  │ • 1 GB storage   │      │ • Pay-per-use                     │
│  └──────────────────┘      └────────────┘                      │
│                                                                  │
│  OPTION 3: CONTAINERIZED                                        │
│  ┌──────────────────────────────────────┐                      │
│  │ Docker Container                     │                      │
│  │  ├─ Dockerfile                       │                      │
│  │  ├─ docker-compose.yml               │                      │
│  │  └─ Deploy to: K8s, ECS, Heroku     │                      │
│  └──────────────────────────────────────┘                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Module Dependencies

```
app.py (Streamlit Web UI)
 ├─ streamlit
 ├─ run_rag_tests.py
 ├─ vector_store.py (build_vector_store, reset_vector_store)
 ├─ document_loader.py (LOADER_MAP, load_document)
 └─ chunker.py (split_into_chunks)

rag_graph.py (LangGraph RAG Pipeline)
 ├─ langgraph
 ├─ langchain_core
 ├─ retriever.py (get_retriever)
 └─ llm.py (get_llm)

run_rag_tests.py (Test Runner)
 ├─ rag_graph.py (rag_app)
 ├─ json
 ├─ pathlib
 └─ time

evaluate_ragas.py (RAGAS Evaluation)
 ├─ ragas
 ├─ datasets
 ├─ llm.py (get_llm)
 ├─ embeddings.py (get_embeddings)
 └─ json

vector_store.py (Vector DB Management)
 ├─ langchain_chroma
 ├─ embeddings.py (get_embeddings)
 ├─ document_loader.py (load_document)
 ├─ chunker.py (split_into_chunks)
 └─ config.py

retriever.py (Similarity Search)
 └─ vector_store.py (load_vector_store)

chunker.py (Text Splitting)
 └─ document_loader.py (load_document)

embeddings.py (Vector Generation)
 └─ config.py (EMBEDDING_MODEL)

document_loader.py (Document Ingestion)
 └─ config.py (SOURCE_DOCX)

llm.py (LLM Interface)
 └─ config.py (LLM_MODEL, OPENROUTER_API_KEY)

config.py (Configuration)
 └─ .env (environment variables)
```

---

## Summary

**RAG QA Project** is a production-ready, free-tier optimized Retrieval-Augmented Generation system featuring:

- ✅ **Multi-format document processing** (5+ file types)
- ✅ **Local embeddings** (zero API cost)
- ✅ **Vector persistence** (Chroma DB)
- ✅ **LangGraph orchestration** (Retrieve → Generate)
- ✅ **LLM integration** (OpenRouter free models)
- ✅ **RAGAS evaluation** (4 quality metrics)
- ✅ **Web interface** (Streamlit)
- ✅ **100% free deployment** (open source)

**Perfect for**:
- RAG pipeline demonstrations
- Document QA systems
- Evaluation frameworks
- Educational projects
- Cost-conscious production deployments
