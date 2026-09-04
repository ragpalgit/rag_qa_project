# RAG QA Architecture - Visual Diagram

## High-Level Architecture Flow

```mermaid
graph TD
    A["📄 Input Documents<br/>(DOCX/PDF/TXT/MD/CSV)"] --> B["Document Loader<br/>(Multi-format)"]
    B --> C["Text Chunker<br/>(800 chars + overlap)"]
    C --> D["Local Embeddings<br/>(HuggingFace<br/>sentence-transformers)"]
    D --> E["Chroma Vector Store<br/>(Persistent DB)"]
    
    F["❓ User Question<br/>(Web UI / CLI)"] --> G["Embed Question<br/>(Local)"]
    G --> H["Retriever<br/>(Cosine Similarity<br/>Top-4 Chunks)"]
    H --> I["LangGraph Pipeline"]
    
    E --> H
    
    I --> J["Retrieve Node<br/>(Format Context)"]
    J --> K["Generate Node<br/>(OpenRouter LLM<br/>Free-tier)"]
    K --> L["💬 Grounded Answer<br/>(Context-based)"]
    
    L --> M["RAGAS Evaluation<br/>4 Metrics"]
    M --> N["📊 Results"]
    N --> O["JSON Dataset<br/>(eval_dataset.json)"]
    
    L --> P["Streamlit Web UI<br/>(Display/Download)"]
    
    style A fill:#e1f5ff
    style B fill:#b3e5fc
    style C fill:#81d4fa
    style D fill:#4fc3f7
    style E fill:#29b6f6
    style F fill:#e1f5ff
    style G fill:#81d4fa
    style H fill:#29b6f6
    style I fill:#0288d1
    style J fill:#0277bd
    style K fill:#01579b
    style L fill:#e8f5e9
    style M fill:#c8e6c9
    style N fill:#81c784
    style O fill:#558b2f
    style P fill:#fff3e0
```

## Component Architecture

```mermaid
graph LR
    subgraph Input["🔵 INPUT LAYER"]
        I1["User<br/>Documents"]
        I2["Test<br/>Cases"]
        I3["Web<br/>Upload"]
    end
    
    subgraph Processing["🟢 PROCESSING LAYER"]
        P1["Document<br/>Loader"]
        P2["Chunker"]
        P3["Embeddings<br/>Generator"]
    end
    
    subgraph Storage["🟡 STORAGE LAYER"]
        S1["Chroma<br/>Vector DB"]
        S2["Metadata<br/>Store"]
    end
    
    subgraph Retrieval["🟠 RETRIEVAL LAYER"]
        R1["Retriever"]
        R2["Similarity<br/>Search"]
    end
    
    subgraph Generation["🔴 GENERATION LAYER"]
        G1["LangGraph<br/>Orchestrator"]
        G2["LLM<br/>OpenRouter"]
    end
    
    subgraph Evaluation["🟣 EVALUATION LAYER"]
        E1["RAGAS<br/>Metrics"]
        E2["Score<br/>Calculator"]
    end
    
    subgraph Output["🔵 OUTPUT LAYER"]
        O1["Results<br/>JSON"]
        O2["Web<br/>Dashboard"]
    end
    
    Input --> Processing
    Processing --> Storage
    Storage --> Retrieval
    Retrieval --> Generation
    Generation --> Evaluation
    Evaluation --> Output
    
    style Input fill:#e3f2fd
    style Processing fill:#f1f8e9
    style Storage fill:#fff8e1
    style Retrieval fill:#ffe0b2
    style Generation fill:#ffccbc
    style Evaluation fill:#f3e5f5
    style Output fill:#e0f2f1
```

## Data Flow - Question to Answer

```mermaid
sequenceDiagram
    participant User as User/App
    participant Embed as Embeddings
    participant Retriever
    participant VectorDB as Chroma DB
    participant RAG as LangGraph
    participant LLM as OpenRouter LLM
    participant RAGAS as RAGAS Eval

    User->>Embed: Question text
    Embed->>Embed: Embed question<br/>(local, 384-dim)
    
    Embed->>Retriever: Embedded vector
    Retriever->>VectorDB: Query similar chunks
    VectorDB->>Retriever: Top-4 chunks
    
    Retriever->>RAG: Question + Context
    
    RAG->>RAG: Retrieve Node
    RAG->>LLM: Format prompt
    LLM->>LLM: Generate response
    LLM->>RAG: Grounded answer
    
    RAG->>User: Answer
    
    RAG->>RAGAS: Question + Answer<br/>+ Context + Reference
    RAGAS->>RAGAS: Compute 4 metrics
    RAGAS->>User: Scores & Metrics
```

## Technology Stack

```mermaid
graph TB
    subgraph Core["CORE FRAMEWORKS"]
        CF1["LangChain 0.3<br/>LangGraph 0.2<br/>RAGAS 0.2"]
    end
    
    subgraph Embeddings["EMBEDDINGS & VECTORS"]
        EMB1["HuggingFace<br/>Transformers"]
        EMB2["sentence-transformers<br/>all-MiniLM-L6-v2"]
        EMB3["Chroma 0.5<br/>Vector DB"]
    end
    
    subgraph LLM["LLM SERVICES"]
        LLM1["OpenRouter API"]
        LLM2["NVIDIA Nemotron<br/>Free-tier"]
    end
    
    subgraph Web["WEB INTERFACE"]
        WEB1["Streamlit 1.x"]
        WEB2["Python 3.9+"]
    end
    
    subgraph Utils["UTILITIES"]
        U1["datasets"]
        U2["python-dotenv"]
        U3["docx2txt"]
    end
    
    Core --> Embeddings
    Core --> LLM
    Core --> Web
    Embeddings --> Utils
    LLM --> Utils
    
    style Core fill:#1565c0
    style Embeddings fill:#6a1b9a
    style LLM fill:#c62828
    style Web fill:#f57f17
    style Utils fill:#004d40
```

## Deployment Architecture

```mermaid
graph TD
    subgraph Local["LOCAL DEVELOPMENT"]
        L1["Virtual Env"]
        L2["Local Files"]
        L3["Chroma DB<br/>disk"]
        L4["Streamlit<br/>:8501"]
    end
    
    subgraph Cloud["CLOUD DEPLOYMENT"]
        C1["Streamlit<br/>Cloud"]
        C2["AWS Lambda<br/>+ RDS"]
        C3["Docker<br/>Container"]
    end
    
    subgraph External["EXTERNAL SERVICES"]
        E1["OpenRouter<br/>LLM API"]
        E2["HuggingFace<br/>Hub"]
    end
    
    Local --> E1
    Local --> E2
    Cloud --> E1
    Cloud --> E2
    
    style Local fill:#c8e6c9
    style Cloud fill:#bbdefb
    style External fill:#ffe0b2
```

## Performance Metrics

```mermaid
graph LR
    subgraph Embedding["EMBEDDINGS"]
        E1["Speed: 50-100<br/>docs/sec<br/>Memory: 2GB"]
    end
    
    subgraph Retrieval["RETRIEVAL"]
        R1["Speed: &lt;100ms<br/>Throughput:<br/>1000+ qps"]
    end
    
    subgraph Generation["GENERATION"]
        G1["Speed: 2-5 sec<br/>Rate Limit:<br/>20 req/min"]
    end
    
    subgraph Evaluation["EVALUATION"]
        EV1["1Q: 5-10 sec<br/>5Q: 25-50 sec<br/>Calls: ~20"]
    end
    
    E1 --> R1 --> G1 --> EV1
    
    style E1 fill:#c8e6c9
    style R1 fill:#bbdefb
    style G1 fill:#ffe0b2
    style EV1 fill:#f8bbd0
```

## Cost Breakdown

```mermaid
pie title "RAG QA PROJECT TOTAL COST (Monthly)"
    "Embeddings": 0
    "Vector DB": 0
    "LLM (Free-tier)": 0
    "Web Hosting": 0
    "Other": 0
```

**Total Cost: $0.00 ✓ 100% FREE**

## Integration Points

```mermaid
graph TB
    subgraph RAG["RAG PIPELINE<br/>(Core System)"]
        RAG1["Document<br/>Processing"]
        RAG2["Embedding &<br/>Storage"]
        RAG3["Retrieval &<br/>Generation"]
        RAG4["Evaluation"]
    end
    
    subgraph External["EXTERNAL INTEGRATIONS"]
        EXT1["OpenRouter API<br/>(LLM Provider)"]
        EXT2["HuggingFace Hub<br/>(Model Hub)"]
        EXT3["File System<br/>(Data Store)"]
        EXT4["Web Browser<br/>(UI)"]
    end
    
    RAG1 -.-> EXT3
    RAG2 -.-> EXT2
    RAG3 -.-> EXT1
    RAG4 -.-> EXT1
    RAG -.-> EXT4
    
    style RAG fill:#e1f5fe
    style External fill:#fff3e0
```

## System Requirements

```mermaid
graph LR
    subgraph Minimal["MINIMAL"]
        MIN1["CPU: 2 cores<br/>RAM: 4 GB<br/>Storage: 10 GB"]
    end
    
    subgraph Recommended["RECOMMENDED"]
        REC1["CPU: 4+ cores<br/>RAM: 8+ GB<br/>Storage: 50 GB"]
    end
    
    subgraph Optimal["OPTIMAL"]
        OPT1["CPU: 8 cores<br/>RAM: 16+ GB<br/>GPU: CUDA"]
    end
    
    MIN1 --> REC1 --> OPT1
    
    style MIN1 fill:#ffccbc
    style REC1 fill:#ffe0b2
    style OPT1 fill:#fff9c4
```

## RAGAS Evaluation Metrics

```mermaid
graph TB
    subgraph RAGASMetrics["RAGAS EVALUATION FRAMEWORK"]
        M1["FAITHFULNESS<br/>Is answer grounded in context?<br/>No hallucinations?<br/>Score: 0.0-1.0"]
        
        M2["ANSWER RELEVANCY<br/>Does answer address question?<br/>Relevant to user intent?<br/>Score: 0.0-1.0"]
        
        M3["CONTEXT PRECISION<br/>Is all retrieved context needed?<br/>No irrelevant chunks?<br/>Score: 0.0-1.0"]
        
        M4["CONTEXT RECALL<br/>Were all necessary chunks retrieved?<br/>Sufficient information?<br/>Score: 0.0-1.0"]
    end
    
    style M1 fill:#c8e6c9
    style M2 fill:#bbdefb
    style M3 fill:#ffe0b2
    style M4 fill:#f8bbd0
```

