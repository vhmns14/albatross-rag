# PRD: Albatross RAG
**Observable, Self-Evaluating Hybrid Retrieval Workbench**

## 1. Problem Statement
Most real-world RAG applications fail in production due to two blind spots:
1. **Retrieval Blindness**: Pure vector search misses exact terms, identifiers, and table data; pure keyword search misses semantic context.
2. **Evaluation Blindness**: Engineers have no quantitative proof of whether their system hallucinates or if new prompts/chunks improve accuracy.

Albatross RAG solves both problems by coupling a high-precision Hybrid Retrieval Pipeline (BM25 + Dense Vectors + Cross-Encoder Reranking) with built-in Latency/Cost Telemetry and Automated RAGAS-style Evaluations.

---

## 2. Ponytail Philosophy & Guardrails
> *"The best code is the code never written. YAGNI first, stdlib/minimal dependencies, zero boilerplate scaffolding, fewest files possible, concrete proof over claims."*

### What We Intentionally Build:
1. **Hybrid Retrieval**: BM25 (sparse) + Dense Vector Search + Reciprocal Rank Fusion (RRF).
2. **Re-ranker Stage**: Cross-encoder scoring to filter Top-K candidates into high-fidelity context.
3. **Pluggable Multi-LLM**: Direct integration with **Claude Opus**, **Mistral**, and **Groq (Free Llama 3.3)**.
4. **Latency & Cost Telemetry**: Sub-millisecond step-by-step waterfall (Embedding, Retrieval, Rerank, Generation) + Token Cost tracker.
5. **Automated Evaluation**: 4 metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall) with comparison scorecard.
6. **Custom 3-Column Studio UI**: Knowledge & Chunks Manager, Reasoning Canvas, and Under-The-Hood Inspector.

### Ponytail Ceilings (YAGNI - Explicitly NOT Built):
- ❌ **No External Daemon DBs**: No Dockerized Milvus/Elasticsearch/Qdrant servers. We use embedded in-process storage (Chroma persistent client / SQLite vector). Upgrade path: switch to hosted cloud vector DB if document volume > 1,000,000 chunks.
- ❌ **No Heavy Framework Bloat**: No 15-layer LangChain chain abstractions. Direct, readable, composable Python functions.
- ❌ **No Multi-Tenancy / Auth**: Single-workspace engineer studio. Add auth only when converting to public SaaS.
- ❌ **No Node/NPM build overhead**: 100% Python stack (Streamlit + custom modern dark CSS) to keep laptop RAM consumption strictly under control.

---

## 3. Core Architecture & Data Flow

```
[Uploaded Document (PDF / TXT)]
           │
           ▼
[Semantic / Recursive Chunker]
           │
     ┌─────┴────────────────┐
     ▼                      ▼
[Dense Embeddings]     [BM25 Index]
(FastEmbed / BGE)      (rank_bm25)
     └─────┬────────────────┘
           │ (Top 20 candidates)
           ▼
[Reciprocal Rank Fusion (RRF)]
           │
           ▼
[Cross-Encoder Reranker] (Top 3-5 candidates)
           │
     ┌─────┴────────────────┐
     ▼                      ▼
[LLM Generation Stream] [Telemetry & Waterfall Timer]
(Claude / Mistral / Groq) (Embedding, Search, Rerank, Gen)
           │
           ▼
[Evaluation Engine (RAGAS Metrics: Faithfulness, Relevance, Precision)]
```

---

## 4. Key Metrics & Acceptance Criteria

| Metric | Target | Measurement Method |
| :--- | :--- | :--- |
| **Context Precision** | ≥ 85% | RAGAS metric comparing retrieved chunks against ground truth |
| **Faithfulness (Anti-Hallucination)** | ≥ 90% | Verification that output statements are strictly grounded |
| **Total Query Latency (Groq)** | < 1.5s | Telemetry waterfall stopwatch |
| **Memory Footprint** | < 1.0 GB RAM | In-process local execution |

---

## 5. File Structure
```
albatross-rag/
├── PRD.md
├── requirements.txt
├── .env.example
├── core/
│   ├── engine.py       # Hybrid retrieval & ingestion
│   ├── providers.py    # Multi-LLM dispatcher (Claude, Mistral, Groq)
│   └── telemetry.py    # Latency waterfall & cost tracking
├── evals/
│   ├── benchmark.py    # RAGAS metrics & scorecard generator
│   └── dataset.json    # Ground truth test dataset
├── app.py              # 3-column Studio Workbench UI
└── tests/
    └── test_retrieval.py # Sanity & ranking unit tests
```
