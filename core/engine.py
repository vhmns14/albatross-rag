"""
engine.py - Core Hybrid Retrieval Pipeline for Albatross RAG.
Security & Concurrency Hardened:
  - Atomic index synchronization under lock
  - Infinite-loop prevention in chunking (step >= 1)
  - Dimension alignment auto-guard on vector dot products
  - Robust zero-result and empty query handling
"""

import os
import re
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from pypdf import PdfReader

from .providers import get_embeddings, get_lightweight_embedding, LLMProvider
from .telemetry import QueryTelemetry, TelemetryTimer, estimate_tokens

@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    source: str
    page: int
    token_count: int
    score: float = 0.0

def split_text_recursive(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Split text respecting sentence boundaries with guaranteed positive step size."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if len(text) > 15 else []

    # Safeguard against overlap >= chunk_size
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)
    step = max(1, chunk_size - chunk_overlap)

    paragraphs = re.split(r'(\n\n+|\.\s+)', text)
    chunks = []
    current = ""

    for part in paragraphs:
        if not part:
            continue
        if len(current) + len(part) <= chunk_size:
            current += part
        else:
            if len(current.strip()) > 15:
                chunks.append(current.strip())
            current = (current[-chunk_overlap:] if len(current) > chunk_overlap else "") + part

    if len(current.strip()) > 15:
        chunks.append(current.strip())

    final_chunks = []
    for c in chunks:
        if len(c) > chunk_size * 1.5:
            start = 0
            while start < len(c):
                end = min(start + chunk_size, len(c))
                final_chunks.append(c[start:end].strip())
                start += step
        else:
            final_chunks.append(c)

    return [c for c in final_chunks if len(c) > 15]

class AlbatrossEngine:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 4)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.corpus_tokenized: List[List[str]] = []
        self._lock = threading.Lock()

    def clear(self):
        """Reset indexed knowledge safely under lock."""
        with self._lock:
            self.chunks.clear()
            self.bm25 = None
            self.dense_embeddings = None
            self.corpus_tokenized.clear()

    # ---------------------------------------------------------
    # 1. Ingestion & Chunking (Thread-Safe)
    # ---------------------------------------------------------
    def load_pdf(self, file_path_or_bytes, source_name: str = "document.pdf") -> int:
        """Extract text from PDF and index into chunks."""
        reader = PdfReader(file_path_or_bytes)
        raw_items = []

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r'[ \t]+', ' ', text).strip()
            if not text:
                continue

            page_chunks = split_text_recursive(text, self.chunk_size, self.chunk_overlap)
            for c_text in page_chunks:
                raw_items.append((c_text, page_idx + 1))

        return self._index_raw_items(raw_items, source_name)

    def load_text(self, text: str, source_name: str = "document.txt") -> int:
        """Index raw string or TXT file."""
        text = re.sub(r'[ \t]+', ' ', text).strip()
        if not text:
            return 0

        text_chunks = split_text_recursive(text, self.chunk_size, self.chunk_overlap)
        raw_items = [(c_text, 1) for c_text in text_chunks]

        return self._index_raw_items(raw_items, source_name)

    def _index_raw_items(self, items: List[Tuple[str, int]], source_name: str) -> int:
        """Atomically create chunks with sequential IDs and rebuild index under lock."""
        if not items:
            return 0

        with self._lock:
            base_id = len(self.chunks)
            new_chunks = []
            for offset, (c_text, page) in enumerate(items):
                new_chunks.append(DocumentChunk(
                    chunk_id=base_id + offset,
                    text=c_text,
                    source=source_name,
                    page=page,
                    token_count=estimate_tokens(c_text)
                ))

            self.chunks.extend(new_chunks)
            # Rebuild sparse & dense indices
            self.corpus_tokenized = [c.text.lower().split() for c in self.chunks]
            self.bm25 = BM25Okapi(self.corpus_tokenized)

            texts = [c.text for c in self.chunks]
            self.dense_embeddings = get_embeddings(texts)

            return len(new_chunks)

    def delete_document(self, source_name: str) -> int:
        """Remove all chunks belonging to a document and rebuild indices."""
        with self._lock:
            self.chunks = [c for c in self.chunks if c.source != source_name]
            if not self.chunks:
                self.bm25 = None
                self.dense_embeddings = None
                self.corpus_tokenized.clear()
            else:
                self.corpus_tokenized = [c.text.lower().split() for c in self.chunks]
                self.bm25 = BM25Okapi(self.corpus_tokenized)
                texts = [c.text for c in self.chunks]
                self.dense_embeddings = get_embeddings(texts)
            return len(self.chunks)

    # ---------------------------------------------------------
    # 2. Retrieval Strategies (Thread-Safe & Dimension-Guarded)
    # ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 4,
        telemetry: Optional[QueryTelemetry] = None
    ) -> List[DocumentChunk]:
        """
        Execute retrieval pipeline safely:
        - Dimension auto-alignment prevents numpy dot product crashes
        - Zero-results protection
        """
        query = (query or "").strip()
        if not query:
            return []

        with self._lock:
            if not self.chunks or self.dense_embeddings is None or self.bm25 is None:
                return []
            chunks_snapshot = list(self.chunks)
            dense_snapshot = self.dense_embeddings
            bm25_snapshot = self.bm25

        # Step 1: Query Embedding
        with TelemetryTimer("Query Embedding", telemetry if telemetry else QueryTelemetry("", "")):
            expected_dim = dense_snapshot.shape[1] if dense_snapshot.ndim == 2 else 1024
            query_vecs = get_embeddings([query], expected_dim=expected_dim)
            if len(query_vecs) == 0:
                return []
            query_vec = query_vecs[0]

            # Auto-align dimensions if unexpected mismatch occurs
            if query_vec.shape[0] != dense_snapshot.shape[1]:
                query_vec = get_lightweight_embedding(query, dim=dense_snapshot.shape[1])

        if mode == "naive":
            # --- NAIVE RAG: Pure Dense Vector Search ---
            with TelemetryTimer("Dense Vector Search", telemetry if telemetry else QueryTelemetry("", "")):
                scores = np.dot(dense_snapshot, query_vec)
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    chunk = chunks_snapshot[idx]
                    results.append(DocumentChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        source=chunk.source,
                        page=chunk.page,
                        token_count=chunk.token_count,
                        score=round(float(scores[idx]), 4)
                    ))
                return results

        # --- ADVANCED HYBRID RAG ---
        # Step 2: Dense Top-20
        with TelemetryTimer("Dense Search (Vector)", telemetry if telemetry else QueryTelemetry("", "")):
            dense_scores = np.dot(dense_snapshot, query_vec)
            dense_top_indices = np.argsort(dense_scores)[::-1][:20]

        # Step 3: Sparse Top-20 (BM25)
        with TelemetryTimer("Sparse Search (BM25)", telemetry if telemetry else QueryTelemetry("", "")):
            tokenized_query = query.lower().split()
            bm25_scores = bm25_snapshot.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:20]

        # Step 4: Reciprocal Rank Fusion (RRF)
        with TelemetryTimer("RRF Fusion", telemetry if telemetry else QueryTelemetry("", "")):
            rrf_k = 60
            rrf_scores: Dict[int, float] = {}

            for rank, idx in enumerate(dense_top_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

            for rank, idx in enumerate(bm25_top_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

            fused_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:15]

        # Step 5: Cross-Score Re-ranking
        with TelemetryTimer("Cross-Score Re-ranker", telemetry if telemetry else QueryTelemetry("", "")):
            reranked_chunks = []
            query_terms = set(re.findall(r'\w+', query.lower()))

            for idx, rrf_score in fused_candidates:
                chunk = chunks_snapshot[idx]
                chunk_terms = set(re.findall(r'\w+', chunk.text.lower()))
                term_overlap = len(query_terms.intersection(chunk_terms)) / max(1, len(query_terms))
                
                dense_sim = float(dense_scores[idx])
                final_score = (0.5 * dense_sim) + (0.3 * term_overlap) + (0.2 * (rrf_score * 10))
                
                chunk_copy = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    page=chunk.page,
                    token_count=chunk.token_count,
                    score=round(final_score, 4)
                )
                reranked_chunks.append(chunk_copy)

            reranked_chunks.sort(key=lambda x: x.score, reverse=True)
            return reranked_chunks[:top_k]

    # ---------------------------------------------------------
    # 3. Prompt Construction & Grounding
    # ---------------------------------------------------------
    def format_context(self, chunks: List[DocumentChunk]) -> str:
        """Format chunks into structured context block with citations."""
        if not chunks:
            return "No relevant context found in the document."
        
        context_parts = []
        for c in chunks:
            context_parts.append(
                f"--- [SOURCE CHUNK #{c.chunk_id} | Page {c.page} | File: {c.source}] ---\n{c.text}"
            )
        return "\n\n".join(context_parts)

    def build_prompt(self, query: str, chunks: List[DocumentChunk]) -> Tuple[str, str]:
        """Construct grounded system and user prompts."""
        system_prompt = (
            "You are Albatross RAG, a high-precision factual assistant. "
            "Your task is to answer the user question STRICTLY using the provided context snippets. "
            "RULES:\n"
            "1. Do not assume or extrapolate beyond facts explicitly mentioned.\n"
            "2. Whenever you state a claim or number, cite the chunk source (e.g. [Chunk #42, Page 2]).\n"
            "3. If the context does not contain sufficient facts to answer, explicitly state: "
            "'Informasi tidak ditemukan dalam dokumen yang tersedia.'"
        )

        context_block = self.format_context(chunks)
        user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION: {query}\n\nGROUNDED ANSWER:"
        return system_prompt, user_prompt
