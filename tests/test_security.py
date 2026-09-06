"""
Security audit and vulnerability regression tests for Albatross RAG.
Deep Dive Tests:
  - Path Traversal prevention on upload
  - File size limits (DoS mitigation)
  - PDF magic header validation
  - Corrupted PDF exception handling (HTTP 400 instead of 500)
  - Disallowed file extensions
  - Empty text file upload rejection (HTTP 400)
  - Query length and enum schema validation
  - Secret redaction in error messages
  - Concurrency safety & zero-norm edge cases
  - Chunking step infinite-loop immunity
  - Vector dimension alignment guard (1024d vs 384d)
  - Empty dataset benchmark zero-division guard
"""

import os
import io
import pytest
import numpy as np
from fastapi.testclient import TestClient

from server import app, MAX_UPLOAD_SIZE
from core.engine import AlbatrossEngine, split_text_recursive
from core.providers import sanitize_error_message, get_lightweight_embedding, get_embeddings
from evals.benchmark import evaluate_metrics, run_benchmark

client = TestClient(app)

# -------------------------------------------------------------
# 1. Path Traversal & Upload Security
# -------------------------------------------------------------
def test_path_traversal_filename_sanitization():
    safe_text = b"Kinerja Finansial Perusahaan 2024: Rasio aman."
    malicious_filename = "../../../tmp/malicious_exploit.txt"
    
    response = client.post(
        "/api/upload",
        files={"file": (malicious_filename, io.BytesIO(safe_text), "text/plain")}
    )
    assert response.status_code == 200
    assert not os.path.exists("/tmp/malicious_exploit.txt")

def test_corrupted_pdf_returns_400_not_500():
    corrupted_pdf = b"%PDF-1.4 truncated corrupted stream"
    response = client.post(
        "/api/upload",
        files={"file": ("corrupted.pdf", io.BytesIO(corrupted_pdf), "application/pdf")}
    )
    assert response.status_code == 400
    assert "Corrupted or unreadable PDF" in response.json()["detail"]

def test_disallowed_file_extension():
    response = client.post(
        "/api/upload",
        files={"file": ("exploit.sh", io.BytesIO(b"echo pwned"), "text/x-sh")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_invalid_pdf_magic_bytes():
    disguised_content = b"This is plain text with no PDF header"
    response = client.post(
        "/api/upload",
        files={"file": ("fake.pdf", io.BytesIO(disguised_content), "application/pdf")}
    )
    assert response.status_code == 400
    assert "missing %PDF- header" in response.json()["detail"]

def test_oversized_file_rejection():
    oversized_data = b"A" * (MAX_UPLOAD_SIZE + 1024)
    response = client.post(
        "/api/upload",
        files={"file": ("large_doc.txt", io.BytesIO(oversized_data), "text/plain")}
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed size limit" in response.json()["detail"]

def test_empty_text_upload_rejection():
    response = client.post(
        "/api/upload",
        files={"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")}
    )
    assert response.status_code == 400
    assert "empty or contains only whitespace" in response.json()["detail"]

# -------------------------------------------------------------
# 2. Input Validation & Schema Enforcement
# -------------------------------------------------------------
def test_query_schema_validation():
    # 1. Query exceeding max length (2000 chars)
    oversized_query = "x" * 2500
    res = client.post("/api/query", json={"query": oversized_query, "mode": "hybrid", "provider": "opus"})
    assert res.status_code == 422

    # 2. Invalid mode
    res = client.post("/api/query", json={"query": "test query", "mode": "injected_mode", "provider": "opus"})
    assert res.status_code == 422

    # 3. Invalid provider
    res = client.post("/api/query", json={"query": "test query", "mode": "hybrid", "provider": "untrusted_ai"})
    assert res.status_code == 422

    # 4. Empty/whitespace query
    res = client.post("/api/query", json={"query": "   ", "mode": "hybrid", "provider": "opus"})
    assert res.status_code == 400 or res.status_code == 422

# -------------------------------------------------------------
# 3. Sensitive Credential Redaction
# -------------------------------------------------------------
def test_credential_sanitization():
    raw_error = "Failed connecting with Bearer sk-ant-api03-secretkey9991283912839 to endpoint."
    sanitized = sanitize_error_message(raw_error)
    assert "secretkey9991283912839" not in sanitized
    assert "[REDACTED]" in sanitized

# -------------------------------------------------------------
# 4. Deep Algorithmic Edge Cases
# -------------------------------------------------------------
def test_engine_empty_query_and_zero_vector():
    engine = AlbatrossEngine()
    res = engine.retrieve("", mode="hybrid")
    assert res == []

    res = engine.retrieve("   ", mode="hybrid")
    assert res == []

    vec = get_lightweight_embedding("")
    assert vec.shape == (1024,)
    assert not any(v != 0.0 for v in vec)

def test_infinite_loop_immunity_in_chunking():
    # If overlap >= chunk_size, step must remain strictly positive
    text = "A quick brown fox jumps over the lazy dog. " * 50
    chunks = split_text_recursive(text, chunk_size=100, chunk_overlap=100)
    assert len(chunks) > 0

    engine = AlbatrossEngine(chunk_size=100, chunk_overlap=150)
    assert engine.chunk_overlap < engine.chunk_size

def test_dimension_alignment_auto_guard():
    engine = AlbatrossEngine(chunk_size=200, chunk_overlap=30)
    engine.load_text("Artificial Intelligence and Machine Learning systems.", source_name="doc.txt")
    # Verify index has 1024 dimensions
    assert engine.dense_embeddings.shape[1] == 1024
    
    # Query must not raise shape mismatch
    results = engine.retrieve("AI and ML", mode="hybrid")
    assert len(results) > 0

def test_empty_dataset_benchmark_zero_division_guard():
    # Empty metrics evaluation
    metrics = evaluate_metrics([], [])
    assert metrics["context_precision"] == 0.0
    assert metrics["context_recall"] == 0.0
