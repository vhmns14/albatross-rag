"""
Unit tests for Albatross RAG engine and telemetry.
Skeptical verification: tests BM25 exact match, RRF rank fusion, and latency stopwatch.
"""

import pytest
from core.engine import AlbatrossEngine, DocumentChunk
from core.telemetry import QueryTelemetry, TelemetryTimer, estimate_tokens

SAMPLE_DOC = """
Laporan Tahunan PT Teknologi Nusantara 2024.
Bab 1: Kinerja Finansial.
Pada kuartal ketiga tahun 2024, rasio solvabilitas perusahaan tercatat sebesar 1.45x, meningkat dari 1.20x pada periode yang sama tahun sebelumnya. Total liabilitas jangka panjang adalah Rp 450 Miliar, sedangkan ekuitas tercatat Rp 650 Miliar.

Bab 2: Kebijakan Privasi dan Kepatuhan UU PDP.
Sesuai Pasal 28 UU Perlindungan Data Pribadi, retensi data pengguna wajib dihapus setelah 5 tahun masa retensi berakhir atau atas permintaan tertulis dari pemilik data.
"""

def test_chunking_and_indexing():
    engine = AlbatrossEngine(chunk_size=300, chunk_overlap=50)
    count = engine.load_text(SAMPLE_DOC, source_name="laporan_2024.txt")
    assert count > 0, "Engine should create at least one chunk"
    assert len(engine.chunks) == count
    assert engine.bm25 is not None
    assert engine.dense_embeddings is not None
    assert engine.dense_embeddings.shape[0] == count

def test_sparse_bm25_exact_keyword():
    engine = AlbatrossEngine(chunk_size=300, chunk_overlap=50)
    engine.load_text(SAMPLE_DOC, source_name="laporan_2024.txt")

    # BM25 must find "solvabilitas 1.45x"
    results = engine.retrieve("rasio solvabilitas 1.45x", mode="hybrid", top_k=2)
    assert len(results) > 0
    top_text = results[0].text.lower()
    assert "solvabilitas" in top_text
    assert "1.45x" in top_text

def test_hybrid_vs_naive_retrieval():
    engine = AlbatrossEngine(chunk_size=300, chunk_overlap=50)
    engine.load_text(SAMPLE_DOC, source_name="laporan_2024.txt")

    # Query with specific legal article
    query = "Pasal 28 retensi data 5 tahun"
    naive_results = engine.retrieve(query, mode="naive", top_k=2)
    hybrid_results = engine.retrieve(query, mode="hybrid", top_k=2)

    assert len(naive_results) > 0
    assert len(hybrid_results) > 0
    top_hybrid_text = hybrid_results[0].text
    assert "Pasal 28" in top_hybrid_text

def test_telemetry_timer_and_cost():
    telemetry = QueryTelemetry(model="llama-3.3-70b-versatile", mode="hybrid")
    with TelemetryTimer("Embedding Step", telemetry):
        x = sum(i * i for i in range(1000))

    assert len(telemetry.timings) == 1
    assert telemetry.timings[0].step_name == "Embedding Step"
    assert telemetry.timings[0].duration_ms >= 0.0

    telemetry.input_tokens = 1000
    telemetry.output_tokens = 500
    cost = telemetry.calculate_cost()
    assert cost > 0.0
    assert estimate_tokens("Hello world from Albatross") > 0
