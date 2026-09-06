"""
server.py - Production FastAPI Backend for Albatross RAG Workbench.
Security Hardened: Path traversal protection, file size limits, empty-content rejection,
input validation, thread safety, and safe CORS configuration.
"""

import os
import time
import json
import tempfile
from typing import Optional, List, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

from core.engine import AlbatrossEngine, DocumentChunk
from core.providers import LLMProvider
from core.telemetry import QueryTelemetry, TelemetryTimer, estimate_tokens
from evals.benchmark import run_benchmark, SAMPLE_CORPUS

# Maximum allowed upload size: 10 MB (prevents memory exhaustion on 16GB RAM laptops)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

app = FastAPI(
    title="Albatross RAG Studio API",
    version="1.3.1",
    docs_url="/api/docs",
    redoc_url=None
)

# Safe CORS: Disallow wildcard credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Global in-memory engine and session stats
engine = AlbatrossEngine(chunk_size=1000, chunk_overlap=150)
engine.load_text(SAMPLE_CORPUS, source_name="laporan_keuangan_2024.txt")

session_stats = {
    "total_queries": 0,
    "total_cost_usd": 0.0,
    "start_time": time.time(),
}

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Albatross RAG API is live</h1>")

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "chunks_indexed": len(engine.chunks),
        "embedding_model": "mistral-embed" if os.getenv("MISTRAL_API_KEY") else "local-hash-cpu",
        "default_model": os.getenv("DEFAULT_MODEL", "opus"),
        "total_queries": session_stats["total_queries"],
        "total_cost_usd": round(session_stats["total_cost_usd"], 6),
        "uptime_seconds": int(time.time() - session_stats["start_time"]),
    }

@app.get("/api/chunks")
async def get_chunks():
    return [
        {
            "chunk_id": c.chunk_id,
            "source": c.source,
            "page": c.page,
            "token_count": c.token_count,
            "snippet": c.text[:200] + ("..." if len(c.text) > 200 else ""),
            "full_text": c.text,
        }
        for c in engine.chunks
    ]

@app.get("/api/documents")
async def get_documents():
    counts = {}
    tokens = {}
    for c in engine.chunks:
        counts[c.source] = counts.get(c.source, 0) + 1
        tokens[c.source] = tokens.get(c.source, 0) + c.token_count
    return [
        {
            "name": name,
            "chunk_count": counts[name],
            "token_count": tokens[name],
            "is_demo": "(Demo)" in name or name == "laporan_keuangan_2024.txt"
        }
        for name in counts
    ]

@app.post("/api/documents/clear")
async def clear_documents():
    engine.clear()
    return {"status": "success", "message": "Knowledge base cleared", "chunks_count": 0}

class DeleteDocRequest(BaseModel):
    name: str

@app.post("/api/documents/delete")
async def delete_document_endpoint(req: DeleteDocRequest):
    remaining = engine.delete_document(req.name)
    return {"status": "success", "remaining_chunks": remaining}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate filename presence and extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing.")

    clean_filename = os.path.basename(file.filename)
    _, ext = os.path.splitext(clean_filename.lower())

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 2. Read with strict size limit to prevent memory DoS
    content = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size limit of {MAX_UPLOAD_SIZE // (1024*1024)}MB."
        )

    # 3. Process securely based on format
    count = 0
    if ext == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Invalid PDF structure (missing %PDF- header).")

        tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp_file.write(content)
            tmp_file.flush()
            tmp_file.close()
            try:
                count = engine.load_pdf(tmp_file.name, source_name=clean_filename)
            except Exception as err:
                raise HTTPException(status_code=400, detail=f"Corrupted or unreadable PDF: {str(err)}")
        finally:
            if os.path.exists(tmp_file.name):
                try:
                    os.remove(tmp_file.name)
                except Exception:
                    pass
    else:
        text_str = content.decode("utf-8", errors="ignore").strip()
        if not text_str:
            raise HTTPException(status_code=400, detail="Uploaded text file is empty or contains only whitespace.")
        count = engine.load_text(text_str, source_name=clean_filename)

    if count == 0:
        raise HTTPException(status_code=400, detail="No readable text chunks could be extracted from the document.")

    return {"status": "success", "added_chunks": count, "total_chunks": len(engine.chunks)}

@app.post("/api/reload-sample")
async def reload_sample():
    engine.clear()
    count = engine.load_text(SAMPLE_CORPUS, source_name="laporan_keuangan_2024.txt")
    return {"status": "success", "total_chunks": count}

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Analytical query text")
    mode: Literal["hybrid", "naive"] = "hybrid"
    provider: Literal["opus", "mistral", "groq"] = "opus"

@app.post("/api/query")
async def execute_query(req: QueryRequest):
    """Execute retrieval + streaming LLM generation via Server-Sent Events (SSE)."""
    sanitized_query = req.query.strip()
    if not sanitized_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty or whitespace.")

    telemetry = QueryTelemetry(
        model="claude-opus-4-6" if req.provider == "opus" else req.provider,
        mode=req.mode
    )
    t_start = time.perf_counter()

    # Step 1: Retrieval (Dynamic Top-K for exhaustive requests)
    is_exhaustive = any(w in sanitized_query.lower() for w in ["semua", "seluruh", "daftar", "list", "lengkap", "siapa saja"])
    target_top_k = min(len(engine.chunks), 15) if is_exhaustive else min(len(engine.chunks), 8)
    retrieved_chunks = engine.retrieve(sanitized_query, mode=req.mode, top_k=max(3, target_top_k), telemetry=telemetry)
    sys_prompt, user_prompt = engine.build_prompt(sanitized_query, retrieved_chunks)
    telemetry.input_tokens = estimate_tokens(sys_prompt + user_prompt)

    def sse_generator():
        nonlocal telemetry
        try:
            initial_meta = {
                "type": "meta",
                "retrieved_chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "page": c.page,
                        "source": c.source,
                        "score": c.score,
                        "text": c.text,
                    }
                    for c in retrieved_chunks
                ],
                "mode": req.mode,
                "provider": req.provider,
            }
            yield f"data: {json.dumps(initial_meta)}\n\n"

            t_gen_start = time.perf_counter()
            full_text = ""
            stream = LLMProvider.generate_stream(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                provider=req.provider
            )

            for chunk in stream:
                full_text += chunk
                yield f"data: {json.dumps({'type': 'delta', 'content': chunk})}\n\n"

            gen_time = (time.perf_counter() - t_gen_start) * 1000.0
            telemetry.add_step("LLM Generation", gen_time)
            telemetry.output_tokens = estimate_tokens(full_text)
            telemetry.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
            cost = telemetry.calculate_cost()

            session_stats["total_queries"] += 1
            session_stats["total_cost_usd"] += cost

            final_telemetry = {
                "type": "telemetry",
                "timings": [{"step": t.step_name, "duration_ms": t.duration_ms} for t in telemetry.timings],
                "total_latency_ms": round(telemetry.total_latency_ms, 2),
                "input_tokens": telemetry.input_tokens,
                "output_tokens": telemetry.output_tokens,
                "cost_usd": cost,
                "total_session_cost": round(session_stats["total_cost_usd"], 6)
            }
            yield f"data: {json.dumps(final_telemetry)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as err:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal inference pipeline error.'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/benchmark")
async def run_benchmark_endpoint():
    summary = run_benchmark()
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
