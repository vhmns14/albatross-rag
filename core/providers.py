"""
providers.py - Multi-LLM Client (Claude Opus 4.6, Mistral, Groq).
Security Hardened: Sanitized error reporting (no secret leakage), safe timeouts,
consistent 1024-dimensional vector fallback, and zero-norm guards.
"""

import os
import re
import json
import httpx
from typing import Generator, List, Dict, Any, Optional
import numpy as np

OPUS_KEY = os.getenv("OPUS_API_KEY", "").strip()
OPUS_BASE_URL = os.getenv("OPUS_BASE_URL", "https://emtf.aipm9527.xyz/v1").strip().rstrip("/")
OPUS_MODEL = os.getenv("OPUS_MODEL", "claude-opus-4-6").strip()

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest").strip()

GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Standard embedding dimensionality for cross-model alignment
EMBEDDING_DIM = 1024

def sanitize_error_message(err_str: str) -> str:
    """Redact any API keys or credentials that might leak in error messages."""
    sanitized = re.sub(r'sk-[a-zA-Z0-9_-]{10,}', 'sk-[REDACTED]', err_str)
    sanitized = re.sub(r'Bearer\s+[a-zA-Z0-9_\.-]+', 'Bearer [REDACTED]', sanitized)
    return sanitized

class LLMProvider:
    """Unified, zero-bloat gateway to Claude Opus 4.6, Mistral, and Groq."""

    @staticmethod
    def generate_stream(
        prompt: str,
        system_prompt: str = "You are Albatross RAG, a high-precision factual assistant. Ground answers strictly in provided context.",
        provider: str = "opus",
        model_name: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream response chunks from selected provider safely."""
        provider = (provider or "opus").lower().strip()
        
        # 1. Claude Opus 4.6 (OpenAI-compatible proxy endpoint)
        if provider in ["opus", "claude", "claude-opus"]:
            key = os.getenv("OPUS_API_KEY", OPUS_KEY)
            base_url = os.getenv("OPUS_BASE_URL", OPUS_BASE_URL).strip().rstrip("/")
            model = model_name or os.getenv("OPUS_MODEL", OPUS_MODEL)

            if not key:
                yield "⚠️ [Claude Opus API Key missing]. Please check OPUS_API_KEY in .env."
                return

            url = f"{base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "stream": True
            }
            try:
                with httpx.Client(timeout=60.0) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            raw_err = response.read().decode('utf-8', errors='ignore')
                            yield f"❌ Opus Error ({response.status_code}): {sanitize_error_message(raw_err)}"
                            return
                        for line in response.iter_lines():
                            if line.startswith("data: ") and not line.endswith("[DONE]"):
                                try:
                                    chunk = json.loads(line[6:])
                                    delta = chunk["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield delta
                                except Exception:
                                    continue
            except Exception as e:
                yield f"❌ Opus Connection Error: {sanitize_error_message(str(e))}"

        # 2. Mistral AI Official API
        elif provider == "mistral":
            key = os.getenv("MISTRAL_API_KEY", MISTRAL_KEY)
            if not key:
                yield "⚠️ [Mistral API Key missing]. Please check MISTRAL_API_KEY in .env."
                return
            model = model_name or os.getenv("MISTRAL_MODEL", MISTRAL_MODEL)
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "stream": True
            }
            try:
                with httpx.Client(timeout=45.0) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            raw_err = response.read().decode('utf-8', errors='ignore')
                            yield f"❌ Mistral Error ({response.status_code}): {sanitize_error_message(raw_err)}"
                            return
                        for line in response.iter_lines():
                            if line.startswith("data: ") and not line.endswith("[DONE]"):
                                try:
                                    chunk = json.loads(line[6:])
                                    delta = chunk["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield delta
                                except Exception:
                                    continue
            except Exception as e:
                yield f"❌ Mistral Connection Error: {sanitize_error_message(str(e))}"

        # 3. Groq (Free / Fast Llama 3.3)
        elif provider == "groq":
            key = os.getenv("GROQ_API_KEY", GROQ_KEY)
            if not key:
                yield "⚠️ [Groq API Key missing]. Set GROQ_API_KEY in .env for free Llama testing."
                return
            model = model_name or "llama-3.3-70b-versatile"
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": True
            }
            try:
                with httpx.Client(timeout=30.0) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            raw_err = response.read().decode('utf-8', errors='ignore')
                            yield f"❌ Groq Error ({response.status_code}): {sanitize_error_message(raw_err)}"
                            return
                        for line in response.iter_lines():
                            if line.startswith("data: ") and not line.endswith("[DONE]"):
                                try:
                                    chunk = json.loads(line[6:])
                                    delta = chunk["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield delta
                                except Exception:
                                    continue
            except Exception as e:
                yield f"❌ Groq Connection Error: {sanitize_error_message(str(e))}"

        else:
            yield f"❌ Unknown provider: {provider}"

    @staticmethod
    def generate_complete(prompt: str, system_prompt: str = "", provider: str = "opus", model_name: Optional[str] = None) -> str:
        """Non-streaming helper for automated benchmarks."""
        chunks = list(LLMProvider.generate_stream(prompt, system_prompt, provider, model_name))
        return "".join(chunks)


# -------------------------------------------------------------
# Dense Embeddings Provider (Mistral Embed + Consistent 1024d Fallback)
# -------------------------------------------------------------

def get_lightweight_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Fast, deterministic hash-projection embedding in pure numpy with zero-norm guard."""
    words = (text or "").lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    if not words:
        return vec
    for word in words:
        h = hash(word) % dim
        vec[h] += 1.0
    for i in range(len(words) - 1):
        h2 = hash(words[i] + "_" + words[i+1]) % dim
        vec[h2] += 1.5
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec

def get_embeddings(texts: List[str], expected_dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Return 2D numpy array of embeddings for given texts safely."""
    if not texts:
        return np.empty((0, expected_dim), dtype=np.float32)

    # Use Mistral Embed if MISTRAL_API_KEY is available
    mistral_key = os.getenv("MISTRAL_API_KEY", MISTRAL_KEY).strip()
    if mistral_key:
        try:
            url = "https://api.mistral.ai/v1/embeddings"
            headers = {"Authorization": f"Bearer {mistral_key}", "Content-Type": "application/json"}
            all_embeddings = []
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch = [t[:4000] for t in texts[i:i+batch_size]]
                res = httpx.post(url, headers=headers, json={"model": "mistral-embed", "input": batch}, timeout=20.0)
                if res.status_code == 200:
                    data = res.json()["data"]
                    for item in data:
                        all_embeddings.append(item["embedding"])
                else:
                    break
            
            if len(all_embeddings) == len(texts):
                matrix = np.array(all_embeddings, dtype=np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms = np.where(norms < 1e-6, 1.0, norms)
                return matrix / norms
        except Exception:
            pass

    # Default fallback: consistent 1024-dim CPU embedding (guarantees dimension alignment)
    return np.array([get_lightweight_embedding(t, dim=expected_dim) for t in texts], dtype=np.float32)
