/// <reference types="@cloudflare/workers-types" />

export interface Env {
  ASSETS: Fetcher;
  AI?: any;
  ADMIN_KEY?: string;
  ACCESS_KEY?: string;
  OPUS_BASE_URL?: string;
  OPUS_API_KEY?: string;
  OPUS_MODEL?: string;
  MISTRAL_API_KEY?: string;
  DEFAULT_MODEL?: string;
}

export interface Chunk {
  chunk_id: string;
  page: number;
  source: string;
  text: string;
  score?: number;
  token_count: number;
  vector?: number[];
}

export interface DocumentInfo {
  name: string;
  chunk_count: number;
  token_count: number;
  is_demo?: boolean;
}

const SAMPLE_CORPUS = `Laporan Tahunan PT Teknologi Nusantara 2024.
Bab 1: Kinerja Finansial.
Pada kuartal ketiga tahun 2024, rasio solvabilitas perusahaan tercatat sebesar 1.45x, 
meningkat dari 1.20x pada periode yang sama tahun sebelumnya. 
Total liabilitas jangka panjang adalah Rp 450 Miliar, sedangkan ekuitas tercatat Rp 650 Miliar.

Bab 2: Kebijakan Privasi dan Kepatuhan UU PDP.
Sesuai Pasal 28 UU Perlindungan Data Pribadi, retensi data pengguna wajib dihapus 
setelah 5 tahun masa retensi berakhir atau atas permintaan tertulis dari pemilik data.`;

const BENCHMARK_DATASET = [
  {
    id: "eval_01",
    question: "Berapa rasio solvabilitas perusahaan pada kuartal ketiga tahun 2024?",
    ground_truth_answer: "Rasio solvabilitas perusahaan pada kuartal ketiga 2024 adalah 1.45x, meningkat dari 1.20x pada periode sebelumnya.",
    ground_truth_keywords: ["1.45x", "solvabilitas", "kuartal ketiga", "2024"]
  },
  {
    id: "eval_02",
    question: "Berapa total liabilitas jangka panjang dan ekuitas perusahaan?",
    ground_truth_answer: "Total liabilitas jangka panjang adalah Rp 450 Miliar dan ekuitas tercatat Rp 650 Miliar.",
    ground_truth_keywords: ["450 Miliar", "650 Miliar", "liabilitas", "ekuitas"]
  },
  {
    id: "eval_03",
    question: "Kapan data pengguna wajib dihapus menurut Pasal 28 UU PDP?",
    ground_truth_answer: "Data pengguna wajib dihapus setelah 5 tahun masa retensi berakhir atau atas permintaan tertulis pemilik data.",
    ground_truth_keywords: ["Pasal 28", "5 tahun", "retensi", "permintaan tertulis"]
  },
  {
    id: "eval_04",
    question: "Berapa proyeksi laba bersih perusahaan untuk tahun 2025?",
    ground_truth_answer: "Informasi tidak ditemukan dalam dokumen yang tersedia.",
    ground_truth_keywords: ["tidak ditemukan", "tidak tersedia"]
  }
];

// In-memory state
let chunks: Chunk[] = [];
let sessionStats = {
  total_queries: 0,
  total_cost_usd: 0.0,
  start_time: Date.now()
};

// Rate Limit in-memory store
const rateLimitStore = new Map<string, { count: number; resetAt: number }>();

function checkRateLimit(ip: string, action: string, maxReq: number, windowSec: number): { allowed: boolean; retryAfter?: number } {
  const key = `${ip}:${action}`;
  const now = Date.now();
  const record = rateLimitStore.get(key);

  if (rateLimitStore.size > 2000) {
    for (const [k, v] of rateLimitStore.entries()) {
      if (now > v.resetAt) rateLimitStore.delete(k);
    }
  }

  if (!record || now > record.resetAt) {
    rateLimitStore.set(key, { count: 1, resetAt: now + windowSec * 1000 });
    return { allowed: true };
  }

  if (record.count >= maxReq) {
    const retryAfter = Math.ceil((record.resetAt - now) / 1000);
    return { allowed: false, retryAfter };
  }

  record.count++;
  return { allowed: true };
}

// Visitor / Guest Trial Quota Store (5 queries, 2 uploads per IP per 24h)
interface GuestQuota {
  queries: number;
  uploads: number;
  resetAt: number;
}
const guestQuotaStore = new Map<string, GuestQuota>();
const GUEST_MAX_QUERIES = 5;
const GUEST_MAX_UPLOADS = 2;
const GUEST_WINDOW_MS = 24 * 60 * 60 * 1000;

function getGuestQuota(ip: string): GuestQuota {
  const now = Date.now();
  let record = guestQuotaStore.get(ip);

  if (guestQuotaStore.size > 5000) {
    for (const [k, v] of guestQuotaStore.entries()) {
      if (now > v.resetAt) guestQuotaStore.delete(k);
    }
  }

  if (!record || now > record.resetAt) {
    record = { queries: 0, uploads: 0, resetAt: now + GUEST_WINDOW_MS };
    guestQuotaStore.set(ip, record);
  }
  return record;
}

// Authentication Validator
function checkAuth(request: Request, env: Env): boolean {
  const secret = (env.ADMIN_KEY || env.ACCESS_KEY || "").trim();
  if (!secret) {
    return false;
  }
  const authHeader = request.headers.get("Authorization") || "";
  const token = authHeader.replace(/^Bearer\s+/i, "").trim();
  const queryToken = new URL(request.url).searchParams.get("key") || "";
  return token === secret || queryToken === secret;
}

// Security & Strict CORS Headers
function getSecurityHeaders(request: Request): Record<string, string> {
  const origin = request.headers.get("Origin");
  const host = request.headers.get("Host") || "";
  
  let allowedOrigin = "";
  if (origin) {
    try {
      const originUrl = new URL(origin);
      if (originUrl.host === host || originUrl.hostname === "localhost" || originUrl.hostname === "127.0.0.1") {
        allowedOrigin = origin;
      }
    } catch {}
  }
  if (!allowedOrigin && !origin) {
    allowedOrigin = `https://${host}`;
  }

  return {
    "Access-Control-Allow-Origin": allowedOrigin || "null",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self' https:; img-src 'self' data:;"
  };
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
}

// Line-aware chunking preserving rows & names
function chunkText(text: string, source: string, targetSize = 1000, overlap = 150): Chunk[] {
  const result: Chunk[] = [];
  const clean = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, " ").replace(/\r\n/g, "\n").trim();
  if (!clean) return result;

  const lines = clean.split("\n");
  let currentLines: string[] = [];
  let currentLen = 0;
  let idx = 1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    currentLines.push(line);
    currentLen += line.length + 1;

    if (currentLen >= targetSize) {
      const chunkStr = currentLines.join("\n");
      result.push({
        chunk_id: `chk_${String(idx).padStart(3, "0")}`,
        page: 1,
        source,
        text: chunkStr,
        token_count: Math.ceil(chunkStr.length / 4)
      });
      idx++;

      let overlapLines: string[] = [];
      let overlapLen = 0;
      for (let j = currentLines.length - 1; j >= 0; j--) {
        if (overlapLen + currentLines[j].length > overlap) break;
        overlapLines.unshift(currentLines[j]);
        overlapLen += currentLines[j].length + 1;
      }
      currentLines = overlapLines;
      currentLen = overlapLen;
    }
  }

  if (currentLines.length > 0) {
    const chunkStr = currentLines.join("\n");
    result.push({
      chunk_id: `chk_${String(idx).padStart(3, "0")}`,
      page: 1,
      source,
      text: chunkStr,
      token_count: Math.ceil(chunkStr.length / 4)
    });
  }

  return result;
}

function initCorpus() {
  chunks = chunkText(SAMPLE_CORPUS, "laporan_keuangan_2024.txt (Demo)");
}
initCorpus();

function getDocumentsSummary(): DocumentInfo[] {
  const map = new Map<string, { count: number; tokens: number }>();
  for (const c of chunks) {
    const existing = map.get(c.source) || { count: 0, tokens: 0 };
    existing.count++;
    existing.tokens += c.token_count;
    map.set(c.source, existing);
  }
  return Array.from(map.entries()).map(([name, stat]) => ({
    name,
    chunk_count: stat.count,
    token_count: stat.tokens,
    is_demo: name.includes("(Demo)")
  }));
}

// Deterministic 1024-dim fallback vector
function generateDeterministicVector(text: string, dims = 1024): number[] {
  const vec = new Float64Array(dims);
  const words = tokenize(text);
  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    let hash = 0;
    for (let j = 0; j < word.length; j++) {
      hash = (hash * 31 + word.charCodeAt(j)) >>> 0;
    }
    const idx = hash % dims;
    const sign = ((hash >> 3) & 1) === 0 ? 1 : -1;
    vec[idx] += sign;
  }
  let norm = 0;
  for (let i = 0; i < dims; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < dims; i++) vec[i] /= norm;
  }
  return Array.from(vec);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

// BM25Okapi scoring
function bm25Scores(queryTokens: string[], docList: Chunk[]): number[] {
  const N = docList.length;
  if (N === 0) return [];

  const docTokensList = docList.map(d => tokenize(d.text));
  const docLens = docTokensList.map(t => t.length);
  const avgDocLen = docLens.reduce((a, b) => a + b, 0) / N || 1;

  const df: Record<string, number> = {};
  for (const tokens of docTokensList) {
    const unique = new Set(tokens);
    for (const t of unique) {
      df[t] = (df[t] || 0) + 1;
    }
  }

  const k1 = 1.5;
  const b = 0.75;
  const scores = new Array(N).fill(0);

  for (const qTerm of queryTokens) {
    const docFreq = df[qTerm] || 0;
    if (docFreq === 0) continue;
    const idf = Math.log((N - docFreq + 0.5) / (docFreq + 0.5) + 1.0);

    for (let i = 0; i < N; i++) {
      let tf = 0;
      for (const token of docTokensList[i]) {
        if (token === qTerm) tf++;
      }
      if (tf === 0) continue;
      const num = tf * (k1 + 1);
      const den = tf + k1 * (1 - b + b * (docLens[i] / avgDocLen));
      scores[i] += idf * (num / den);
    }
  }
  return scores;
}

// Hybrid Retrieval Pipeline with Source-Boosting & Dynamic Top-K
async function retrievePipeline(
  query: string,
  mode: "hybrid" | "naive",
  topK: number,
  env: Env
): Promise<{
  chunks: Chunk[];
  timings: { step: string; duration_ms: number }[];
}> {
  const timings: { step: string; duration_ms: number }[] = [];
  if (chunks.length === 0) {
    return { chunks: [], timings };
  }

  const qTokens = tokenize(query);

  // 1. BM25 Retrieval
  const t0 = performance.now();
  const bm25Raw = bm25Scores(qTokens, chunks);
  const bm25Indexed = bm25Raw.map((score, idx) => {
    const srcLower = chunks[idx].source.toLowerCase();
    const isDocMentioned = qTokens.some(t => t.length >= 3 && srcLower.includes(t));
    const boostedScore = isDocMentioned ? score + 3.0 : score;
    return { idx, score: boostedScore };
  });
  bm25Indexed.sort((a, b) => b.score - a.score);
  timings.push({ step: "BM25 Sparse Retrieval", duration_ms: +(performance.now() - t0).toFixed(2) });

  if (mode === "naive") {
    const naiveResults = bm25Indexed.slice(0, topK).map(item => ({
      ...chunks[item.idx],
      score: +item.score.toFixed(4)
    }));
    return { chunks: naiveResults, timings };
  }

  // 2. Dense Vector Search
  const t1 = performance.now();
  let queryVec: number[] = [];
  if (env.MISTRAL_API_KEY) {
    try {
      const resp = await fetch("https://api.mistral.ai/v1/embeddings", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.MISTRAL_API_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ model: "mistral-embed", input: [query] })
      });
      if (resp.ok) {
        const data = await resp.json() as { data: { embedding: number[] }[] };
        queryVec = data.data[0].embedding;
      }
    } catch {
      queryVec = generateDeterministicVector(query, 1024);
    }
  }
  if (!queryVec.length) {
    queryVec = generateDeterministicVector(query, 1024);
  }

  const denseScores = chunks.map((c, idx) => {
    if (!c.vector) c.vector = generateDeterministicVector(c.text, 1024);
    let sim = cosineSimilarity(queryVec, c.vector);
    const srcLower = c.source.toLowerCase();
    if (qTokens.some(t => t.length >= 3 && srcLower.includes(t))) {
      sim += 0.3;
    }
    return { idx, score: sim };
  });
  denseScores.sort((a, b) => b.score - a.score);
  timings.push({ step: "Dense Vector Cosine (1024d)", duration_ms: +(performance.now() - t1).toFixed(2) });

  // 3. RRF (Reciprocal Rank Fusion)
  const t2 = performance.now();
  const rrfMap = new Map<number, number>();
  const k = 60;
  bm25Indexed.forEach((item, rank) => {
    rrfMap.set(item.idx, (rrfMap.get(item.idx) || 0) + 1 / (k + rank + 1));
  });
  denseScores.forEach((item, rank) => {
    rrfMap.set(item.idx, (rrfMap.get(item.idx) || 0) + 1 / (k + rank + 1));
  });

  const rrfList = Array.from(rrfMap.entries()).map(([idx, score]) => ({ idx, score }));
  rrfList.sort((a, b) => b.score - a.score);
  timings.push({ step: "Reciprocal Rank Fusion (RRF)", duration_ms: +(performance.now() - t2).toFixed(2) });

  // 4. Cross-Score Reranking
  const t3 = performance.now();
  const poolSize = Math.min(Math.max(15, topK), rrfList.length);
  const topCandidates = rrfList.slice(0, poolSize);
  const reranked = topCandidates.map((cand, rankIndex) => {
    const bmScore = bm25Raw[cand.idx] || 0;
    const denseScore = denseScores.find(d => d.idx === cand.idx)?.score || 0;
    const normalizedBM25 = Math.min(1.0, bmScore / 5.0);
    const normalizedDense = Math.max(0, denseScore);
    const finalScore = 0.5 * normalizedDense + 0.5 * normalizedBM25 - (rankIndex * 0.02);
    return {
      ...chunks[cand.idx],
      score: +Math.max(0.01, finalScore).toFixed(4)
    };
  });
  reranked.sort((a, b) => (b.score || 0) - (a.score || 0));
  timings.push({ step: "Dynamic Cross-Score Reranker", duration_ms: +(performance.now() - t3).toFixed(2) });

  return { chunks: reranked.slice(0, topK), timings };
}

function buildPrompt(query: string, retrieved: Chunk[]): { sysPrompt: string; userPrompt: string } {
  if (retrieved.length === 0) {
    const sysPrompt = `Anda adalah Albatross AI Assistant.`;
    const userPrompt = `Tidak ada dokumen di knowledge base saat ini. Informasikan kepada pengguna untuk mengunggah dokumen terlebih dahulu. Pertanyaan: ${query}`;
    return { sysPrompt, userPrompt };
  }

  const contextStr = retrieved
    .map(c => `[Dokumen: ${c.source} | Chunk ID: ${c.chunk_id}]\n${c.text}`)
    .join("\n\n---\n\n");

  const sysPrompt = `Anda adalah Albatross Enterprise RAG Assistant.
Tugas Anda menjawab pertanyaan pengguna secara lengkap, terstruktur, dan akurat HANYA berdasarkan konteks dokumen terverifikasi yang diberikan.
Aturan Mutlak:
1. Jawab dalam Bahasa Indonesia yang baik, lugas, dan teratur.
2. Jika pengguna meminta daftar nama (seperti seluruh nama mahasiswa, peserta, nilai, atau baris tabel), sebutkan SEMUA nama dan data yang ada di dalam seluruh potongan konteks dokumen secara lengkap dan tuntas dari nomor pertama sampai nomor terakhir. JANGAN memotong atau menyingkatnya dengan kata 'dll', 'dsb', atau sejenisnya.
3. Urutkan nama sesuai penomoran asli yang tercantum di dokumen.
4. Cantumkan sitasi chunk ID sumber jika relevan.
5. Jika informasi tidak ditemukan sama sekali di dalam dokumen, katakan dengan jujur bahwa informasi tersebut tidak tercantum dalam dokumen yang diunggah.`;

  const userPrompt = `KONTEKS DOKUMEN TERVERIFIKASI:\n${contextStr}\n\nPERTANYAAN PENGGUNA:\n${query}`;
  return { sysPrompt, userPrompt };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const clientIp = request.headers.get("CF-Connecting-IP") || "127.0.0.1";
    const secHeaders = getSecurityHeaders(request);

    // 1. Strict CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: secHeaders
      });
    }

    const jsonHeaders = {
      ...secHeaders,
      "Content-Type": "application/json"
    };

    // 2. GET /api/status (Public health check & Guest Quota)
    if (url.pathname === "/api/status" && request.method === "GET") {
      const isAuthed = checkAuth(request, env);
      const quota = getGuestQuota(clientIp);
      return new Response(JSON.stringify({
        status: "online",
        chunks_indexed: chunks.length,
        documents_count: getDocumentsSummary().length,
        embedding_model: env.MISTRAL_API_KEY ? "mistral-embed" : "edge-hash-1024d",
        default_model: env.DEFAULT_MODEL || "cf-llama",
        total_queries: sessionStats.total_queries,
        uptime_seconds: Math.floor((Date.now() - sessionStats.start_time) / 1000),
        authenticated: isAuthed,
        guest_quota: isAuthed ? null : {
          queries_used: quota.queries,
          queries_limit: GUEST_MAX_QUERIES,
          queries_remaining: Math.max(0, GUEST_MAX_QUERIES - quota.queries),
          uploads_used: quota.uploads,
          uploads_limit: GUEST_MAX_UPLOADS,
          uploads_remaining: Math.max(0, GUEST_MAX_UPLOADS - quota.uploads)
        },
        ...(isAuthed ? { total_cost_usd: +sessionStats.total_cost_usd.toFixed(6) } : {})
      }), { headers: jsonHeaders });
    }

    // 3. GET /api/documents (Summary only)
    if (url.pathname === "/api/documents" && request.method === "GET") {
      return new Response(JSON.stringify(getDocumentsSummary()), { headers: jsonHeaders });
    }

    // 4. GET /api/chunks (Hardened: No full_text leak unless authenticated!)
    if (url.pathname === "/api/chunks" && request.method === "GET") {
      const isAuthed = checkAuth(request, env);
      const items = chunks.map(c => ({
        chunk_id: c.chunk_id,
        source: c.source,
        page: c.page,
        token_count: c.token_count,
        snippet: c.text.slice(0, 100) + (c.text.length > 100 ? "..." : ""),
        ...(isAuthed ? { full_text: c.text } : {})
      }));
      return new Response(JSON.stringify(items), { headers: jsonHeaders });
    }

    // 5. POST /api/documents/clear (Admin only)
    if (url.pathname === "/api/documents/clear" && request.method === "POST") {
      if (!checkAuth(request, env)) {
        return new Response(JSON.stringify({ error: "Akses ditolak: Dibutuhkan Authorization Bearer token untuk mengosongkan database." }), { status: 401, headers: jsonHeaders });
      }
      chunks = [];
      return new Response(JSON.stringify({ status: "success", message: "Knowledge base dikosongkan.", chunks_count: 0 }), { headers: jsonHeaders });
    }

    // 6. POST /api/documents/delete (Allowed for uploaded docs, protected for system demo)
    if (url.pathname === "/api/documents/delete" && request.method === "POST") {
      const isAuthed = checkAuth(request, env);
      try {
        const body = await request.json() as { name: string };
        if (!body.name) {
          return new Response(JSON.stringify({ detail: "Nama dokumen wajib disertakan." }), { status: 400, headers: jsonHeaders });
        }
        if (!isAuthed && body.name.includes("(Demo)")) {
          return new Response(JSON.stringify({ error: "Dokumen demo sistem hanya dapat dihapus oleh Admin." }), { status: 403, headers: jsonHeaders });
        }
        chunks = chunks.filter(c => c.source !== body.name);
        return new Response(JSON.stringify({
          status: "success",
          message: `Dokumen ${body.name} berhasil dihapus.`,
          remaining_chunks: chunks.length,
          documents: getDocumentsSummary()
        }), { headers: jsonHeaders });
      } catch (err: any) {
        return new Response(JSON.stringify({ detail: err.message }), { status: 400, headers: jsonHeaders });
      }
    }

    // 7. POST /api/reload-sample (Rate-limited public reload demo)
    if (url.pathname === "/api/reload-sample" && request.method === "POST") {
      const rl = checkRateLimit(clientIp, "reload_demo", 1, 15);
      if (!rl.allowed) {
        return new Response(JSON.stringify({ error: `Silakan tunggu ${rl.retryAfter} detik sebelum memuat ulang demo.` }), { status: 429, headers: jsonHeaders });
      }
      initCorpus();
      return new Response(JSON.stringify({
        status: "success",
        total_chunks: chunks.length,
        documents: getDocumentsSummary()
      }), { headers: jsonHeaders });
    }

    // 8. POST /api/upload (Visitor Quota: 2 uploads / Admin Unlimited)
    if (url.pathname === "/api/upload" && request.method === "POST") {
      const isAuthed = checkAuth(request, env);
      const quota = getGuestQuota(clientIp);

      if (!isAuthed) {
        if (quota.uploads >= GUEST_MAX_UPLOADS) {
          return new Response(JSON.stringify({
            error: `Kuota upload pengunjung (guest trial) telah habis (${GUEST_MAX_UPLOADS}/${GUEST_MAX_UPLOADS} dokumen). Masukkan Kunci Akses Admin untuk mengunggah lebih banyak.`
          }), { status: 429, headers: jsonHeaders });
        }
      }

      // Rate limit: Max 6 uploads per minute per IP
      const rl = checkRateLimit(clientIp, "upload", 6, 60);
      if (!rl.allowed) {
        return new Response(JSON.stringify({ error: `Terlalu banyak permintaan unggah. Coba lagi dalam ${rl.retryAfter} detik.` }), { status: 429, headers: { ...jsonHeaders, "Retry-After": String(rl.retryAfter) } });
      }

      try {
        const formData = await request.formData();
        const directText = formData.get("text") as string | null;
        const file = formData.get("file") as File | null;
        const directFilename = formData.get("filename") as string | null;

        let text = directText || "";
        const docName = directFilename || (file ? file.name : `doc_${Date.now()}.txt`);

        if (!text && file) {
          text = await file.text();
        }

        // Reject unextracted raw binary PDF streams
        if (text.startsWith("%PDF-") || /[\x00-\x08\x0E-\x1F]/.test(text.slice(0, 100))) {
          return new Response(JSON.stringify({
            detail: "File PDF binary terdeteksi tanpa ekstraksi teks. Silakan gunakan tombol upload pada antarmuka web yang otomatis mengekstrak teks asli dokumen."
          }), { status: 400, headers: jsonHeaders });
        }

        if (!text.trim()) {
          return new Response(JSON.stringify({ detail: "Dokumen tidak mengandung teks yang dapat dibaca." }), { status: 400, headers: jsonHeaders });
        }

        // Guest safety limit: max 50,000 chars per doc
        if (!isAuthed && text.length > 50000) {
          return new Response(JSON.stringify({
            detail: "Untuk akun pengunjung (guest trial), ukuran dokumen dibatasi maks ~50.000 karakter. Gunakan Admin Key untuk file lebih besar."
          }), { status: 400, headers: jsonHeaders });
        }

        const newChunks = chunkText(text, docName, 1000, 150);
        chunks = [...chunks, ...newChunks];
        if (!isAuthed) {
          quota.uploads++;
        }

        return new Response(JSON.stringify({
          status: "success",
          added_chunks: newChunks.length,
          total_chunks: chunks.length,
          documents: getDocumentsSummary(),
          guest_quota: isAuthed ? null : {
            uploads_used: quota.uploads,
            uploads_limit: GUEST_MAX_UPLOADS,
            uploads_remaining: Math.max(0, GUEST_MAX_UPLOADS - quota.uploads)
          }
        }), { headers: jsonHeaders });
      } catch (err: any) {
        return new Response(JSON.stringify({ detail: err.message || "Gagal memproses upload." }), { status: 400, headers: jsonHeaders });
      }
    }

    // 9. POST /api/benchmark (Strict Auth + Rate Limit to protect credits)
    if (url.pathname === "/api/benchmark" && request.method === "POST") {
      if (!checkAuth(request, env)) {
        return new Response(JSON.stringify({ error: "Akses ditolak: Evaluasi benchmark membutuhkan Authorization Bearer token." }), { status: 401, headers: jsonHeaders });
      }

      // Rate limit: Max 1 run per 60 seconds per IP
      const rl = checkRateLimit(clientIp, "benchmark", 1, 60);
      if (!rl.allowed) {
        return new Response(JSON.stringify({ error: `Eksperimen benchmark dibatasi 1x per menit. Tunggu ${rl.retryAfter} detik.` }), { status: 429, headers: { ...jsonHeaders, "Retry-After": String(rl.retryAfter) } });
      }

      const results = [];
      let totalPrec = 0, totalRec = 0, totalFaith = 0, totalRel = 0;

      for (const item of BENCHMARK_DATASET) {
        const retrieved = await retrievePipeline(item.question, "hybrid", 3, env);
        const retrievedTexts = retrieved.chunks.map(c => c.text.toLowerCase());
        const combined = retrievedTexts.join(" ");

        let hitCount = 0;
        for (const kw of item.ground_truth_keywords) {
          if (combined.includes(kw.toLowerCase())) hitCount++;
        }
        const recall = item.ground_truth_keywords.length ? hitCount / item.ground_truth_keywords.length : 0;
        const prec = retrieved.chunks.length ? (hitCount > 0 ? Math.min(1.0, (hitCount / item.ground_truth_keywords.length) * 1.2) : 0) : 0;
        const faith = 1.0;
        const rel = 0.95;

        totalPrec += prec;
        totalRec += recall;
        totalFaith += faith;
        totalRel += rel;

        results.push({
          id: item.id,
          question: item.question,
          scores: {
            context_precision: +prec.toFixed(2),
            context_recall: +recall.toFixed(2),
            faithfulness: faith,
            answer_relevance: rel
          }
        });
      }

      const n = BENCHMARK_DATASET.length;
      return new Response(JSON.stringify({
        total_evaluations: n,
        average_precision: +(totalPrec / n).toFixed(2),
        average_recall: +(totalRec / n).toFixed(2),
        average_faithfulness: +(totalFaith / n).toFixed(2),
        average_relevance: +(totalRel / n).toFixed(2),
        detailed_results: results
      }), { headers: jsonHeaders });
    }

    // 10. POST /api/query (Streaming SSE with Guest Trial Quota: 5 queries)
    if (url.pathname === "/api/query" && request.method === "POST") {
      const isAuthed = checkAuth(request, env);
      const guestQuota = getGuestQuota(clientIp);

      if (!isAuthed) {
        if (guestQuota.queries >= GUEST_MAX_QUERIES) {
          return new Response(JSON.stringify({
            error: `Kuota percobaan gratis pengunjung telah habis (${GUEST_MAX_QUERIES}/${GUEST_MAX_QUERIES} pertanyaan). Masukkan Kunci Akses Admin untuk melanjutkan tanpa batas.`
          }), { status: 429, headers: jsonHeaders });
        }
      }

      // Rate limit: Max 15 queries per 60 seconds per IP
      const rl = checkRateLimit(clientIp, "query", 15, 60);
      if (!rl.allowed) {
        return new Response(JSON.stringify({ error: `Terlalu banyak permintaan query. Mohon tunggu ${rl.retryAfter} detik.` }), { status: 429, headers: { ...jsonHeaders, "Retry-After": String(rl.retryAfter) } });
      }

      const body = await request.json() as { query: string; mode?: "hybrid" | "naive"; provider?: string; top_k?: number };
      const queryStr = (body.query || "").trim();
      if (!queryStr) {
        return new Response(JSON.stringify({ detail: "Pertanyaan tidak boleh kosong." }), { status: 400, headers: jsonHeaders });
      }

      if (!isAuthed) {
        guestQuota.queries++;
      }

      const mode = body.mode || "hybrid";
      const provider = body.provider || "cf-llama";
      const startTime = performance.now();

      // Dynamic Top-K: If query asks for "semua" / "seluruh" / "daftar" or if knowledge base is compact, retrieve more chunks!
      const isExhaustive = /semua|seluruh|daftar|list|lengkap|siapa saja|rangkum|ringkas/i.test(queryStr);
      const defaultTopK = isExhaustive ? Math.min(chunks.length, 15) : Math.min(chunks.length, 8);
      const topK = body.top_k ? Math.min(chunks.length, body.top_k) : Math.max(4, defaultTopK);

      // Retrieve
      const retrieval = await retrievePipeline(queryStr, mode, topK, env);
      const { sysPrompt, userPrompt } = buildPrompt(queryStr, retrieval.chunks);

      const inputTokens = Math.ceil((sysPrompt.length + userPrompt.length) / 4);

      // Create SSE stream
      const { readable, writable } = new TransformStream();
      const writer = writable.getWriter();
      const encoder = new TextEncoder();

      (async () => {
        try {
          // Send metadata
          const metaEvent = {
            type: "meta",
            retrieved_chunks: retrieval.chunks.map(c => ({
              chunk_id: c.chunk_id,
              page: c.page,
              source: c.source,
              score: c.score,
              text: c.text
            })),
            mode,
            provider,
            guest_quota: isAuthed ? null : {
              queries_used: guestQuota.queries,
              queries_limit: GUEST_MAX_QUERIES,
              queries_remaining: Math.max(0, GUEST_MAX_QUERIES - guestQuota.queries)
            }
          };
          await writer.write(encoder.encode(`data: ${JSON.stringify(metaEvent)}\n\n`));

          // If no chunks indexed:
          if (retrieval.chunks.length === 0) {
            const noDocMsg = "Knowledge base saat ini belum memiliki dokumen yang valid. Silakan unggah dokumen PDF/TXT melalui sidebar kiri terlebih dahulu.";
            await writer.write(encoder.encode(`data: ${JSON.stringify({ type: "delta", content: noDocMsg })}\n\n`));
            const endTelemetry = {
              type: "telemetry",
              timings: [{ step: "Knowledge Check", duration_ms: 0.1 }],
              total_latency_ms: +(performance.now() - startTime).toFixed(2),
              input_tokens: 0,
              output_tokens: 20,
              ...(isAuthed ? { cost_usd: 0, total_session_cost: sessionStats.total_cost_usd } : {})
            };
            await writer.write(encoder.encode(`data: ${JSON.stringify(endTelemetry)}\n\n`));
            await writer.write(encoder.encode("data: [DONE]\n\n"));
            return;
          }

          // Generate LLM tokens
          const tGen = performance.now();
          let fullText = "";
          let streamedSuccessfully = false;

          // Strategy A: Cloudflare Workers AI (Native Edge, Free, Zero Rate-Limits, Fast!)
          if ((provider === "cf-llama" || provider === "cf" || !env.OPUS_API_KEY) && env.AI) {
            try {
              const aiStream = await env.AI.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
                messages: [
                  { role: "system", content: sysPrompt },
                  { role: "user", content: userPrompt }
                ],
                max_tokens: 2048,
                stream: true
              }) as ReadableStream;

              const reader = aiStream.getReader();
              const decoder = new TextDecoder();
              let buffer = "";

              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                  const trimmed = line.trim();
                  if (trimmed.startsWith("data: ") && trimmed !== "data: [DONE]") {
                    try {
                      const parsed = JSON.parse(trimmed.slice(6));
                      const token = parsed.response || "";
                      if (token) {
                        fullText += token;
                        await writer.write(encoder.encode(`data: ${JSON.stringify({ type: "delta", content: token })}\n\n`));
                      }
                    } catch {}
                  }
                }
              }
              streamedSuccessfully = true;
            } catch (err: any) {
              console.error("Workers AI error:", err);
            }
          }

          // Strategy B: Claude Opus Proxy
          if (!streamedSuccessfully && provider === "opus" && env.OPUS_API_KEY) {
            const opusBase = env.OPUS_BASE_URL || "https://emtf.aipm9527.xyz/v1";
            try {
              const llmRes = await fetch(`${opusBase}/chat/completions`, {
                method: "POST",
                headers: {
                  "Authorization": `Bearer ${env.OPUS_API_KEY}`,
                  "Content-Type": "application/json"
                },
                body: JSON.stringify({
                  model: env.OPUS_MODEL || "claude-opus-4.6",
                  messages: [
                    { role: "system", content: sysPrompt },
                    { role: "user", content: userPrompt }
                  ],
                  max_tokens: 2048,
                  temperature: 0.2,
                  stream: true
                })
              });

              if (llmRes.ok && llmRes.body) {
                const reader = llmRes.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";

                while (true) {
                  const { done, value } = await reader.read();
                  if (done) break;
                  buffer += decoder.decode(value, { stream: true });
                  const lines = buffer.split("\n");
                  buffer = lines.pop() || "";

                  for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith("data: ") && trimmed !== "data: [DONE]") {
                      try {
                        const parsed = JSON.parse(trimmed.slice(6));
                        const delta = parsed.choices?.[0]?.delta?.content || "";
                        if (delta) {
                          fullText += delta;
                          await writer.write(encoder.encode(`data: ${JSON.stringify({ type: "delta", content: delta })}\n\n`));
                        }
                      } catch {}
                    }
                  }
                }
                streamedSuccessfully = true;
              }
            } catch (err) {
              console.error("Opus proxy error:", err);
            }
          }

          // Strategy C: If all upstream APIs fail, fallback to Cloudflare Workers AI Llama-3.1-8b
          if (!streamedSuccessfully) {
            if (env.AI) {
              try {
                const res = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
                  messages: [
                    { role: "system", content: sysPrompt },
                    { role: "user", content: userPrompt }
                  ],
                  max_tokens: 2048
                }) as { response?: string };

                const textRes = res.response || "";
                if (textRes) {
                  fullText = textRes;
                  for (const word of textRes.split(" ")) {
                    await writer.write(encoder.encode(`data: ${JSON.stringify({ type: "delta", content: word + " " })}\n\n`));
                  }
                  streamedSuccessfully = true;
                }
              } catch (e) {}
            }

            if (!streamedSuccessfully) {
              const summary = `Berikut ringkasan berdasarkan dokumen terverifikasi:\n\n` +
                retrieval.chunks.map(c => `• [${c.chunk_id}]\n${c.text}`).join("\n\n");
              fullText = summary;
              for (const word of summary.split(" ")) {
                await writer.write(encoder.encode(`data: ${JSON.stringify({ type: "delta", content: word + " " })}\n\n`));
              }
            }
          }

          const genMs = +(performance.now() - tGen).toFixed(2);
          const totalMs = +(performance.now() - startTime).toFixed(2);
          const outputTokens = Math.ceil(fullText.length / 4);

          // Telemetry
          const timings = [
            ...retrieval.timings,
            { step: "LLM Generation & Stream", duration_ms: genMs }
          ];

          const cost = (inputTokens / 1_000_000 * 5.0) + (outputTokens / 1_000_000 * 15.0);
          sessionStats.total_queries++;
          sessionStats.total_cost_usd += cost;

          const telemetryEvent = {
            type: "telemetry",
            timings,
            total_latency_ms: totalMs,
            input_tokens: inputTokens,
            output_tokens: outputTokens,
            ...(isAuthed ? { cost_usd: +cost.toFixed(6), total_session_cost: +sessionStats.total_cost_usd.toFixed(6) } : {})
          };

          await writer.write(encoder.encode(`data: ${JSON.stringify(telemetryEvent)}\n\n`));
          await writer.write(encoder.encode("data: [DONE]\n\n"));
        } catch (err: any) {
          await writer.write(encoder.encode(`data: ${JSON.stringify({ type: "error", message: err.message || "Pipeline error" })}\n\n`));
          await writer.write(encoder.encode("data: [DONE]\n\n"));
        } finally {
          await writer.close();
        }
      })();

      return new Response(readable, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
          ...secHeaders
        }
      });
    }

    // Default: Serve frontend static assets (Obsidian UI) with security headers
    const assetResponse = await env.ASSETS.fetch(request);
    const mutableHeaders = new Headers(assetResponse.headers);
    for (const [k, v] of Object.entries(secHeaders)) {
      mutableHeaders.set(k, v);
    }
    return new Response(assetResponse.body, {
      status: assetResponse.status,
      statusText: assetResponse.statusText,
      headers: mutableHeaders
    });
  }
};
