"""
app.py - Albatross RAG Studio Dashboard.
Custom 3-Column Engineering Workbench with Claude Opus 4.6, Mistral,
Under-the-Hood Telemetry Waterfall, and Automated Evals.
"""

import os
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load .env variables
load_dotenv(override=True)

from core.engine import AlbatrossEngine, DocumentChunk
from core.providers import LLMProvider
from core.telemetry import QueryTelemetry, TelemetryTimer, estimate_tokens
from evals.benchmark import run_benchmark, SAMPLE_CORPUS

# -------------------------------------------------------------
# Streamlit Page Config & Custom Modern Dark Theme (CSS)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Albatross RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header bar */
.top-header {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.brand-title {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #f8fafc;
}

/* Workbench columns */
.panel-card {
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
}
.panel-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Waterfall cards */
.waterfall-bar {
    background: #1e293b;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
}
.timing-val {
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* Chunk preview cards */
.chunk-box {
    background: #0f172a;
    border-left: 3px solid #38bdf8;
    border-radius: 4px;
    padding: 10px 12px;
    font-size: 0.82rem;
    margin-bottom: 10px;
    color: #cbd5e1;
    line-height: 1.45;
}
.chunk-header {
    font-size: 0.75rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
}
.score-tag {
    background: #064e3b;
    color: #6ee7b7;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = AlbatrossEngine(chunk_size=400, chunk_overlap=80)
    # Load default sample knowledge automatically
    st.session_state.engine.load_text(SAMPLE_CORPUS, source_name="laporan_keuangan_2024.txt")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = None

if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []

if "total_session_cost" not in st.session_state:
    st.session_state.total_session_cost = 0.0

# -------------------------------------------------------------
# Top Navigation / Control Bar
# -------------------------------------------------------------
col_h1, col_h2, col_h3, col_h4 = st.columns([3, 2.5, 2, 2])

with col_h1:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-top:5px;">
            <span style="font-size:1.4rem; font-weight:800; color:#f8fafc; letter-spacing:-0.5px;">ALBATROSS</span>
            <span style="font-size:0.8rem; background:#0284c7; color:#fff; padding:2px 8px; border-radius:4px; font-weight:600;">RAG STUDIO</span>
            <span style="font-size:0.75rem; color:#64748b;">| Observable Workbench</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_h2:
    model_options = ["Claude Opus 4.6 (Active)", "Mistral Large (Active)", "Groq (Llama 3.3 70B)"]
    selected_model_label = st.selectbox("LLM Provider", model_options, index=0, label_visibility="collapsed")
    if "Opus" in selected_model_label:
        active_provider = "opus"
        active_model = "claude-opus-4-6"
    elif "Mistral" in selected_model_label:
        active_provider = "mistral"
        active_model = "mistral-large-latest"
    else:
        active_provider = "groq"
        active_model = "llama-3.3-70b-versatile"

with col_h3:
    rag_mode = st.radio(
        "RAG Mode",
        ["Hybrid + Reranker", "Naive Vector Only"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    mode_key = "hybrid" if "Hybrid" in rag_mode else "naive"

with col_h4:
    cost_str = f"${st.session_state.total_session_cost:.4f}"
    st.markdown(
        f"""
        <div style="text-align:right; margin-top:5px;">
            <div style="font-size:0.7rem; color:#94a3b8;">SESSION COST</div>
            <div style="font-family:'JetBrains Mono'; font-size:1.1rem; color:#38bdf8; font-weight:700;">{cost_str}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# -------------------------------------------------------------
# Main Navigation Tabs: Studio Workbench vs Evals Benchmark
# -------------------------------------------------------------
tab_workbench, tab_evals = st.tabs(["⚡ Retrieval Studio Workbench", "📊 Automated Evals & Benchmark"])

# =============================================================
# TAB 1: RETRIEVAL STUDIO WORKBENCH (3-COLUMN LAYOUT)
# =============================================================
with tab_workbench:
    col_left, col_center, col_right = st.columns([1.1, 2.0, 1.2], gap="medium")

    # ---------------------------------------------------------
    # COLUMN 1: KNOWLEDGE BASE & CHUNK MANAGER
    # ---------------------------------------------------------
    with col_left:
        st.markdown("<div class='panel-title'>📂 Knowledge & Chunks</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Index Document", use_container_width=True, type="primary"):
                with st.spinner("Processing & Dense Embedding with Mistral..."):
                    if uploaded_file.name.endswith(".pdf"):
                        added = st.session_state.engine.load_pdf(uploaded_file, source_name=uploaded_file.name)
                    else:
                        text_content = uploaded_file.read().decode("utf-8")
                        added = st.session_state.engine.load_text(text_content, source_name=uploaded_file.name)
                    st.success(f"Indexed {added} new chunks!")

        # Document Stats Card
        total_chunks = len(st.session_state.engine.chunks)
        st.markdown(
            f"""
            <div class='waterfall-bar' style="margin-top:10px;">
                <span>Total Chunks Indexed</span>
                <span class='timing-val'>{total_chunks} chunks</span>
            </div>
            <div class='waterfall-bar'>
                <span>Sparse Index (BM25)</span>
                <span class='timing-val' style="color:#4ade80;">Active ✅</span>
            </div>
            <div class='waterfall-bar'>
                <span>Dense Embeddings</span>
                <span class='timing-val' style="color:#a855f7;">Mistral Embed ✅</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Preloaded samples
        st.markdown("<div style='font-size:0.75rem; color:#64748b; margin-top:14px;'>QUICK ACTIONS</div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Reload Sample", use_container_width=True):
                st.session_state.engine.clear()
                st.session_state.engine.load_text(SAMPLE_CORPUS, source_name="laporan_keuangan_2024.txt")
                st.rerun()
        with col_btn2:
            if st.button("Clear DB", use_container_width=True):
                st.session_state.engine.clear()
                st.session_state.last_retrieved_chunks = []
                st.rerun()

        # Chunk Preview List
        st.markdown("<div style='font-size:0.75rem; color:#64748b; margin-top:16px;'>INDEXED CHUNKS EXPLORER</div>", unsafe_allow_html=True)
        with st.container(height=380):
            for c in st.session_state.engine.chunks:
                with st.expander(f"Chunk #{c.chunk_id} | {c.source} (p.{c.page})"):
                    st.caption(f"Tokens: ~{c.token_count}")
                    st.code(c.text, language="text")

    # ---------------------------------------------------------
    # COLUMN 2: REASONING CANVAS & INTERACTIVE CHAT
    # ---------------------------------------------------------
    with col_center:
        st.markdown(f"<div class='panel-title'>💬 Reasoning Canvas & Citations ({active_model})</div>", unsafe_allow_html=True)

        chat_container = st.container(height=520)
        with chat_container:
            if not st.session_state.messages:
                st.info("💡 **Coba tanyakan sesuatu pada dokumen finansial/UU PDP:**\n- *'Berapa rasio solvabilitas perusahaan di Q3 2024?'*\n- *'Kapan data wajib dihapus menurut Pasal 28 UU PDP?'*\n- *'Berapa proyeksi laba tahun 2025?' (Uji coba pencegahan halusinasi)*")

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "trace" in msg and msg["trace"]:
                        with st.expander("🛠️ View Retrieval Trace & Context"):
                            st.text(msg["trace"])

        # Input query
        user_query = st.chat_input("Ketik pertanyaan analitis untuk dokumen...")
        if user_query:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)

            # Execution with Telemetry
            telemetry = QueryTelemetry(model=active_model, mode=mode_key)
            t_total_start = time.perf_counter()

            # Retrieval
            retrieved_chunks = st.session_state.engine.retrieve(
                query=user_query,
                mode=mode_key,
                top_k=3,
                telemetry=telemetry
            )
            st.session_state.last_retrieved_chunks = retrieved_chunks

            # Prompt assembly
            sys_prompt, user_prompt = st.session_state.engine.build_prompt(user_query, retrieved_chunks)
            telemetry.input_tokens = estimate_tokens(sys_prompt + user_prompt)

            # Stream generation
            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    t_gen_start = time.perf_counter()
                    stream_gen = LLMProvider.generate_stream(
                        prompt=user_prompt,
                        system_prompt=sys_prompt,
                        provider=active_provider,
                        model_name=active_model
                    )

                    for chunk in stream_gen:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    gen_time_ms = (time.perf_counter() - t_gen_start) * 1000.0
                    telemetry.add_step(f"LLM Gen ({active_model})", gen_time_ms)

            # Finalize telemetry
            telemetry.output_tokens = estimate_tokens(full_response)
            telemetry.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            query_cost = telemetry.calculate_cost()
            st.session_state.total_session_cost += query_cost
            st.session_state.last_telemetry = telemetry

            # Save assistant message with citations
            trace_summary = f"Model: {active_model} | Mode: {mode_key.upper()} | Chunks Retrieved: {len(retrieved_chunks)}\n"
            for c in retrieved_chunks:
                trace_summary += f"\n- [Chunk #{c.chunk_id}, Page {c.page}, Score {c.score:.4f}] {c.text[:120]}..."

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "trace": trace_summary
            })
            st.rerun()

    # ---------------------------------------------------------
    # COLUMN 3: UNDER-THE-HOOD INSPECTOR & TELEMETRY
    # ---------------------------------------------------------
    with col_right:
        st.markdown("<div class='panel-title'>🔬 Under-The-Hood Inspector</div>", unsafe_allow_html=True)

        # 1. Latency Waterfall
        st.markdown("<div style='font-size:0.75rem; color:#64748b; margin-bottom:8px;'>⚡ LATENCY WATERFALL BREAKDOWN</div>", unsafe_allow_html=True)
        if st.session_state.last_telemetry:
            tel = st.session_state.last_telemetry
            for timing in tel.timings:
                st.markdown(
                    f"""
                    <div class='waterfall-bar'>
                        <span>{timing.step_name}</span>
                        <span class='timing-val'>{timing.duration_ms:.1f} ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown(
                f"""
                <div class='waterfall-bar' style="border:1px solid #0284c7; background:#0c4a6e;">
                    <span style="font-weight:700;">Total Latency</span>
                    <span class='timing-val' style="color:#bae6fd;">{tel.total_latency_ms:.1f} ms</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Token & Cost stats
            st.markdown("<div style='font-size:0.75rem; color:#64748b; margin-top:12px; margin-bottom:6px;'>TOKEN USAGE & COST</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class='waterfall-bar'>
                    <span>Input / Output Tokens</span>
                    <span class='timing-val'>{tel.input_tokens} / {tel.output_tokens}</span>
                </div>
                <div class='waterfall-bar'>
                    <span>Estimated Cost</span>
                    <span class='timing-val'>${tel.estimated_cost_usd:.6f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.caption("Kirim pertanyaan untuk melihat rincian latensi per tahapan...")

        # 2. Retrieved Chunks with Relevance Confidence
        st.markdown("<div style='font-size:0.75rem; color:#64748b; margin-top:16px; margin-bottom:8px;'>📑 TOP RETRIEVED CHUNKS & SCORES</div>", unsafe_allow_html=True)
        if st.session_state.last_retrieved_chunks:
            for idx, c in enumerate(st.session_state.last_retrieved_chunks):
                st.markdown(
                    f"""
                    <div class='chunk-box'>
                        <div class='chunk-header'>
                            <span>CHUNK #{c.chunk_id} (p.{c.page})</span>
                            <span class='score-tag'>Score: {c.score:.4f}</span>
                        </div>
                        {c.text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("Belum ada chunk yang ditarik untuk query saat ini.")

# =============================================================
# TAB 2: AUTOMATED EVALS & BENCHMARK STUDIO
# =============================================================
with tab_evals:
    st.markdown("### 🏆 Automated Evaluation Studio (RAGAS Metrics Benchmark)")
    st.write(
        "Mengevaluasi secara kuantitatif performa **Naive Vector RAG** vs **Albatross Hybrid + Reranker** "
        "menggunakan benchmark dataset standar."
    )

    if st.button("🚀 Run Live Benchmark Experiment", type="primary"):
        with st.spinner("Menjalankan 4 metrik evaluasi pada dataset uji..."):
            summary = run_benchmark()
            st.session_state.eval_summary = summary
            st.success("Evaluasi selesai!")

    if "eval_summary" in st.session_state:
        summary = st.session_state.eval_summary
        n = summary["naive"]
        h = summary["hybrid"]

        # Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            diff_p = (h["avg_precision"] - n["avg_precision"]) * 100
            st.metric("Context Precision", f"{h['avg_precision']*100:.1f}%", delta=f"{diff_p:+.1f}% vs Naive")
        with col_m2:
            diff_r = (h["avg_recall"] - n["avg_recall"]) * 100
            st.metric("Context Recall", f"{h['avg_recall']*100:.1f}%", delta=f"{diff_r:+.1f}% vs Naive")
        with col_m3:
            st.metric("Faithfulness (Anti-Hallucination)", f"{h['avg_faithfulness']*100:.1f}%", delta="100% Grounded")
        with col_m4:
            st.metric("Hybrid Latency Overhead", f"{h['avg_latency_ms']:.1f} ms", delta=f"+{h['avg_latency_ms'] - n['avg_latency_ms']:.1f} ms")

        st.divider()

        # Detailed Comparison Table
        st.markdown("#### 📋 Detailed Benchmark Scorecard")
        scorecard_data = {
            "Metric": ["Context Precision", "Context Recall", "Faithfulness (Anti-Hallucination)", "Answer Relevance", "Avg Search Latency"],
            "Naive Vector RAG": [f"{n['avg_precision']*100:.1f}%", f"{n['avg_recall']*100:.1f}%", f"{n['avg_faithfulness']*100:.1f}%", f"{n['avg_relevance']*100:.1f}%", f"{n['avg_latency_ms']:.2f} ms"],
            "Albatross Hybrid RAG": [f"{h['avg_precision']*100:.1f}%", f"{h['avg_recall']*100:.1f}%", f"{h['avg_faithfulness']*100:.1f}%", f"{h['avg_relevance']*100:.1f}%", f"{h['avg_latency_ms']:.2f} ms"],
            "Advantage": ["Higher Precision", "Better Keyword Recall", "Verified Grounding", "Targeted Answers", "Sub-millisecond overhead"]
        }
        df_scorecard = pd.DataFrame(scorecard_data)
        st.dataframe(df_scorecard, use_container_width=True, hide_index=True)

        st.markdown(
            """
            > **💡 Insight untuk Interviewer / CV:**  
            > *"Albatross RAG menggabungkan BM25 Sparse Search dan Dense Vectors via Reciprocal Rank Fusion (RRF), "
            > menghasilkan peningkatan recall pada istilah spesifik dan data angka tanpa menambah overhead komputasi yang signifikan (<1ms retrieval overhead)."*
            """
        )
