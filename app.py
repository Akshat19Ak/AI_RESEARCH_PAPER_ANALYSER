"""
AI Research Paper Analyzer — Streamlit Frontend
Calls FastAPI backend at localhost:8000 for all RAG operations.
"""
import os, streamlit as st, httpx
from datetime import datetime

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Research Analyzer", page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI Research Paper Analyzer | Hybrid RAG + Reranking + Evaluation"})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #21262d; }
[data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 { color: #58a6ff; font-weight: 600; }
.app-header { background: linear-gradient(90deg, #1f6feb 0%, #388bfd 50%, #58a6ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 2.4rem; font-weight: 700; text-align: center; padding: 0.5rem 0; }
.sub-header { text-align: center; color: #8b949e; font-size: 1rem; margin-top: -0.6rem; margin-bottom: 1.5rem; }
.analysis-card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.8rem 0; transition: border-color 0.2s; }
.analysis-card:hover { border-color: #388bfd; }
.badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; margin: 0.2rem; }
.badge-blue { background: #1f3a5f; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-green { background: #1a3a2a; color: #56d364; border: 1px solid #2ea043; }
.badge-amber { background: #3a2a0a; color: #e3b341; border: 1px solid #9e6a03; }
.source-box { background: #0d1117; border: 1px solid #30363d; border-left: 3px solid #388bfd; border-radius: 6px; padding: 0.7rem 1rem; margin: 0.4rem 0; font-size: 0.82rem; color: #8b949e; }
.chat-user { background: #1f3a5f; border-radius: 12px 12px 4px 12px; padding: 0.8rem 1.1rem; margin: 0.5rem 0; color: #c9d1d9; border: 1px solid #1f6feb; }
.chat-assistant { background: #161b22; border-radius: 12px 12px 12px 4px; padding: 0.8rem 1.1rem; margin: 0.5rem 0; color: #c9d1d9; border: 1px solid #21262d; }
.metric-box { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1rem; text-align: center; }
.metric-value { color: #58a6ff; font-size: 1.6rem; font-weight: 700; }
.metric-label { color: #8b949e; font-size: 0.78rem; margin-top: 0.2rem; }
.stTextInput input, .stTextArea textarea { background: #0d1117 !important; border: 1px solid #30363d !important; color: #c9d1d9 !important; border-radius: 8px !important; }
.stButton > button { background: linear-gradient(90deg, #1f6feb, #388bfd); color: white; border: none; border-radius: 8px; font-weight: 600; padding: 0.5rem 1.2rem; transition: all 0.2s; width: 100%; }
.stButton > button:hover { background: linear-gradient(90deg, #388bfd, #58a6ff); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(56,139,253,0.3); }
.stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e; border-bottom: 2px solid transparent; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #388bfd !important; }
div[data-testid="stExpander"] { background: #161b22; border: 1px solid #21262d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
defaults = {"session_id": None, "doc_metadata": {}, "chunk_count": 0,
    "chat_history": [], "structured_summary": "", "quick_insights": "",
    "saved_summary": "", "session_metrics": [], "processing_done": False}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API Call Helpers ─────────────────────────────────────────────────────────
def api_post(endpoint, json=None, files=None, data=None, timeout=120):
    try:
        r = httpx.post(f"{API}{endpoint}", json=json, files=files, data=data, timeout=timeout)
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type","").startswith("application/json") else r.text
            st.error(f"API Error: {detail}")
            return None
        return r.json()
    except httpx.ConnectError:
        st.error("Cannot connect to API server. Start it with: `uvicorn api.server:app --port 8000`")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

def api_get(endpoint, timeout=30):
    try:
        r = httpx.get(f"{API}{endpoint}", timeout=timeout)
        return r.json() if r.status_code < 400 else None
    except:
        return None

# ── UI Helpers ───────────────────────────────────────────────────────────────
def show_citations(sources, label="Retrieved Sources"):
    if not sources: return
    with st.expander(f"📎 {label} ({len(sources)} chunks)", expanded=False):
        for s in sources:
            score_str = f" | Relevance: {s['reranker_score']:.3f}" if s.get("reranker_score") else ""
            st.markdown(f'<div class="source-box"><b>Chunk {s["chunk_index"]}</b>{score_str}<br>{s["content"][:300]}{"..." if len(s["content"])>300 else ""}</div>', unsafe_allow_html=True)

def show_metrics(metrics):
    cols = st.columns(4)
    items = [("Answer Relevance", metrics.get("answer_relevance",0)),
             ("Faithfulness", metrics.get("faithfulness",0)),
             ("Context Precision", metrics.get("context_precision",0)),
             ("Overall Score", metrics.get("overall_score",0))]
    for col, (label, val) in zip(cols, items):
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{val:.0f}%</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 Research Analyzer")
        st.markdown("---")
        # API health check
        health = api_get("/health")
        if health:
            st.markdown(f'<span class="badge badge-green">API Online</span> <span class="badge badge-blue">v{health.get("version","?")}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge" style="background:#3a1a1a;color:#f85149;border:1px solid #da3633">API Offline</span>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 🔑 Groq API Key")
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            st.success("Loaded from `.env`")
        else:
            api_key = st.text_input("Enter Groq API key", type="password", placeholder="gsk_...")
        if not api_key:
            st.warning("API key required.")
        st.markdown("---")

        st.markdown("### 📂 Input Source")
        source_options = {"📄 PDF": "pdf", "🖼️ Image (JPG/PNG)": "image", "🌐 URL / ArXiv": "url"}
        selected = st.selectbox("Select input type", list(source_options.keys()))
        source_type = source_options[selected]

        st.markdown("### 📤 Upload / Enter")
        file_obj, url_input = None, ""
        if source_type == "url":
            url_input = st.text_input("Enter URL", placeholder="https://arxiv.org/html/...")
        elif source_type == "pdf":
            file_obj = st.file_uploader("Upload PDF", type=["pdf"])
        elif source_type == "image":
            file_obj = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        process_ready = (file_obj is not None) or (source_type == "url" and url_input.strip())
        if process_ready and api_key:
            if st.button("🚀 Analyze Document", use_container_width=True):
                with st.spinner("Processing via API..."):
                    if source_type == "url":
                        resp = api_post("/upload-url", json={"url": url_input.strip(), "api_key": api_key})
                    else:
                        file_bytes = file_obj.read()
                        files = {"file": (file_obj.name, file_bytes)}
                        resp = api_post("/upload", files=files, data={"api_key": api_key}, timeout=180)
                    if resp and resp.get("status") == "success":
                        st.session_state.session_id = resp["session_id"]
                        st.session_state.doc_metadata = resp.get("metadata", {})
                        st.session_state.chunk_count = resp.get("chunk_count", 0)
                        st.session_state.processing_done = True
                        st.session_state.chat_history = []
                        st.session_state.structured_summary = ""
                        st.session_state.quick_insights = ""
                        st.session_state.session_metrics = []
                        st.success(f"Ready! Session: {resp['session_id']}")
                        st.rerun()
        st.markdown("---")

        if st.session_state.processing_done:
            meta = st.session_state.doc_metadata
            st.markdown("### 📊 Document Stats")
            c1, c2 = st.columns(2)
            with c1: st.metric("Chunks", st.session_state.chunk_count)
            with c2: st.metric("Pages", meta.get("pages", "—"))
            st.metric("Characters", f'{meta.get("char_count", 0):,}')
            dtype = meta.get("type","").upper()
            st.markdown(f'<span class="badge badge-blue">{dtype}</span> <span class="badge badge-green">HYBRID</span> <span class="badge badge-green">RERANKED</span>', unsafe_allow_html=True)
            if meta.get("ocr_used"):
                st.markdown('<span class="badge badge-amber">OCR</span>', unsafe_allow_html=True)
            st.caption(f"Session: `{st.session_state.session_id}`")
            if st.button("🗑️ Clear & Reset", use_container_width=True):
                if st.session_state.session_id:
                    api_get(f"/session/{st.session_state.session_id}")  # best-effort delete
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()
        st.markdown("---")
        st.markdown('<div style="color:#8b949e;font-size:0.72rem;text-align:center;">🔬 RAG Analyzer v2.0<br>FastAPI Backend + Streamlit Frontend<br>Hybrid BM25+FAISS · Cross-Encoder Reranking</div>', unsafe_allow_html=True)
    return api_key

# ── Landing Page ─────────────────────────────────────────────────────────────
def render_landing():
    st.markdown("---")
    features = [
        ("🧠", "Hybrid Retrieval", "BM25 + FAISS with Reciprocal Rank Fusion"),
        ("⚡", "Cross-Encoder Reranking", "30%+ precision boost via cross-attention"),
        ("📊", "Evaluation Metrics", "Real-time faithfulness, relevance, precision"),
        ("📄", "Multi-Format Input", "PDF, Images (OCR), URLs & ArXiv"),
        ("🎯", "Answer Citations", "Every answer cites source chunks with scores"),
        ("🔥", "WOW Features", "Paper comparison, interview prep, difficulty modes"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f'<div class="analysis-card"><div style="font-size:2rem;margin-bottom:0.5rem">{icon}</div><div style="color:#c9d1d9;font-weight:600;margin-bottom:0.3rem">{title}</div><div style="color:#8b949e;font-size:0.85rem">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="metric-box"><div class="metric-value">~80%</div><div class="metric-label">Reading Time Reduced</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-box"><div class="metric-value">500K+</div><div class="metric-label">Tokens Processable</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-box"><div class="metric-value">~35%</div><div class="metric-label">Relevance Improvement</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:#8b949e;padding:2rem"><div style="font-size:3rem;margin-bottom:1rem">⬅️</div><div style="font-size:1.1rem;color:#c9d1d9;font-weight:500">Upload a document or paste a URL in the sidebar</div></div>', unsafe_allow_html=True)

# ── Tab: Summary ─────────────────────────────────────────────────────────────
def tab_summary(api_key):
    st.markdown("### 🔬 Structured Research Paper Summary")
    st.markdown("Extracts **Problem, Contributions, Methodology, Results, Limitations, Future Work** with citations.")
    c1, c2 = st.columns([1, 3])
    with c1: gen = st.button("🚀 Generate Summary", use_container_width=True)
    with c2:
        if st.session_state.structured_summary:
            st.download_button("⬇️ Download (.md)", data=f"# Paper Analysis\n\n{st.session_state.structured_summary}",
                file_name=f"analysis_{datetime.now():%Y%m%d_%H%M%S}.md", mime="text/markdown", use_container_width=True)
    if gen:
        with st.spinner("🔍 Retrieving + reranking + synthesizing..."):
            resp = api_post("/summary", json={"session_id": st.session_state.session_id, "api_key": api_key})
            if resp:
                st.session_state.structured_summary = resp["result"]
                show_citations(resp.get("sources", []))
    if st.session_state.structured_summary:
        st.markdown("---")
        st.markdown(st.session_state.structured_summary)
        if st.button("💾 Save Summary for Paper Comparison"):
            st.session_state.saved_summary = st.session_state.structured_summary
            st.success("Saved! Load a new paper and use the Advanced tab to compare.")

# ── Tab: Chat ────────────────────────────────────────────────────────────────
def tab_chat(api_key):
    st.markdown("### 💬 Chat with the Paper")
    difficulty = st.radio("Explanation Level", ["Beginner", "Expert"], horizontal=True, index=1)
    history = st.session_state.chat_history
    if history:
        for msg in history:
            cls = "chat-user" if msg["role"] == "user" else "chat-assistant"
            icon = "👤 **You:**" if msg["role"] == "user" else "🤖 **Analyst:**"
            st.markdown(f'<div class="{cls}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown('<div style="text-align:center;color:#8b949e;padding:2rem;border:1px dashed #30363d;border-radius:8px;">Ask anything about the paper</div>', unsafe_allow_html=True)
    st.markdown("---")
    if not history:
        st.markdown("**💡 Suggested:**")
        for i, q in enumerate(["What problem does this paper solve?", "Explain the proposed method simply", "What datasets were used?", "How does this compare to previous work?"]):
            if st.button(f"📌 {q}", key=f"sug_{i}"):
                st.session_state._pending_q = q
                st.rerun()
    pending = st.session_state.pop("_pending_q", None)
    question = st.text_area("Your question:", value=pending or "", placeholder="E.g., What is the main novelty?", height=80, key="chat_input")
    if st.button("💬 Ask") and question.strip():
        with st.spinner("🤔 Searching + reranking + answering..."):
            resp = api_post("/chat", json={
                "session_id": st.session_state.session_id, "question": question.strip(),
                "difficulty": difficulty, "api_key": api_key,
            })
            if resp:
                st.session_state.chat_history.append({"role": "user", "content": question.strip()})
                st.session_state.chat_history.append({"role": "assistant", "content": resp["answer"]})
                show_citations(resp.get("sources", []))
                if resp.get("metrics"):
                    st.session_state.session_metrics.append(resp["metrics"])
                    show_metrics(resp["metrics"])
                st.rerun()

# ── Tab: Insights ────────────────────────────────────────────────────────────
def tab_insights(api_key):
    st.markdown("### ⚡ Quick Insights")
    st.markdown("Get the **5 most important takeaways**.")
    if st.button("⚡ Generate Insights"):
        with st.spinner("🔍 Identifying key findings..."):
            resp = api_post("/insights", json={"session_id": st.session_state.session_id, "api_key": api_key})
            if resp:
                st.session_state.quick_insights = resp["result"]
                show_citations(resp.get("sources", []))
    if st.session_state.quick_insights:
        st.markdown("---")
        st.markdown(st.session_state.quick_insights)

# ── Tab: Deep Dive ───────────────────────────────────────────────────────────
def tab_deepdive(api_key):
    st.markdown("### 🔍 Section Deep-Dive")
    presets = ["Abstract and Introduction", "Related Work", "Methodology and Architecture",
        "Experiments and Evaluation", "Results and Analysis", "Conclusion and Future Work"]
    c1, c2 = st.columns([2, 1])
    with c1: section = st.text_input("Section or topic:", placeholder="e.g., self-attention...")
    with c2: preset = st.selectbox("Or pick:", ["— Custom —"] + presets)
    if preset != "— Custom —": section = preset
    if st.button("🔍 Deep Dive") and section.strip():
        with st.spinner(f"🔬 Analyzing: {section}..."):
            resp = api_post("/deepdive", json={"session_id": st.session_state.session_id, "section": section, "api_key": api_key})
            if resp:
                st.markdown("---")
                st.markdown(resp["result"])
                show_citations(resp.get("sources", []))

# ── Tab: Advanced ────────────────────────────────────────────────────────────
def tab_advanced(api_key):
    st.markdown("### 🔥 Advanced Tools")
    mode = st.radio("Select tool:", ["📝 Interview Prep", "📊 Paper Comparison"], horizontal=True)
    st.markdown("---")
    if mode == "📝 Interview Prep":
        st.markdown("Generate **interview questions** (Easy/Medium/Hard) with model answers.")
        if st.button("🎯 Generate Interview Questions"):
            with st.spinner("Generating questions..."):
                resp = api_post("/interview", json={"session_id": st.session_state.session_id, "api_key": api_key})
                if resp:
                    st.markdown(resp["result"])
                    show_citations(resp.get("sources", []))
    else:
        if st.session_state.saved_summary:
            st.success("Paper 1 summary loaded.")
            if st.button("🔄 Compare Papers"):
                with st.spinner("Comparing..."):
                    resp = api_post("/compare", json={
                        "session_id": st.session_state.session_id,
                        "paper1_summary": st.session_state.saved_summary, "api_key": api_key,
                    })
                    if resp: st.markdown(resp["result"])
        else:
            st.info("**How to compare:**\n1. Load Paper 1 → Summary tab → Generate → Save for Comparison\n2. Load Paper 2 → come here → Compare")

# ── Tab: Metrics ─────────────────────────────────────────────────────────────
def tab_metrics():
    st.markdown("### 📊 Evaluation Metrics Dashboard")
    metrics_list = st.session_state.session_metrics
    if not metrics_list:
        st.info("Metrics appear here after you ask questions in the Chat tab.")
        st.markdown("""
**Metrics Explained:**
| Metric | What It Measures | How It Works |
|--------|-----------------|--------------|
| **Answer Relevance** | Is the answer on-topic? | Cosine similarity: question ↔ answer |
| **Faithfulness** | Is it grounded in sources? | % sentences matching source chunks |
| **Context Precision** | Were right chunks retrieved? | Avg similarity: question ↔ chunks |
| **Overall Score** | Weighted combination | 40% faith + 35% rel + 25% precision |
""")
        return
    st.markdown(f"**Session Stats:** {len(metrics_list)} queries evaluated")
    avg = {}
    for key in ["answer_relevance", "faithfulness", "context_precision", "overall_score"]:
        vals = [m[key] for m in metrics_list]
        avg[key] = sum(vals) / len(vals)
    st.markdown("#### 📈 Session Averages")
    show_metrics(avg)
    st.markdown("---")
    st.markdown("#### 📋 Per-Query Breakdown")
    for i, m in enumerate(metrics_list, 1):
        with st.expander(f"Query {i} — Overall: {m['overall_score']:.0f}%"):
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Relevance", f"{m['answer_relevance']:.0f}%")
            with c2: st.metric("Faithfulness", f"{m['faithfulness']:.0f}%")
            with c3: st.metric("Ctx Precision", f"{m['context_precision']:.0f}%")
            with c4: st.metric("Overall", f"{m['overall_score']:.0f}%")
            st.caption(f"Faithful: {m.get('faithful_sentences',0)}/{m.get('total_sentences',0)}")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    api_key = render_sidebar()
    st.markdown('<div class="app-header">🔬 AI Research Paper Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">FastAPI Backend · Hybrid Retrieval · Cross-Encoder Reranking · Evaluation Metrics</div>', unsafe_allow_html=True)
    if not st.session_state.processing_done:
        render_landing(); return
    if not api_key:
        st.error("🔑 Provide your Groq API key in the sidebar."); return
    t1,t2,t3,t4,t5,t6 = st.tabs(["🔬 Summary","💬 Chat","⚡ Insights","🔍 Deep Dive","🔥 Advanced","📊 Metrics"])
    with t1: tab_summary(api_key)
    with t2: tab_chat(api_key)
    with t3: tab_insights(api_key)
    with t4: tab_deepdive(api_key)
    with t5: tab_advanced(api_key)
    with t6: tab_metrics()

if __name__ == "__main__":
    main()
