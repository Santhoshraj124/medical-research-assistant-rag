import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_loader import load_and_split
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #f8fafc; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }

    .main-header p {
        color: #b8d4f0;
        margin: 0.3rem 0 0;
        font-size: 0.95rem;
    }

    .answer-box {
        background: #1e1e1e;
        color: #ffffff !important;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.8;
        font-size: 16px;
        white-space: pre-wrap;
    }

    .answer-box * {
        color: #ffffff !important;
    }

    .citation-box {
        background: #1f2937;
        color: white;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }

    .citation-box * {
        color: white !important;
    }

    /* FIX TAB TEXT COLOR */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #111827 !important;
        font-size: 16px;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"] p {
        color: #111827 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #dc2626 !important;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_store():
    return VectorStore("data/index")


def get_engine():
    return RAGEngine(get_store())


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Medical RAG")
    st.caption("LangChain · FAISS · Ollama · Free & Local")
    st.divider()

    st.success("✅ Using Ollama (local — no API key needed)")

    st.divider()
    st.markdown("### 📄 Upload Papers")

    uploaded = st.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        store = get_store()
        existing = [s["source_file"] for s in store.get_all_sources()]
        for uf in uploaded:
            if uf.name in existing:
                st.info(f"Already indexed: {uf.name[:30]}")
                continue
            with st.spinner(f"Processing {uf.name[:25]}…"):
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False,
                    dir="data/papers"
                ) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name

                # rename to original filename so source_file is clean
                clean_path = os.path.join("data/papers", uf.name)
                os.replace(tmp_path, clean_path)

                try:
                    docs = load_and_split(clean_path)
                    added = store.add_documents(docs)
                    st.success(f"✅ {uf.name[:25]} — {added} chunks")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    store = get_store()
    sources = store.get_all_sources()
    st.markdown(f"### 📚 Library ({len(sources)} papers)")
    for s in sources:
        st.caption(f"📃 {s['source_file']}")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧬 Medical Research Assistant</h1>
  <p>LangChain · FAISS · Ollama · Fully Local & Free</p>
</div>
""", unsafe_allow_html=True)

store = get_store()
if store.is_empty:
    st.info("👈 Upload at least one PDF research paper to get started.")
    st.stop()

engine = get_engine()
sources = store.get_all_sources()
source_names = [s["source_file"] for s in sources]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Q&A", "📝 Summarise", "⚖️ Compare", "📖 Literature Review", "🔍 Key Findings"
])

# ── Tab 1: Q&A ─────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Ask a Research Question")
    st.caption("Ask anything about your papers. Answers include citations.")

    filter_src = st.selectbox(
        "Filter to a specific paper (optional)",
        ["All papers"] + source_names,
        key="qa_filter",
    )
    question = st.text_area(
        "Your question",
        height=90,
        placeholder="e.g. What is this paper about?",
    )

    if st.button("🔍 Get Answer", type="primary", disabled=not question.strip()):
        src_filter = None if filter_src == "All papers" else filter_src

        with st.spinner("Generating answer…"):
            try:
                stream, citations = engine.stream_answer(question, source_filter=src_filter)
                full = ""
                for chunk in stream:
                    if chunk:
                        full += chunk

                st.markdown("#### Answer")
                if full.strip():
                    st.markdown(
                        f'<div class="answer-box">{full}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("No answer generated. Make sure Ollama is running and try again.")

                if citations:
                    st.markdown("#### Sources")
                    for c in citations:
                        st.markdown(
                            f'<div class="citation-box">'
                            f'[{c["index"]}] <b>{c["source_file"]}</b> | Page {c["page"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.error(f"Error: {e}\n\nMake sure Ollama is running: `ollama run llama3.2`")

# ── Tab 2: Summarise ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Summarise a Paper")
    st.caption("Get a structured summary: objective, methods, findings, clinical relevance.")

    selected_sum = st.selectbox(
        "Choose a paper",
        source_names,
        key="sum_sel",
    )

    if st.button("📝 Generate Summary", type="primary"):
        with st.spinner("Generating summary…"):
            try:
                stream = engine.stream_summary(selected_sum)
                full = "".join(chunk for chunk in stream if chunk)
                st.markdown("#### Summary")
                if full.strip():
                    st.markdown(f'<div class="answer-box">{full}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No summary generated. Try again.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Tab 3: Compare ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Compare Papers")
    st.caption("Select two or more papers for a structured head-to-head comparison.")

    if len(source_names) < 2:
        st.warning("Upload at least 2 papers to use this feature.")
    else:
        selected_cmp = st.multiselect(
            "Select papers to compare",
            source_names,
            default=source_names[:2],
            key="cmp_sel",
        )
        cmp_topic = st.text_input(
            "Comparison focus (optional)",
            placeholder="e.g. efficacy and safety of the intervention",
        )

        if st.button("⚖️ Compare", type="primary", disabled=len(selected_cmp) < 2):
            with st.spinner("Comparing papers…"):
                try:
                    stream = engine.stream_comparison(selected_cmp, topic=cmp_topic)
                    full = "".join(chunk for chunk in stream if chunk)
                    st.markdown("#### Comparison")
                    if full.strip():
                        st.markdown(f'<div class="answer-box">{full}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No comparison generated. Try again.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 4: Literature Review ────────────────────────────────────────────────────
with tab4:
    st.markdown("### Generate Literature Review")
    st.caption("Synthesise evidence across all uploaded papers on a chosen topic.")

    lr_topic = st.text_input(
        "Research topic",
        placeholder="e.g. euglycemic diabetic ketoacidosis in burn patients",
    )
    lr_filter = st.multiselect(
        "Restrict to specific papers (optional)",
        source_names,
        key="lr_filter",
    )

    if st.button("📖 Generate Review", type="primary", disabled=not lr_topic.strip()):
        with st.spinner("Generating literature review…"):
            try:
                stream, citations = engine.stream_literature_review(
                    lr_topic, source_filter=lr_filter or None
                )
                full = "".join(chunk for chunk in stream if chunk)
                st.markdown("#### Literature Review")
                if full.strip():
                    st.markdown(f'<div class="answer-box">{full}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No review generated. Try again.")

                if citations:
                    with st.expander("📚 Sources used"):
                        for c in citations:
                            st.caption(f"[{c['index']}] {c['source_file']} | Page {c['page']}")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Tab 5: Key Findings ─────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Extract Key Findings")
    st.caption("Precise bullet-point extraction of results, statistics, and outcomes.")

    selected_kf = st.selectbox(
        "Choose a paper",
        source_names,
        key="kf_sel",
    )

    if st.button("🔍 Extract Findings", type="primary"):
        with st.spinner("Extracting findings…"):
            try:
                stream = engine.stream_key_findings(selected_kf)
                full = "".join(chunk for chunk in stream if chunk)
                st.markdown("#### Key Findings")
                if full.strip():
                    st.markdown(f'<div class="answer-box">{full}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No findings extracted. Try again.")
            except Exception as e:
                st.error(f"Error: {e}")