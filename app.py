"""
Streamlit UI — so reviewers can try the system without Postman.

Two tabs:
  * Ask       : question -> grounded answer, expandable citations (file + section
                + snippet), confidence meter, and a visible human-review flag.
  * Contradict: pick two documents + a topic -> conflict verdict with each
                document's position and the reasoning.

Run:  streamlit run app.py   (after `python -m scripts.ingest`)
"""
from __future__ import annotations

import streamlit as st

from rag.config import settings
from rag.llm import LLMNotConfigured
from rag.pipeline import build_system

st.set_page_config(page_title="Micro-Irrigation Doc Q&A", page_icon="💧", layout="wide")

st.markdown(
    """
    <style>
      /* ---- base ---- */
      .stApp { background: #ffffff; }
      .block-container { padding-top: 1.6rem; max-width: 1080px; }
      html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      }

      /* ---- hero header ---- */
      .hero {
        background: linear-gradient(135deg, #ecfeff 0%, #eff6ff 100%);
        border: 1px solid #cffafe;
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 22px;
      }
      .hero-title {
        font-size: 2rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;
      }
      .hero-title span { color: #0891b2; }
      .hero-sub { color: #475569; font-size: 1.02rem; margin-top: 6px; }
      .chips { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
      .chip {
        background: #ffffff; border: 1px solid #cbd5e1; color: #0f172a;
        padding: 5px 12px; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
      }

      /* ---- sidebar ---- */
      [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }

      /* ---- buttons ---- */
      .stButton > button {
        border-radius: 10px; border: 1px solid #cbd5e1; background: #ffffff;
        color: #0f172a; font-weight: 600; transition: all .15s ease;
      }
      .stButton > button:hover {
        border-color: #0891b2; color: #0891b2; background: #ecfeff;
      }
      [data-testid="baseButton-primary"] {
        background: #0891b2; color: #ffffff; border: none;
      }
      [data-testid="baseButton-primary"]:hover { background: #0e7490; color: #fff; }

      /* ---- text input ---- */
      .stTextInput input {
        border-radius: 10px; border: 1px solid #cbd5e1; padding: 10px 12px;
      }
      .stTextInput input:focus {
        border-color: #0891b2; box-shadow: 0 0 0 3px rgba(8,145,178,.15);
      }

      /* ---- tabs ---- */
      [data-baseweb="tab"] { font-weight: 600; }

      /* ---- expanders as cards ---- */
      [data-testid="stExpander"] {
        border: 1px solid #e2e8f0; border-radius: 12px; background: #ffffff;
        box-shadow: 0 1px 2px rgba(15,23,42,.04); margin-bottom: 8px;
      }

      /* ---- alerts / containers ---- */
      [data-testid="stAlert"] { border-radius: 12px; }
      h3 { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models and vector store…")
def get_system():
    return build_system(offline=False)


# Acronyms we want to keep upper-cased when prettifying file names.
_ACRONYMS = {"pmksy", "pdmc", "dbt", "nabard"}


def pretty_source(filename: str) -> str:
    """'02_maharashtra_drip_subsidy_scheme.md' -> '2. Maharashtra Drip Subsidy Scheme'."""
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    num = ""
    if parts and parts[0].isdigit():
        num = str(int(parts[0]))
        parts = parts[1:]
    words = [w.upper() if w.lower() in _ACRONYMS else w.capitalize() for w in parts]
    title = " ".join(words)
    return f"{num}. {title}" if num else title


def confidence_badge(conf: float, review: bool) -> None:
    pct = int(conf * 100)
    if review:
        st.warning(f"⚠️ Confidence {pct}% — flagged for human review "
                   f"(threshold {int(settings.low_confidence_threshold*100)}%).")
    else:
        st.success(f"✅ Confidence {pct}%")
    st.progress(conf)


st.markdown(
    """
    <div class="hero">
      <div class="hero-title">💧 Micro-Irrigation Policy <span>Q&amp;A</span></div>
      <div class="hero-sub">Citation-first RAG over six Indian drip-irrigation policy
      &amp; technical documents · Groq Llama&nbsp;3.3 · ChromaDB · cross-encoder reranker</div>
      <div class="chips">
        <span class="chip">📎 Grounded citations</span>
        <span class="chip">🛡️ No hallucination</span>
        <span class="chip">🌐 English · हिंदी · मराठी</span>
        <span class="chip">↕️ Reranked retrieval</span>
        <span class="chip">👤 Human-in-the-loop</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    system = get_system()
    sources = system.store.list_sources()
except Exception as e:  # pragma: no cover - UI guard
    st.error(f"Could not initialise the system: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Indexed documents")
    for s in sources:
        st.markdown(f"**{pretty_source(s)}**")
        st.caption(s)
    st.divider()
    st.caption(f"{system.store.count()} chunks indexed")
    st.caption("No-hallucination gate + confidence-based human-in-the-loop are ON.")

tab_ask, tab_contradict = st.tabs(["🔎 Ask", "⚖️ Contradict"])

with tab_ask:
    st.markdown("Ask in **English, हिंदी, or मराठी** — the answer comes back in the "
                "same language.")
    q = st.text_input("Your question",
                      placeholder="e.g. What subsidy do small farmers get in Maharashtra?")
    examples = [
        "What subsidy do small and marginal farmers get under the central PDMC scheme?",
        "महाराष्ट्रात लहान शेतकऱ्यांना किती अनुदान मिळते?",
        "How does drip irrigation reduce emitter clogging?",
        "What is the DBT disbursement timeline for the subsidy?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex[:28] + "…", use_container_width=True):
            q = ex

    if q:
        try:
            with st.spinner("Retrieving and reasoning…"):
                ans = system.qa.ask(q)
        except LLMNotConfigured as e:
            st.error(str(e))
            st.stop()

        st.markdown(f"**Detected language:** {ans.language}")
        if not ans.answered:
            st.info("🚫 " + ans.answer + "  \n*(The system refused rather than "
                    "guess — this is the no-hallucination guard working.)*")
        else:
            with st.container(border=True):
                st.markdown("### Answer")
                st.write(ans.answer)
        confidence_badge(ans.confidence, ans.needs_human_review)

        if ans.citations:
            st.markdown("### Citations")
            for c in ans.citations:
                with st.expander(f"[{c.marker}] {c.source_file} — {c.section}"):
                    st.write(c.snippet)
                    st.caption(f"chunk id: {c.chunk_id} · doc id: {c.doc_id}")

        with st.expander("Debug: retrieved chunks & confidence signals"):
            st.json({"signals": ans.signals, "retrieved": ans.retrieved})

with tab_contradict:
    st.markdown("Check whether two documents **conflict** on a topic.")
    c1, c2 = st.columns(2)
    doc_a = c1.selectbox("Document A", sources, index=1 if len(sources) > 1 else 0)
    doc_b = c2.selectbox("Document B", sources, index=2 if len(sources) > 2 else 0)
    topic = st.text_input("Topic (optional)",
                          placeholder="e.g. effective subsidy percentage")

    if st.button("Compare documents", type="primary"):
        if doc_a == doc_b:
            st.warning("Pick two different documents.")
        else:
            try:
                with st.spinner("Comparing…"):
                    res = system.contradict.compare(doc_a, doc_b, topic)
            except LLMNotConfigured as e:
                st.error(str(e)); st.stop()
            except ValueError as e:
                st.error(str(e)); st.stop()

            if res.conflict:
                st.error(f"⚠️ Conflict detected on: **{res.topic}**")
            else:
                st.success(f"✅ No genuine conflict on: **{res.topic}**")
            a_col, b_col = st.columns(2)
            a_col.markdown(f"**{res.document_a}**\n\n{res.document_a_position}")
            b_col.markdown(f"**{res.document_b}**\n\n{res.document_b_position}")
            st.markdown("**Reasoning**")
            st.write(res.reasoning)
            with st.expander("Excerpts compared"):
                st.markdown("**A:**")
                for x in res.excerpts_a:
                    st.caption("• " + x)
                st.markdown("**B:**")
                for x in res.excerpts_b:
                    st.caption("• " + x)
