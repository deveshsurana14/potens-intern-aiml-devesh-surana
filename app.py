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


@st.cache_resource(show_spinner="Loading models and vector store…")
def get_system():
    return build_system(offline=False)


def confidence_badge(conf: float, review: bool) -> None:
    pct = int(conf * 100)
    if review:
        st.warning(f"⚠️ Confidence {pct}% — flagged for human review "
                   f"(threshold {int(settings.low_confidence_threshold*100)}%).")
    else:
        st.success(f"✅ Confidence {pct}%")
    st.progress(conf)


st.title("💧 Micro-Irrigation Policy — Document Q&A with Citations")
st.caption("RAG over 6 Indian drip-irrigation subsidy & technical documents · "
           "Groq Llama 3.3 · ChromaDB · cross-encoder reranker")

try:
    system = get_system()
    sources = system.store.list_sources()
except Exception as e:  # pragma: no cover - UI guard
    st.error(f"Could not initialise the system: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Indexed documents")
    for s in sources:
        st.write(f"• {s}")
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
