import os
from pathlib import Path
from typing import List, Dict

import streamlit as st

from rag_core import ingest_file, load_store, is_on_topic, retrieve, support_strength
from llm_providers import generate_doc_answer, generate_topical_answer, provider_names

st.set_page_config(page_title="Chat with RAG", page_icon="💬", layout="wide")
st.title("💬 Chat with RAG (PDF/PPT)")

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", provider_names(), index=0)
    st.caption("Set API keys in Settings → Secrets on Streamlit Cloud")

    st.subheader("Relevance & Support Gates")
    on_topic_thresh = st.slider("On-topic threshold", 0.0, 1.0, 0.25, 0.01)
    support_thresh = st.slider("Doc support threshold", 0.0, 1.0, 0.35, 0.01)

    st.markdown("---")
    st.subheader("🎯 Try these questions")
    st.caption("Use after uploading a sample doc")
    example_qs = [
        "What is the scope of this document?",
        "List key findings with page references",
        "What assumptions were made?",
        "Summarize the limitations",
        "Give an executive summary in 5 bullets",
        "What are future work items?",
    ]
    for q in example_qs:
        st.code(q, language="text")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

uploaded = st.file_uploader("Upload a PDF or PPT/PPTX", type=["pdf", "ppt", "pptx"], accept_multiple_files=False)

col1, col2 = st.columns([1, 2])
with col1:
    if uploaded is not None and st.button("Index document", use_container_width=True):
        tmp_path = Path("/tmp") / uploaded.name
        tmp_path.write_bytes(uploaded.read())
        try:
            doc_id = ingest_file(str(tmp_path))
            st.session_state.doc_id = doc_id
            st.success(f"Document indexed ✅  (doc_id: {doc_id[:8]}…)")
        except Exception as e:
            st.error(f"Failed to index: {e}")

with col2:
    if st.session_state.doc_id:
        st.info("Ask on-topic questions for doc-grounded answers with citations. If the model needs related info beyond the document, it will label it as **Off-doc (related)**. Irrelevant questions return an error.")

q = st.text_input("Ask a question…", placeholder="e.g., What are the key risks identified?")
ask = st.button("Ask", type="primary")

if ask:
    if not st.session_state.doc_id:
        st.error("Please upload and index a document first.")
    elif not q.strip():
        st.error("Please enter a question.")
    else:
        try:
            doc = load_store("store", st.session_state.doc_id)
            on_topic, topic_sim = is_on_topic(q, doc.centroid, on_topic_thresh)
            st.write(f"**Topic similarity:** {topic_sim:.3f}")

            if not on_topic:
                st.error("irrelevant")
            else:
                hits = retrieve(doc, q, k=6)
                support = support_strength(hits)
                st.write(f"**Doc support:** {support:.3f}")

                if support >= support_thresh:
                    ans = generate_doc_answer(hits, q, provider)
                    st.markdown(ans)

                    with st.expander("Sources (retrieved chunks)"):
                        for h in hits:
                            st.markdown(f"**{h['ref']}** — score={h['score']:.3f}")
                            st.write(h["text"])
                            st.markdown("---")
                else:
                    ans = generate_topical_answer(q, provider)
                    st.markdown(ans)
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.subheader("ℹ️ Notes")
st.markdown(
    """
- Answers may include citations like `[ref: page_3, page_4]` or `[ref: slide_2]`.
- Thresholds can be tuned from the sidebar.
- This demo stores indices in the ephemeral `/store` folder on Streamlit Cloud.
    """
)