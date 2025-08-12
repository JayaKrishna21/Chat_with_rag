# streamlit_app.py
import os
from pathlib import Path
from typing import Dict, List

import streamlit as st

from rag_core import ingest_file, load_store, retrieve
from llm_providers import generate_doc_answer, generate_topical_answer, provider_names

st.set_page_config(page_title="Chat with RAG", page_icon="💬", layout="wide")
st.title("💬 Chat with RAG (PDF/PPT)")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", provider_names(), index=0)
    st.caption("Set keys in Streamlit Cloud → Settings → Secrets")

    st.markdown("---")
    st.subheader("🧹 Utilities")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.success("Chat cleared.")
    if st.button("Forget indexed document"):
        st.session_state.doc_id = None
        st.success("Document cleared.")

# ---------- Session state ----------
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict] = []

# ---------- Upload & indexing ----------
uploaded = st.file_uploader(
    "Upload a PDF or PPT/PPTX",
    type=["pdf", "ppt", "pptx"],
    accept_multiple_files=False,
    help="After indexing, you can chat below."
)

col_left, col_right = st.columns([1, 2])

with col_left:
    if uploaded is not None and st.button("Index document", use_container_width=True):
        tmp_path = Path("/tmp") / uploaded.name
        tmp_path.write_bytes(uploaded.read())
        try:
            doc_id = ingest_file(str(tmp_path))
            st.session_state.doc_id = doc_id
            st.success(f"Indexed ✅  (doc_id: {doc_id[:8]}…)")
        except Exception as e:
            st.error(f"Failed to index: {e}")

with col_right:
    if st.session_state.doc_id:
        st.info("Ask anything. The app first answers from the document (with citations). "
                "If the info isn’t in the file, it will answer as **Off-doc (related)**.")

# ---------- Show chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources", expanded=False):
                st.markdown(", ".join(sorted(set(msg["citations"]))))

# ---------- Chat input ----------
question = st.chat_input("Ask a question about the uploaded document…")

if question:
    # show the user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # guard: require a document
    if not st.session_state.doc_id:
        assistant_text = "Please upload and index a document first."
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
    else:
        try:
            # 1) Retrieve top chunks (no thresholds—always attempt doc-grounded first)
            doc = load_store("store", st.session_state.doc_id)
            hits = retrieve(doc, question, k=6)

            # 2) Try to answer ONLY from the document excerpts
            ans_doc = generate_doc_answer(hits, question, provider)

            # Our doc-only prompt says: "I couldn't find an answer in the document."
            # If that appears, fall back to a general knowledge answer.
            if "couldn't find an answer in the document" in ans_doc.lower():
                ans = generate_topical_answer(question, provider)
                mode = "off-doc"
                citations = []
            else:
                ans = ans_doc
                mode = "doc"
                citations = [h["ref"] for h in hits]

            # 3) Show + store assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": ans, "citations": citations, "mode": mode}
            )
            with st.chat_message("assistant"):
                st.markdown(ans)
                if citations:
                    with st.expander("Sources", expanded=False):
                        st.markdown(", ".join(sorted(set(citations))))

        except Exception as e:
            err = f"⚠️ Error while answering: {e}"
            st.session_state.messages.append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.error(err)

# ---------- Footer note ----------
st.markdown("---")
st.caption(
    "Tip: Ask follow-ups naturally. The history stays visible above like a chatbot. "
    "Answers use document citations when available; otherwise you'll see an Off-doc (related) answer."
)
