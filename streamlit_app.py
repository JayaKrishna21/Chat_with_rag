# streamlit_app.py (simple chatbot)
from pathlib import Path
from typing import List, Dict

import streamlit as st
from rag_core import ingest_file, load_store, retrieve
from llm_providers import generate_doc_answer, generate_topical_answer  # uses Groq

st.set_page_config(page_title="Chat with RAG", page_icon="💬", layout="wide")
st.title("💬 Chat with RAG")
st.caption("Upload a PDF/PPT, index it, then chat. Answers use the document when possible; otherwise they say Off-doc (related).")

# --- session state ---
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict] = []

# --- upload & index ---
uploaded = st.file_uploader("Upload a file", type=["pdf", "ppt", "pptx"], accept_multiple_files=False)

col1, col2 = st.columns([1, 2])
with col1:
    if uploaded is not None and st.button("Index"):
        tmp = Path("/tmp") / uploaded.name
        tmp.write_bytes(uploaded.read())
        try:
            doc_id = ingest_file(str(tmp))
            st.session_state.doc_id = doc_id
            st.session_state.messages = []  # start fresh chat for this doc
            st.success("Indexed ✅ — you can start chatting below.")
        except Exception as e:
            st.error(f"Failed to index: {e}")

with col2:
    if st.session_state.doc_id:
        st.info("Doc is indexed. Ask questions below.")

# --- show history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        cits = m.get("citations") or []
        if cits:
            st.caption("Sources: " + ", ".join(sorted(set(cits))))

# --- chat input ---
q = st.chat_input("Type your question…")
if q:
    # show user msg
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # answer
    if not st.session_state.doc_id:
        reply = "Please upload and index a document first."
        citations: List[str] = []
    else:
        try:
            doc = load_store("store", st.session_state.doc_id)
            hits = retrieve(doc, q, k=6)

            # try doc-grounded first
            ans_doc = generate_doc_answer(hits, q, provider="Groq")
            if "couldn't find an answer in the document" in ans_doc.lower():
                reply = generate_topical_answer(q, provider="Groq")
                citations = []
            else:
                reply = ans_doc
                citations = [h["ref"] for h in hits]
        except Exception as e:
            reply = f"⚠️ Error: {e}"
            citations = []

    # show assistant msg
    st.session_state.messages.append({"role": "assistant", "content": reply, "citations": citations})
    with st.chat_message("assistant"):
        st.markdown(reply)
        if citations:
            st.caption("Sources: " + ", ".join(sorted(set(citations))))
