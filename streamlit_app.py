# streamlit_app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from rag_core import ingest_file, load_store, retrieve
from llm_providers import generate_doc_answer, generate_topical_answer, provider_names


# ------------------------- Page & Style -------------------------
st.set_page_config(page_title="Chat with RAG", page_icon="💬", layout="wide")

CUSTOM_CSS = """
<style>
/* tighter chat bubbles */
.block-container { padding-top: 1rem; }
.chat-citation-chip {
    display:inline-block; padding:4px 8px; border-radius:12px;
    border:1px solid rgba(0,0,0,0.1); margin: 0 6px 6px 0; font-size: 0.85rem;
    background: #f6f6f9;
}
.small-muted { color:#6b7280; font-size: 0.85rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("💬 Chat with RAG (PDF/PPT)")
st.caption("Upload a document, ask questions like a chatbot. Answers are grounded in the file when possible, otherwise marked **Off-doc (related)**.")


# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", provider_names(), index=0)
    st.caption("Add keys in Streamlit → Settings → Secrets")

    st.markdown("---")
    st.subheader("🧹 Utilities")
    col_util_a, col_util_b = st.columns(2)
    with col_util_a:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.success("Cleared chat history.")
    with col_util_b:
        if st.button("Clear Doc"):
            st.session_state.doc_id = None
            st.session_state.doc_name = None
            st.session_state.doc_refs = []  # unique page_#/slide_#
            st.success("Cleared indexed document.")

    st.markdown("---")
    st.subheader("💡 Sample questions")
    st.caption("Click to insert below")
    SAMPLE_QS = [
        "Give a 5-bullet executive summary.",
        "List key findings with page references.",
        "What assumptions were stated?",
        "What are the limitations?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QS):
        if cols[i % 2].button(q, key=f"sample_{i}"):
            st.session_state.pending_question = q

    st.markdown("---")
    debug = st.toggle("Show retrieval debug", value=False, help="Shows top retrieved chunks and scores for the last answer.")


# ------------------------- Session State -------------------------
def _init_state():
    if "doc_id" not in st.session_state:
        st.session_state.doc_id: Optional[str] = None
    if "doc_name" not in st.session_state:
        st.session_state.doc_name: Optional[str] = None
    if "doc_refs" not in st.session_state:
        st.session_state.doc_refs: List[str] = []
    if "messages" not in st.session_state:
        # messages: [{"role":"user"/"assistant","content":str,"citations":[...],"retrieved":[{ref,text,score},...] }]
        st.session_state.messages: List[Dict] = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question: Optional[str] = None

_init_state()


# ------------------------- Upload & Indexing -------------------------
uploaded = st.file_uploader(
    "Upload a PDF or PPT/PPTX",
    type=["pdf", "ppt", "pptx"],
    accept_multiple_files=False,
    help="After indexing, chat below."
)

col_left, col_right = st.columns([1, 2])

with col_left:
    if uploaded is not None and st.button("📚 Index document", use_container_width=True):
        tmp_path = Path("/tmp") / uploaded.name
        tmp_path.write_bytes(uploaded.read())
        try:
            with st.spinner("Indexing…"):
                doc_id = ingest_file(str(tmp_path))
            st.session_state.doc_id = doc_id
            st.session_state.doc_name = uploaded.name

            # load refs summary for the sidebar "Document" info
            doc = load_store("store", doc_id)
            unique_refs = sorted({c["ref"] for c in doc.chunks})
            st.session_state.doc_refs = unique_refs

            st.success(f"Indexed ✅  **{uploaded.name}**  (doc_id: {doc_id[:8]}…)")
        except Exception as e:
            st.error(f"Failed to index: {e}")

with col_right:
    if st.session_state.doc_id:
        st.info(
            f"**Active document:** {st.session_state.doc_name or 'Untitled'}  \n"
            f"Refs detected: {len(st.session_state.doc_refs)} (e.g., "
            + ", ".join(st.session_state.doc_refs[:5]) + ("…" if len(st.session_state.doc_refs) > 5 else ")")
        )


# ------------------------- Chat History -------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            chips = " ".join([f"<span class='chat-citation-chip'>{ref}</span>" for ref in sorted(set(msg["citations"]))])
            st.markdown(chips, unsafe_allow_html=True)


# ------------------------- Compose question -------------------------
def ask_and_answer(question: str):
    """Core flow: retrieve → try doc-only answer → fallback off-doc → append to history."""
    # Echo user
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not st.session_state.doc_id:
        reply = "Please upload and index a document first."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        return

    try:
        with st.spinner("Thinking…"):
            # Retrieve top chunks (always try doc-grounded first)
            doc = load_store("store", st.session_state.doc_id)
            hits = retrieve(doc, question, k=6)  # [{'id','ref','text','score'}]

            # Try doc-only answer first
            ans_doc = generate_doc_answer(hits, question, provider)

        # If doc-only template says it couldn't find an answer → fallback
        if "couldn't find an answer in the document" in ans_doc.lower():
            with st.spinner("Answering with related knowledge…"):
                ans = generate_topical_answer(question, provider)
            mode = "off-doc"
            citations: List[str] = []
        else:
            ans = ans_doc
            mode = "doc"
            citations = [h["ref"] for h in hits]

        # Show assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ans,
                "citations": citations,
                "mode": mode,
                "retrieved": hits,  # keep for debug panel
            }
        )
        with st.chat_message("assistant"):
            st.markdown(ans)
            if citations:
                chips = " ".join([f"<span class='chat-citation-chip'>{ref}</span>" for ref in sorted(set(citations))])
                st.markdown(chips, unsafe_allow_html=True)

    except Exception as e:
        err = f"⚠️ Error while answering: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)


# Send question either from chat_input or sample buttons
question = st.chat_input("Type a message…")
if st.session_state.pending_question and not question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    ask_and_answer(question)


# ------------------------- Right-side Interactive Panels -------------------------
with st.expander("📄 Document overview", expanded=bool(st.session_state.doc_id)):
    if not st.session_state.doc_id:
        st.caption("No document indexed yet.")
    else:
        st.markdown(f"**Name:** {st.session_state.doc_name or 'Untitled'}")
        st.markdown(f"**Refs found:** {len(st.session_state.doc_refs)}")
        if st.session_state.doc_refs:
            st.code(", ".join(st.session_state.doc_refs[:50]), language="text")

with st.expander("🔎 Retrieval debug (last answer)", expanded=debug):
    # show top hits for the most recent assistant message
    last_assistant_msgs = [m for m in st.session_state.messages if m.get("role") == "assistant"]
    if not last_assistant_msgs:
        st.caption("Ask a question to see retrieval results.")
    else:
        last = last_assistant_msgs[-1]
        hits = last.get("retrieved") or []
        if not hits:
            st.caption("No retrieval data recorded.")
        else:
            for h in hits:
                st.markdown(f"**{h['ref']}** — score: `{h.get('score', 0):.3f}`")
                st.write(h["text"])
                st.markdown("---")

# ------------------------- Export Chat -------------------------
st.markdown("---")
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    if st.session_state.messages:
        md = []
        for m in st.session_state.messages:
            role = "User" if m["role"] == "user" else "Assistant"
            md.append(f"**{role}:** {m['content']}")
            cits = m.get("citations") or []
            if cits:
                md.append(f"_Citations:_ {', '.join(sorted(set(cits)))}")
            md.append("")
        md_text = "\n".join(md)
        st.download_button("💾 Download Chat (Markdown)", data=md_text, file_name="chat.md", mime="text/markdown")
with col_dl2:
    if st.session_state.messages:
        json_blob = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button("💾 Download Chat (JSON)", data=json_blob, file_name="chat.json", mime="application/json")
