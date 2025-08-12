import os
from typing import List, Dict

# Gemini (google-genai)
try:
    from google import genai
    _has_genai = True
except Exception:
    _has_genai = False

# Groq (official SDK)
try:
    from groq import Groq
    _has_groq = True
except Exception:
    _has_groq = False


def provider_names() -> List[str]:
    return ["Gemini", "Groq"]


DOC_PROMPT = (
    "You are a careful assistant. Answer ONLY using the provided DOCUMENT EXCERPTS.\n"
    "If the excerpts do not contain the answer, say: \"I couldn't find an answer in the document.\"\n"
    "Always include inline citations like [ref: <page_or_slide_ids>]. Keep answers concise."
)

TOPICAL_PROMPT = (
    "You are a helpful assistant. The user's question is on-topic for the uploaded document,\n"
    "but the document lacks the needed details. Give a concise, correct answer from your general knowledge.\n"
    "Start your answer with: Off-doc (related): and DO NOT fabricate citations."
)

def _make_context(hits: List[Dict], question: str, max_chars=6000) -> str:
    blocks, used = [], set()
    for h in hits:
        if h["id"] in used:
            continue
        blocks.append(f"[{h['ref']}]\n{h['text']}\n")
        used.add(h["id"])
    ctx = ("\n---\n".join(blocks))[:max_chars]
    return f"DOCUMENT EXCERPTS:\n{ctx}\n\nQUESTION: {question}"


# ---------- Gemini ----------
def _gemini_client():
    if not _has_genai:
        raise RuntimeError("google-genai not installed; add it to requirements.txt")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in secrets")
    return genai.Client(api_key=key)

def _gemini_doc_answer(hits: List[Dict], question: str) -> str:
    client = _gemini_client()
    prompt = DOC_PROMPT + "\n\n" + _make_context(hits, question)
    rsp = client.responses.generate(model="gemini-2.5-flash", input=prompt)
    return rsp.output_text.strip()

def _gemini_topical_answer(question: str) -> str:
    client = _gemini_client()
    prompt = TOPICAL_PROMPT + "\n\nQ: " + question
    rsp = client.responses.generate(model="gemini-2.5-pro", input=prompt)
    return rsp.output_text.strip()


# ---------- Groq ----------
_GROQ_MODEL_DOC = os.getenv("GROQ_MODEL_DOC", "llama-3.1-8b-instant")        # fast
_GROQ_MODEL_TOP = os.getenv("GROQ_MODEL_TOPICAL", "llama-3.3-70b-versatile")  # stronger

def _groq_client():
    if not _has_groq:
        raise RuntimeError("groq package not installed; add it to requirements.txt")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing in secrets")
    return Groq(api_key=key)

def _groq_doc_answer(hits: List[Dict], question: str) -> str:
    client = _groq_client()
    content = DOC_PROMPT + "\n\n" + _make_context(hits, question)
    rsp = client.chat.completions.create(
        model=_GROQ_MODEL_DOC,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
    )
    return rsp.choices[0].message.content.strip()

def _groq_topical_answer(question: str) -> str:
    client = _groq_client()
    content = TOPICAL_PROMPT + "\n\nQ: " + question
    rsp = client.chat.completions.create(
        model=_GROQ_MODEL_TOP,
        messages=[{"role": "user", "content": content}],
        temperature=0.6,
    )
    return rsp.choices[0].message.content.strip()


# ---------- Unified ----------
def generate_doc_answer(hits: List[Dict], question: str, provider: str) -> str:
    if provider == "Gemini":
        return _gemini_doc_answer(hits, question)
    if provider == "Groq":
        return _groq_doc_answer(hits, question)
    raise ValueError(f"Unknown provider: {provider}")

def generate_topical_answer(question: str, provider: str) -> str:
    if provider == "Gemini":
        return _gemini_topical_answer(question)
    if provider == "Groq":
        return _groq_topical_answer(question)
    raise ValueError(f"Unknown provider: {provider}")
