# 💬 Chat with RAG (PDF/PPT) — Streamlit

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red.svg)](https://YOUR_STREAMLIT_APP_URL)
[![Open in GitHub](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/YOUR_HANDLE/chat-with-rag)

Upload a PDF/PPT and chat with it.
- On-topic questions → **doc-grounded** answers with citations
- On-topic but not in doc → **Off-doc (related)** answers from model knowledge
- Irrelevant questions → `"irrelevant"` error

https://github.com/YOUR_HANDLE/YOUR_HANDLE — add the **Live Demo** link to your profile README and pin this repo.

## ✨ Demo GIF
Add a short GIF at `docs/demo.gif` showing: upload → ask → doc-grounded answer + citations → off-doc route → irrelevant example.

## 🚀 Quickstart (Local)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
