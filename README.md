# JOSA Day 2 - RAG & Grounded Chatbots

**JOSA AI Bootcamp 2026 - Day 2: AI Architectures: RAG & Grounded Chatbots**

Hands-on lab for building a **grounded chatbot** with **Retrieval-Augmented Generation (RAG)**. The chatbot answers from a custom FAQ / knowledge base instead of relying only on the LLM’s memory.

---

## Learning goals

By the end of this lab, you should be able to:

- Explain what RAG is and why **grounding** reduces hallucinations
- Chunk documents, create embeddings, and store them in a **vector database**
- Retrieve relevant context for a user question
- Write prompts that answer **only** from retrieved context
- Handle missing context with safe **“I don’t know”** behavior
- (Enhanced track) Tune retrieval, detect failure modes, and add basic logging/evaluation

---

## RAG pipeline

```text
User question
     ↓
Embed question → search vector DB
     ↓
Retrieve top-k relevant chunks
     ↓
LLM answers using only that context
     ↓
Grounded answer (+ evidence / confidence in enhanced track)
```

---

## Project structure

| Track | Folder | What it covers |
|-------|--------|----------------|
| **Basic** | `Basic_Rag/` | Simple chunking, ChromaDB, grounded Q&A via Streamlit or notebook |
| **Enhanced** | `Enhanced_Rag/` | Q&A chunking, metadata filters, context compression, structured answers, logs & evaluation |

```
josa-day2-rag-grounded-chatbots/
├── Basic_Rag/
│   ├── Basic_app.py              # Streamlit app
│   └── Basic_rag_chatbot.ipynb   # Same pipeline in Jupyter
├── Enhanced_Rag/
│   ├── Enhanced_app.py            # Streamlit app (production-style features)
│   └── Enhanced_rag_chatbot.ipynb
├── data/
│   ├── Basic_FQA.txt             # Basic track knowledge base
│   ├── full_detailed_data.txt    # Enhanced track (Category / Q / A metadata)
│   └── sport_FQA.txt             # Optional extra FAQ
├── requirements.txt
└── .env                          # You create this (not committed)
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**2. OpenRouter API key**

Get a key at [openrouter.ai](https://openrouter.ai/) and create `.env` in the project root:

```env
OPENROUTER_API_KEY=your-key-here
```

**3. Run from the project root** (so `data/` paths resolve correctly)

**Basic - Streamlit**

```bash
streamlit run Basic_Rag/Basic_app.py
```

**Basic - Jupyter:** open `Basic_Rag/Basic_rag_chatbot.ipynb`

**Enhanced - Streamlit**

```bash
streamlit run Enhanced_Rag/Enhanced_app.py
```

Point `FAQ_FILE_PATH` in `Enhanced_app.py` to `data/full_detailed_data.txt` if you use that file (the notebook already does).

**Enhanced - Jupyter:** open `Enhanced_Rag/Enhanced_rag_chatbot.ipynb`

In each app, use the sidebar to **ingest documents** before asking questions.

---

## Data files

- **`data/Basic_FQA.txt`** - used by the basic track
- **`data/full_detailed_data.txt`** - structured FAQ with `Category:`, `Language:`, `Version:` headers (enhanced track)
- **`data/sport_FQA.txt`** - optional extra content for experiments

---

## Why grounding matters (bootcamp context)

LLMs can sound confident while being wrong (**hallucination**). A grounded chatbot retrieves trusted snippets first (FAQs, policies, manuals), then instructs the model to answer from that context only-improving reliability for real-world Q&A.

---

## Tech stack

- [OpenRouter](https://openrouter.ai/) (chat + embeddings via OpenAI-compatible API)
- [ChromaDB](https://www.trychroma.com/) (vector store)
- [Streamlit](https://streamlit.io/) (UI)
- Python 3.9+
