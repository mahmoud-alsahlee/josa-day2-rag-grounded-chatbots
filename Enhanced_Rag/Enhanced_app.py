import os
import re
import csv
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st
import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. Load environment variables (OpenRouter)
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. Get a key at https://openrouter.ai/ "
        "and add it to your .env file."
    )

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


# ============================================================
# 2. Configuration
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"


def openrouter_model(model: str) -> str:
    """Map short model names to OpenRouter provider/model slugs."""
    if "/" in model:
        return model
    return f"openai/{model}"

FAQ_FILE_PATH = "data/faq.txt"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "grounded_faq_rag"

LOG_DIR = "logs"
QUERY_LOG_PATH = os.path.join(LOG_DIR, "query_log.csv")
FEEDBACK_LOG_PATH = os.path.join(LOG_DIR, "feedback_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# 3. Initialize ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# 4. Load documents
# ============================================================

def load_text_file(file_path: str) -> str:
    """
    Loads a text document from disk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# ============================================================
# 5. Metadata extraction
# ============================================================

def extract_metadata_from_block(block: str) -> Dict[str, Any]:
    """
    Extracts simple metadata from a text block.

    Example supported metadata:
    Category: refund
    Language: en
    Version: 2026-05
    """

    metadata = {
        "source": FAQ_FILE_PATH,
        "category": "general",
        "language": "en",
        "version": "unknown"
    }

    category_match = re.search(r"Category:\s*(.+)", block, re.IGNORECASE)
    language_match = re.search(r"Language:\s*(.+)", block, re.IGNORECASE)
    version_match = re.search(r"Version:\s*(.+)", block, re.IGNORECASE)

    if category_match:
        metadata["category"] = category_match.group(1).strip()

    if language_match:
        metadata["language"] = language_match.group(1).strip()

    if version_match:
        metadata["version"] = version_match.group(1).strip()

    return metadata


# ============================================================
# 6. Better chunking
# ============================================================

def split_into_qa_chunks(text: str) -> List[Dict[str, Any]]:
    """
    Splits FAQ text into Question + Answer chunks.

    This is better than splitting blindly by character count because
    each chunk keeps the question and its answer together.
    """

    chunks = []

    sections = re.split(r"\n(?=Category:)", text)

    for section in sections:
        section = section.strip()

        if not section:
            continue

        metadata = extract_metadata_from_block(section)

        qa_pairs = re.findall(
            r"(Q:\s*.*?)(?=\nQ:|\Z)",
            section,
            flags=re.DOTALL | re.IGNORECASE
        )

        for qa in qa_pairs:
            clean_chunk = qa.strip()

            if clean_chunk:
                chunks.append({
                    "text": clean_chunk,
                    "metadata": metadata
                })

    return chunks


def split_into_fallback_chunks(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Fallback fixed-size chunking if FAQ-style chunks are not detected.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": FAQ_FILE_PATH,
                    "category": "general",
                    "language": "en",
                    "version": "unknown"
                }
            })

        start += chunk_size - overlap

    return chunks


def create_chunks(text: str) -> List[Dict[str, Any]]:
    """
    Uses Q&A chunking first.
    Falls back to fixed-size chunking if no Q&A chunks are found.
    """

    qa_chunks = split_into_qa_chunks(text)

    if qa_chunks:
        return qa_chunks

    return split_into_fallback_chunks(text)


# ============================================================
# 7. Embedding
# ============================================================

def get_embedding(text: str) -> List[float]:
    """
    Converts text into an embedding vector via OpenRouter.
    """

    response = client.embeddings.create(
        model=openrouter_model(EMBEDDING_MODEL),
        input=text
    )

    return response.data[0].embedding


# ============================================================
# 8. Ingest documents
# ============================================================

def clear_collection() -> None:
    """
    Deletes and recreates the ChromaDB collection.
    Useful when documents are updated.
    """

    global collection

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def ingest_documents(force_reingest: bool = False) -> str:
    """
    Loads, chunks, embeds, and stores the FAQ document.
    """

    if force_reingest:
        clear_collection()

    if collection.count() > 0:
        return f"Documents already ingested. Current chunks: {collection.count()}"

    text = load_text_file(FAQ_FILE_PATH)
    chunks = create_chunks(text)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())

        metadata = chunk["metadata"].copy()
        metadata["chunk_index"] = index
        metadata["ingested_at"] = datetime.now().isoformat()

        ids.append(chunk_id)
        documents.append(chunk["text"])
        embeddings.append(get_embedding(chunk["text"]))
        metadatas.append(metadata)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return f"Ingestion completed. Stored {len(chunks)} chunks."


# ============================================================
# 9. Context optimization
# ============================================================

def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes duplicated retrieved chunks.
    """

    seen = set()
    unique_chunks = []

    for chunk in chunks:
        text = chunk["text"].strip()

        if text not in seen:
            seen.add(text)
            unique_chunks.append(chunk)

    return unique_chunks


def compress_context(
    chunks: List[Dict[str, Any]],
    max_chars: int = 3000
) -> str:
    """
    Keeps only relevant snippets and prevents the prompt from becoming too long.
    Most relevant chunks should already be first.
    """

    context_parts = []
    current_length = 0

    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source", "unknown")
        category = chunk["metadata"].get("category", "general")
        version = chunk["metadata"].get("version", "unknown")

        chunk_text = chunk["text"].strip()

        formatted_chunk = f"""
[Context {i}]
Source: {source}
Category: {category}
Version: {version}
Content:
{chunk_text}
""".strip()

        if current_length + len(formatted_chunk) > max_chars:
            break

        context_parts.append(formatted_chunk)
        current_length += len(formatted_chunk)

    return "\n\n".join(context_parts)


# ============================================================
# 10. Retrieval tuning
# ============================================================

def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3,
    similarity_threshold: float = 0.35,
    category_filter: str = "All"
) -> List[Dict[str, Any]]:
    """
    Retrieves relevant chunks from ChromaDB.

    Includes:
    - top_k retrieval
    - metadata filtering
    - similarity threshold
    - deduplication
    """

    question_embedding = get_embedding(question)

    where_filter = None

    if category_filter and category_filter != "All":
        where_filter = {
            "category": category_filter
        }

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, metadata, distance in zip(documents, metadatas, distances):
        similarity_score = 1 - distance

        if similarity_score >= similarity_threshold:
            retrieved_chunks.append({
                "text": doc,
                "metadata": metadata,
                "distance": distance,
                "similarity_score": similarity_score
            })

    retrieved_chunks = deduplicate_chunks(retrieved_chunks)

    return retrieved_chunks


# ============================================================
# 11. Grounded prompt engineering
# ============================================================

def build_grounded_prompt(
    question: str,
    context: str
) -> str:
    """
    Creates a strict grounded prompt.
    """

    return f"""
You are a grounded FAQ assistant.

Your role:
Answer user questions using only the retrieved context.

Source rule:
Use only the context provided below.
Do not use outside knowledge.
Do not guess.
Do not invent facts.
Do not answer beyond the provided context.

Missing context rule:
If the answer is not clearly supported by the context, say:
"I don't know based on the provided documents."

Partial support rule:
If the context only partially answers the question:
- Answer only the supported part.
- Clearly say what information is missing.

Consistency rule:
Always use this answer format:

Answer:
<short answer>

Evidence:
- Source: <source>
- Relevant text: <short evidence text>

Confidence:
High / Medium / Low

Retrieved context:
{context}

User question:
{question}
""".strip()


# ============================================================
# 12. Generate grounded answer
# ============================================================

def generate_grounded_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    temperature: float = 0.0
) -> str:
    """
    Generates an answer using only retrieved context.
    """

    if not retrieved_chunks:
        return """
Answer:
I don't know based on the provided documents.

Evidence:
- Source: No relevant source found.
- Relevant text: No relevant context was retrieved.

Confidence:
Low
""".strip()

    context = compress_context(retrieved_chunks)

    prompt = build_grounded_prompt(
        question=question,
        context=context
    )

    response = client.chat.completions.create(
        model=openrouter_model(GENERATION_MODEL),
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content


# ============================================================
# 13. RAG failure detection
# ============================================================

def detect_rag_failure(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> List[str]:
    """
    Detects common RAG failure modes.
    """

    failures = []

    if not retrieved_chunks:
        failures.append("Missing answer: no relevant chunks retrieved.")

    if "I don't know based on the provided documents" in answer:
        failures.append("Refusal triggered: answer not found in retrieved context.")

    if len(answer) > 2000:
        failures.append("Long answer: possible too much context or weak prompt control.")

    if "Evidence:" not in answer:
        failures.append("Inconsistent format: answer does not include evidence section.")

    if "Confidence:" not in answer:
        failures.append("Inconsistent format: answer does not include confidence section.")

    return failures


# ============================================================
# 14. Logging
# ============================================================

def append_csv_row(file_path: str, row: Dict[str, Any]) -> None:
    """
    Appends a row to a CSV file.
    """

    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def log_query(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    latency: float,
    failures: List[str]
) -> None:
    """
    Logs query, answer, context, latency, and failure signals.
    """

    sources = [
        chunk["metadata"].get("source", "unknown")
        for chunk in retrieved_chunks
    ]

    categories = [
        chunk["metadata"].get("category", "general")
        for chunk in retrieved_chunks
    ]

    similarities = [
        round(chunk["similarity_score"], 4)
        for chunk in retrieved_chunks
    ]

    row = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "num_retrieved_chunks": len(retrieved_chunks),
        "sources": " | ".join(sources),
        "categories": " | ".join(categories),
        "similarity_scores": " | ".join(map(str, similarities)),
        "latency_seconds": round(latency, 3),
        "failures": " | ".join(failures)
    }

    append_csv_row(QUERY_LOG_PATH, row)


def log_feedback(
    question: str,
    answer: str,
    feedback: str
) -> None:
    """
    Logs user feedback.
    """

    row = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "feedback": feedback
    }

    append_csv_row(FEEDBACK_LOG_PATH, row)


# ============================================================
# 15. Simple evaluation metrics
# ============================================================

def evaluate_single_response(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Basic automatic evaluation signals.
    These are not perfect, but useful for a beginner lab.
    """

    context_text = " ".join([chunk["text"] for chunk in retrieved_chunks]).lower()
    answer_text = answer.lower()

    has_context = len(retrieved_chunks) > 0
    has_evidence = "evidence:" in answer_text
    has_confidence = "confidence:" in answer_text
    refused = "i don't know based on the provided documents" in answer_text

    if has_context and not refused:
        groundedness = "Needs human review"
    elif refused:
        groundedness = "Safe refusal"
    else:
        groundedness = "No context retrieved"

    return {
        "question": question,
        "has_context": has_context,
        "has_evidence_section": has_evidence,
        "has_confidence_section": has_confidence,
        "refused_when_missing": refused,
        "groundedness_signal": groundedness
    }


# ============================================================
# 16. Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Grounded RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title("Grounded RAG Chatbot")
st.write(
    "This version focuses on hallucination reduction, consistency, context optimization, "
    "metadata filtering, evaluation, and basic production hardening."
)


# ---------------- Sidebar ----------------

with st.sidebar:
    st.header("RAG Controls")

    st.subheader("Document Ingestion")

    if st.button("Ingest Documents"):
        with st.spinner("Ingesting documents..."):
            message = ingest_documents(force_reingest=False)
            st.success(message)

    if st.button("Force Re-ingest"):
        with st.spinner("Re-ingesting documents..."):
            message = ingest_documents(force_reingest=True)
            st.success(message)

    st.divider()

    st.subheader("Retrieval Tuning")

    top_k = st.slider(
        "Top-k retrieved chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=-1.0,
        max_value=1.0,
        value=0.35,
        step=0.05
    )

    category_filter = st.selectbox(
        "Metadata category filter",
        options=["All", "refund", "billing", "support", "general"]
    )

    st.divider()

    st.subheader("Consistency Control")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )

    st.caption("Use temperature = 0 for more stable and predictable answers.")


# ---------------- Main app ----------------

if collection.count() == 0:
    st.warning("Please ingest documents first from the sidebar.")

question = st.text_input("Ask a question:")

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []


if st.button("Ask") and question:
    if collection.count() == 0:
        st.error("No documents found. Please ingest documents first.")
    else:
        start_time = time.time()

        with st.spinner("Retrieving relevant chunks..."):
            retrieved_chunks = retrieve_relevant_chunks(
                question=question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                category_filter=category_filter
            )

        with st.spinner("Generating grounded answer..."):
            answer = generate_grounded_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
                temperature=temperature
            )

        latency = time.time() - start_time

        failures = detect_rag_failure(
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )

        log_query(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            latency=latency,
            failures=failures
        )

        st.session_state.last_question = question
        st.session_state.last_answer = answer
        st.session_state.last_chunks = retrieved_chunks

        st.subheader("Answer")
        st.write(answer)

        st.caption(f"Latency: {latency:.2f} seconds")

        if failures:
            st.subheader("Detected RAG Failure Signals")
            for failure in failures:
                st.warning(failure)

        st.subheader("Evaluation Signals")
        evaluation = evaluate_single_response(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )
        st.json(evaluation)

        with st.expander("Retrieved Context"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(chunk["text"])
                st.json({
                    "metadata": chunk["metadata"],
                    "similarity_score": round(chunk["similarity_score"], 4),
                    "distance": round(chunk["distance"], 4)
                })


# ---------------- Feedback ----------------

st.divider()
st.subheader("User Feedback")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("👍 Helpful"):
        if st.session_state.last_answer:
            log_feedback(
                st.session_state.last_question,
                st.session_state.last_answer,
                "thumbs_up"
            )
            st.success("Feedback saved.")

with col2:
    if st.button("👎 Not Helpful"):
        if st.session_state.last_answer:
            log_feedback(
                st.session_state.last_question,
                st.session_state.last_answer,
                "thumbs_down"
            )
            st.success("Feedback saved.")

with col3:
    rating = st.selectbox(
        "Star rating",
        options=["", "1", "2", "3", "4", "5"]
    )

    if rating:
        if st.session_state.last_answer:
            log_feedback(
                st.session_state.last_question,
                st.session_state.last_answer,
                f"star_{rating}"
            )
            st.success("Rating saved.")


# ---------------- Logs viewer ----------------

st.divider()
st.subheader("Logs and Observability")

log_tab, feedback_tab = st.tabs(["Query Logs", "Feedback Logs"])

with log_tab:
    if os.path.exists(QUERY_LOG_PATH):
        logs_df = pd.read_csv(QUERY_LOG_PATH)
        st.dataframe(logs_df)
    else:
        st.info("No query logs yet.")

with feedback_tab:
    if os.path.exists(FEEDBACK_LOG_PATH):
        feedback_df = pd.read_csv(FEEDBACK_LOG_PATH)
        st.dataframe(feedback_df)
    else:
        st.info("No feedback logs yet.")