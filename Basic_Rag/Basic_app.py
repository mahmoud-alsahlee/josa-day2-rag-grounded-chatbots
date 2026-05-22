import os
import uuid
from typing import List

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------------------------------
# 1. Load environment variables (OpenRouter)
# ----------------------------------------------------
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


# ----------------------------------------------------
# 2. Basic configuration
# ----------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"


def openrouter_model(model: str) -> str:
    """Map short model names to OpenRouter provider/model slugs."""
    if "/" in model:
        return model
    return f"openai/{model}"

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "faq_rag_collection"

FAQ_FILE_PATH = "data/Basic_FQA.txt"


# ----------------------------------------------------
# 3. Initialize ChromaDB
# ----------------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ----------------------------------------------------
# 4. Load FAQ documents
# ----------------------------------------------------
def load_faq_file(file_path: str) -> str:
    """
    Loads the FAQ document from a text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FAQ file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# ----------------------------------------------------
# 5. Split document into chunks
# ----------------------------------------------------
def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Splits long text into overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ----------------------------------------------------
# 6. Convert chunks into embeddings
# ----------------------------------------------------
def get_embedding(text: str) -> List[float]:
    """
    Creates an embedding vector via OpenRouter.
    """
    response = client.embeddings.create(
        model=openrouter_model(EMBEDDING_MODEL),
        input=text,
    )

    return response.data[0].embedding


# ----------------------------------------------------
# 7. Store chunks in vector database
# ----------------------------------------------------
def ingest_faq_documents():
    """
    Loads FAQ file, chunks it, embeds each chunk,
    and stores everything in ChromaDB.
    """

    existing_count = collection.count()

    if existing_count > 0:
        return f"Documents already ingested. Current chunks in DB: {existing_count}"

    faq_text = load_faq_file(FAQ_FILE_PATH)
    chunks = split_text_into_chunks(faq_text)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())

        ids.append(chunk_id)
        documents.append(chunk)
        embeddings.append(get_embedding(chunk))
        metadatas.append({
            "source": FAQ_FILE_PATH,
            "chunk_index": index
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return f"Ingestion completed. Stored {len(chunks)} chunks in ChromaDB."


# ----------------------------------------------------
# 8. Retrieve relevant chunks
# ----------------------------------------------------
def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3
) -> List[str]:
    """
    Embeds the user question and retrieves the most relevant chunks.
    """

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]

    return retrieved_docs


# ----------------------------------------------------
# 9. Generate answer using only retrieved context
# ----------------------------------------------------
def generate_answer(question: str, context_chunks: List[str]) -> str:
    """
    Generates an answer via OpenRouter, grounded only in retrieved context.
    """

    context = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model=openrouter_model(GENERATION_MODEL),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful RAG chatbot. "
                    "Answer using ONLY the provided context. "
                    "If the answer is not in the context, say exactly: "
                    "I do not know based on the provided documents."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ----------------------------------------------------
# 10. Streamlit UI
# ----------------------------------------------------
st.set_page_config(
    page_title="Basic RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("Basic RAG Chatbot")
st.write("This chatbot answers questions using only the uploaded FAQ knowledge base.")

with st.sidebar:
    st.header("RAG Pipeline")
    st.write("1. Load FAQ documents")
    st.write("2. Split into chunks")
    st.write("3. Convert chunks into embeddings")
    st.write("4. Store in vector database")
    st.write("5. Retrieve relevant chunks")
    st.write("6. Answer using retrieved context")

    if st.button("Ingest FAQ Documents"):
        with st.spinner("Ingesting documents..."):
            message = ingest_faq_documents()
            st.success(message)

question = st.text_input("Ask a question about the FAQ document:")

if st.button("Ask") and question:
    if collection.count() == 0:
        st.warning("Please ingest the FAQ documents first from the sidebar.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved_chunks = retrieve_relevant_chunks(question)

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, retrieved_chunks)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)