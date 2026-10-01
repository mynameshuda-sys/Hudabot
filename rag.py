import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(__file__).parent / "data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    """
    Get the Groq API key.

    Local:
        GROQ_API_KEY environment variable

    Streamlit Cloud:
        Streamlit Secrets
    """

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if api_key:
            return api_key

    except Exception:
        pass

    return None


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_documents():
    documents = []

    if not DATA_DIR.exists():
        raise ValueError(
            f"Data folder not found: {DATA_DIR}"
        )

    files = list(DATA_DIR.glob("*.txt"))

    if not files:
        raise ValueError(
            "No .txt documents were found in the data folder. "
            "Make sure your TXT files are uploaded to GitHub inside data/."
        )

    for file_path in files:

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append({
            "source": file_path.name,
            "text": text
        })

    if not documents:
        raise ValueError(
            "TXT files were found, but they are empty."
        )

    return documents


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=700,
    overlap=100
):
    """
    Divide text into overlapping word-based chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def build_knowledge_base():

    documents = load_documents()

    chunks = []
    sources = []

    # ----------------------------------------
    # Chunk every document
    # ----------------------------------------

    for document in documents:

        document_chunks = chunk_text(
            document["text"]
        )

        for chunk in document_chunks:

            chunks.append(chunk)

            sources.append(
                document["source"]
            )

    if not chunks:
        raise ValueError(
            "No text chunks were created from the documents."
        )

    # ----------------------------------------
    # Load embedding model
    # ----------------------------------------

    embedder = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # ----------------------------------------
    # Generate embeddings
    # ----------------------------------------

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    # ----------------------------------------
    # Create FAISS index
    # ----------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    # ----------------------------------------
    # Return everything needed by the app
    # ----------------------------------------

    return (
        embedder,
        index,
        chunks,
        sources
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question,
    embedder,
    index,
    chunks,
    sources,
    top_k=4
):

    # ----------------------------------------
    # Convert question to embedding
    # ----------------------------------------

    question_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # ----------------------------------------
    # Search FAISS
    # ----------------------------------------

    number_to_retrieve = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        number_to_retrieve
    )

    retrieved = []

    # ----------------------------------------
    # Collect retrieved chunks
    # ----------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        retrieved.append({
            "text": chunks[idx],
            "source": sources[idx],
            "score": float(score)
        })

    return retrieved


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved,
    history
):

    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    client = Groq(
        api_key=api_key
    )

    # ----------------------------------------
    # Retrieved context
    # ----------------------------------------

    context = "\n\n".join(
        f"[Source: {item['source']}]\n"
        f"{item['text']}"
        for item in retrieved
    )

    # ----------------------------------------
    # Recent conversation
    # ----------------------------------------

    conversation = "\n".join(
        f"{message['role']}: "
        f"{message['content']}"
        for message in history[-6:]
    )

    # ----------------------------------------
    # Prompt engineering
    # ----------------------------------------

    prompt = f"""
You are HudaBot, a personalized AI knowledge assistant.

Answer the user's question using the retrieved context below.

IMPORTANT RULES:

1. Use the retrieved context as your primary knowledge source.
2. Do not invent personal facts.
3. If the answer is not supported by the retrieved context, say:
   "I don't have that information in my knowledge base."
4. You may combine information from multiple retrieved chunks.
5. Keep answers clear and concise.
6. Do not mention internal implementation details unless the user asks.

RETRIEVED CONTEXT:
{context}

RECENT CONVERSATION:
{conversation}

USER QUESTION:
{question}

ANSWER:
"""

    # ----------------------------------------
    # Groq LLM request
    # ----------------------------------------

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are HudaBot, a helpful "
                    "personal knowledge assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content


# ============================================================
# COMPLETE QUESTION-ANSWER PIPELINE
# ============================================================

def ask_hudabot(
    question,
    embedder,
    index,
    chunks,
    sources,
    history
):

    # Step 1: retrieve relevant information
    retrieved = retrieve(
        question,
        embedder,
        index,
        chunks,
        sources,
        top_k=4
    )

    # Step 2: generate answer using retrieved information
    answer = generate_answer(
        question,
        retrieved,
        history
    )

    return answer, retrieved
