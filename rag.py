import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai


DATA_DIR = Path("data")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    documents = []

    for file_path in DATA_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_knowledge_base():
    documents = load_documents()

    chunks = []
    sources = []

    for document in documents:
        document_chunks = chunk_text(document["text"])

        for chunk in document_chunks:
            chunks.append(chunk)
            sources.append(document["source"])

    embedder = SentenceTransformer(MODEL_NAME)

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return embedder, index, chunks, sources


def retrieve(question, embedder, index, chunks, sources, top_k=4):
    question_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(question_embedding, top_k)

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        retrieved.append({
            "text": chunks[idx],
            "source": sources[idx],
            "score": float(score)
        })

    return retrieved


def generate_answer(question, retrieved, history):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    context = "\n\n".join(
        f"[Source: {item['source']}]\n{item['text']}"
        for item in retrieved
    )

    conversation = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[-6:]
    )

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def ask_hudabot(question, embedder, index, chunks, sources, history):
    retrieved = retrieve(
        question,
        embedder,
        index,
        chunks,
        sources,
        top_k=4
    )

    answer = generate_answer(
        question,
        retrieved,
        history
    )

    return answer, retrieved
