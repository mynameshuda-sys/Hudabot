# HudaBot - Personal RAG Chatbot

HudaBot is a personalized Retrieval-Augmented Generation chatbot built for an NLP semester project.

## Architecture

Documents -> Text Chunking -> Sentence Transformer Embeddings -> FAISS -> Similarity Retrieval -> Prompt Engineering -> Gemini LLM -> Streamlit Chatbot

## Features

- Custom personal dataset
- Document loading and preprocessing
- Text chunking
- Sentence Transformer embeddings
- FAISS vector database
- Similarity-based retrieval
- Gemini LLM generation
- Prompt engineering
- Conversation history
- Streamlit web interface
- Retrieved-source display

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export Grok="YOUR_API_KEY"
streamlit run app.py
```

## Dataset

The `data/` directory contains custom-created personal profile, education, skills, project, and HudaBot information.

## Deployment

Deploy the GitHub repository using Streamlit Community Cloud. Add `Grok_API_KEY` as a Streamlit secret.

## Important

Do not commit or upload your API key.
