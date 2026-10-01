import os
import streamlit as st

from rag import build_knowledge_base, ask_hudabot


st.set_page_config(
    page_title="HudaBot",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 HudaBot")
st.caption("Personal Retrieval-Augmented Generation (RAG) Assistant")


with st.sidebar:
    st.header("About HudaBot")

    st.write(
        "HudaBot answers questions using a custom personal "
        "knowledge base with RAG."
    )

    st.divider()

    st.write("**RAG Pipeline**")
    st.write("✓ Document preprocessing")
    st.write("✓ Text chunking")
    st.write("✓ Sentence Transformer embeddings")
    st.write("✓ FAISS vector search")
    st.write("✓ LLM response generation")
    st.write("✓ Conversation history")

    st.divider()

    # Check whether Groq API key is available
    if os.getenv("GROQ_API_KEY"):
        st.success("LLM API configured")
    else:
        try:
            if st.secrets.get("GROQ_API_KEY"):
                st.success("LLM API configured")
            else:
                st.warning("Add GROQ_API_KEY to run the chatbot.")
        except Exception:
            st.warning("Add GROQ_API_KEY to run the chatbot.")


@st.cache_resource
def initialize_rag():
    return build_knowledge_base()


# Build the RAG knowledge base
try:
    embedder, index, chunks, sources = initialize_rag()

except Exception as e:
    st.error(f"Could not initialize the RAG system: {e}")
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Ask HudaBot something about the knowledge base..."
)


if question:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching the knowledge base..."):

            try:

                answer, retrieved = ask_hudabot(
                    question,
                    embedder,
                    index,
                    chunks,
                    sources,
                    st.session_state.messages
                )

                st.markdown(answer)

                # Show retrieved RAG sources
                with st.expander("Retrieved Sources"):

                    for item in retrieved:

                        st.write(
                            f"**{item['source']}** "
                            f"(similarity: {item['score']:.3f})"
                        )

                        st.write(item["text"])


                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"Error generating response: {e}"
                )
