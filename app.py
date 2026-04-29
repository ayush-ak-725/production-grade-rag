import streamlit as st
import time
from rag_pipeline import RAGPipeline

# --- Page Config ---
st.set_page_config(page_title="BioAge RAG", page_icon="🧬", layout="centered")

# --- Cold Start: Initialize Pipeline ---
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

rag = get_pipeline()

# --- UI Layout ---
st.title("🧬 Longevity Research Assistant")
st.markdown("Ask anything about Metformin, Rapamycin, or Age-related biology.")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How does Metformin affect aging?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate RAG Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing research papers..."):
            # Execute your RAG Pipeline
            response = rag.query(prompt)

            # Display Text
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # 3. Generate and Stream Voice
        with st.spinner("Synthesizing voice..."):
            audio_path = rag.speak_response(response)

            if audio_path:
                # Minimalistic Audio Player
                st.audio(audio_path, format="audio/wav", autoplay=True)
                st.success("Audio generated via Sarvam AI (Bulbul v3)")

# --- Sidebar Logs ---
with st.sidebar:
    st.header("Pipeline Logs")
    if "tokens_before" in st.session_state:
        st.metric("Context Tokens", st.session_state.tokens_before)
    st.info("Using LLMLingua-2 Compression & Cross-Encoder Reranking.")