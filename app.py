import streamlit as st
import logging
from langchain_core.messages import HumanMessage, AIMessage
from rag import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Indian History RAG Assistant",
    page_icon="📜",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    with st.spinner("Loading AI Models and Vector Database... (This may take a moment)"):
        try:
            st.session_state.rag_pipeline = RAGPipeline()
            logger.info("RAG Pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            st.error(f"Failed to load the AI models. Please check your environment and logs.\n\nError: {e}")
            st.session_state.rag_pipeline = None

# --- UI Layout ---
st.title("📜 Indian History Query Assistant")
st.caption("Ask questions about Indian History using our RAG pipeline.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input Handling ---
if prompt := st.chat_input("Ask about Indian History..."):
    pipeline = st.session_state.rag_pipeline

    if pipeline is None:
        st.error("RAG Pipeline is not loaded. Please refresh the page.")
        st.stop()

    # 1. Add user message to UI and state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # 2. Prepare chat history for the pipeline
    chat_history_msgs = []
    for msg in st.session_state.messages[:-1]:  
        if msg["role"] == "user":
            chat_history_msgs.append(HumanMessage(content=msg["content"]))
        else:
            chat_history_msgs.append(AIMessage(content=msg["content"]))

    # 3. Invoke the RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = pipeline.invoke_query(prompt, chat_history_msgs)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.exception("Error during invocation")