# Streamlit UI for GenAI Knowledge Assistant

import streamlit as st
import os
from pathlib import Path
import tempfile
import logging

from rag_pipeline import RAGPipeline
from config import MAX_UPLOAD_SIZE, ALLOWED_EXTENSIONS, MAX_HISTORY
from utils import load_document

# Configure logging
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GenAI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "show_retrieval_details" not in st.session_state:
    st.session_state.show_retrieval_details = False


def format_chat_message(role: str, content: str) -> str:
    """Format a chat message for display."""
    if role == "user":
        return f"👤 **You:** {content}"
    else:
        return f"🤖 **Assistant:** {content}"


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary location.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to saved file
    """
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown(
        '<h1 class="main-header">🤖 GenAI Knowledge Assistant</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666;">Powered by LangChain + RAG + FAISS</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
        if uploaded_files:
            if st.button("📤 Process & Add Documents", use_container_width=True):
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Save file
                        file_path = save_uploaded_file(uploaded_file)
                        
                        # Add to pipeline
                        success = st.session_state.rag_pipeline.add_document(file_path)
                        
                        if success:
                            st.session_state.uploaded_files.append(uploaded_file.name)
                            st.success(f"✅ Processed: {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process: {uploaded_file.name}")
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        error_str = str(e)
                        # Display specific error messages
                        if "quota" in error_str.lower() or "429" in error_str:
                            st.error("⚠️ **API Quota Error**\n\nYour account has insufficient credits. Check your API key settings.")
                        elif "not found" in error_str.lower() or "404" in error_str:
                            st.error("⚠️ **Model Not Found Error**\n\nThe specified model is not available. Please check your setup.")
                        elif "permission" in error_str.lower() or "403" in error_str or "unauthorized" in error_str.lower():
                            st.error("⚠️ **Authorization Error**\n\nYour API key doesn't have the required permissions.")
                        elif "no readable text" in error_str.lower() or "no pages" in error_str.lower():
                            st.error(f"⚠️ **Document Error**\n\n{uploaded_file.name} appears to be empty or corrupted. Please use a valid PDF with text content.")
                        else:
                            st.error(f"❌ Error: {error_str[:200]}")
        
        # Display uploaded documents
        if st.session_state.uploaded_files:
            st.subheader("📄 Loaded Documents")
            for doc in st.session_state.uploaded_files:
                st.text(f"✓ {doc}")
        
        # Pipeline statistics
        st.subheader("📊 Pipeline Stats")
        stats = st.session_state.rag_pipeline.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Documents",
                stats["vector_store"]["document_count"]
            )
        with col2:
            st.metric(
                "Status",
                "Active" if stats["vector_store"]["status"] == "active" else "Empty"
            )
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.rag_pipeline.clear_documents()
            st.session_state.uploaded_files = []
            st.session_state.chat_history = []
            st.success("Documents cleared!")
            st.rerun()
    
    # Main content area
    if not st.session_state.uploaded_files:
        st.info(
            "👈 **Get Started:** Upload PDF or text documents using the sidebar. "
            "Then ask questions about them!",
            icon="ℹ️"
        )
        
        # Example usage
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Upload Documents**: Click the file uploader in the sidebar to select PDF or text files
            2. **Process**: Click "Process & Add Documents" to add them to the knowledge base
            3. **Ask Questions**: Type your question below and get answers based on your documents
            4. **View Sources**: Check which documents your answer came from
            
            **Example Questions:**
            - "Summarize the main points"
            - "What are the key findings?"
            - "Explain the technical details"
            """)
    else:
        st.success("✅ Documents loaded successfully. You can now ask questions about the uploaded files.")
        
        # Chat interface
        st.subheader("💬 Ask Questions")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📌 Source Documents"):
                            for source in message["sources"]:
                                st.write(f"**Source:** {source['source']}")
                                st.caption(f"Preview: {source['content_preview']}")
        
        # Input area
        question = st.chat_input(
            "Ask a question about your documents...",
            key="question_input"
        )
        
        if question:
            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents..."):
                    result = st.session_state.rag_pipeline.query(question)
                
                # Display answer
                answer = result["answer"]
                st.markdown(answer)
                
                # Display sources if available
                if result["source_documents"]:
                    with st.expander("📌 Source Documents"):
                        for source in result["source_documents"]:
                            st.write(f"**Source:** {source['source']}")
                            st.caption(f"Preview: {source['content_preview']}")
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": result["source_documents"]
                })
        
        # Limit chat history
        if len(st.session_state.chat_history) > MAX_HISTORY * 2:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY*2:]
        
        # Clear chat button
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
