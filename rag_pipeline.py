# RAG Pipeline orchestration using LangChain

import logging
from typing import List, Dict, Any
import requests
import json

# Modern langchain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from config import (
    HF_API_KEY, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_DOCS_TO_RETRIEVE, TEMPERATURE, MAX_TOKENS
)
from vector_store import VectorStore
from prompts import RAG_PROMPT, SYSTEM_PROMPT
from utils import load_document, split_text

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """Simple wrapper for Hugging Face Inference API."""
    
    def __init__(self, model_id: str, api_key: str, temperature: float = 0.7, max_tokens: int = 2000):
        """Initialize HF LLM."""
        self.model_id = model_id
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    def __call__(self, prompt: str) -> str:
        """Generate text using HF API."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_length": self.max_tokens,
                    "do_sample": True
                }
            }
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return str(result[0])
            return str(result)
        except Exception as e:
            logger.error(f"Error calling HF API: {e}")
            raise


class RAGPipeline:
    """Main RAG pipeline for document-based question answering."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.vector_store = VectorStore()
        self.llm = HuggingFaceLLM(
            model_id=LLM_MODEL,
            api_key=HF_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Memory will be initialized on first use
        self.memory = None
        self.qa_chain = None
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Set up the QA chain with retrieval."""
        # Initialize as None - will use manual retrieval pattern
    
    def _setup_qa_chain(self):
        """Set up the QA chain with retrieval."""
        # Initialize as None - will use manual retrieval pattern
        self.qa_chain = None
    
    def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a document to the RAG pipeline.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata for the document
            
        Returns:
            True if successful
        """
        try:
            # Load document
            content = load_document(file_path)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create Document objects with metadata
            documents = []
            file_name = file_path.split("\\")[-1]
            
            for i, chunk in enumerate(chunks):
                meta = metadata or {}
                meta.update({
                    "source": file_name,
                    "chunk": i,
                    "total_chunks": len(chunks)
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=meta
                ))
            
            # Add to vector store
            success = self.vector_store.add_documents(documents)
            
            if success:
                # Rebuild QA chain with new documents
                self._setup_qa_chain()
                logger.info(f"Successfully added document: {file_name}")
                return True
            return False
            
        except Exception as e:
            error_msg = str(e)
            # Check for API quota errors and re-raise with user-friendly message
            if "insufficient_quota" in error_msg.lower() or "429" in error_msg:
                logger.error(f"OpenAI API Quota Error: {error_msg}")
                raise
            logger.error(f"Error adding document: {e}")
            return False
    
    def add_documents_batch(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Add multiple documents to the RAG pipeline.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            Dictionary with results for each file
        """
        results = {}
        for file_path in file_paths:
            results[file_path] = self.add_document(file_path)
        return results
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.vector_store.vector_store is None:
            logger.warning("No documents in vector store. Cannot answer question.")
            return {
                "answer": "No documents available. Please upload documents first.",
                "source_documents": [],
                "success": False
            }
        
        try:
            # Use RetrievalQA if available
            if self.qa_chain is not None:
                result = self.qa_chain({"query": question})
                # Format source documents
                sources = []
                if "source_documents" in result:
                    for doc in result["source_documents"]:
                        sources.append({
                            "source": doc.metadata.get("source", "Unknown"),
                            "chunk": doc.metadata.get("chunk", 0),
                            "content_preview": doc.page_content[:200] + "..."
                        })
                
                return {
                    "answer": result.get("result", ""),
                    "source_documents": sources,
                    "success": True
                }
            else:
                # Manual retrieval mode
                retrieved_docs = self.retrieve_context(question)
                if not retrieved_docs:
                    return {
                        "answer": "Could not find relevant information in documents.",
                        "source_documents": [],
                        "success": False
                    }
                
                # Combine context
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Generate answer
                answer_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer: """
                
                response = self.llm.invoke(answer_prompt)
                answer_text = response.content if hasattr(response, 'content') else str(response)
                
                # Format source documents
                sources = []
                for doc in retrieved_docs:
                    sources.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "chunk": doc.metadata.get("chunk", 0),
                        "content_preview": doc.page_content[:200] + "..."
                    })
                
                return {
                    "answer": answer_text,
                    "source_documents": sources,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Error querying RAG pipeline: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def retrieve_context(self, question: str, k: int = MAX_DOCS_TO_RETRIEVE) -> List[Document]:
        """
        Retrieve relevant documents for a question without generating an answer.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        try:
            return self.vector_store.search(question, k=k)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def clear_documents(self):
        """Clear all documents from the RAG pipeline."""
        try:
            self.vector_store.clear_store()
            self.qa_chain = None
            logger.info("All documents cleared from RAG pipeline")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        return {
            "vector_store": self.vector_store.get_stats(),
            "model": LLM_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "max_retrieve": MAX_DOCS_TO_RETRIEVE
        }
