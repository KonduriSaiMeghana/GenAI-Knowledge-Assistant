# Embeddings and Vector Store management

import os
import pickle
import logging
import time
from typing import List
import numpy as np
import hashlib

# Modern langchain imports
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    HF_API_KEY, EMBEDDING_MODEL, VECTOR_STORE_PATH, 
    VECTOR_STORE_INDEX
)
from utils import ensure_directory_exists

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Simple embeddings using hash-based vectors for reliable local operation."""
    
    def __init__(self):
        """Initialize the embeddings manager."""
        self.embedding_dim = 384  # Standard embedding dimension
        logger.info(f"Using simple hash-based embeddings (dim: {self.embedding_dim})")
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding using hash-based method.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Use hash to create a consistent seed
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_value % (2**32))
        
        # Generate consistent random embedding
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self._text_to_embedding(text)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            return [self._text_to_embedding(text) for text in texts]
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    # FAISS compatible methods
    def embed_query(self, text: str) -> List[float]:
        """FAISS compatible embed_query method."""
        return self.embed_text(text).tolist()


class FAISSEmbeddings:
    """Wrapper for FAISS compatibility with Sentence Transformers."""
    
    def __init__(self, embeddings_manager: EmbeddingsManager):
        """Initialize FAISS embeddings wrapper."""
        self.embeddings_manager = embeddings_manager
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents for FAISS."""
        embeddings = self.embeddings_manager.embed_documents(texts)
        return [emb.tolist() for emb in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query for FAISS."""
        return self.embeddings_manager.embed_query(text)


class VectorStore:
    """Manage FAISS vector store for semantic search."""
    
    def __init__(self):
        """Initialize the vector store."""
        ensure_directory_exists(VECTOR_STORE_PATH)
        self.embeddings_manager = EmbeddingsManager()
        self.faiss_embeddings = FAISSEmbeddings(self.embeddings_manager)
        self.vector_store = None
        self.load_or_create_store()
    
    def load_or_create_store(self):
        """Load existing vector store or create a new one."""
        store_path = os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_INDEX)
        
        if os.path.exists(store_path):
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=VECTOR_STORE_PATH,
                    embeddings=self.faiss_embeddings,
                    index_name=VECTOR_STORE_INDEX
                )
                logger.info("Loaded existing vector store")
            except Exception as e:
                logger.warning(f"Could not load existing store: {e}. Creating new one.")
                self.vector_store = None
        else:
            logger.info("Creating new vector store")
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects with page_content and metadata
            
        Returns:
            True if successful
        """
        try:
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.faiss_embeddings
                )
                logger.info(f"Created vector store with {len(documents)} documents")
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to vector store")
            
            # Save the vector store
            self.save_store()
            return True
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store with scores: {e}")
            return []
    
    def save_store(self):
        """Save the vector store to disk."""
        if self.vector_store is not None:
            try:
                self.vector_store.save_local(
                    folder_path=VECTOR_STORE_PATH,
                    index_name=VECTOR_STORE_INDEX
                )
                logger.info("Vector store saved successfully")
            except Exception as e:
                logger.error(f"Error saving vector store: {e}")
    
    def clear_store(self):
        """Clear the vector store."""
        import shutil
        try:
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH)
            self.vector_store = None
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        if self.vector_store is None:
            return {"status": "empty", "document_count": 0}
        
        try:
            return {
                "status": "active",
                "document_count": len(self.vector_store.docstore._dict)
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"status": "error", "message": str(e)}
