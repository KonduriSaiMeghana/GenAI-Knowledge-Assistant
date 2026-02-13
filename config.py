# Configuration for GenAI Knowledge Assistant

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HF_API_KEY = os.getenv("HF_API_KEY")

# Validate API key
if not HF_API_KEY:
    print("\n" + "="*60)
    print("⚠️  WARNING: HF_API_KEY not found!")
    print("="*60)
    print("\nTo use this application, you need to:")
    print("1. Create a .env file in the project directory")
    print("2. Add your Hugging Face API key: HF_API_KEY=...\n")
    print("Get your API key from: https://huggingface.co/settings/tokens")
    print("="*60 + "\n")

# Model Configuration
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_DOCS_TO_RETRIEVE = 4

# Vector Store
VECTOR_STORE_PATH = "./vector_store"
VECTOR_STORE_INDEX = "faiss_index"

# Streamlit Configuration
MAX_UPLOAD_SIZE = 50  # MB
ALLOWED_EXTENSIONS = [".pdf", ".txt", ".md"]

# Chat Configuration
MAX_HISTORY = 10
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# Logging
LOG_LEVEL = "INFO"
