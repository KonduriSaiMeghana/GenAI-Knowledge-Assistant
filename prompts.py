# Prompt templates for RAG pipeline

from langchain_core.prompts import PromptTemplate

# Main RAG prompt template
RAG_PROMPT_TEMPLATE = """Answer the question using only the context below. If the answer is not found in the context, say "Not found in documents".

Context:
{context}

Question:
{question}

Instructions:
- Base your answer strictly on the provided context
- If answer is not found, say "Not found in documents"
- Keep answer short and precise
- If relevant, cite which document the information comes from

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE
)

# System prompt for the chat
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base of documents. 
Your role is to answer questions based on the provided documents.

When responding:
- Only use information from the provided context
- Be clear and concise
- If information is not available, clearly state "Not found in documents"
- Think step-by-step before providing your answer
- Cite sources when relevant"""

# Summarization prompt
SUMMARIZE_PROMPT_TEMPLATE = """Summarize the following text concisely:

{text}

Summary:"""

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=SUMMARIZE_PROMPT_TEMPLATE
)

# Question refinement prompt
REFINE_QUESTION_TEMPLATE = """Given the conversation history and a new question, refine the question to be more specific and contextual.

Conversation history:
{history}

New question:
{question}

Refined question:"""

REFINE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["history", "question"],
    template=REFINE_QUESTION_TEMPLATE
)
