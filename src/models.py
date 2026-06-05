"""
models.py - Initialize AI models for the pipeline.

- Text LLM:   Groq `llama-3.1-8b-instant` (summarize text and tables)
- Vision LLM: Groq `meta-llama/llama-4-scout-17b-16e-instruct` (images → captions / multimodal QA)
- Embeddings: Google Gemini `gemini-embedding-001` (text → vectors)
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def get_text_llm() -> ChatGroq:
    """Fast text LLM for summarizing narrative text and HTML tables."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )


def get_vision_llm() -> ChatGroq:
    """Multimodal (vision) chat model on Groq."""
    return ChatGroq(model=_GROQ_VISION_MODEL, temperature=0)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Google Gemini embedding model - maps text to dense vectors."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
