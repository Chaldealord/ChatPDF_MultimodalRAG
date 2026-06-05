"""
rag_pipeline.py - Orchestrator of the full Multimodal RAG pipeline.

Structure mirrors Multi-Modal-RAG (extract → summarize with batched LLM → multi-vector index).
Tuning knobs live in `src.config`.
"""

import os

# Must run before `extractor` → unstructured/transformers import chain.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from pathlib import Path

from src.extractor import extract_pdf_elements
from src.generator import answer_question  # noqa: F401  (re-export)
from src.models import get_text_llm, get_vision_llm
from src.retriever import build_retriever
from src.summarizer import summarize_images, summarize_tables, summarize_texts


def process_pdf(file_path: str, progress_callback=None) -> dict:
    """
    Run Extract → Summarize → Store (vectors in-memory only).

    Returns:
        {
            "retriever": MultiVectorRetriever,
            "stats": {"texts": int, "tables": int, "images": int},
        }
    """

    def update(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    file_path = str(Path(file_path).resolve())

    update(0.05, "Analyzing PDF and extracting text, tables, images...")
    texts, tables, images, text_pages, table_pages, image_pages = extract_pdf_elements(
        file_path
    )
    update(
        0.30,
        f"Extracted: {len(texts)} text chunks · {len(tables)} tables · {len(images)} images",
    )

    text_llm = get_text_llm()
    vision_llm = get_vision_llm()

    update(0.38, f"Summarizing {len(texts)} text chunks...")
    text_summaries = summarize_texts(texts, text_llm)

    update(0.55, f"Summarizing {len(tables)} tables...")
    table_summaries = summarize_tables(tables, text_llm)

    update(0.68, f"Summarizing {len(images)} images (vision model)...")
    image_summaries = summarize_images(images, vision_llm)

    update(0.88, "Indexing summaries in ChromaDB...")
    retriever = build_retriever(
        texts,
        tables,
        images,
        text_summaries,
        table_summaries,
        image_summaries,
        text_pages=text_pages,
        table_pages=table_pages,
        image_pages=image_pages,
    )

    update(1.0, "Done.")
    return {
        "retriever": retriever,
        "stats": {
            "texts": len(texts),
            "tables": len(tables),
            "images": len(images),
        },
    }
