"""Tunable pipeline constants.

Chunking defaults match Multi-Modal-RAG `extract_data.extract_chunks`;
summarization uses LangChain `Runnable.batch` + `max_concurrency` like their `summarize.py`.
"""

from __future__ import annotations

# unstructured.partition.pdf - see extract_data.py in Multi-Modal-RAG
PDF_MAX_CHARACTERS = 10_000
PDF_COMBINE_TEXT_UNDER_N_CHARS = 2_000
PDF_NEW_AFTER_N_CHARS = 6_000

# Runnable.batch (text/table summarization - Groq)
SUMMARY_TEXT_TABLE_MAX_CONCURRENCY = 3

# Vision image summarization - Groq
SUMMARY_IMAGE_MAX_CONCURRENCY = 2

# Retries for 429 / transient Groq errors (summarizer `.with_retry`)
GROQ_SUMMARY_STOP_AFTER_ATTEMPT = 8

# MultiVectorRetriever search breadth
RETRIEVER_SEARCH_K = 8
