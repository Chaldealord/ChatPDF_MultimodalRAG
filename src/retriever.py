"""
retriever.py - Build a Multi-Vector Retriever with ChromaDB.

Two-tier storage (in-memory only; no `data/` directory written):
  - VectorStore (Chroma EphemeralClient): embeddings of chunk **summaries** → semantic search
  - DocStore (InMemoryStore): **raw** chunk content → returned on retrieve

At question time:
  1. Embed the question → nearest summaries in Chroma
  2. Read doc_id from each summary → fetch the original document from DocStore
  3. Return raw text / table HTML / image payload to the answering LLM
"""

import uuid

import chromadb
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import RETRIEVER_SEARCH_K
from src.models import get_embeddings

DOC_ID_KEY = "doc_id"


def build_retriever(
    texts: list[str],
    tables: list[str],
    images: list[str],
    text_summaries: list[str],
    table_summaries: list[str],
    image_summaries: list[str],
    *,
    text_pages: list[int | None] | None = None,
    table_pages: list[int | None] | None = None,
    image_pages: list[int | None] | None = None,
) -> MultiVectorRetriever:
    """
    Build a MultiVectorRetriever from raw contents and their LLM summaries.

    Returns:
        retriever: call `retriever.invoke(question)` to fetch context documents
    """
    chroma_client = chromadb.EphemeralClient()
    # Unique collection per ingest so vectors never mix across PDFs.
    collection_name = f"mm_rag_{uuid.uuid4().hex[:16]}"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        client=chroma_client,
    )
    docstore = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=DOC_ID_KEY,
        search_kwargs={"k": RETRIEVER_SEARCH_K},
    )

    _add_to_retriever(
        retriever, texts, text_summaries, content_type="text", pages=text_pages
    )
    _add_to_retriever(
        retriever, tables, table_summaries, content_type="table", pages=table_pages
    )
    _add_to_retriever(
        retriever, images, image_summaries, content_type="image", pages=image_pages
    )

    return retriever


def _add_to_retriever(
    retriever: MultiVectorRetriever,
    raw_contents: list[str],
    summaries: list[str],
    content_type: str,
    pages: list[int | None] | None = None,
) -> None:
    """
    Add one modality (text / table / image) to the retriever.

    - Summaries → vectorstore (embedded and searched)
    - Raw contents → docstore (fetched after a hit)
    """
    if not raw_contents:
        return

    if pages is None or len(pages) != len(raw_contents):
        page_list: list[int | None] = [None] * len(raw_contents)
    else:
        page_list = list(pages)

    doc_ids = [str(uuid.uuid4()) for _ in raw_contents]

    # Persist summaries in the vectorstore with doc_id links to docstore rows.
    summary_docs = [
        Document(
            page_content=summary,
            metadata={DOC_ID_KEY: doc_id, "type": content_type},
        )
        for summary, doc_id in zip(summaries, doc_ids)
    ]
    retriever.vectorstore.add_documents(summary_docs)

    # Persist raw payloads in the docstore under the same ids.
    raw_docs = [
        Document(
            page_content=content,
            metadata={"type": content_type, "page": page},
        )
        for content, page in zip(raw_contents, page_list)
    ]
    retriever.docstore.mset(list(zip(doc_ids, raw_docs)))
