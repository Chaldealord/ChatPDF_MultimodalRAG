"""
summarizer.py - LLM summaries for every extracted PDF element.

Each modality uses a dedicated model role:
  - Text: Groq `llama-3.1-8b-instant` (chat, text-only)
  - Tables: same model on HTML fragments
  - Images: Groq `llama-4-scout` vision model

Text/table batches mirror Multi-Modal-RAG `summarize.summarize_elements` concurrency knobs from `src.config`.
"""

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.config import (
    GROQ_SUMMARY_STOP_AFTER_ATTEMPT,
    SUMMARY_IMAGE_MAX_CONCURRENCY,
    SUMMARY_TEXT_TABLE_MAX_CONCURRENCY,
)

_RETRY = {
    "stop_after_attempt": GROQ_SUMMARY_STOP_AFTER_ATTEMPT,
    "wait_exponential_jitter": True,
}


def summarize_texts(texts: list[str], llm) -> list[str]:
    """Summarize each narrative chunk into 1-3 sentences."""
    if not texts:
        return []

    prompt = ChatPromptTemplate.from_template(
        "Summarize the following passage briefly (1-3 sentences). "
        "Keep the most important facts intact:\n\n{text}"
    )
    chain = (prompt | llm | StrOutputParser()).with_retry(**_RETRY)
    return chain.batch(
        [{"text": t} for t in texts],
        {"max_concurrency": SUMMARY_TEXT_TABLE_MAX_CONCURRENCY},
    )


def summarize_tables(tables: list[str], llm) -> list[str]:
    """Produce a short natural-language description for each HTML table."""
    if not tables:
        return []

    prompt = ChatPromptTemplate.from_template(
        "Briefly describe the content and intent of this HTML table:\n\n{table}"
    )
    chain = (prompt | llm | StrOutputParser()).with_retry(**_RETRY)
    return chain.batch(
        [{"table": t} for t in tables],
        {"max_concurrency": SUMMARY_TEXT_TABLE_MAX_CONCURRENCY},
    )


def summarize_images(images: list[str], vision_llm) -> list[str]:
    """
    Run the vision LLM per image (base64 → textual summary).
    Images are attached as JPEG data URLs inside the chat message.
    """
    if not images:
        return []

    def describe_one(img_b64: str) -> str:
        msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image briefly (1-3 sentences). "
                        "Highlight important numbers, labels, or charts when visible."
                    ),
                },
            ]
        )
        out = vision_llm.invoke([msg]).content
        return out if isinstance(out, str) else str(out)

    return (
        RunnableLambda(describe_one)
        .with_retry(**_RETRY)
        .batch(images, {"max_concurrency": SUMMARY_IMAGE_MAX_CONCURRENCY})
    )
