"""
generator.py - Answer questions using retrieved context + multimodal LLM.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.models import get_text_llm, get_vision_llm

MediaFocusMode = Literal["both", "tables_only", "images_only"]

_PROMPT_CORE = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer the question.\n"
    "2) If the answer is not in the context, say: "
    '"I don\'t know based on the provided documents"\n'
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) After each factual claim (or one short sentence with a single idea from the docs), "
    "append an inline citation exactly as: [Doc: <filename>, Page: <n>]. Use the filename and "
    "page from the passage markers in the context (text, PDF_TABLE_*, or FIGURE_* lines). "
    "For claims grounded in an attached diagram, cite the Page from that FIGURE line. "
    "For claims grounded in table HTML, cite the Page from that PDF_TABLE line. "
    "If the only matching marker shows Page: unknown, cite Page: unknown. "
    "Do not invent page numbers.\n"
)

_PROMPT_OUTPUT_SHAPE = (
    "5) Do NOT invent new tables and do NOT format answers as markdown/ASCII tables "
    "(no | column grids, no alignment rows with dashes). Use plain sentences or bullet "
    "points only. Numeric or structural detail from tables must come from the PDF excerpts "
    "in the context, rewritten as prose - not as a newly drawn table.\n"
    "6) The application may show the original HTML tables from the PDF separately. "
    "Do not echo internal markers like PDF_TABLE_1 or raw HTML tags in your answer.\n"
    "7) When figures are attached as images:\n"
    "   - The context lists FIGURE 1, FIGURE 2, ... with [Doc: ..., Page: ...]. "
    "The first image in this message is FIGURE 1, the second is FIGURE 2, same order.\n"
    "   - If you describe or infer from a diagram, cite exactly that figure's Page from its "
    "FIGURE line (not Page: unknown unless that line says unknown).\n"
    "   - For claims from narrative text or tables, use the passage or PDF_TABLE markers. "
    "Do not echo raw HTML or internal labels like PDF_TABLE_1 in the answer body.\n"
)

_MODE_RULES: dict[MediaFocusMode, str] = {
    "tables_only": (
        "8) MODE - Tables: The question targets tables or tabular data. "
        "Prioritize facts from the extracted table HTML and nearby narrative text. "
        "Do not discuss figures, charts, photos, or diagrams. "
        "Every number you state must appear in the provided context.\n"
    ),
    "images_only": (
        "8) MODE - Figures: The question targets figures, images, charts, or diagrams. "
        "Use the attached images together with any text context. Describe what you see in each "
        "relevant figure before conclusions. "
        "Do not invent axis values, legend labels, or data points you cannot see in the image "
        "or read verbatim in the text. If the image is unclear, say so briefly.\n"
    ),
    "both": (
        "8) MODE - Mixed: Use narrative text, table excerpts, and any attached figures "
        "when they are present in this context. "
        "If this message has no images, do not describe figure details. "
        "If there are no table excerpts, do not cite specific table cells or numbers.\n"
    ),
}

# Max images passed to the vision LLM (API / payload limits)
MAX_CONTEXT_IMAGES = 2
# Max images shown in the UI (retrieved figures)
MAX_DISPLAY_IMAGES = 8
# Max HTML tables rendered under an answer (wide PDFs can have many rows)
MAX_DISPLAY_TABLES = 6


def _question_has_visual_figure_intent(question: str) -> bool:
    q = (question or "").strip().lower()
    return bool(
        re.search(
            r"\b(figures?|figs?\.?|images?|photos?|pictures?|plots?|charts?|"
            r"illustrations?|diagrams?|screenshots?)\b",
            q,
        )
        # Vietnamese figure/image cues (keep literals for multilingual UX).
        or "ảnh" in q
        or "hình ảnh" in q
        or re.search(r"\bhình\s*\d", q)
    )


def _media_focus_mode(question: str) -> MediaFocusMode:
    """
    If the user asks only about tables or only about figures, narrow context + UI
    so we do not mix in the other modality (e.g. \"show all tables\" -> no figure tiles).
    """
    q = (question or "").strip().lower()

    has_table = bool(
        re.search(r"\btables?\b|\btabular\b", q)
        or "bảng" in q  # Vietnamese "table"
        or re.search(r"\bbang\b", q)  # ASCII transliteration without diacritics
    )
    has_image = _question_has_visual_figure_intent(question)

    if has_table and not has_image:
        return "tables_only"
    if has_image and not has_table:
        return "images_only"
    return "both"


def _suppress_retrieved_images_for_question(question: str) -> bool:
    """
    Metrics/results-style questions without figure/chart wording should not use
    retrieved image chunks (avoids unrelated diagrams next to tabular answers).
    """
    if _media_focus_mode(question) != "both":
        return False
    q = (question or "").strip().lower()
    texty = bool(
        re.search(
            r"\b(results?|mrr|metrics?|accuracy|scores?|performance|benchmark|"
            r"evaluation|numbers?|values?|compare|comparison)\b",
            q,
        )
        or "kết quả" in q  # Vietnamese "results"
        or re.search(r"\bket\s+qua\b", q)  # ASCII transliteration
    )
    if not texty:
        return False
    return not _question_has_visual_figure_intent(question)


def _question_wants_tabular_or_metric_evidence(question: str) -> bool:
    """True when the user likely needs numbers, comparisons, or explicit table content."""
    q = (question or "").strip().lower()
    if re.search(
        r"\btables?\b|\btabular\b|\bcolumns?\b|\brows?\b|\bcells?\b",
        q,
    ) or "bảng" in q or re.search(r"\bbang\b", q):  # Vietnamese "table"
        return True
    if re.search(
        r"\b(results?|mrr|metrics?|accuracy|scores?|performance|benchmark|"
        r"evaluation|numbers?|values?|compare|comparison|percentage|percent)\b",
        q,
    ) or "%" in q or "kết quả" in q or re.search(r"\bket\s+qua\b", q):  # "results"
        return True
    if re.search(r"\bversus\b|\bvs\.?\b|\bhigher\b|\blower\b|\bbest\b|\bworse\b", q):
        return True
    return False


def _suppress_retrieved_tables_for_question(question: str) -> bool:
    """
    Definitional / conceptual questions should not attach unrelated table tiles.
    When MODE is \"both\" and the question does not ask for metrics/tables, drop table chunks.
    """
    if _media_focus_mode(question) != "both":
        return False
    if _question_wants_tabular_or_metric_evidence(question):
        return False
    return True


_SOURCE_CITATION_RE = re.compile(
    r"\[Doc:\s*[^\]]+,\s*Page:\s*[^\]]+?\]",
    re.IGNORECASE,
)


def _page_label(page: int | None) -> str:
    if page is None:
        return "unknown"
    try:
        return str(int(page))
    except (TypeError, ValueError):
        return "unknown"


def _ensure_source_footer(text: str, source_filename: str | None) -> str:
    """Append `[Doc: ..., Page: unknown]` when the model omitted all citations."""
    if not (text and str(text).strip()) or not (source_filename or "").strip():
        return text
    if _is_no_info_answer(text):
        return text
    if _SOURCE_CITATION_RE.search(text):
        return text
    fn = source_filename.strip()
    return f"{text.rstrip()}\n\n[Doc: {fn}, Page: unknown]"


def _is_no_info_answer(text: str) -> bool:
    """True when the model refused or said the docs contain no answer."""
    t = (text or "").lower()
    return (
        "don't know based on the provided documents" in t
        or "no passages were retrieved" in t
    )


def _apply_prompt_template(
    context: str, question: str, focus: MediaFocusMode
) -> str:
    """Fill template without str.format (PDF text may contain ``{`` / ``}``)."""
    body = (
        _PROMPT_CORE
        + _PROMPT_OUTPUT_SHAPE
        + _MODE_RULES[focus]
        + "\nContext:\n{context}\n\nQuestion: {question}"
    )
    return body.replace("{context}", context).replace("{question}", question)


_MD_TABLE_SEP = re.compile(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")


def _strip_markdown_pipe_tables(text: str) -> str:
    """Remove github-style markdown table blocks (| ... |) from model output."""
    if not text or "|" not in text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if "|" in line and i + 1 < n and _MD_TABLE_SEP.match(lines[i + 1]):
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _wrap_context_with_metadata(
    text_context: str,
    source_filename: str | None,
    *,
    figures_attached: bool,
) -> str:
    blocks: list[str] = []
    if source_filename:
        blocks.append(
            f"Cite using passage headers, PDF_TABLE_* lines, and FIGURE_* lines below; "
            "each FIGURE n matches the n-th attached image in this message when images are present."
        )
    blocks.append(text_context)
    if figures_attached:
        blocks.append(
            "Additional context: one or more figures from the same document are attached "
            "as images in this message; treat them as part of the context."
        )
    return "\n\n".join(blocks)


@dataclass
class AnswerBundle:
    """Text answer plus optional rendered assets (figures + table HTML)."""

    text: str
    image_base64_list: list[str]
    table_html_list: list[str] = field(default_factory=list)
    media_source_note: str = ""


def _invoke_chat_to_text(llm, messages: list) -> str:
    """LCEL parity with Multi-Modal-RAG: ``chat_model | StrOutputParser()``."""
    return (llm | StrOutputParser()).invoke(messages).strip()


def _answer_text_only(
    question: str, context_block: str, focus: MediaFocusMode
) -> str:
    llm = get_text_llm()
    prompt = _apply_prompt_template(context_block, question, focus)
    return _invoke_chat_to_text(llm, [HumanMessage(content=prompt)])


def _build_media_source_note(
    doc_label: str,
    image_passages: list[dict],
    context_tables: list[dict],
) -> str:
    parts: list[str] = []
    if image_passages:
        fig_lbl = ", ".join(
            f"Fig. {i} p.{_page_label(r.get('page'))}"
            for i, r in enumerate(image_passages, start=1)
        )
        parts.append(f"Retrieved figures ({doc_label}): {fig_lbl}")
    if context_tables:
        tbl_lbl = ", ".join(
            f"Table {i} p.{_page_label(r.get('page'))}"
            for i, r in enumerate(context_tables, start=1)
        )
        parts.append(f"Retrieved tables: {tbl_lbl}")
    return " · ".join(parts)


def answer_question_bundle(
    question: str,
    retriever,
    source_filename: str | None = None,
) -> AnswerBundle:
    """
    Retrieve context, generate an answer, and return retrieved image payloads for the UI.

    Args:
        question: user question
        retriever: MultiVectorRetriever
        source_filename: original PDF name for (source: filename) citations
    """
    vision_llm = get_vision_llm()

    docs = retriever.invoke(question)

    passages: list[dict] = []
    for doc in docs:
        passages.append(
            {
                "type": doc.metadata.get("type", "text"),
                "content": doc.page_content,
                "page": doc.metadata.get("page"),
            }
        )

    focus = _media_focus_mode(question)
    if focus == "tables_only":
        passages = [p for p in passages if p["type"] != "image"]
    elif focus == "images_only":
        passages = [p for p in passages if p["type"] != "table"]
    else:
        if _suppress_retrieved_images_for_question(question):
            passages = [p for p in passages if p["type"] != "image"]
        if _suppress_retrieved_tables_for_question(question):
            passages = [p for p in passages if p["type"] != "table"]

    context_texts = [p for p in passages if p["type"] == "text"]
    context_tables = [p for p in passages if p["type"] == "table"]
    context_images = [p["content"] for p in passages if p["type"] == "image"]

    if not context_texts and not context_tables and not context_images:
        return AnswerBundle(
            "I don't know based on the provided documents (no passages were retrieved).",
            [],
            media_source_note="",
        )

    seen_img: set[str] = set()
    display_images: list[str] = []
    for raw in context_images:
        if raw not in seen_img:
            seen_img.add(raw)
            display_images.append(raw)
    display_images = display_images[:MAX_DISPLAY_IMAGES]

    seen_tbl: set[str] = set()
    display_tables: list[str] = []
    for row in context_tables:
        html = (row.get("content") or "").strip()
        if html and html not in seen_tbl:
            seen_tbl.add(html)
            display_tables.append(html)
    display_tables = display_tables[:MAX_DISPLAY_TABLES]

    doc_label = source_filename or "document"
    image_passages_ret = [p for p in passages if p["type"] == "image"]
    media_note = _build_media_source_note(doc_label, image_passages_ret, context_tables)

    text_blocks: list[str] = []
    for row in context_texts:
        body = row.get("content") or ""
        pg = _page_label(row.get("page"))
        text_blocks.append(
            f"[Doc: {doc_label}, Page: {pg}] - passage begins:\n{body}"
        )
    # Image bytes are attached separately; without inline headers the model cites Page: unknown.
    figure_headers: list[str] = []
    for i, row in enumerate(image_passages_ret, start=1):
        pg = _page_label(row.get("page"))
        figure_headers.append(
            f"FIGURE {i} [Doc: {doc_label}, Page: {pg}] - "
            "same order as attached images (first attachment = FIGURE 1)."
        )
    if figure_headers and len(image_passages_ret) > MAX_CONTEXT_IMAGES:
        figure_headers.insert(
            0,
            f"Note: only the first {MAX_CONTEXT_IMAGES} figures are attached as pixel images; "
            "all FIGURE lines below still give the correct page to cite.",
        )
    if figure_headers:
        text_blocks.append("\n".join(figure_headers))
    if context_tables:
        table_chunks = [
            f"--- PDF_TABLE_{idx} [Doc: {doc_label}, Page: {_page_label(row.get('page'))}] "
            f"(HTML from the source PDF only; do not repeat this label or the HTML in your answer) ---\n"
            f"{row.get('content', '')}"
            for idx, row in enumerate(context_tables, start=1)
        ]
        text_blocks.append(
            "Below is table content exactly as extracted from the PDF (HTML). "
            "Use it for facts only; reply in plain text or bullets - never as a new markdown table.\n\n"
            + "\n\n".join(table_chunks)
        )
    text_context = (
        "\n\n---\n\n".join(text_blocks) if text_blocks else "(No text/table context)"
    )

    figures_attached = bool(context_images)
    context_block = _wrap_context_with_metadata(
        text_context,
        source_filename,
        figures_attached=False,
    )

    if not context_images:
        out = _answer_text_only(question, context_block, focus)
        out = out or "The model returned an empty answer. Check your Groq API key and limits."
        if _is_no_info_answer(out):
            return AnswerBundle(out, [], [], media_source_note=media_note)
        if context_tables:
            out = _strip_markdown_pipe_tables(out)
        out = _ensure_source_footer(out, source_filename)
        return AnswerBundle(out, [], display_tables, media_source_note=media_note)

    context_block = _wrap_context_with_metadata(
        text_context,
        source_filename,
        figures_attached=True,
    )
    full_prompt = _apply_prompt_template(context_block, question, focus)

    content: list = []
    for img_b64 in context_images[:MAX_CONTEXT_IMAGES]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    content.append({"type": "text", "text": full_prompt})

    answer_text = ""
    vision_succeeded = False
    try:
        answer_text = _invoke_chat_to_text(
            vision_llm, [HumanMessage(content=content)]
        )
        if answer_text:
            vision_succeeded = True
    except Exception:
        answer_text = ""

    if not answer_text:
        answer_text = _answer_text_only(question, context_block, focus)
        if answer_text:
            answer_text = (
                answer_text
                + "\n\n_(Answer used text context only; image understanding was skipped.)_"
            )

    if not answer_text:
        answer_text = (
            "The model returned an empty answer. Check your Groq API key and rate limits."
        )

    if context_tables:
        answer_text = _strip_markdown_pipe_tables(answer_text)

    if _is_no_info_answer(answer_text):
        return AnswerBundle(answer_text, [], [], media_source_note=media_note)

    # Do not show retrieved tiles if the answer was not actually grounded in vision,
    # or vision ran but produced nothing (avoids unrelated figures next to text/table answers).
    if display_images and not vision_succeeded:
        display_images = []

    answer_text = _ensure_source_footer(answer_text, source_filename)
    return AnswerBundle(
        answer_text, display_images, display_tables, media_source_note=media_note
    )


def answer_question(
    question: str,
    retriever,
    source_filename: str | None = None,
) -> str:
    """Backward-compatible: text only."""
    return answer_question_bundle(question, retriever, source_filename).text


def decode_base64_image(data: str) -> bytes | None:
    """Decode extracted base64; return None if invalid."""
    if not data or not str(data).strip():
        return None
    try:
        return base64.b64decode(data, validate=False)
    except Exception:
        return None
