"""
extractor.py - Extract PDF content using `unstructured` only.

- Text, tables, images: `partition_pdf` (hi_res, by_title).
- Page numbers: metadata plus `CompositeElement` children (min page); images prefer element/block page;
  back-fill missing text/table pages via substring / word-overlap against `pypdf` extract_text;
  fill remaining gaps on images & tables using neighboring pages in each list.
"""

import os
import re

from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf

from src.config import (
    PDF_COMBINE_TEXT_UNDER_N_CHARS,
    PDF_MAX_CHARACTERS,
    PDF_NEW_AFTER_N_CHARS,
)
from src.table_merge import merge_adjacent_split_tables_with_pages

_TESSERACT_PATHS = "C:\Program Files\Tesseract-OCR\tesseract.exe",

_POPPLER_PATHS = "C:\poppler\Library\bin",



def _configure_windows_tools() -> None:
    for path in _TESSERACT_PATHS:
        if os.path.isfile(path):
            try:
                import unstructured_pytesseract

                unstructured_pytesseract.pytesseract.tesseract_cmd = path
            except Exception:
                pass
            break

    for path in _POPPLER_PATHS:
        if os.path.isdir(path) and path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            break


_configure_windows_tools()


def _page_from_meta(meta) -> int | None:
    if meta is None:
        return None
    p = getattr(meta, "page_number", None)
    if p is None and hasattr(meta, "get"):
        p = meta.get("page_number")  # type: ignore[union-attr]
    if p is None:
        return None
    try:
        return int(p)
    except (TypeError, ValueError):
        return None


def _normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _search_snippet_from_text_body(body: str) -> str:
    """Short signature from chunk text to match against pypdf per-page extract_text."""
    if not (body or "").strip():
        return ""
    take = body.strip()[:800]
    take = re.sub(r"\s+", " ", take)
    if len(take) < 24:
        return take
    return take[:240]


def _strip_html_to_plain(html: str) -> str:
    h = html or ""
    h = re.sub(r"(?is)<script.*?>.*?</script>", " ", h)
    h = re.sub(r"(?is)<style.*?>.*?</style>", " ", h)
    h = re.sub(r"<[^>]+>", " ", h)
    return _normalize_for_match(h)


def _load_normalized_page_texts(file_path: str) -> list[str] | None:
    try:
        reader = PdfReader(file_path)
        out: list[str] = []
        for page in reader.pages:
            try:
                out.append(_normalize_for_match(page.extract_text() or ""))
            except Exception:
                out.append("")
        return out
    except Exception:
        return None


def _best_page_by_word_overlap(
    body: str, page_strings: list[str], *, min_hits: int
) -> int | None:
    words = re.findall(r"[a-z][a-z0-9]{2,}", (body or "").lower())
    if len(words) < 4:
        return None
    word_set = set(words[:120])
    best_p, best = None, 0
    for i, pg in enumerate(page_strings):
        if not pg:
            continue
        pw = set(re.findall(r"[a-z][a-z0-9]{2,}", pg))
        inter = len(word_set & pw)
        if inter > best:
            best = inter
            best_p = i + 1
    if best >= min_hits:
        return best_p
    if best >= 4 and len(word_set) < 25:
        return best_p
    return None


def _resolve_page_for_body(body: str, page_strings: list[str]) -> int | None:
    if not (body or "").strip():
        return None
    sig_norm = _normalize_for_match(_search_snippet_from_text_body(body))
    if len(sig_norm) >= 18:
        for page_one_based, pg in enumerate(page_strings, start=1):
            if sig_norm and sig_norm in pg:
                return page_one_based
    return _best_page_by_word_overlap(body, page_strings, min_hits=6)


def enrich_text_pages_from_pdf_loose(
    texts: list[str],
    text_pages: list[int | None],
    page_strings: list[str] | None,
) -> None:
    """Fill missing text_pages via substring / word-overlap against pypdf extract_text."""
    if not texts or len(texts) != len(text_pages) or not page_strings:
        return
    if all(p is not None for p in text_pages):
        return
    for i, body in enumerate(texts):
        if text_pages[i] is not None:
            continue
        found = _resolve_page_for_body(body, page_strings)
        if found is not None:
            text_pages[i] = found


def enrich_table_pages_from_pdf_loose(
    tables: list[str],
    table_pages: list[int | None],
    page_strings: list[str] | None,
) -> None:
    if not tables or len(tables) != len(table_pages) or not page_strings:
        return
    if all(p is not None for p in table_pages):
        return
    for i, html in enumerate(tables):
        if table_pages[i] is not None:
            continue
        plain = _strip_html_to_plain(html)
        if len(plain) < 12:
            continue
        found = _resolve_page_for_body(plain, page_strings)
        if found is not None:
            table_pages[i] = found


def fill_page_gaps_neighbor(pages: list[int | None]) -> None:
    """Fill None using nearest known page (forward, then backward pass)."""
    if not pages:
        return
    last: int | None = None
    for i, p in enumerate(pages):
        if p is not None:
            last = p
        elif last is not None:
            pages[i] = last
    nxt: int | None = None
    for i in range(len(pages) - 1, -1, -1):
        if pages[i] is not None:
            nxt = pages[i]
        elif nxt is not None:
            pages[i] = nxt


def extract_pdf_elements(
    file_path: str,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[int | None],
    list[int | None],
    list[int | None],
]:
    """
    Split the PDF into text blocks, HTML tables, and base64 images (with page hints when known).

    Returns:
        texts, tables, images, text_pages, table_pages, image_pages - aligned by index per list.
    """
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=PDF_MAX_CHARACTERS,
        combine_text_under_n_chars=PDF_COMBINE_TEXT_UNDER_N_CHARS,
        new_after_n_chars=PDF_NEW_AFTER_N_CHARS,
    )

    texts: list[str] = []
    tables: list[str] = []
    images: list[str] = []
    text_pages: list[int | None] = []
    table_pages: list[int | None] = []
    image_pages: list[int | None] = []

    # Unstructured often omits page_number on Image while parent/sibling elements have it.
    # Carry forward the last known page in document order (hi_res / by_title order).
    carry_page: int | None = None

    for chunk in chunks:
        chunk_type = str(type(chunk))
        chunk_meta = getattr(chunk, "metadata", None)
        chunk_p = _page_from_meta(chunk_meta)
        if chunk_p is not None:
            carry_page = chunk_p

        if "Table" in chunk_type:
            html = getattr(chunk.metadata, "text_as_html", None)
            tables.append(html if html else str(chunk))
            tp = chunk_p
            if tp is None:
                tp = carry_page
            table_pages.append(tp)
            if chunk_p is not None:
                carry_page = chunk_p

        elif "CompositeElement" in chunk_type:
            # Aggregate pages from all children; use min as section start (better than None).
            orig = getattr(chunk.metadata, "orig_elements", None)
            page_pool: list[int] = []
            if orig:
                for el in orig:
                    p_el = _page_from_meta(getattr(el, "metadata", None))
                    if p_el is not None:
                        page_pool.append(p_el)
            block_page = min(page_pool) if page_pool else None
            if block_page is None:
                block_page = chunk_p
            if block_page is None:
                block_page = carry_page

            text_page: int | None = block_page
            run_page = block_page if block_page is not None else carry_page
            if orig:
                for el in orig:
                    el_meta = getattr(el, "metadata", None)
                    p_el = _page_from_meta(el_meta)
                    if p_el is not None:
                        run_page = p_el
                        carry_page = p_el
                    if "Image" in str(type(el)):
                        b64 = getattr(el_meta, "image_base64", None) if el_meta else None
                        if b64:
                            images.append(b64)
                            img_p = p_el if p_el is not None else block_page
                            if img_p is None:
                                img_p = run_page
                            if img_p is None:
                                img_p = chunk_p
                            if img_p is None:
                                img_p = carry_page
                            image_pages.append(img_p)
            if text_page is None:
                text_page = chunk_p
            if text_page is None:
                text_page = carry_page
            texts.append(str(chunk))
            text_pages.append(text_page)
            if page_pool:
                carry_page = max(page_pool)

        elif "Image" in chunk_type and "Composite" not in chunk_type:
            b64 = getattr(chunk_meta, "image_base64", None) if chunk_meta else None
            if b64:
                images.append(b64)
                ip = chunk_p if chunk_p is not None else carry_page
                image_pages.append(ip)
                if chunk_p is not None:
                    carry_page = chunk_p

        elif "Table" not in chunk_type and "CompositeElement" not in chunk_type:
            # Stand-alone narrative blocks (uncommon with by_title, but fills gaps).
            if any(
                t in chunk_type
                for t in (
                    "Title",
                    "NarrativeText",
                    "ListItem",
                    "Header",
                    "Formula",
                )
            ):
                texts.append(str(chunk))
                tp = chunk_p if chunk_p is not None else carry_page
                text_pages.append(tp)
                if chunk_p is not None:
                    carry_page = chunk_p

    tables, table_pages = merge_adjacent_split_tables_with_pages(tables, table_pages)
    page_blob = _load_normalized_page_texts(file_path)
    if page_blob:
        enrich_text_pages_from_pdf_loose(texts, text_pages, page_blob)
        enrich_table_pages_from_pdf_loose(tables, table_pages, page_blob)
    fill_page_gaps_neighbor(image_pages)
    fill_page_gaps_neighbor(table_pages)
    return texts, tables, images, text_pages, table_pages, image_pages
