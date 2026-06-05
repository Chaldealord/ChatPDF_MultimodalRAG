import base64
import io
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()
# Quieter transformers when unstructured pulls layout/YOLO models (alias __path__ warnings).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st

st.set_page_config(
    page_title="ChatPDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

PREVIEW_MAX_PAGES = 7

# Session-state keys - upload cache + isolation of right-panel reruns (@st.fragment)
_SS_UPLOAD_FILE_ID = "chatpdf_sess_upload_file_id"
_SS_UPLOAD_BYTES = "chatpdf_sess_upload_bytes"
_SS_UPLOAD_DISP_NAME = "chatpdf_sess_upload_display_name"

# Deferred ingest feedback after st.rerun() - avoids leftover progress/spinner in @st.fragment
_SS_FEED_INGEST_OK = "chatpdf_feed_ingest_ok"
_SS_FEED_INGEST_ERR = "chatpdf_feed_ingest_err"


def _pdf_page_count(pdf_bytes: bytes) -> int | None:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception:
        return None


def _pdf_first_n_pages(pdf_bytes: bytes, n: int) -> bytes | None:
    """Return a new PDF containing the first n pages (or fewer if the doc is shorter)."""
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = len(reader.pages)
        if total == 0:
            return None
        take = min(max(n, 1), total)
        writer = PdfWriter()
        for i in range(take):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except Exception:
        return None


def display_pdf(pdf_bytes: bytes) -> None:
    """Same pattern as Multi-Modal-RAG `app.display_pdf` (iframe, base64 data URL)."""
    if not pdf_bytes:
        return
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
        'width="100%" height="800" type="application/pdf"></iframe>'
    )
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_pdf_smart(pdf_bytes: bytes, *, download_name: str = "document.pdf") -> None:
    """
    Inline preview: at most the first PREVIEW_MAX_PAGES pages (iframe base64).
    Process PDF still uses the full upload; trimming is UI-only.
    """
    if not pdf_bytes:
        return
    pages = _pdf_page_count(pdf_bytes)
    snippet = _pdf_first_n_pages(pdf_bytes, PREVIEW_MAX_PAGES)
    safe_name = (download_name or "document.pdf").strip() or "document.pdf"

    if not snippet:
        st.warning(
            "Could not build a page-limited preview (PDF may be encrypted or invalid). "
            "Download the file to open it locally, or try another export."
        )
        st.download_button(
            label=f"Download PDF ({safe_name})",
            data=pdf_bytes,
            file_name=safe_name,
            mime="application/pdf",
            key="chatpdf_download_full_pdf",
        )
        return

    try:
        from pypdf import PdfReader

        shown = len(PdfReader(io.BytesIO(snippet)).pages)
    except Exception:
        shown = min(PREVIEW_MAX_PAGES, pages) if pages is not None else PREVIEW_MAX_PAGES

    trimmed = pages is None or pages > PREVIEW_MAX_PAGES
    if trimmed:
        label_pages = pages if pages is not None else "?"
        st.info(
            f"**Preview:** first **{shown}** page(s)"
            f"{f' of **{label_pages}**' if label_pages != '?' else ''}"
            ". **Process PDF** uses the **full** uploaded file."
        )

    display_pdf(snippet)

    if trimmed:
        st.download_button(
            label=f"Download full PDF ({safe_name})",
            data=pdf_bytes,
            file_name=safe_name,
            mime="application/pdf",
            key="chatpdf_download_full_pdf",
        )


def _env_or_input(env_name: str, text_value: str) -> str:
    t = (text_value or "").strip()
    if t:
        return t
    return (os.environ.get(env_name) or "").strip()


def _bind_upload_session(uploaded_file) -> None:
    """Refresh cached PDF bytes once per uploaded file id (cheap full-script reruns)."""
    if uploaded_file is None:
        st.session_state.pop(_SS_UPLOAD_FILE_ID, None)
        st.session_state.pop(_SS_UPLOAD_BYTES, None)
        st.session_state.pop(_SS_UPLOAD_DISP_NAME, None)
        return
    if st.session_state.get(_SS_UPLOAD_FILE_ID) != uploaded_file.file_id:
        st.session_state[_SS_UPLOAD_FILE_ID] = uploaded_file.file_id
        st.session_state[_SS_UPLOAD_BYTES] = uploaded_file.getvalue()
        st.session_state[_SS_UPLOAD_DISP_NAME] = uploaded_file.name or "document.pdf"


@st.fragment
def _chatpdf_right_panel() -> None:
    st.header("API Keys and Interaction")

    if st.session_state.pop(_SS_FEED_INGEST_OK, False):
        st.success("PDF processed successfully!")
        if st.session_state.pdf_stats:
            s = st.session_state.pdf_stats
            st.caption(
                f"Indexed: {s['texts']} text chunks · {s['tables']} tables · "
                f"{s['images']} images"
            )

    _ingest_err = st.session_state.pop(_SS_FEED_INGEST_ERR, None)
    if _ingest_err is not None:
        st.error(f"Error processing PDF:\n\n{_ingest_err}")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Required unless GROQ_API_KEY is set in `.env`.",
    )
    google_key = st.text_input(
        "Google API Key (Gemini embeddings)",
        type="password",
        help="Required unless GOOGLE_API_KEY is set in `.env`.",
    )
    langchain_key = st.text_input(
        "LangChain API Key (optional)",
        type="password",
        help="Enables LangSmith tracing when set.",
    )

    upload_bytes = st.session_state.get(_SS_UPLOAD_BYTES)
    upload_name_disp = (
        st.session_state.get(_SS_UPLOAD_DISP_NAME) or "document.pdf"
    )

    if st.button("Process PDF", type="primary"):
        groq_eff = _env_or_input("GROQ_API_KEY", groq_key)
        google_eff = _env_or_input("GOOGLE_API_KEY", google_key)
        if upload_bytes is None:
            st.error("Please upload a PDF first (left column).")
        elif not groq_eff or not google_eff:
            st.error(
                "Please provide Groq and Google API keys here, or set "
                "GROQ_API_KEY and GOOGLE_API_KEY in `.env`."
            )
        else:
            os.environ["GROQ_API_KEY"] = groq_eff
            os.environ["GOOGLE_API_KEY"] = google_eff
            if (langchain_key or "").strip():
                os.environ["LANGCHAIN_API_KEY"] = langchain_key.strip()
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
            else:
                os.environ.pop("LANGCHAIN_TRACING_V2", None)

            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(upload_bytes)
                    tmp_path = tmp.name

                progress_bar = st.progress(0.0)
                status_placeholder = st.empty()

                def update_progress(pct: float, msg: str) -> None:
                    progress_bar.progress(min(max(pct, 0.0), 1.0))
                    status_placeholder.caption(f"⏳ {msg}")

                from rag_pipeline import process_pdf

                result = process_pdf(tmp_path, update_progress)

                st.session_state.retriever = result["retriever"]
                st.session_state.pdf_stats = result["stats"]
                st.session_state.pdf_name = upload_name_disp
                st.session_state.messages = []
                st.session_state.last_media_source_note = ""
                st.session_state[_SS_FEED_INGEST_OK] = True
            except Exception as e:
                st.session_state[_SS_FEED_INGEST_ERR] = str(e)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                st.rerun()

    if st.session_state.retriever is not None:
        st.divider()
        st.subheader("Ask Questions")
        if st.session_state.pdf_name and st.session_state.pdf_stats:
            s = st.session_state.pdf_stats
            st.caption(
                f"Document: **{st.session_state.pdf_name}** - "
                f"{s['texts']} texts, {s['tables']} tables, {s['images']} images"
            )

        with st.form("qa_form", clear_on_submit=False):
            question = st.text_input(
                "Enter your question",
                key="mm_rag_question",
                placeholder="Type here and press Enter or click the button",
            )
            submitted = st.form_submit_button("Get Answer")

        if submitted:
            from src.app_utils import log_line, run_with_timeout

            prompt = (question or "").strip()
            if not prompt:
                st.warning("Please enter a question")
            else:
                ASK_TIMEOUT_SEC = 180.0
                retriever_snapshot = st.session_state.retriever
                source_name = st.session_state.pdf_name

                def _do_answer():
                    from src.generator import answer_question_bundle

                    if retriever_snapshot is None:
                        raise RuntimeError(
                            "Retriever is missing. Click Process PDF again."
                        )
                    return answer_question_bundle(
                        prompt,
                        retriever_snapshot,
                        source_filename=source_name,
                    )

                log_line(f"QA start | q={prompt[:300]!r}")
                answer = ""

                try:
                    with st.spinner("Generating answer..."):
                        bundle = run_with_timeout(_do_answer, ASK_TIMEOUT_SEC)
                except TimeoutError:
                    answer = (
                        f"**Timeout:** no response after {ASK_TIMEOUT_SEC:.0f}s. "
                        "Try again; details are logged to the server stderr."
                    )
                    log_line(f"QA TIMEOUT after {ASK_TIMEOUT_SEC:.0f}s")
                except Exception as e:
                    answer = f"**Error:** {e}"
                    log_line(f"QA EXCEPTION {repr(e)}")
                else:
                    answer = bundle.text
                    st.session_state.last_media_source_note = (
                        getattr(bundle, "media_source_note", None) or ""
                    )

                if not (answer and str(answer).strip()):
                    answer = "(Empty response - check API keys and `.env`.)"

                log_line(f"QA done | answer_len={len(str(answer))}")

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

        if st.session_state.messages:
            last = st.session_state.messages[-1]
            if last.get("role") == "assistant":
                st.write("Answer:", last.get("content", ""))
                if st.session_state.get("last_media_source_note"):
                    st.caption(
                        "**Retrieved media (pages in index):** "
                        + st.session_state.last_media_source_note
                    )
            if len(st.session_state.messages) > 2:
                with st.expander("Conversation history"):
                    for msg in st.session_state.messages[:-2]:
                        role = msg.get("role", "")
                        label = "You" if role == "user" else "Assistant"
                        st.markdown(f"**{label}:** {msg.get('content', '')}")

        if st.button("Clear & upload new"):
            st.session_state.retriever = None
            st.session_state.pdf_stats = None
            st.session_state.pdf_name = None
            st.session_state.messages = []
            st.session_state.last_media_source_note = ""
            st.session_state.pop(_SS_UPLOAD_FILE_ID, None)
            st.session_state.pop(_SS_UPLOAD_BYTES, None)
            st.session_state.pop(_SS_UPLOAD_DISP_NAME, None)
            st.session_state.pop(_SS_FEED_INGEST_OK, None)
            st.session_state.pop(_SS_FEED_INGEST_ERR, None)
            st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "last_media_source_note" not in st.session_state:
    st.session_state.last_media_source_note = ""

st.title("ChatPDF")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("PDF Upload and Preview")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    _bind_upload_session(uploaded_file)
    cached = st.session_state.get(_SS_UPLOAD_BYTES)
    if cached is not None:
        display_pdf_smart(
            cached,
            download_name=st.session_state.get(_SS_UPLOAD_DISP_NAME)
            or "document.pdf",
        )

with right_col:
    _chatpdf_right_panel()
