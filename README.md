# Multimodal RAG System for PDF Documents

A Streamlit-based application that processes PDF files to extract text, tables, and images; summarizes the extracted data; and uses a retrieval-augmented generation (RAG) pipeline to answer user questions based on the document content. This project leverages multiple APIs and libraries such as Gemini, LangChain, and unstructured to provide a multimodal interface for document understanding.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture (brief)](#architecture-brief)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)



## Overview

This project is designed to:

- **Extract** content from PDFs, including text, tables, and images using the `unstructured` library.
- **Summarize** the extracted content using custom pipelines that integrate with ChatGroq and ChatOpenAI.
- **Summarize** text and tables with **Groq** (`llama-3.1-8b-instant`), and images with **Groq vision** (`meta-llama/llama-4-scout-17b-16e-instruct`).
- **Index** with a **MultiVectorRetriever**: embeddings of **summaries** in **Chroma**; **raw** chunks (text / table HTML / image base64) in an **in-memory docstore**, linked by `doc_id`, with **page** metadata for citations.
- **Answer** questions with retrieval-augmented generation, optional **vision** when image context is present, and strict **inline citations** `[Doc: …, Page: …]`.


## Features

- **PDF upload & preview** (first **7 pages** in-browser preview via `pypdf`; full file still used for processing).
- **Multimodal extraction** (text, tables, figures) and **table merge** heuristics for page-split tables.
- **Semantic search** over summaries; **grounded answers** from original passages / tables / images.
- **Streamlit UI** for API keys (or `.env`), processing progress, Q&A, and optional **retrieved media page** hints.

## Project Structure

```text
ChatPDF/                        (project root)
├── app.py                      # Streamlit UI: upload, preview, Process PDF, Q&A
├── rag_pipeline.py             # Orchestrator: extract → summarize → build retriever
├── requirements.txt            # Python dependencies
├── .env                        # API keys (do NOT commit — use .env.example as template)
├── .streamlit/
│   └── config.toml             # Streamlit theme and performance config
├── papers/                     # Sample PDF for testing
└── src/
    ├── __init__.py
    ├── config.py               # Chunk sizes, concurrency limits, retriever k
    ├── models.py               # ChatGroq + Gemini embeddings initialization
    ├── extractor.py            # unstructured partition_pdf + page enrichment
    ├── table_merge.py          # Adjacent HTML table merge across pages
    ├── summarizer.py           # Text / table / image summaries via Groq
    ├── retriever.py            # MultiVectorRetriever (Chroma + InMemoryStore)
    ├── generator.py            # Retrieve → prompt → text/vision LLM → AnswerBundle
    └── app_utils.py            # Logging helpers
```

## Architecture (brief)

1. **Ingest:** `extract_pdf_elements` → lists of texts, tables, images + parallel **page** lists.  
2. **Summarize:** `summarize_texts` / `summarize_tables` / `summarize_images`.  
3. **Index:** `build_retriever` → Chroma vectors on **summaries**; docstore holds **raw** content + `type` + `page`.  
4. **Query:** `answer_question_bundle` → retrieve → modality filters → LLM (vision if images) → citations + optional `media_source_note`.

## Prerequisites

- **Python 3.10+** (3.11 recommended).
- **Groq** API key and **Google** API key (for **Gemini embeddings**).
- For reliable **`hi_res`** PDF parsing with `unstructured` on **Windows**:
  - **Poppler** (PDF utilities; e.g. add `…\Library\bin` to `PATH`, or adjust paths in `src/extractor.py`).
  - **Tesseract OCR** (optional but useful for scanned PDFs; paths are probed in `src/extractor.py`).

On **Linux**, install OS packages such as `tesseract-ocr` and `poppler-utils` and ensure they are on `PATH`.

## Installation

### 1. Clone or copy the project

```bash
git clone https://github.com/Chaldealord/ChatPDF_MultimodalRAG.git
cd ChatPDF_MultimodalRAG

```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
```

If activating the venv fails with an **execution policy** error:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

**Alternative (no PowerShell scripts):** use **cmd**:
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip and install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment variables

Create a **`.env`** file in the project root (see [Configuration](#configuration)). Never commit real keys to git.

### 5. Optional: Streamlit config

If you use `.streamlit/config.toml`, keep it when deploying (e.g. `server.maxUploadSize`).

## Configuration

Create **`.env`** in the project root with at least:

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_generative_ai_key
```

Optional:

```env
LANGCHAIN_API_KEY=...        # LangSmith tracing
LANGCHAIN_TRACING_V2=true
```

Tuning knobs (chunk sizes, summarizer concurrency, `RETRIEVER_SEARCH_K`) live in **`src/config.py`**.

## Usage

### Running the web app

With the virtual environment **activated**:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default **http://localhost:8501**). Upload a PDF, click **Process PDF** (wait for indexing), then ask questions in the right column.

