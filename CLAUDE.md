# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agente Aduaneiro is an AI-powered customs compliance auditor that validates NCM (Nomenclatura Comum do Mercosul) product classifications. It uses RAG (Retrieval-Augmented Generation) with NESH explanatory notes indexed in FAISS, combined with GPT-4, to generate technical audit reports.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app (opens at http://localhost:8501)
streamlit run app.py

# Run all tests
python -m pytest tests/ -v

# Run a specific test class
python -m pytest tests/test_validador.py::TestFormatoNCM -v

# Run a single test
python -m pytest tests/test_validador.py::TestFormatoNCM::test_ncm_valida -v
```

## Architecture

**Entry point:** `app.py` — Streamlit web UI with three tabs: single validation, batch processing (CSV/Excel), and cost analysis.

**Core pipeline:**
1. User submits a product description + NCM code
2. `RAGEngine.search()` (`src/rag_engine.py`) retrieves relevant NESH chunks from a FAISS vector store using `sentence-transformers` (all-MiniLM-L6-v2) embeddings
3. `ValidadorNCM.validar()` (`src/llm_agent.py`) constructs prompts with RAG context and calls the OpenAI API (GPT-4) with retry logic
4. The LLM returns a structured JSON response with: `status` (APROVADO/RISCO/ATENCAO), `mensagem`, `justificativa`, `ncm_sugerida_alternativa`, `regras_citadas`
5. Results are logged to daily JSONL files in `outputs/logs/`

**Key modules:**
- `src/prompts.py` — System and user prompt templates (Portuguese)
- `src/utils.py` — NCM format validation, cost calculation, token counting, JSONL logging, report export

## Key Conventions

- **Language:** All code names, comments, docstrings, and UI text are in Portuguese
- **NCM format:** 8-digit code normalized to `XXXX.XX.XX` pattern
- **Configuration:** OpenAI API key and model set via `.env` file (see `.env.example`)
- **Caching:** Streamlit `@st.cache_resource` is used for expensive RAG engine initialization
- **Logging:** Python `logging` module; validation results appended to daily JSONL files
- **Vector store:** Pre-built FAISS index and numpy chunks live in `data/vectorstore/`; PDFs sourced from `data/nesh/`
- **Test data:** `data/exemplos_ncm.json` contains 10 curated test cases (3 APROVADO, 4 RISCO, 3 ATENCAO)
- **No linter/formatter configured** — no flake8, black, or pylint config exists
