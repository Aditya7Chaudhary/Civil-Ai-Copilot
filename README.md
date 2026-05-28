# Civil Engineering AI Co-Pilot

Local-first structural co-pilot: IS code compliance (RAG), BOQ cost lookup (CSR 2024), and RFI drafting, orchestrated with **LangGraph** and exposed via **FastAPI**.

Uses **Qdrant**, **HuggingFace embeddings**, and **Groq** instead of Azure (no cloud account required).

## Prerequisites

- Python 3.11+
- IS standard PDFs in `data/standards/` (not committed to git)
- [Groq API key](https://console.groq.com/)

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

- Set `GROQ_API_KEY`
- Set `API_SECRET_KEY` to a strong local secret (recommended), **or** set `DEBUG_SKIP_AUTH=true` for quick local testing only

## Index the standards (first run)

```bash
python scripts/ingest.py
```

## Run the API

```bash
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

## Run the frontend (local URL)

This repo’s frontend is a static page in `app/frontend/`, served locally so you get a clickable URL in the terminal.

```bash
python scripts/run_frontend.py
```

You should see a link like:

- `http://127.0.0.1:5173/index.html`

Optional:

- Open automatically: `python scripts/run_frontend.py --open`
- Use a different port: `python scripts/run_frontend.py --port 3000`

Endpoints:

| Method | Path | Auth |
|--------|------|------|
| GET | `/health`, `/api/health` | None |
| POST | `/invoke`, `/api/invoke` | Bearer token |
| GET | `/api/boq/download` | Bearer token |

### Authentication (no Azure)

1. **API key (default for assignments):**  
   `Authorization: Bearer <your API_SECRET_KEY>`  
   Returns **401** if missing or wrong.

2. **Debug bypass:** `DEBUG_SKIP_AUTH=true` in `.env` — any bearer string is accepted.

3. **Optional MSAL:** If `AZURE_CLIENT_ID` is set, validates Azure AD JWTs (for teams with a Microsoft tenant).

## Chat UI (Teams-ready static tab)

Open `app/frontend/index.html` in a browser, or sideload as a Teams personal tab (see `teams/appPackage/`).

- Set **API base URL** (default `http://127.0.0.1:8000`)
- Enter the same **API key** as `API_SECRET_KEY` in `.env` (stored in session for the tab session)

## Quick API test

```bash
# With API_SECRET_KEY set in .env:
set API_KEY=your-secret-from-env
python test_api.py
```

## Project layout

- `app/graphs/main_graph.py` — LangGraph router (compliance / cost / RFI / chat)
- `app/tools/` — retrieval, compliance, cost, RFI tools
- `app/api/main.py` — async FastAPI gateway
- `data/qdrant_storage/` — vector index
- `data/csr/csr_rates.csv` — CSR 2024 rate table (~82 items)
- `data/temp/boq/` — temporary BOQ CSV outputs (per user; cleared on next query)

## BOQ download behavior

When a COST/BOQ query runs:

- The backend generates the BOQ result via **NLP → SQL** over a local SQLite copy of `data/csr/csr_rates.csv`.
- A temporary CSV is created and is available at `GET /api/boq/download`.
- On the **next query** from the same user (any intent), the previous BOQ CSV is deleted.
