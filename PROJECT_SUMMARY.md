## Civil Engineering AI Co-Pilot (Project Summary)

This project is a local-first civil/structural engineering assistant that routes user queries into focused workflows:

- **Compliance auditing (RAG)**: retrieves relevant clause text from ingested Indian Standard documents and answers based on that evidence.
- **BOQ / cost estimation**: estimates cost from CSR (CPWD Schedule of Rates) line items.
- **RFI drafting**: creates a formal Request For Information draft based on a described site issue.
- **General chat**: fallback for out-of-scope conversation.

### Architecture

- **Frontend**: static Teams-ready UI in `app/frontend/index.html`
  - Sends `POST /api/invoke` with a user query
  - Displays response text, citations, sources
  - If BOQ CSV is available, shows **Download BOQ CSV**
- **Backend**: FastAPI gateway in `app/api/main.py`
  - Auth supports API key (default) or optional Azure AD validation; dev bypass is available via `DEBUG_SKIP_AUTH=true`
  - Runs the LangGraph workflow to classify intent and execute the correct toolchain
- **Orchestration**: LangGraph router in `app/graphs/main_graph.py`
  - `COMPLIANCE`, `COST`, `RFI`, `CHAT` routes
- **Standards retrieval**: Qdrant + embeddings (ingested via `scripts/ingest.py`)

### BOQ / cost estimation details

- **NLP → SQL**: the COST route generates a SQLite `SELECT` query (guardrailed to SELECT/WITH only) to compute line totals by joining the user-requested items against `csr_rates`.
- **CSR rates store**:
  - Source of truth remains `data/csr/csr_rates.csv`
  - A local SQLite copy is created on-demand at `data/csr/csr_rates.sqlite3`
- **Temporary BOQ CSV**:
  - After generating BOQ results, the backend writes a temporary CSV file under `data/temp/boq/`
  - It becomes downloadable at `GET /api/boq/download`
  - The temp BOQ CSV is **deleted automatically on the next query** from that user (any intent)

### Key endpoints

- `GET /api/health`: health check
- `POST /api/invoke`: run the agent on a query (auth required)
- `GET /api/boq/download`: download the most recent BOQ CSV for the current authenticated user (auth required)

