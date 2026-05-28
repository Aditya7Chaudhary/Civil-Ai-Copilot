import asyncio
import logging
import os
import secrets
from typing import Any, Dict

import jwt
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.graphs.main_graph import co_pilot_agent
from app.tools.boq_temp_store import clear_latest_csv, get_latest_csv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI_Backend")

app = FastAPI(
    title="Civil Engineering Co-Pilot API",
    description=(
        "Async FastAPI service exposing the LangGraph co-pilot. "
        "Supports local API-key auth (no Azure) or optional Azure AD JWT validation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security_bearer = HTTPBearer(auto_error=True)

TENANT_ID = os.getenv("AZURE_TENANT_ID", "common")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "").strip()
DEBUG_SKIP_AUTH = os.getenv("DEBUG_SKIP_AUTH", "").lower() == "true"


def _auth_mode() -> str:
    if DEBUG_SKIP_AUTH:
        return "debug_bypass"
    if AZURE_CLIENT_ID:
        return "azure_msal"
    if API_SECRET_KEY:
        return "api_key"
    return "none"


def verify_request(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)) -> Dict[str, Any]:
    """Reject unauthenticated clients. Azure MSAL is optional; default path uses API_SECRET_KEY."""
    token = credentials.credentials
    mode = _auth_mode()

    if mode == "debug_bypass":
        logger.warning("Auth bypass active (DEBUG_SKIP_AUTH=true). Do not use in production.")
        return {"user": "local_debug_engineer", "auth": mode}

    if mode == "none":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentication required. Set API_SECRET_KEY in .env and send "
                "Authorization: Bearer <key>, or configure AZURE_CLIENT_ID for MSAL."
            ),
        )

    if mode == "api_key":
        if not token or not secrets.compare_digest(token, API_SECRET_KEY):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
            )
        return {"user": "api_key_client", "auth": mode}

    # Azure AD / MSAL (optional — only when AZURE_CLIENT_ID is configured)
    try:
        discovery_url = f"https://login.microsoftonline.com/{TENANT_ID}/v2.0/.well-known/openid-configuration"
        jwks_uri = requests.get(discovery_url, timeout=10).json()["jwks_uri"]
        jwks_keys = requests.get(jwks_uri, timeout=10).json()["keys"]

        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        rsa_key = next((key for key in jwks_keys if key["kid"] == kid), None)
        if not rsa_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token signing key identity.",
            )

        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=AZURE_CLIENT_ID,
            issuer=f"https://login.microsoftonline.com/{TENANT_ID}/v2.0",
        )
        payload["auth"] = mode
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token has expired.",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation rejection: {exc}",
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Security validation failure.",
        )


class UserInquiryPayload(BaseModel):
    query: str


def _format_agent_response(final_state: dict) -> dict:
    intent = final_state.get("intent", "UNKNOWN")
    boq_available = intent == "COST" and bool(final_state.get("boq_csv_available", False))
    return {
        "success": True,
        "intent_routed": intent,
        "response": final_state.get("final_output", ""),
        "citations": final_state.get("citations", []),
        "sources": final_state.get("sources", []),
        "compliance_status": final_state.get("compliance_status", ""),
        "boq_csv_available": boq_available,
        "boq_csv_download_url": "/api/boq/download" if boq_available else None,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Human-friendly landing page (JSON endpoints look blank in some embedded browsers)."""
    mode = _auth_mode()
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Civil AI Co-Pilot API</title>
<style>
  body {{ font-family: Segoe UI, system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
  a {{ color: #4f46e5; }}
  .ok {{ color: #059669; }}
</style></head>
<body>
  <h1>Civil AI Co-Pilot API</h1>
  <p class="ok">Server is running.</p>
  <p>Auth mode: <code>{mode}</code></p>
  <ul>
    <li><a href="/docs">Swagger UI</a> — try <code>POST /api/invoke</code> here</li>
    <li><a href="/api/health">/api/health</a> — JSON health check</li>
    <li>Chat UI: open <code>app/frontend/index.html</code> in your browser (not this URL)</li>
  </ul>
  <p><small><code>/api/health</code> returns raw JSON; Cursor&apos;s browser may look empty — that is normal. Use links above or Chrome/Edge.</small></p>
</body></html>"""


@app.get("/api/health")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "CivilAI-Copilot-Engine",
        "auth_mode": _auth_mode(),
    }


async def _run_agent(user_query: str, claims: dict) -> dict:
    logger.info(
        "Processing query for auth=%s: '%s'",
        claims.get("auth", claims.get("user", "unknown")),
        user_query[:120],
    )
    user_id = (claims.get("oid") or claims.get("sub") or claims.get("user") or "anonymous")
    # Any new query invalidates the previous BOQ download for this user.
    clear_latest_csv(user_id)
    initial_state = {"query": user_query, "user_id": user_id}
    final_state = await asyncio.to_thread(co_pilot_agent.invoke, initial_state)
    return _format_agent_response(final_state)


@app.post("/api/invoke")
@app.post("/invoke")
async def invoke_agent(
    payload: UserInquiryPayload,
    token_claims: dict = Depends(verify_request),
):
    """Async invoke endpoint for the LangGraph co-pilot (Teams tab / MCP-style HTTP gateway)."""
    user_query = payload.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query text payload cannot be empty.")

    try:
        return await _run_agent(user_query, token_claims)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Agent execution error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Internal agent engine processing error: {exc}",
        )


@app.get("/api/boq/download")
async def download_latest_boq_csv(token_claims: dict = Depends(verify_request)):
    """
    Download the most recent BOQ CSV generated for this authenticated user.
    The file is overwritten per user (previous temp CSV is deleted when a new BOQ query runs).
    """
    user_id = (token_claims.get("oid") or token_claims.get("sub") or token_claims.get("user") or "anonymous")
    csv_path = get_latest_csv(user_id)
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="No BOQ CSV is available for download yet.")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="boq_results.csv",
    )
