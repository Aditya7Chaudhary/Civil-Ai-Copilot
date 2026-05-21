import os
import logging
import jwt
import requests
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your unified LangGraph orchestrator compiled in Task 1
from app.graphs.main_graph import co_pilot_agent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI_Backend")

app = FastAPI(
    title="Civil Engineering Co-Pilot MCP Backend",
    description="Production FastAPI service wrapping LangGraph agent workflows with MSAL validation."
)

# Enable Cross-Origin Resource Sharing (CORS) for Teams Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific Teams App Domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security_bearer = HTTPBearer()

# Configuration variables for Azure AD / MSAL validation
TENANT_ID = os.getenv("AZURE_TENANT_ID", "common")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")

def verify_msal_token(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)) -> dict:
    """Middle-tier security barrier confirming incoming bearer tokens against Microsoft identity keys."""
    token = credentials.credentials
    
    # Bypass authorization ONLY if explicitly running in a local unauthenticated debug configuration
    if os.getenv("DEBUG_SKIP_AUTH") == "true":
        logger.warning("⚠️ SECURITY WARNING: Auth validation bypassed via DEBUG_SKIP_AUTH flags.")
        return {"user": "local_debug_engineer"}

    if not CLIENT_ID:
        raise HTTPException(status_code=500, detail="Server misconfigured: AZURE_CLIENT_ID is missing.")

    try:
        # 1. Fetch current Microsoft open ID configuration signing keys
        discovery_url = f"https://login.microsoftonline.com/{TENANT_ID}/v2.0/.well-known/openid-configuration"
        jwks_uri = requests.get(discovery_url).json()["jwks_uri"]
        jwks_keys = requests.get(jwks_uri).json()["keys"]
        
        # 2. Extract unverified JWT token header tracking info
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        # 3. Locate matching cryptographic key
        rsa_key = next((key for key in jwks_keys if key["kid"] == kid), None)
        if not rsa_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signing token key identity.")
            
        # 4. Strictly validate signatures, token expiration, and audience claims
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=CLIENT_ID,
            issuer=f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"
        )
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization token has expired.")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token validation rejection: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Security validation subsystem failure.")


# ==========================================
# API ENDPOINT SCHEMAS AND CONTROLLERS
# ==========================================

class UserInquiryPayload(BaseModel):
    query: str

@app.get("/api/health")
def health_check():
    """Liveness endpoint for service health monitoring."""
    return {"status": "healthy", "service": "CivilAI-Copilot-Engine"}

@app.post("/api/invoke")
def invoke_agent(payload: UserInquiryPayload, token_claims: dict = Depends(verify_msal_token)):
    """Secure endpoint processing natural language engineering parameters through the LangGraph architecture."""
    user_query = payload.query.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query text payload cannot be empty.")
        
    try:
        logger.info(f"Processing structural task request: '{user_query}' for user context.")
        
        # Inject query payload into compiled LangGraph state workflow tracking arrays
        initial_state = {"query": user_query}
        final_state = co_pilot_agent.invoke(initial_state)
        
        # Return cleanly structured response payload back to client surface
        return {
            "success": True,
            "intent_routed": final_state.get("intent", "UNKNOWN"),
            "response": final_state.get("final_output", ""),
            "citations": final_state.get("citations", []),
            "sources": final_state.get("sources", []),
            "compliance_status": final_state.get("compliance_status", ""),
        }
        
    except Exception as e:
        logger.error(f"Execution boundary error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal agent engine processing error: {str(e)}")