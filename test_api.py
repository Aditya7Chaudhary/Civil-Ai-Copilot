import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
url = f"{BASE}/api/invoke"

api_key = os.getenv("API_SECRET_KEY", "").strip()
if api_key:
    auth_token = api_key
elif os.getenv("DEBUG_SKIP_AUTH", "").lower() == "true":
    auth_token = "mock-dev-token"
else:
    raise SystemExit(
        "Set API_SECRET_KEY or DEBUG_SKIP_AUTH=true in .env before running test_api.py"
    )

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_token}",
}

payload = {"query": "Give me an estimate for 100 cum of CONC-M30 columns."}

print("Sending request to Civil AI Co-Pilot Backend...")

try:
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    print(f"Status: {response.status_code}\n")
    print(json.dumps(response.json(), indent=2))
except Exception as exc:
    print(f"Request failed: {exc}")
