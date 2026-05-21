import requests
import json

# Define the target local server URL
url = "http://127.0.0.1:8000/api/invoke"

# Set headers matching the FastAPI endpoint requirements
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer mock-dev-token"  # Passes DEBUG_SKIP_AUTH check
}

# The payload dictionary handled natively by Python
payload = {
    "query": "Give me an estimate for 100 cum of CONC-M30 columns."
}

print("📡 Sending request to Civil AI Co-Pilot Backend...")

try:
    # Send the POST request
    response = requests.post(url, json=payload, headers=headers)
    
    print(f"📥 Response Received! Status Code: {response.status_code}\n")
    
    # Format and print the JSON reply
    print(json.dumps(response.json(), indent=4))

except Exception as e:
    print(f"❌ Failed to communicate with the server: {str(e)}")