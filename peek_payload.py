import os
from qdrant_client import QdrantClient

# Define path to local persistent folder
QDRANT_PATH = os.path.join(os.getcwd(), "data", "qdrant_storage")
# ⚠️ Change this to your active collection name from Case B!
COLLECTION_NAME = "civil_codes" 

print(f"🧐 Peeking inside collection: '{COLLECTION_NAME}'...")

try:
    client = QdrantClient(path=QDRANT_PATH)
    
    # Scroll and retrieve exactly 1 point out of storage to inspect its inner structures
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=True,
        with_vectors=False
    )
    
    if not records:
        print("❌ Zero points found in this collection. It is completely empty!")
    else:
        point = records[0]
        print("\n🎯 Found a Data Point! Here is how your ingestion script saved it:")
        print("-----------------------------------------------------------------")
        print(f"Point ID: {point.id}")
        print("\nPayload Keys & Values:")
        import json
        print(json.dumps(point.payload, indent=4))
        print("-----------------------------------------------------------------")
        
except Exception as e:
    print(f"❌ Failed to read storage structures: {str(e)}")