import os
from qdrant_client import QdrantClient

# Define path to local persistent folder
QDRANT_PATH = os.path.join(os.getcwd(), "data", "qdrant_storage")
print(f"Checking storage folder at: {QDRANT_PATH}")

if not os.path.exists(QDRANT_PATH):
    print("❌ ERROR: The directory 'data/qdrant_storage' does not exist at all!")
else:
    try:
        client = QdrantClient(path=QDRANT_PATH)
        collections = client.get_collections().collections
        
        if not collections:
            print("❌ DB IS EMPTY: No collections exist inside your local Qdrant engine.")
        else:
            for col in collections:
                col_info = client.get_collection(collection_name=col.name)
                print(f"✅ Found Collection: '{col.name}' | Total Embedded Vectors: {col_info.points_count}")
                
    except Exception as e:
        print(f"❌ Failed to read database files: {str(e)}")