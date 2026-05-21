import os
from qdrant_client import QdrantClient

QDRANT_PATH = os.path.join(os.getcwd(), "data", "qdrant_storage")
COLLECTION_NAME = "civil_codes"

try:
    client = QdrantClient(path=QDRANT_PATH)
    
    # 1. Fetch the exact total count of points across the entire collection
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    total_points = collection_info.points_count
    print(f"📊 Total Cumulative Chunks in Database: {total_points}")
    print("--------------------------------------------------")

    # 2. Page through the database to count chunks per document
    document_counts = {}
    next_page_offset = None
    total_scanned = 0
    
    print("Scanning data records...")
    while True:
        records, next_page_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=next_page_offset
        )
        
        if not records:
            break
            
        for point in records:
            meta = point.payload.get("metadata", {})
            source = meta.get("source_doc") or meta.get("source") or "Unknown Document"
            document_counts[source] = document_counts.get(source, 0) + 1
            
        total_scanned += len(records)
        if next_page_offset is None:
            break

    print("\n📦 Chunks Successfully Indexed Per File:")
    print("--------------------------------------------------")
    for doc, count in document_counts.items():
        print(f"🔹 {doc}: {count} chunks")
    print("--------------------------------------------------")

except Exception as e:
    print(f"❌ Diagnostic failed: {str(e)}")