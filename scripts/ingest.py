import os
import sys
import argparse
import pickle
import logging
import tiktoken

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader

from app.tools.chunking import process_pdf_to_chunks, parse_standard_id

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "standards")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
QDRANT_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")

BM25_INDEX_PATH = os.path.join(QDRANT_PATH, "bm25_index.pkl")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "civil_codes"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(QDRANT_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "ingestion.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def token_len(text):
    return len(tokenizer.encode(text))


def run_ingestion_and_search(force: bool = False):
    logging.info("Building multi-standard hybrid RAG ingestion pipeline...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = QdrantClient(path=QDRANT_PATH)
    collection_names = [c.name for c in client.get_collections().collections]

    skip_ingestion = False
    if not force and COLLECTION_NAME in collection_names and os.path.exists(BM25_INDEX_PATH):
        if client.get_collection(COLLECTION_NAME).points_count > 0:
            logging.info("Existing index found. Use --force to rebuild with improved chunking.")
            skip_ingestion = True

    if not skip_ingestion:
        if force and COLLECTION_NAME in collection_names:
            logging.info("Force rebuild: deleting existing collection...")
            client.delete_collection(COLLECTION_NAME)

        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            logging.error(f"No PDFs found in {DATA_DIR}")
            return

        all_final_chunks = []
        for file in sorted(pdf_files):
            sid = parse_standard_id(file)
            logging.info(f"Processing {file} (standard_id={sid})...")
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            pages = loader.load()
            file_chunks = process_pdf_to_chunks(
                pages, file, token_len, CHUNK_SIZE, CHUNK_OVERLAP
            )
            logging.info(f"  -> {len(file_chunks)} chunks from {len(pages)} pages")
            all_final_chunks.extend(file_chunks)

        bm25_corpus = []
        for i, chunk in enumerate(all_final_chunks):
            chunk.metadata["chunk_id"] = i
            bm25_corpus.append(chunk.page_content)
            t_count = token_len(chunk.page_content)
            if t_count > (CHUNK_SIZE + 80):
                logging.warning(f"Chunk {i} ({chunk.metadata.get('standard_id')}) has {t_count} tokens")

        tokenized_corpus = [doc.lower().split() for doc in bm25_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        id_mapping = [c.metadata["chunk_id"] for c in all_final_chunks]
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25_obj": bm25, "id_mapping": id_mapping}, f)

        if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        logging.info("Embedding and uploading to Qdrant...")
        qdrant_store = QdrantVectorStore(
            client=client, collection_name=COLLECTION_NAME, embedding=embeddings
        )
        qdrant_store.add_documents(all_final_chunks)
        logging.info(f"Loaded {len(all_final_chunks)} chunks into '{COLLECTION_NAME}'.")

        by_std = {}
        for c in all_final_chunks:
            sid = c.metadata.get("standard_id", "?")
            by_std[sid] = by_std.get(sid, 0) + 1
        logging.info(f"Chunks per standard: {by_std}")

    else:
        qdrant_store = QdrantVectorStore(
            client=client, collection_name=COLLECTION_NAME, embedding=embeddings
        )
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
            bm25 = bm25_data["bm25_obj"] if isinstance(bm25_data, dict) else bm25_data

    logging.info("Smoke test: IS 13920 ductile detailing column hinge region")
    raw_query = "special confining reinforcement spacing in column hinge region IS 13920"
    vector_results = qdrant_store.similarity_search_with_score(raw_query, k=8)
    for i, (doc, score) in enumerate(vector_results[:5], 1):
        logging.info(
            f"[{i}] score={score:.3f} std={doc.metadata.get('standard_id')} "
            f"clause={doc.metadata.get('clause_ref')} src={doc.metadata.get('source_doc')}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest IS standard PDFs into Qdrant")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the vector index with improved chunking",
    )
    args = parser.parse_args()
    run_ingestion_and_search(force=args.force)
