import os
import re
import pickle
import logging
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader

# =====================================================================
# 1. PATHS & INITIAL CONFIGURATION
# =====================================================================
DATA_DIR = "D:\Programming\civil-ai-copilot\data\standards"
LOGS_DIR = "D:\Programming\civil-ai-copilot\logs"
QDRANT_PATH = "D:\Programming\civil-ai-copilot\data\qdrant_storage"

BM25_INDEX_PATH = os.path.join(QDRANT_PATH, "bm25_index.pkl")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" 
COLLECTION_NAME = "civil_codes"
CHUNK_SIZE = 800       # Maximum target token limit per chunk
CHUNK_OVERLAP = 100

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(QDRANT_PATH, exist_ok=True)

# Logging configuration setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "ingestion.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

tokenizer = tiktoken.get_encoding("cl100k_base")
def token_len(text):
    return len(tokenizer.encode(text))

# =====================================================================
# 2. STREAM-BASED CLEANING & STRUCTURAL REGEX
# =====================================================================
def clean_pdf_text(text):
    """Removes layout noise while safely preserving core engineering tables."""
    # REMOVED: The table stripping line has been deleted to keep Table 16, Table 21, etc.
    
    # Clean out standard page headers/footers
    text = re.sub(r"IS\s+\d+(?:\s*:\s*\d+)?", "", text, flags=re.IGNORECASE)
    
    # Wipe standalone floating page numbers
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    
    return text

# HYPER-SENSITIVE STRUCTURAL REGEX
# 1. (?:^|\n)\s* -> Triggers on any new line, bypassing margin spaces.
# 2. (\d+(?:\.\d+)*)     -> FIX: Matches single integers ("3", "26") AND decimals ("26.4.1").
# 3. \s+                 -> Expects spacing between the clause ID and its heading name.
# 4. ([A-Z][A-Za-z0-9\s,\-\(\)\/\.]{2,100}) -> Captures the title (starts with a capital letter).
# 5. (?=\n|\s{2,}|$)     -> Looks ahead to stop before body text or wide spacing columns.
CLAUSE_REGEX = re.compile(
    r"(?:^|\n)\s*(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9\s,\-\(\)\/\.]{2,100})(?=\n|\s{2,}|$)"
)

# =====================================================================
# 3. CLAUSE SEGMENTATION & FAILSAFE SUB-CHUNKING
# =====================================================================

def segment_document_by_clauses(full_text, filename):
    """Splits text streams into clean structural clauses, keeping children with parents."""
    matches = list(CLAUSE_REGEX.finditer(full_text))
    structured_docs = []
    
    if not matches:
        return [Document(page_content=full_text, metadata={"clause_ref": "General", "heading": "Full Document", "source_doc": filename})]
        
    for idx, match in enumerate(matches):
        clause_ref = match.group(1)
        heading = match.group(2).strip()
        start_pos = match.start()
        
        # FIX: Force the slice window to read continuously through sub-blocks 
        # until the next distinct matching clause structure is found
        end_pos = matches[idx + 1].start() if (idx + 1) < len(matches) else len(full_text)
        clause_body = full_text[start_pos:end_pos].strip()
        
        if len(clause_body) > 10:
            structured_docs.append(Document(
                page_content=clause_body,
                metadata={"clause_ref": clause_ref, "heading": heading, "source_doc": filename}
            ))
            
    return structured_docs

def sub_chunk_oversized_clauses(clause_documents):
    """Ensures huge concepts are safely sliced down to target limits."""
    processed_chunks = []
    
    # Failsafe character splitter to catch any layout overruns
    sub_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len
    )
    
    for doc in clause_documents:
        tokens = token_len(doc.page_content)
        
        if tokens <= CHUNK_SIZE:
            # Concept fits perfectly inside the token ceiling
            processed_chunks.append(doc)
        else:
            # Trigger safe sub-chunking for oversized text blocks
            logging.info(f"✂️ Clause {doc.metadata['clause_ref']} is quite large ({tokens} tokens). Sub-chunking...")
            sub_docs = sub_splitter.split_documents([doc])
            
            for sub_idx, sub_doc in enumerate(sub_docs):
                # Ensure child fragments retain original source metadata properties
                sub_doc.metadata["clause_ref"] = doc.metadata["clause_ref"]
                sub_doc.metadata["heading"] = f"{doc.metadata['heading']} (Part {sub_idx + 1})"
                sub_doc.metadata["source_doc"] = doc.metadata["source_doc"]
                processed_chunks.append(sub_doc)
                
    return processed_chunks

# =====================================================================
# 4. PIPELINE EXECUTION & STRUCTURAL PACKAGING
# =====================================================================
def run_ingestion_and_search():
    logging.info("🚀 Building Stream-Bounded Hybrid RAG Ingestion Pipeline...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = QdrantClient(path=QDRANT_PATH)
    collection_names = [c.name for c in client.get_collections().collections]

    skip_ingestion = False
    if COLLECTION_NAME in collection_names and os.path.exists(BM25_INDEX_PATH):
        if client.get_collection(COLLECTION_NAME).points_count > 0:
            logging.info("⏭️ Verified local persistent assets. Bypassing extraction loops.")
            skip_ingestion = True

    if not skip_ingestion:
        logging.info("🔨 Initializing vector engine collection matrices...")
        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            logging.error(f"❌ Core processing failed: No PDFs matching criteria inside {DATA_DIR}")
            return

        all_final_chunks = []
        for file in pdf_files:
            logging.info(f"📖 Processing text layout profiles for: {file}")
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            pages = loader.load()
            
            # Combine individual page streams into a continuous layout string
            raw_full_text = "\n".join([page.page_content for page in pages])
            
            # Executing Pipeline Pipeline Pipeline
            cleaned_text = clean_pdf_text(raw_full_text)
            clause_documents = segment_document_by_clauses(cleaned_text, file)
            final_file_chunks = sub_chunk_oversized_clauses(clause_documents)
            
            all_final_chunks.extend(final_file_chunks)

        # Build cross-indexing maps for score fusion
        bm25_corpus = []
        for i, chunk in enumerate(all_final_chunks):
            chunk.metadata["chunk_id"] = i
            bm25_corpus.append(chunk.page_content)
            
            # Safety Check Validation
            t_count = token_len(chunk.page_content)
            if t_count > (CHUNK_SIZE + 50):
                logging.warning(f"⚠️ Failsafe Alert: Chunk ID {i} is drifting high ({t_count} tokens)!")

        logging.info("🔤 Engineering lexical term weights for BM25 calculations...")
        tokenized_corpus = [doc.lower().split() for doc in bm25_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25, f)
        
        logging.info("🧠 Commencing heavy vector model matrix transformations...")
        qdrant_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
        qdrant_store.add_documents(all_final_chunks)
        logging.info(f"✅ Setup success! Loaded {len(all_final_chunks)} dense conceptual chunks.")

    else:
        qdrant_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)

    # =====================================================================
    # 5. RETRIEVAL SMOKE TEST VERIFICATION
    # =====================================================================
    logging.info("🔥 EXECUTING HYBRID RETRIEVAL SMOKE TEST")
    raw_query = "minimum cover for reinforcement in slabs IS 456"
    
    vector_results = qdrant_store.similarity_search_with_score(raw_query, k=10)
    tokenized_query = raw_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

    fused_results = []
    v_scores = [s for _, s in vector_results]
    v_max, v_min = max(v_scores), min(v_scores)
    
    for idx, (doc, v_score) in enumerate(vector_results):
        chunk_id = doc.metadata.get("chunk_id")
        b_score = bm25_scores[chunk_id]
        
        norm_v = (v_score - v_min) / (v_max - v_min + 1e-5) if (v_max - v_min) > 0 else v_score
        norm_b = b_score / max_bm25
        final_score = (0.6 * norm_v) + (0.4 * norm_b)
        
        heading = doc.metadata.get("heading", "unknown").lower()
        for term in raw_query.lower().split():
            if len(term) > 3 and term in heading: 
                final_score += 0.15
                break 
                
        fused_results.append((final_score, idx, doc, norm_v, norm_b))
        
    fused_results.sort(key=lambda x: x[0], reverse=True)

    for i, (final_score, idx, doc, v, b) in enumerate(fused_results[:3], 1):
        output_block = (
            f"\n[RANK MATCH {i}]\n"
            f"Source Document: {doc.metadata.get('source_doc')}\n"
            f"Clause Reference: {doc.metadata.get('clause_ref')} | Heading: {doc.metadata.get('heading')}\n"
            f"Tokens: {token_len(doc.page_content)} | Combined Score: {final_score:.2f}\n"
            f"Parsed Content Chunk:\n{doc.page_content[:220].strip()}...\n"
        )
        logging.info(output_block)

if __name__ == "__main__":
    run_ingestion_and_search()