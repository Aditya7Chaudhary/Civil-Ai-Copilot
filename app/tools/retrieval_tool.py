import os
import re
import pickle
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# Resolve paths relative to this tool's position inside /app/tools/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QDRANT_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
BM25_INDEX_PATH = os.path.join(QDRANT_PATH, "bm25_index.pkl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "civil_codes"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetrievalTool")

class EngineeringRetriever:
    def __init__(self):
        logger.info("Initializing Hybrid Engineering Retriever Client...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.client = QdrantClient(path=QDRANT_PATH)
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=COLLECTION_NAME, 
            embedding=self.embeddings
        )
        
        if not os.path.exists(BM25_INDEX_PATH):
            raise FileNotFoundError(f"Missing lexical index assets at: {BM25_INDEX_PATH}. Please run ingestion first.")
            
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

    def _normalize_query(self, query: str) -> str:
        """Cleans and normalizes query text for consistent cross-index scoring."""
        query = query.lower().strip()
        query = re.sub(r"\b(is)\s*456\b", "", query)  # Remove structural names that skew math weights
        return query

    def retrieve_evidence(self, query: str, k: int = 6) -> dict:
        """Executes a hybrid score-fused search across vector space and lexical indexes."""
        cleaned_query = self._normalize_query(query)
        tokenized_query = cleaned_query.split()
        
        # 1. Gather Dense Vector Options
        vector_results = self.vector_store.similarity_search_with_score(query, k=k * 2)
        if not vector_results:
            return {"retrieved_clauses": [], "citations": [], "sources": []}
            
        # 2. Extract Lexical Lexicon Score Weights
        bm25_scores = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        
        v_scores = [score for _, score in vector_results]
        v_max, v_min = max(v_scores), min(v_scores)
        v_range = v_max - v_min + 1e-5
        
        fused_results = []
        for idx, (doc, v_score) in enumerate(vector_results):
            chunk_id = doc.metadata.get("chunk_id")
            
            # Extract corresponding lexical footprint matching this structural chunk ID
            b_score = bm25_scores[chunk_id] if chunk_id is not None else 0
            
            # Apply Min-Max Normalization Scales
            norm_v = (v_score - v_min) / v_range
            norm_b = b_score / max_bm25
            
            # Hybrid Calculation (60% Dense Representation / 40% Keyword Representation)
            final_score = (0.6 * norm_v) + (0.4 * norm_b)
            
            # Meta Heading Term Matching Boost (+0.15 Score bump)
            heading = doc.metadata.get("heading", "").lower()
            for token in tokenized_query:
                if len(token) > 3 and token in heading:
                    final_score += 0.15
                    break
                    
            fused_results.append((final_score, doc))
            
        # Re-rank elements by their fused combination metrics
        fused_results.sort(key=lambda x: x[0], reverse=True)
        top_matches = fused_results[:k]
        
        # 3. Serialize Output Payload Contracts
        retrieved_clauses = []
        citations = []
        sources = set()
        
        for score, doc in top_matches:
            clause_ref = doc.metadata.get("clause_ref", "General")
            heading = doc.metadata.get("heading", "Unknown Section")
            source_doc = doc.metadata.get("source_doc", "Standard")
            
            citation_string = f"Clause {clause_ref} ({heading})"
            citations.append(citation_string)
            sources.add(source_doc)
            
            retrieved_clauses.append({
                "clause_id": clause_ref,
                "title": heading,
                "text": doc.page_content,
                "confidence_score": round(score, 3),
                "origin_file": source_doc
            })
            
        self.client.close() 
        return {
            "retrieved_clauses": retrieved_clauses,
            "citations": list(dict.fromkeys(citations)),
            "sources": list(sources)
        }