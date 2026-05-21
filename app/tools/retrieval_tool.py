import os
import re
import pickle
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

from app.tools.chunking import (
    detect_standards_from_query,
    source_doc_matches_standard,
    parse_standard_id,
    get_file_marker,
    get_retrieval_boost,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QDRANT_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
BM25_INDEX_PATH = os.path.join(QDRANT_PATH, "bm25_index.pkl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "civil_codes"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetrievalTool")

_retriever_instance = None


def get_engineering_retriever() -> "EngineeringRetriever":
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = EngineeringRetriever()
    return _retriever_instance


class EngineeringRetriever:
    def __init__(self):
        logger.info("Initializing Native Hybrid Engineering Retriever Client...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.client = QdrantClient(path=QDRANT_PATH)

        if not os.path.exists(BM25_INDEX_PATH):
            raise FileNotFoundError(f"Missing lexical index at: {BM25_INDEX_PATH}. Run scripts/ingest.py first.")

        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25_data = pickle.load(f)
            if isinstance(self.bm25_data, dict):
                self.bm25 = self.bm25_data.get("bm25_obj")
                self.corpus_ids = self.bm25_data.get("id_mapping", [])
            else:
                self.bm25 = self.bm25_data
                try:
                    corpus_size = len(self.bm25.get_scores([""]))
                except Exception:
                    corpus_size = 0
                self.corpus_ids = list(range(corpus_size))

    def _normalize_query(self, query: str) -> str:
        return query.lower().strip()

    def _standard_augmented_query(self, query: str, standard_id: str) -> str:
        return f"{query} Indian Standard {standard_id} structural code provisions clauses"

    def _vector_search(self, query_vector: List[float], limit: int, source_filter: Optional[str] = None) -> list:
        qfilter = None
        if source_filter:
            qfilter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_doc",
                        match=MatchText(text=source_filter),
                    )
                ]
            )

        for kwargs_variant in (
            {"query": query_vector},
            {"query_vector": query_vector},
        ):
            try:
                kwargs = dict(
                    collection_name=COLLECTION_NAME,
                    limit=limit,
                    **kwargs_variant,
                )
                if qfilter:
                    kwargs["query_filter"] = qfilter
                response = self.client.query_points(**kwargs)
                return response.points
            except Exception:
                continue

        try:
            return self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit,
                query_filter=qfilter,
            )
        except Exception:
            if qfilter:
                return self._vector_search(query_vector, limit, source_filter=None)
            return []

    def _score_candidates(
        self,
        search_results: list,
        tokenized_query: List[str],
        target_standards: List[str],
    ) -> List[Tuple[float, str, dict]]:
        if not search_results:
            return []

        bm25_scores = self.bm25.get_scores(tokenized_query) if tokenized_query else []
        max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1

        v_scores = [p.score for p in search_results]
        v_max, v_min = max(v_scores), min(v_scores)
        v_range = v_max - v_min if (v_max - v_min) > 0 else 1e-5

        fused = []
        for point in search_results:
            payload = point.payload or {}
            page_content = payload.get("page_content") or payload.get("text") or payload.get("content") or ""
            if not page_content.strip():
                continue

            metadata = payload.get("metadata", payload)
            standard_id = metadata.get("standard_id") or parse_standard_id(
                metadata.get("source_doc", "")
            )
            chunk_id = metadata.get("chunk_id", point.id)

            b_score = 0
            try:
                if self.corpus_ids and chunk_id in self.corpus_ids:
                    b_score = bm25_scores[self.corpus_ids.index(chunk_id)]
                elif isinstance(chunk_id, int) and chunk_id < len(bm25_scores):
                    b_score = bm25_scores[chunk_id]
            except Exception:
                b_score = 0

            norm_v = (point.score - v_min) / v_range
            norm_b = b_score / max_bm25
            final_score = (0.55 * norm_v) + (0.45 * norm_b)

            heading = str(metadata.get("heading", "")).lower()
            content_lower = page_content.lower()
            for token in tokenized_query:
                if len(token) > 3 and (token in heading or token in content_lower):
                    final_score += 0.08
                    break

            if target_standards:
                if standard_id in target_standards:
                    final_score += 0.22
                source_doc = metadata.get("source_doc", "")
                if any(source_doc_matches_standard(source_doc, sid) for sid in target_standards):
                    final_score += 0.18

            fused.append((final_score, page_content, metadata))

        fused.sort(key=lambda x: x[0], reverse=True)
        return fused

    def _balanced_select(
        self, fused: List[Tuple[float, str, dict]], k: int, target_standards: List[str]
    ) -> List[Tuple[float, str, dict]]:
        if not target_standards:
            return fused[:k]

        buckets: Dict[str, List] = {sid: [] for sid in target_standards}
        other: List = []

        for item in fused:
            meta = item[2]
            sid = meta.get("standard_id") or parse_standard_id(meta.get("source_doc", ""))
            placed = False
            for ts in target_standards:
                if sid == ts or source_doc_matches_standard(meta.get("source_doc", ""), ts):
                    buckets[ts].append(item)
                    placed = True
                    break
            if not placed:
                other.append(item)

        selected: List[Tuple[float, str, dict]] = []
        seen_keys = set()

        def add_item(item):
            key = (item[2].get("source_doc"), item[2].get("clause_ref"), item[1][:80])
            if key not in seen_keys:
                seen_keys.add(key)
                selected.append(item)

        # Single-standard queries: return only that code's clauses (no IS 456 dilution)
        if len(target_standards) == 1:
            ts = target_standards[0]
            target_only = buckets[ts]
            if len(target_only) >= 3:
                for item in target_only:
                    if len(selected) >= k:
                        break
                    add_item(item)
                return selected[:k]

        min_target_slots = max(k - 2, int(k * 0.8))
        for ts in target_standards:
            for item in buckets[ts]:
                if len(selected) >= min_target_slots:
                    break
                add_item(item)

        for item in other:
            if len(selected) >= k:
                break
            add_item(item)

        for ts in target_standards:
            for item in buckets[ts]:
                if len(selected) >= k:
                    break
                add_item(item)

        return selected[:k]

    def retrieve_evidence(self, query: str, k: int = 8) -> dict:
        try:
            cleaned_query = self._normalize_query(query)
            tokenized_query = [t for t in cleaned_query.split() if len(t) > 1]
            target_standards = detect_standards_from_query(query)

            pool_limit = max(k * 8, 60) if target_standards else max(k * 5, 40)
            query_vector = self.embeddings.embed_query(query)

            all_points = {}
            primary = self._vector_search(query_vector, pool_limit)
            for p in primary:
                all_points[p.id] = p

            # Targeted passes per standard (prevents IS 456/800 corpus from drowning IS 13920)
            for sid in target_standards:
                marker = get_file_marker(sid)
                boost_text = get_retrieval_boost(sid)
                aug_queries = [
                    self._standard_augmented_query(query, sid),
                    f"{query} {boost_text}",
                ]
                for aug_q in aug_queries:
                    aug_vector = self.embeddings.embed_query(aug_q)
                    filtered = self._vector_search(aug_vector, max(k * 5, 35), source_filter=marker)
                    for p in filtered:
                        all_points[p.id] = p

            if not all_points:
                logger.warning("Vector database returned 0 results.")
                return {"retrieved_clauses": [], "citations": [], "sources": []}

            fused = self._score_candidates(list(all_points.values()), tokenized_query, target_standards)
            top_matches = self._balanced_select(fused, k, target_standards)

            retrieved_clauses = []
            citations = []
            sources = set()

            for score, text, meta in top_matches:
                clause_ref = meta.get("clause_ref") or meta.get("clause") or "General"
                heading = meta.get("heading") or meta.get("title") or "Section Reference"
                source_doc = meta.get("source_doc") or meta.get("source") or "Standard Document"
                standard_id = meta.get("standard_id") or parse_standard_id(source_doc)

                citation_string = f"[{standard_id}] Clause {clause_ref} ({heading})"
                citations.append(citation_string)
                sources.add(source_doc)

                retrieved_clauses.append({
                    "clause_id": str(clause_ref),
                    "title": str(heading),
                    "text": str(text),
                    "confidence_score": round(float(score), 3),
                    "origin_file": str(source_doc),
                    "standard_id": str(standard_id),
                })

            return {
                "retrieved_clauses": retrieved_clauses,
                "citations": list(dict.fromkeys(citations)),
                "sources": list(sources),
            }

        except Exception as e:
            logger.error(f"Critical failure in retrieval engine: {str(e)}")
            return {"retrieved_clauses": [], "citations": [], "sources": []}
