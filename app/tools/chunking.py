"""
Shared PDF chunking utilities for civil standards ingestion.
Handles varied IS document layouts (IS 456, IS 800, IS 13920, IS 875, etc.).
"""
import re
from typing import List, Optional, Tuple
from langchain_core.documents import Document

# Registry: filename fragment -> standard id + query detection aliases
STANDARD_REGISTRY = [
    {
        "id": "IS456",
        "file_markers": ["is456"],
        "source_match": ["is456"],
        "aliases": ["is 456", "is:456", "plain and reinforced concrete", "concrete code"],
        "retrieval_boost": "IS 456 reinforced concrete cover slab beam column shear",
    },
    {
        "id": "IS800",
        "file_markers": ["is800"],
        "source_match": ["is800"],
        "aliases": ["is 800", "steel structure", "structural steel", "hot rolled steel"],
        "retrieval_boost": "IS 800 structural steel slenderness tension compression member",
    },
    {
        "id": "IS13920",
        "file_markers": ["is13920", "13920"],
        "source_match": ["is13920", "13920"],
        "aliases": [
            "is 13920", "is13920", "13920",
            "ductile detailing", "ductile detail", "seismic detailing",
            "special moment resisting", "smrf", "confining reinforcement",
            "hinge region", "strong column weak beam", "development length seismic",
        ],
        "retrieval_boost": (
            "IS 13920 ductile detailing reinforced concrete seismic zone "
            "column beam joint confining stirrups hinge lap splice anchorage"
        ),
    },
    {
        "id": "IS1343",
        "file_markers": ["is1343", "1343"],
        "source_match": ["is1343", "1343"],
        "aliases": ["is 1343", "prestress", "prestressed concrete"],
        "retrieval_boost": "IS 1343 prestressed concrete tendon anchorage",
    },
    {
        "id": "IS875",
        "file_markers": ["is875"],
        "source_match": ["is875", "875"],
        "aliases": ["is 875", "wind load", "imposed load", "dead load", "live load"],
        "retrieval_boost": "IS 875 wind load imposed load dead load live load",
    },
]

CLAUSE_PATTERNS = [
    # 26.5.2 Minimum reinforcement (IS 456 style)
    re.compile(
        r"(?:^|\n)\s*(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z0-9\s,\-\(\)\/\.]{2,120})(?=\n|\s{2,}|$)"
    ),
    # Section 7 / SECTION 7
    re.compile(
        r"(?:^|\n)\s*(?:SECTION|Section|Sec\.?)\s*(\d+(?:\.\d+)*)\s*[-–:]?\s*([A-Z][A-Za-z0-9\s,\-\(\)\/\.]{2,120})",
        re.IGNORECASE,
    ),
    # Annex A / ANNEX A
    re.compile(
        r"(?:^|\n)\s*(?:ANNEX|Annex)\s+([A-Z]|\d+)\s*[-–:]?\s*([A-Za-z][A-Za-z0-9\s,\-\(\)\/\.]{2,120})",
        re.IGNORECASE,
    ),
    # 7.3.1 - Title with dash
    re.compile(
        r"(?:^|\n)\s*(\d+(?:\.\d+)+)\s*[-–]\s*([A-Z][A-Za-z0-9\s,\-\(\)\/\.]{2,120})(?=\n|$)"
    ),
]

LICENSE_WATERMARK = re.compile(
    r"SUPPLIED BY BOOK SUPPLY BUREAU.*?(?:\n|$)",
    re.IGNORECASE,
)


def parse_standard_id(filename: str) -> str:
    lower = filename.lower()
    for entry in STANDARD_REGISTRY:
        if any(m in lower for m in entry["file_markers"]):
            return entry["id"]
    return "IS_GENERAL"


TOPIC_STANDARD_HINTS = [
    ("IS13920", [
        "ductile", "hinge", "confining", "seismic", "smrf", "special moment",
        "beam column joint", "lap splice", "sbc", "zone iv", "zone v",
        "development length", "stirrups spacing", "13920",
    ]),
    ("IS875", ["wind speed", "gust", "terrain category", "imposed load", "live load"]),
    ("IS800", ["slenderness", "tension member", "compression member", "girder", "purlin"]),
    ("IS456", ["cover", "slab", "beam", "column", "shear", "development length", "lap length"]),
    ("IS1343", ["prestress", "prestressed", "tendon", "anchorage"]),
]


def detect_standards_from_query(query: str) -> List[str]:
    q = query.lower().replace("-", " ")
    found = []
    for entry in STANDARD_REGISTRY:
        if any(alias in q for alias in entry["aliases"]):
            found.append(entry["id"])
            continue
        num = entry["id"].lower().replace("is", "")
        if num.isdigit() and (f"is {num}" in q or f"is{num}" in q.replace(" ", "")):
            found.append(entry["id"])

    for sid, keywords in TOPIC_STANDARD_HINTS:
        if sid not in found and sum(1 for kw in keywords if kw in q) >= 2:
            found.append(sid)
        elif sid not in found and any(kw in q for kw in keywords if len(kw) > 8):
            found.append(sid)

    return list(dict.fromkeys(found))


def get_file_marker(standard_id: str) -> str:
    for entry in STANDARD_REGISTRY:
        if entry["id"] == standard_id:
            return entry["file_markers"][0]
    return standard_id.lower()


def get_retrieval_boost(standard_id: str) -> str:
    for entry in STANDARD_REGISTRY:
        if entry["id"] == standard_id:
            return entry.get("retrieval_boost", standard_id)
    return standard_id


def source_doc_matches_standard(source_doc: str, standard_id: str) -> bool:
    lower = (source_doc or "").lower()
    for entry in STANDARD_REGISTRY:
        if entry["id"] == standard_id:
            return any(m in lower for m in entry["source_match"])
    return False


def build_chunk_text(standard_id: str, clause_ref: str, heading: str, body: str) -> str:
    """Prefix chunk body so embeddings and BM25 align with standard + clause."""
    header = f"[{standard_id}] Clause {clause_ref} — {heading.strip()}\n\n"
    return header + body.strip()


def clean_pdf_text(text: str) -> str:
    text = LICENSE_WATERMARK.sub("", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _collect_clause_matches(full_text: str) -> List[Tuple[int, int, str, str]]:
    """Merge clause matches from all patterns, sorted by position."""
    matches = []
    seen_starts = set()
    for pattern in CLAUSE_PATTERNS:
        for m in pattern.finditer(full_text):
            start = m.start()
            if start in seen_starts:
                continue
            seen_starts.add(start)
            ref = m.group(1).strip()
            heading = m.group(2).strip()
            if len(heading) < 3:
                continue
            matches.append((start, m.end(), ref, heading))
    matches.sort(key=lambda x: x[0])
    return matches


def segment_document_by_clauses(full_text: str, filename: str, standard_id: str) -> List[Document]:
    matches = _collect_clause_matches(full_text)
    structured_docs = []

    if not matches:
        return []

    for idx, (start_pos, _, clause_ref, heading) in enumerate(matches):
        end_pos = matches[idx + 1][0] if idx + 1 < len(matches) else len(full_text)
        clause_body = full_text[start_pos:end_pos].strip()
        if len(clause_body) > 30:
            structured_docs.append(
                Document(
                    page_content=build_chunk_text(standard_id, clause_ref, heading, clause_body),
                    metadata={
                        "clause_ref": clause_ref,
                        "heading": heading,
                        "source_doc": filename,
                        "standard_id": standard_id,
                    },
                )
            )
    return structured_docs


def segment_by_pages(pages: List, filename: str, standard_id: str) -> List[Document]:
    """Page-level fallback when global clause regex under-segments a PDF."""
    docs = []
    for page_num, page in enumerate(pages, start=1):
        text = clean_pdf_text(page.page_content or "")
        if len(text.strip()) < 40:
            continue
        page_clauses = segment_document_by_clauses(text, filename, standard_id)
        if len(page_clauses) >= 2:
            docs.extend(page_clauses)
        else:
            docs.append(
                Document(
                    page_content=build_chunk_text(
                        standard_id, f"{page_num}", f"Page {page_num}", text
                    ),
                    metadata={
                        "clause_ref": str(page_num),
                        "heading": f"Page {page_num}",
                        "source_doc": filename,
                        "standard_id": standard_id,
                    },
                )
            )
    return docs


def should_use_page_fallback(clause_docs: List[Document], page_count: int) -> bool:
    if page_count <= 0:
        return False
    if len(clause_docs) < max(8, int(page_count * 0.25)):
        return True
    giant = sum(1 for d in clause_docs if len(d.page_content) > 12000)
    if giant > len(clause_docs) * 0.15:
        return True
    return False


def process_pdf_to_chunks(pages: List, filename: str, token_len_fn, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Full pipeline for one PDF: clause segmentation with page-hybrid fallback + sub-chunking.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    standard_id = parse_standard_id(filename)
    raw_full_text = "\n".join(p.page_content or "" for p in pages)
    cleaned_text = clean_pdf_text(raw_full_text)

    clause_documents = segment_document_by_clauses(cleaned_text, filename, standard_id)
    if should_use_page_fallback(clause_documents, len(pages)):
        clause_documents = segment_by_pages(pages, filename, standard_id)

    if not clause_documents:
        clause_documents = [
            Document(
                page_content=build_chunk_text(standard_id, "General", "Full Document", cleaned_text[:50000]),
                metadata={
                    "clause_ref": "General",
                    "heading": "Full Document",
                    "source_doc": filename,
                    "standard_id": standard_id,
                },
            )
        ]

    sub_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_len_fn,
    )

    processed = []
    for doc in clause_documents:
        tokens = token_len_fn(doc.page_content)
        if tokens <= chunk_size:
            processed.append(doc)
        else:
            sub_docs = sub_splitter.split_documents([doc])
            for sub_idx, sub_doc in enumerate(sub_docs):
                ref = doc.metadata["clause_ref"]
                heading = doc.metadata["heading"]
                body = sub_doc.page_content
                if body.startswith("["):
                    parts = body.split("\n\n", 1)
                    body = parts[1] if len(parts) > 1 else body
                sub_doc.page_content = build_chunk_text(
                    standard_id, ref, f"{heading} (Part {sub_idx + 1})", body
                )
                sub_doc.metadata = {**doc.metadata, "heading": f"{heading} (Part {sub_idx + 1})"}
                processed.append(sub_doc)
    return processed
