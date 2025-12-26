#PDF to chunks to embeddings to FAISS
import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
DATA_PDF = BASE_DIR / "data" / "systemdesign.pdf"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "faiss.index"
META_PATH = STORAGE_DIR / "meta.json"
STATE_PATH = STORAGE_DIR / "ingest_state.json"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def file_md5(path: Path) -> str:
    m = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            m.update(chunk)
    return m.hexdigest()

def extract_pdf_text(pdf_path: Path) -> List[Dict]:
    """Return a list of pages with text; keeps page numbers for citations."""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

def chunk_text(text: str, approx_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Approximate 256–512 token chunks with ~10–20% overlap (RAG best practice)."""
    chunks = []
    n = len(text); i = 0
    while i < n:
        end = min(i + approx_chars, n)
        chunks.append(text[i:end])
        if end == n: break
        i = end - overlap
    return chunks

def build_index_from_pdf(pdf_path: Path) -> Tuple[int, int]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    pages = extract_pdf_text(pdf_path)

    corpus, meta = [], []
    for p in pages:
        for ci, ch in enumerate(chunk_text(p["text"])):
            corpus.append(ch)
            meta.append({"source": str(pdf_path), "page": p["page"], "chunk_id": ci})

    # encode + normalize (cosine via inner product)
    embs = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact search first
    index.add(embs.astype(np.float32))

    faiss.write_index(index, str(INDEX_PATH))            # persistent FAISS
    with open(META_PATH, "w", encoding="utf-8") as f:    # sidecar metadata
        json.dump({"meta": meta, "corpus": corpus}, f, ensure_ascii=False)

    state = {
        "pdf_path": str(pdf_path),
        "md5": file_md5(pdf_path),
        "count_chunks": len(corpus)
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

    return len(pages), len(corpus)

def ensure_index_uptodate() -> bool:
    """Rebuild index if PDF changed."""
    if not DATA_PDF.exists():
        raise FileNotFoundError(f"PDF not found: {DATA_PDF}")
    current_md5 = file_md5(DATA_PDF)
    if not STATE_PATH.exists() or not INDEX_PATH.exists() or not META_PATH.exists():
        build_index_from_pdf(DATA_PDF)
        return True
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    if state.get("md5") != current_md5:
        build_index_from_pdf(DATA_PDF)
        return True
    return False

def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)
    return index, store["meta"], store["corpus"]
