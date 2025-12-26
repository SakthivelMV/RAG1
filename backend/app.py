
# FastAPI: retrieval + SSE chat to Grok

import os
import json
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

import faiss
import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from ingest import (
    DATA_PDF, ensure_index_uptodate, load_index, build_index_from_pdf
)


#Resolve the path to backend/.env
BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:5173")

print("[env] .env path:", ENV_PATH)
print("[env] XAI_API_KEY present:", bool(XAI_API_KEY))



app = FastAPI(title="RAG FAQ Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global index cache
_index: faiss.Index = None
_meta: List[Dict[str, Any]] = []
_corpus: List[str] = []

def _reload_index_if_needed():
    """Ensure FAISS index is up to date and loaded."""
    changed = ensure_index_uptodate()
    global _index, _meta, _corpus
    if changed or _index is None:
        _index, _meta, _corpus = load_index()

class SearchReq(BaseModel):
    query: str
    top_k: int = 5

@app.on_event("startup")
def startup():
    # Don’t crash the app if the PDF is missing; show it via /health
    try:
        _reload_index_if_needed()
    except FileNotFoundError as e:
        print(f"[startup] {e}")


# ---------- Convenience routes ----------

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
      <head><title>RAG FAQ Chatbot (Backend)</title></head>
      <body style="font-family: system-ui; padding: 16px;">
        <h2>Backend is running </h2>
        <ul>
          <li><a href="/docs">/docs</a> – API docs</li>
          <li><a href="/health">/health</a> – Health check (PDF path & md5)</li>
        </ul>
      </body>
    </html>
    """

@app.get("/health")
def health():
    pdf_path = Path(DATA_PDF)
    exists = pdf_path.exists()
    md5 = None
    if exists:
        from ingest import file_md5
        md5 = file_md5(pdf_path)
    return {
        "status": "ok" if exists else "missing_pdf",
        "pdf_path": str(pdf_path),
        "exists": exists,
        "md5": md5,
    }

# ---------- Index management ----------

@app.post("/index/rebuild")
def index_rebuild(pdf_path: str | None = None):
    """Force rebuild. Optional pdf_path lets you index another file."""
    path = Path(pdf_path) if pdf_path else Path(DATA_PDF)
    if not path.exists():
        return JSONResponse({"error": f"PDF not found: {path}"}, status_code=400)
    pages, chunks = build_index_from_pdf(path)
    _reload_index_if_needed()
    return {"ok": True, "pages": pages, "chunks": chunks, "pdf_path": str(path)}

# ---------- Retrieval ----------

@app.post("/search")
def search(req: SearchReq):
    _reload_index_if_needed()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([req.query], normalize_embeddings=True)
    D, I = _index.search(np.array(q_emb, dtype=np.float32), req.top_k)
    results = []
    for i, score in zip(I[0], D[0]):
        if i == -1:
            continue
        m = _meta[i]
        results.append({
            "text": _corpus[i],
            "score": float(score),
            "path": m["source"],
            "page": m["page"],
            "chunk_id": m["chunk_id"],
        })
    return {"results": results}

def _prompt_from_results(query: str, results: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, r in enumerate(results):
        blocks.append(
            f"Source {i+1} ({r['path']}#p{r['page']}:{r['chunk_id']}):\n{r['text']}"
        )
    return (
        f"Question: {query}\n\n"
        f"Relevant sources:\n{'\n\n'.join(blocks)}\n\n"
        "Instructions:\n"
        "- Answer only using the sources above.\n"
        "- If insufficient, say you don't know.\n"
        "- Include citations like [S1], [S2] that map to Source numbers.\n\n"
        "Answer:"
    )

# ---------- Chat (SSE to xAI Grok) ----------

@app.get("/chat/stream")
async def chat_stream(q: str = Query(..., min_length=2), top_k: int = 5):
    if not XAI_API_KEY:
        return JSONResponse({"error": "XAI_API_KEY missing"}, status_code=500)

    # --- retrieval (with error wrapping) ---
    try:
        _reload_index_if_needed()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q_emb = model.encode([q], normalize_embeddings=True)
        D, I = _index.search(np.array(q_emb, dtype=np.float32), top_k)
        results = []
        for i, score in zip(I[0], D[0]):
            if i == -1:
                continue
            m = _meta[i]
            results.append({
                "text": _corpus[i],
                "score": float(score),
                "path": m["source"],
                "page": m["page"],
                "chunk_id": m["chunk_id"],
            })
    except Exception as e:
        async def err_gen():
            yield f"data: {json.dumps({'error': f'retrieval_failed: {str(e)}'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    prompt = _prompt_from_results(q, results)

    async def event_gen():
        # send sources first (named event)
        yield "event: sources\n"
        yield f"data: {json.dumps({'sources': [{'id': f'S{i+1}','path': r['path'],'chunk_id': r['chunk_id'],'preview': (r['text'][:280]+'…') if len(r['text'])>280 else r['text'],'score': r['score']} for i, r in enumerate(results)]})}\n\n"

        # model call
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        body = {"model": "grok-4-0709",
                "messages": [
                    {"role": "system", "content": "You are an internal FAQ assistant. Answer only from provided sources. Cite as [S1], [S2]."},
                    {"role": "user", "content": prompt}
                ]}

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post("https://api.x.ai/v1/chat/completions", headers=headers, json=body)
                if resp.status_code != 200:
                    err = f"xAI error {resp.status_code}: {resp.text}"
                    print(err)
                    yield f"data: {json.dumps({'error': err})}\n\n"
                    return
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    yield f"data: {json.dumps({'error': 'empty_content_from_model'})}\n\n"
                    return
                yield f"data: {json.dumps({'token': content})}\n\n"
        except httpx.ReadTimeout:
            yield f"data: {json.dumps({'error': 'model_timeout'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'model_call_failed: {str(e)}'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# 2)@app.get("/chat/stream")
# async def chat_stream(q: str = Query(..., min_length=2), top_k: int = 5):
#     if not XAI_API_KEY:
#         return JSONResponse({"error": "XAI_API_KEY missing"}, status_code=500)

#     # 1) Retrieve top-k chunks
#     _reload_index_if_needed()
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     q_emb = model.encode([q], normalize_embeddings=True)
#     D, I = _index.search(np.array(q_emb, dtype=np.float32), top_k)
#     results = []
#     for i, score in zip(I[0], D[0]):
#         if i == -1:
#             continue
#         m = _meta[i]
#         results.append({
#             "text": _corpus[i],
#             "score": float(score),
#             "path": m["source"],
#             "page": m["page"],
#             "chunk_id": m["chunk_id"],
#         })

#     prompt = _prompt_from_results(q, results)

#     async def event_gen():
#         # SSE requires text/event-stream framing
#         # Send sources first (named event) so UI renders chips immediately
#         sources_payload = {
#             "sources": [
#                 {
#                     "id": f"S{i+1}",
#                     "path": r["path"],
#                     "chunk_id": r["chunk_id"],
#                     "preview": (r["text"][:280] + "…") if len(r["text"]) > 280 else r["text"],
#                     "score": r["score"],
#                 } for i, r in enumerate(results)
#             ]
#         }
#         yield "event: sources\n"
#         yield f"data: {json.dumps(sources_payload)}\n\n"

#         # 2) xAI Grok Chat Completions (OpenAI-style)
#         headers = {
#             "Authorization": f"Bearer {XAI_API_KEY}",
#             "Content-Type": "application/json"
#         }
#         body = {
#             "model": "grok-4-0709",  # update to your current Grok model if needed
#             "messages": [
#                 {"role": "system",
#                  "content": "You are an internal FAQ assistant. Answer only from provided sources. Cite as [S1], [S2]."},
#                 {"role": "user", "content": prompt}
#             ]
#         }

#         async with httpx.AsyncClient(timeout=60) as client:
#             resp = await client.post("https://api.x.ai/v1/chat/completions",
#                                      headers=headers, json=body)
#             if resp.status_code != 200:
#                 yield f"data: {json.dumps({'error': resp.text})}\n\n"
#                 return

#             data = resp.json()
#             content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

#             # One-shot SSE message (xAI Grok non-streaming in this template)
#             yield f"data: {json.dumps({'token': content})}\n\n"

#     return StreamingResponse(event_gen(), media_type="text/event-stream")


#endpoint
from fastapi import Body
class ChatReq(BaseModel):
    query: str
    selected_sources: List[str] = []  # e.g., ["S1", "S3"]

@app.post("/chat/answer")
async def chat_answer(req: ChatReq):
    """Answer using ONLY the selected sources (S1, S2, ...). Returns JSON."""
    if not XAI_API_KEY:
        return JSONResponse({"error": "XAI_API_KEY missing"}, status_code=500)

    # 1) Retrieve top-k (we’ll use the same top_k=5 for now)
    try:
        _reload_index_if_needed()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q_emb = model.encode([req.query], normalize_embeddings=True)
        D, I = _index.search(np.array(q_emb, dtype=np.float32), 5)
        results = []
        for i, score in zip(I[0], D[0]):
            if i == -1:
                continue
            m = _meta[i]
            results.append({
                "text": _corpus[i],
                "score": float(score),
                "path": m["source"],
                "page": m["page"],
                "chunk_id": m["chunk_id"],
            })
    except Exception as e:
        return JSONResponse({"error": f"retrieval_failed: {str(e)}"}, status_code=500)

    # 2) Map top-k results to S-IDs and filter by selection
    #    Top-k order determines S1, S2, ... consistently with the SSE chips.
    s_mapping = {f"S{idx+1}": r for idx, r in enumerate(results)}
    filtered = []
    if req.selected_sources:
        for sid in req.selected_sources:
            if sid in s_mapping:
                filtered.append(s_mapping[sid])
    else:
        filtered = results  # fallback: use all top-k

    if not filtered:
        return JSONResponse({"error": "no_selected_sources"}, status_code=400)

    # 3) Build prompt ONLY from filtered sources
    blocks = []
    for i, r in enumerate(filtered):
        blocks.append(
            f"Source sel-{i+1} ({r['path']}#p{r['page']}:{r['chunk_id']}):\n{r['text']}"
        )
    prompt = (
        f"Question: {req.query}\n\n"
        f"Selected sources:\n{'\n\n'.join(blocks)}\n\n"
        "Instructions:\n"
        "- Answer only using the selected sources above.\n"
        "- If insufficient, say you don't know.\n"
        "- Include citations like [S1], [S2] to refer to the selected sources in the order they appear.\n\n"
        "Answer:"
    )

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "grok-4-0709",
        "messages": [
            {"role": "system",
             "content": "You are an internal FAQ assistant. Answer only from the provided sources. Cite as [S1], [S2]."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post("https://api.x.ai/v1/chat/completions",
                                     headers=headers, json=body)
            if resp.status_code != 200:
                return JSONResponse({"error": f"xAI {resp.status_code}: {resp.text}"}, status_code=500)
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return JSONResponse({"error": "empty_content_from_model"}, status_code=500)
            return {"answer": content}
    except httpx.ReadTimeout:
        return JSONResponse({"error": "model_timeout"}, status_code=504)
    except Exception as e:
        return JSONResponse({"error": f"model_call_failed: {str(e)}"}, status_code=500)

