# api/index.py
import os
import json
import uuid
import threading
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
import httpx

# Try Redis; fall back to simple in-memory store for dev if REDIS_URL is not set.
REDIS_URL = os.getenv("REDIS_URL", "").strip() or None
USE_REDIS = bool(REDIS_URL)

if USE_REDIS:
    try:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        # If Redis import/connect fails, fall back to memory mode (but log)
        print("[WARN] Redis import/connect failed, falling back to in-memory store:", e)
        USE_REDIS = False
        redis_client = None

# In-memory store (thread-safe)
_mem_lock = threading.Lock()
_mem_store: Dict[str, Dict[str, Any]] = {}  # job_key -> dict with meta, raw, processed

JOB_PREFIX = os.getenv("JOB_PREFIX", "lithybrid:job:")

app = FastAPI(title="LitHybrid - Vercel-safe API")

# ---------------------------
# Models
# ---------------------------
class ProjectRequest(BaseModel):
    title: str
    max_papers: Optional[int] = 7000
    per_run: Optional[int] = 100
    expected_time_minutes: Optional[int] = 30

# ---------------------------
# Helpers: store abstraction
# ---------------------------
def _key(job_id: str, suffix: str = "") -> str:
    return f"{JOB_PREFIX}{job_id}{suffix}"

def store_hset(job_id: str, mapping: Dict[str, Any]):
    if USE_REDIS and redis_client:
        redis_client.hset(_key(job_id), mapping=mapping)
    else:
        with _mem_lock:
            s = _mem_store.setdefault(job_id, {})
            meta = s.get("meta", {})
            meta.update(mapping)
            s["meta"] = meta

def store_set(key: str, value: Any):
    if USE_REDIS and redis_client:
        redis_client.set(key, json.dumps(value))
    else:
        # key passed as full redis-style key; keep simple mapping
        with _mem_lock:
            job_id = key.replace(JOB_PREFIX, "").split(":")[0]
            s = _mem_store.setdefault(job_id, {})
            # store raw or processed depending on suffix
            if key.endswith(":raw"):
                s["raw"] = value
            elif key.endswith(":processed"):
                s["processed"] = value
            else:
                # generic store
                s[key] = value

def store_get(key: str):
    if USE_REDIS and redis_client:
        val = redis_client.get(key)
        return json.loads(val) if val else None
    else:
        job_id = key.replace(JOB_PREFIX, "").split(":")[0]
        with _mem_lock:
            s = _mem_store.get(job_id, {})
            if key.endswith(":raw"):
                return s.get("raw")
            if key.endswith(":processed"):
                return s.get("processed")
            return s.get(key)

def store_hgetall(job_id: str) -> Dict[str, Any]:
    if USE_REDIS and redis_client:
        return redis_client.hgetall(_key(job_id))
    else:
        with _mem_lock:
            s = _mem_store.get(job_id, {})
            return dict(s.get("meta", {}))

def exists_key(key: str) -> bool:
    if USE_REDIS and redis_client:
        return redis_client.exists(key)
    else:
        job_id = key.replace(JOB_PREFIX, "").split(":")[0]
        with _mem_lock:
            return job_id in _mem_store

# ---------------------------
# Lightweight OpenAlex fetcher (one page)
# ---------------------------
OPENALEX_WORKS = "https://api.openalex.org/works"

def fetch_openalex_page(query: str, page: int = 1, per_page: int = 25) -> Dict[str, Any]:
    headers = {"User-Agent": "LitHybrid/0.1 (mailto:you@example.com)"}
    params = {"search": query, "per-page": per_page, "page": page}
    with httpx.Client(timeout=20, headers=headers) as client:
        r = client.get(OPENALEX_WORKS, params=params)
        r.raise_for_status()
        return r.json()

def normalize_record(p: Dict[str, Any]) -> Dict[str, Any]:
    ids = p.get("ids") or {}
    host = p.get("host_venue") or {}
    return {
        "id": p.get("id"),
        "title": p.get("title") or p.get("display_name") or "",
        "publication_year": p.get("publication_year") or p.get("year") or None,
        "authorships": p.get("authorships") or [],
        "doi": (ids.get("doi") if isinstance(ids, dict) else p.get("doi")) or "",
        "journal": (host.get("display_name") if isinstance(host, dict) else None) or p.get("venue") or "",
        "abstract": p.get("abstract") or "",
        "url": p.get("id") or p.get("url") or ""
    }

# ---------------------------
# API endpoints
# ---------------------------
@app.get("/")
def health():
    return {"ok": True, "service": "LitHybrid (vercel-safe)"}

@app.post("/api/projects")
def create_project(req: ProjectRequest):
    project_id = str(uuid.uuid4())
    meta = {
        "project_id": project_id,
        "title": req.title,
        "max_papers": str(req.max_papers),
        "per_run": str(req.per_run),
        "status": "queued",
        "progress": "0",
        "next_offset": "0",
        "expected_time_minutes": str(req.expected_time_minutes or 30)
    }
    # store meta
    store_hset(project_id, meta)
    # initialize raw and processed buckets
    store_set(_key(project_id, ":raw"), [])
    store_set(_key(project_id, ":processed"), [])
    return {"project_id": project_id, "status": "queued", "expected_time_minutes": meta["expected_time_minutes"]}

@app.get("/api/projects/{project_id}/status")
def project_status(project_id: str):
    if not exists_key(_key(project_id)):
        raise HTTPException(status_code=404, detail="project not found")
    data = store_hgetall(project_id)
    # convert numeric strings where possible
    for k in ("max_papers", "per_run", "next_offset"):
        if k in data:
            try:
                data[k] = int(data[k])
            except Exception:
                pass
    return data

@app.get("/api/projects/{project_id}/papers")
def project_papers(project_id: str, page: int = 1, per_page: int = 100):
    processed_raw = store_get(_key(project_id, ":processed")) or []
    try:
        papers = processed_raw if isinstance(processed_raw, list) else json.loads(processed_raw)
    except Exception:
        papers = processed_raw or []
    total = len(papers)
    start = (page - 1) * per_page
    end = start + per_page
    return {"project_id": project_id, "page": page, "per_page": per_page, "total": total, "papers": papers[start:end]}

@app.get("/api/projects/{project_id}/summary")
def project_summary(project_id: str):
    meta = store_hgetall(project_id)
    if not meta:
        raise HTTPException(status_code=404, detail="project not found")
    summary_raw = meta.get("summary")
    if not summary_raw:
        raise HTTPException(status_code=404, detail="summary not ready")
    try:
        return {"project_id": project_id, "summary": json.loads(summary_raw)}
    except Exception:
        return {"project_id": project_id, "summary": summary_raw}

# ---------------------------
# Short idempotent worker step (safe for serverless)
# Call repeatedly (cron or manual) until job status == 'done'
# ---------------------------
@app.post("/api/worker_step")
async def worker_step(req: Request):
    """
    Body optional: {"project_id":"..."} to process just one project.
    Without body, scans for queued/processing jobs.
    """
    payload = {}
    try:
        payload = await req.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    project_id = payload.get("project_id")

    candidates: List[str] = []
    if project_id:
        candidates = [project_id]
    else:
        # scan in-memory or Redis keys; keep small & safe
        if USE_REDIS and redis_client:
            for key in redis_client.scan_iter(match=JOB_PREFIX + "*"):
                pid = key.replace(JOB_PREFIX, "")
                if ":" in pid:
                    continue
                meta = redis_client.hgetall(_key(pid))
                if meta and meta.get("status", "") in ("queued", "discovering", "processing", "summarizing"):
                    candidates.append(pid)
        else:
            with _mem_lock:
                for pid, data in _mem_store.items():
                    meta = data.get("meta", {})
                    if meta and meta.get("status", "") in ("queued", "discovering", "processing", "summarizing"):
                        candidates.append(pid)

    processed = []
    for pid in candidates:
        meta = store_hgetall(pid)
        if not meta:
            continue
        status = meta.get("status", "queued")
        try:
            max_papers = int(meta.get("max_papers", 7000))
        except Exception:
            max_papers = 7000
        try:
            per_run = int(meta.get("per_run", 100))
        except Exception:
            per_run = 100
        offset = int(meta.get("next_offset", 0) or 0)

        raw_list = store_get(_key(pid, ":raw")) or []
        if isinstance(raw_list, str):
            try:
                raw_list = json.loads(raw_list)
            except Exception:
                raw_list = []

        # DISCOVERY: fetch one small page if raw_list insufficient
        if status in ("queued", "discovering") and len(raw_list) < max_papers:
            try:
                page_number = (len(raw_list) // 25) + 1
                resp = fetch_openalex_page(meta.get("title", ""), page=page_number, per_page=min(25, per_run))
                results = resp.get("results", []) or []
                for r in results:
                    raw_list.append(r)
                    if len(raw_list) >= max_papers:
                        break
                store_set(_key(pid, ":raw"), raw_list)
                # update status/progress
                progress = int((len(raw_list) / max(1, max_papers)) * 100)
                store_hset(pid, {"status": "discovering", "progress": str(min(progress, 50))})
            except Exception as e:
                store_hset(pid, {"status": "discovering", "progress": "0", "error": str(e)})
                continue

        # PROCESSING: normalize next chunk
        if len(raw_list) > offset:
            to_process = raw_list[offset: offset + per_run]
            normalized = []
            for item in to_process:
                try:
                    normalized.append(normalize_record(item))
                except Exception:
                    continue

            existing = store_get(_key(pid, ":processed")) or []
            if isinstance(existing, str):
                try:
                    existing = json.loads(existing)
                except Exception:
                    existing = []
            existing.extend(normalized)
            store_set(_key(pid, ":processed"), existing)

            new_offset = offset + len(to_process)
            progress = int((new_offset / max(1, max_papers)) * 100)
            store_hset(pid, {"next_offset": str(new_offset), "progress": str(min(progress, 99)), "status": "processing"})

            # finished condition
            if new_offset >= max_papers or new_offset >= len(raw_list):
                # compose short extractive summary
                snippets = []
                for rec in existing:
                    a = rec.get("abstract", "") or ""
                    if a:
                        s = a.split(".")
                        snippet = ".".join(s[:2]).strip()
                        if snippet:
                            snippets.append(snippet + ".")
                composed = {
                    "title": meta.get("title"),
                    "num_papers": len(existing),
                    "composed_summary": ("Automatic reduce summary for: " + meta.get("title", "") + "\n\n" +
                                        "\n".join(["- " + s.replace("\n", " ")[:400] for s in snippets[:20]]))
                }
                store_hset(pid, {"summary": json.dumps(composed), "status": "done", "progress": "100"})
                # also store processed into raw for convenience
                store_set(_key(pid, ":raw"), existing)

            processed.append(pid)

    return {"processed": processed, "count": len(processed)}
