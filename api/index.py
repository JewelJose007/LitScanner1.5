import os
import json
import uuid
import threading
import traceback
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import redis
from dotenv import load_dotenv

from summarizer import generate_summary
from topic_modeling import generate_topic_clusters

load_dotenv()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="LitHybrid - Vercel-safe API")

# ---------------------------
# Error Handling
# ---------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_details = traceback.format_exc()
    print("SERVER ERROR:", error_details)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Something went wrong, but the server is still running.",
            "details": str(exc)
        }
    )

@app.middleware("http")
async def catch_404_middleware(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Endpoint not found"}
        )
    return response

# Health check
@app.get("/")
def health():
    return {"message": "Server is up and running!"}

# ---------------------------
# Redis Cache Setup
# ---------------------------
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url)

# ---------------------------
# Pydantic Models
# ---------------------------
class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, str]] = None
    max_results: int = 10

class PaperSummaryRequest(BaseModel):
    title: str
    abstract: str

class TopicClusterRequest(BaseModel):
    papers: List[Dict[str, Any]]

# ---------------------------
# Helper Functions
# ---------------------------
async def fetch_openalex_page(query: str, filters: Optional[Dict[str, str]] = None, per_page: int = 200, cursor: str = "*") -> dict:
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": per_page,
        "cursor": cursor
    }
    if filters:
        params.update(filters)
    async with httpx.AsyncClient() as client:
        r = await client.get(base_url, params=params)
        r.raise_for_status()
        return r.json()

def normalize_record(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "abstract": record.get("abstract"),
        "authors": [auth.get("author", {}).get("display_name") for auth in record.get("authorships", [])],
        "year": record.get("publication_year"),
        "doi": record.get("doi"),
        "url": record.get("primary_location", {}).get("source", {}).get("url"),
    }

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/api/search")
async def search_papers(request: SearchRequest):
    cache_key = f"search:{request.query}:{json.dumps(request.filters)}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    results = []
    cursor = "*"
    while len(results) < request.max_results:
        data = await fetch_openalex_page(request.query, request.filters, per_page=200, cursor=cursor)
        page_results = [normalize_record(r) for r in data.get("results", [])]
        results.extend(page_results)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    results = results[:request.max_results]
    redis_client.setex(cache_key, 3600, json.dumps(results))
    return results

@app.post("/api/summarize")
async def summarize_paper(request: PaperSummaryRequest):
    summary = generate_summary(request.title, request.abstract)
    return {"summary": summary}

@app.post("/api/topics")
async def topic_clusters(request: TopicClusterRequest):
    clusters = generate_topic_clusters(request.papers)
    return {"clusters": clusters}

