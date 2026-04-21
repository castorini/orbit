#!/usr/bin/env python3
"""
ddgs_web_search.py — Production DDGS retrieval server for orbit

Provides a FastAPI /retrieve endpoint compatible with the Search-R1 /
verl-tool retrieval schema. Designed for high-concurrency GRPO training where
many trajectories issue simultaneous search queries.

Key design choices:
  - All backends called in parallel (fan-out) — total latency = max(backends),
    not sum. This eliminates the 54–80s mean tool latency from sequential fallback.
  - Per-thread DDGS clients (not thread-safe to share).
  - Async LRU cache with configurable TTL; short negative-cache TTL avoids
    long-lived poisoning of transient "empty result" states.
  - In-flight deduplication prevents stampeding on identical concurrent queries.

Usage:
    python ddgs_web_search.py --port 8280 --topk 5 \
        --backend "google,brave,bing,wikipedia,grokipedia"
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import random
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from ddgs import DDGS
from fastapi import FastAPI
from pydantic import BaseModel

# ── Tunables ──────────────────────────────────────────────────────────────────
DEFAULT_TOPK = 5
DEFAULT_PORT = 8280
DEFAULT_BACKENDS: List[str] = ["google", "brave", "bing", "wikipedia", "grokipedia"]

THREADPOOL_MAX_WORKERS = 128
CACHE_MAX_SIZE = 1_000
CACHE_TTL_SECONDS = 3_600          # 1 hour
NEGATIVE_CACHE_TTL_SECONDS = 30    # short TTL for empty / error results
SEM_MAX_CONCURRENT = 128           # parallel fan-out headroom
DDGS_TIMEOUT = 10                  # seconds per backend; keeps total latency bounded


# ── Async LRU cache ───────────────────────────────────────────────────────────

class AsyncLRUCache:
    """Thread-safe async LRU cache with per-entry TTL."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        async with self._lock:
            self._cache.pop(key, None)
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
            self._cache[key] = (time.time() + max(1, int(ttl)), value)


# ── Request schema ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


# ── Core retriever ────────────────────────────────────────────────────────────

class DDGSRetrievalService:
    """
    Manages per-backend DDGS clients, threadpool, caching, and concurrency
    control for high-throughput GRPO training workloads.
    """

    def __init__(
        self,
        backends: List[str],
        topk: int = DEFAULT_TOPK,
        threadpool_workers: int = THREADPOOL_MAX_WORKERS,
        cache_size: int = CACHE_MAX_SIZE,
        cache_ttl: int = CACHE_TTL_SECONDS,
        negative_cache_ttl: int = NEGATIVE_CACHE_TTL_SECONDS,
        sem_limit: int = SEM_MAX_CONCURRENT,
        ddgs_timeout: int = DDGS_TIMEOUT,
    ) -> None:
        self.backends = backends or DEFAULT_BACKENDS
        self.backend_str = ",".join(self.backends)
        self.topk = topk
        self.timeout = ddgs_timeout
        self.negative_cache_ttl = negative_cache_ttl

        # DDGS is not thread-safe; use a per-thread client
        self._local = threading.local()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=threadpool_workers)
        self._semaphore = asyncio.Semaphore(sem_limit)
        self._cache = AsyncLRUCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self._inflight_lock = asyncio.Lock()
        self._inflight: Dict[str, asyncio.Future] = {}

    def _client(self) -> DDGS:
        if not hasattr(self._local, "ddg"):
            self._local.ddg = DDGS(timeout=self.timeout)
        return self._local.ddg

    @staticmethod
    def _cache_key(query: str, topk: int, backend_str: str) -> str:
        return f"{backend_str}|{topk}|{query.strip()}"

    def _query_sync(self, query: str, topk: int, backend: str) -> List[Dict[str, str]]:
        """Blocking DDGS call. Returns docs with strict schema: {"contents": str}."""
        docs: List[Dict[str, str]] = []
        try:
            for item in self._client().text(query, max_results=topk, backend=backend):
                title = item.get("title", "") or ""
                snippet = item.get("body") or item.get("snippet") or ""
                docs.append({"contents": f'"{title}"\n{snippet}'.strip()})
        except Exception as exc:
            if "no results" not in str(exc).lower():
                print(f"[DDGS backend={backend}] {exc}")
        return docs

    async def _query_backend(self, query: str, topk: int, backend: str) -> List[Dict[str, str]]:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._query_sync, query, topk, backend)

    async def _fan_out(self, query: str, topk: int) -> List[Dict[str, str]]:
        """
        Call all backends in parallel and merge deduplicated results.
        Total latency = max(backend latencies) instead of their sum.
        """
        results = await asyncio.gather(
            *[self._query_backend(query, topk, b) for b in self.backends],
            return_exceptions=True,
        )
        seen: set = set()
        merged: List[Dict[str, str]] = []
        for result in results:
            if isinstance(result, Exception) or not result:
                continue
            for doc in result:
                key = doc.get("contents", "")[:80]
                if key and key not in seen:
                    seen.add(key)
                    merged.append(doc)
                    if len(merged) >= topk:
                        return merged
        return merged[:topk]

    async def _search(self, query: str, topk: int) -> List[Dict[str, str]]:
        """
        Cached, in-flight-deduplicated parallel fan-out search.
        Falls back to one jittered retry when all backends return empty.
        """
        key = self._cache_key(query, topk, self.backend_str)

        cached = await self._cache.get(key)
        if cached is not None:
            return cached

        async with self._inflight_lock:
            if key in self._inflight:
                fut = self._inflight[key]
                owner = False
            else:
                fut = asyncio.get_running_loop().create_future()
                self._inflight[key] = fut
                owner = True

        if not owner:
            return await fut

        try:
            docs = await self._fan_out(query, topk)
            if not docs:
                await asyncio.sleep(0.5 + random.random())
                docs = await self._fan_out(query, topk)

            ttl = None if docs else self.negative_cache_ttl
            await self._cache.set(key, docs, ttl_seconds=ttl)
            fut.set_result(docs)
            return docs
        except Exception:
            if not fut.done():
                fut.set_result([])
            return []
        finally:
            async with self._inflight_lock:
                self._inflight.pop(key, None)

    async def batch_search(
        self, queries: List[str], topk: int, return_scores: bool
    ) -> Tuple[List[List[Dict[str, str]]], List[List[float]]]:
        topk = topk or self.topk
        docs_list = await asyncio.gather(*[self._search(q, topk) for q in queries])
        scores_list = [[0.0] * len(d) for d in docs_list]
        return list(docs_list), scores_list

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)


# ── FastAPI app ───────────────────────────────────────────────────────────────

_service: DDGSRetrievalService  # set during lifespan startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: nothing to do — service is created in main() before uvicorn
    yield
    # shutdown: release threadpool resources
    _service.close()


app = FastAPI(
    title="orbit DDGS Retrieval Server",
    description="Search-R1 / verl-tool compatible /retrieve endpoint",
    lifespan=lifespan,
)


@app.post("/retrieve")
async def retrieve(req: QueryRequest):
    """
    Request:  {"queries": [...], "topk": 5, "return_scores": false}
    Response: {"result": [[{"document": {"contents": "..."}, "score": 0.0}, ...], ...]}
    """
    topk = req.topk or _service.topk
    docs_list, scores_list = await _service.batch_search(req.queries, topk, req.return_scores)

    result = []
    for docs, scores in zip(docs_list, scores_list):
        if req.return_scores:
            result.append([{"document": doc, "score": s} for doc, s in zip(docs, scores)])
        else:
            result.append([{"document": doc} for doc in docs])
    return {"result": result}


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="orbit DDGS retrieval server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument(
        "--backend",
        type=str,
        default=",".join(DEFAULT_BACKENDS),
        help='Comma-separated backends, e.g. "google,brave,bing,wikipedia,grokipedia"',
    )
    parser.add_argument("--workers", type=int, default=1, help="uvicorn worker processes")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backend.split(",") if b.strip()] or DEFAULT_BACKENDS

    print(f"Backends : {backends}")
    print(f"Top-k    : {args.topk}")
    print(f"Port     : {args.port}")

    global _service
    _service = DDGSRetrievalService(backends=backends, topk=args.topk)

    # Warm-up: one lightweight query per backend to reduce first-call latency
    async def _warm() -> None:
        await _service._fan_out("test", topk=1)

    try:
        asyncio.run(_warm())
    except Exception:
        pass

    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=args.workers, log_level="info")


if __name__ == "__main__":
    main()
