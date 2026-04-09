from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, HTTPException

from server.models import (
    MODEL_ID,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)


def _worker_urls() -> list[str]:
    raw = os.getenv("HACKATHON_ROUTER_WORKER_URLS", "").strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    n = max(1, int(os.getenv("HACKATHON_DATA_PARALLEL_REPLICAS", "8")))
    base = int(os.getenv("HACKATHON_WORKER_BASE_PORT", "8100"))
    return [f"http://127.0.0.1:{base + i}" for i in range(n)]


def _router_policy() -> str:
    raw = os.getenv("HACKATHON_ROUTER_POLICY", "round_robin").strip().lower()
    if raw in {"rr", "round-robin", "round_robin"}:
        return "round_robin"
    if raw in {"least_pending", "least-pending", "least"}:
        return "least_pending"
    return "round_robin"


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = aiohttp.ClientTimeout(total=600)
    app.state.http = aiohttp.ClientSession(timeout=timeout)
    app.state.worker_urls = _worker_urls()
    app.state.pending = [0] * len(app.state.worker_urls)
    app.state.rr = 0
    app.state.policy = _router_policy()
    app.state.max_retries = max(0, int(os.getenv("HACKATHON_ROUTER_MAX_RETRIES", "1")))
    app.state.lock = asyncio.Lock()
    app.state.stats = {
        "routed_requests": 0,
        "router_errors": 0,
        "router_retries": 0,
        "max_pending_seen": 0,
        "routed_per_worker": [0] * len(app.state.worker_urls),
        "failed_per_worker": [0] * len(app.state.worker_urls),
    }
    try:
        yield
    finally:
        await app.state.http.close()


app = FastAPI(title="Hackathon Router", version="0.1.0", lifespan=lifespan)


async def _pick_worker_index(exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    async with app.state.lock:
        available = [i for i in range(len(app.state.pending)) if i not in exclude]
        if not available:
            available = list(range(len(app.state.pending)))

        if app.state.policy == "least_pending":
            min_pending = min(app.state.pending[i] for i in available)
            candidates = [i for i in available if app.state.pending[i] == min_pending]
            idx = candidates[app.state.rr % len(candidates)]
        else:
            idx = available[app.state.rr % len(available)]
        app.state.rr = (app.state.rr + 1) % len(app.state.pending)
        app.state.pending[idx] += 1
        app.state.stats["max_pending_seen"] = max(app.state.stats["max_pending_seen"], app.state.pending[idx])
        return idx


async def _release_worker_index(idx: int) -> None:
    async with app.state.lock:
        app.state.pending[idx] = max(0, app.state.pending[idx] - 1)


@app.get("/health")
async def health() -> dict[str, str]:
    ready = 0
    for base in app.state.worker_urls:
        try:
            async with app.state.http.get(f"{base}/health") as resp:
                if resp.status == 200:
                    ready += 1
        except Exception:
            pass
    return {
        "status": "ok" if ready == len(app.state.worker_urls) else "degraded",
        "backend": "ProcessRouter",
        "workers_ready": str(ready),
        "workers_total": str(len(app.state.worker_urls)),
    }


@app.get("/metrics")
async def metrics() -> dict:
    return {
        **app.state.stats,
        "backend": "ProcessRouter",
        "router_policy": app.state.policy,
        "router_max_retries": app.state.max_retries,
        "worker_urls": app.state.worker_urls,
        "pending_per_worker": app.state.pending,
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")

    total_workers = len(app.state.worker_urls)
    max_attempts = min(total_workers, 1 + app.state.max_retries)
    tried: set[int] = set()
    last_error: HTTPException | None = None

    for attempt in range(max_attempts):
        idx = await _pick_worker_index(exclude=tried)
        tried.add(idx)
        base = app.state.worker_urls[idx]
        try:
            async with app.state.http.post(
                f"{base}/v1/chat/completions",
                json=req.model_dump(),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    app.state.stats["router_errors"] += 1
                    app.state.stats["failed_per_worker"][idx] += 1
                    if resp.status >= 500 and attempt + 1 < max_attempts:
                        app.state.stats["router_retries"] += 1
                        continue
                    raise HTTPException(status_code=resp.status, detail=body)
                payload = ChatCompletionResponse.model_validate_json(body)
                app.state.stats["routed_requests"] += 1
                app.state.stats["routed_per_worker"][idx] += 1
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    created=int(time.time()),
                    model=req.model or MODEL_ID,
                    object="chat.completion",
                    choices=payload.choices,
                    usage=payload.usage,
                )
        except HTTPException as exc:
            last_error = exc
            if exc.status_code < 500 or attempt + 1 >= max_attempts:
                raise
            app.state.stats["router_retries"] += 1
        except Exception as exc:
            app.state.stats["router_errors"] += 1
            app.state.stats["failed_per_worker"][idx] += 1
            last_error = HTTPException(status_code=503, detail=f"worker request failed: {exc}")
            if attempt + 1 >= max_attempts:
                break
            app.state.stats["router_retries"] += 1
        finally:
            await _release_worker_index(idx)

    if last_error is not None:
        raise last_error
    raise HTTPException(status_code=503, detail="worker request failed")


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run hackathon process router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("server.router:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
