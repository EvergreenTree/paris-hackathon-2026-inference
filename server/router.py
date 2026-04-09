from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(..., gt=0, le=8192)
    temperature: float = 0.0
    top_p: float = 1.0


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


def _worker_urls() -> list[str]:
    raw = os.getenv("HACKATHON_ROUTER_WORKER_URLS", "").strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    n = max(1, int(os.getenv("HACKATHON_DATA_PARALLEL_REPLICAS", "8")))
    base = int(os.getenv("HACKATHON_WORKER_BASE_PORT", "8100"))
    return [f"http://127.0.0.1:{base + i}" for i in range(n)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = aiohttp.ClientTimeout(total=600)
    app.state.http = aiohttp.ClientSession(timeout=timeout)
    app.state.worker_urls = _worker_urls()
    app.state.pending = [0] * len(app.state.worker_urls)
    app.state.rr = 0
    app.state.lock = asyncio.Lock()
    app.state.stats = {
        "routed_requests": 0,
        "router_errors": 0,
        "max_pending_seen": 0,
    }
    try:
        yield
    finally:
        await app.state.http.close()


app = FastAPI(title="Hackathon Router", version="0.1.0", lifespan=lifespan)


async def _pick_worker_index() -> int:
    async with app.state.lock:
        min_pending = min(app.state.pending)
        candidates = [i for i, p in enumerate(app.state.pending) if p == min_pending]
        idx = candidates[app.state.rr % len(candidates)]
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
        "worker_urls": app.state.worker_urls,
        "pending_per_worker": app.state.pending,
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")

    idx = await _pick_worker_index()
    base = app.state.worker_urls[idx]
    try:
        async with app.state.http.post(
            f"{base}/v1/chat/completions",
            json=req.model_dump(),
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                app.state.stats["router_errors"] += 1
                raise HTTPException(status_code=resp.status, detail=body)
            payload = ChatCompletionResponse.model_validate_json(body)
            app.state.stats["routed_requests"] += 1
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=req.model or MODEL_ID,
                object="chat.completion",
                choices=payload.choices,
                usage=payload.usage,
            )
    except HTTPException:
        raise
    except Exception as exc:
        app.state.stats["router_errors"] += 1
        raise HTTPException(status_code=503, detail=f"worker request failed: {exc}") from exc
    finally:
        await _release_worker_index(idx)


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
