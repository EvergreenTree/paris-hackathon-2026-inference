from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.legacy_engine_runtime import EngineRequest, InferenceEngine

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = InferenceEngine()
    await app.state.engine.start()
    try:
        yield
    finally:
        await app.state.engine.stop()


app = FastAPI(title="Hackathon Inference Server", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "backend": app.state.engine.backend_name}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")

    messages_payload = [{"role": m.role, "content": m.content} for m in req.messages]
    engine_req = EngineRequest(
        messages=messages_payload,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    engine_resp = await app.state.engine.submit(engine_req)

    choice = ChatChoice(
        index=0,
        message=ChatMessage(role="assistant", content=engine_resp.content),
        finish_reason=engine_resp.finish_reason,
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=req.model or MODEL_ID,
        choices=[choice],
        usage=Usage(
            prompt_tokens=engine_resp.prompt_tokens,
            completion_tokens=engine_resp.completion_tokens,
            total_tokens=engine_resp.prompt_tokens + engine_resp.completion_tokens,
        ),
    )


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run hackathon API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
