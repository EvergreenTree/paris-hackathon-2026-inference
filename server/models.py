from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=1024, gt=0, le=8192)
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
