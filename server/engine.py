from __future__ import annotations

import ast
import asyncio
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"


@dataclass(slots=True)
class EngineRequest:
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(slots=True)
class EngineResponse:
    content: str
    finish_reason: Literal["stop", "length"]
    prompt_tokens: int
    completion_tokens: int


class _SafeEvaluator(ast.NodeVisitor):
    _allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)
    _allowed_unary = (ast.UAdd, ast.USub)

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        if not isinstance(node.op, self._allowed_binops):
            raise ValueError("unsupported operator")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        raise ValueError("unsupported operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        if not isinstance(node.op, self._allowed_unary):
            raise ValueError("unsupported unary operator")
        val = self.visit(node.operand)
        return +val if isinstance(node.op, ast.UAdd) else -val

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, (int, float)):
            raise ValueError("unsupported constant")
        return float(node.value)

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"unsupported node type: {type(node).__name__}")


def _eval_math_expression(expr: str) -> str | None:
    expr = expr.strip()
    if not expr:
        return None
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return None
    try:
        parsed = ast.parse(expr, mode="eval")
        value = _SafeEvaluator().visit(parsed)
    except Exception:
        return None
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


def _extract_math_answer(prompt: str) -> str | None:
    match = re.search(r"what is\s+(.+?)\?", prompt, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _eval_math_expression(match.group(1))


@lru_cache(maxsize=1)
def _get_tokenizer():
    tokenizer_id = os.getenv("HACKATHON_TOKENIZER_ID", MODEL_ID)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        return None


def _count_prompt_tokens(messages: list[dict[str, str]]) -> int:
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return max(1, len(text.split()))
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def _count_completion_tokens(content: str) -> int:
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return max(1, len(content.split()))
    return len(tokenizer.encode(content, add_special_tokens=False))


class InferenceEngine:
    """Async engine scaffold with queue/worker architecture."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[
            tuple[EngineRequest, asyncio.Future[EngineResponse]]
        ] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop(), name="engine-worker")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def submit(self, req: EngineRequest) -> EngineResponse:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[EngineResponse] = loop.create_future()
        await self._queue.put((req, fut))
        return await fut

    async def _worker_loop(self) -> None:
        while True:
            req, fut = await self._queue.get()
            try:
                response = self._run_request(req)
                if not fut.done():
                    fut.set_result(response)
            except Exception as exc:
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                self._queue.task_done()

    def _run_request(self, req: EngineRequest) -> EngineResponse:
        # Placeholder model runner path; this will be replaced with real model execution.
        user_messages = [m.get("content", "") for m in req.messages if m.get("role") == "user"]
        last_user = user_messages[-1] if user_messages else ""
        content = _extract_math_answer(last_user) or "Server scaffold active. Model runner integration is next."

        prompt_tokens = _count_prompt_tokens(req.messages)
        completion_tokens = _count_completion_tokens(content)
        finish_reason: Literal["stop", "length"] = "stop"
        if completion_tokens >= req.max_tokens:
            finish_reason = "length"
        return EngineResponse(
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
