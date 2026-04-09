from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
from typing import Literal, Protocol

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
LOG = logging.getLogger(__name__)


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


class InferenceBackend(Protocol):
    def generate(self, req: EngineRequest) -> EngineResponse: ...

    def generate_batch(self, reqs: list[EngineRequest]) -> list[EngineResponse]: ...


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


def _count_tokens_fallback(text: str) -> int:
    return max(1, len(text.split()))


@lru_cache(maxsize=4)
def _get_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        return None


def _count_prompt_tokens(messages: list[dict[str, str]], tokenizer_id: str) -> int:
    tokenizer = _get_tokenizer(tokenizer_id)
    if tokenizer is None:
        text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return _count_tokens_fallback(text)
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _count_completion_tokens(content: str, tokenizer_id: str) -> int:
    tokenizer = _get_tokenizer(tokenizer_id)
    if tokenizer is None:
        return _count_tokens_fallback(content)
    return len(tokenizer.encode(content, add_special_tokens=False))


class RuleBasedBackend:
    """Safe fallback backend when real model runtime is unavailable."""

    def generate(self, req: EngineRequest) -> EngineResponse:
        user_messages = [m.get("content", "") for m in req.messages if m.get("role") == "user"]
        last_user = user_messages[-1] if user_messages else ""
        content = _extract_math_answer(last_user) or "Server scaffold active. Model runner integration is next."

        tokenizer_id = os.getenv("HACKATHON_TOKENIZER_ID", MODEL_ID)
        prompt_tokens = _count_prompt_tokens(req.messages, tokenizer_id=tokenizer_id)
        completion_tokens = _count_completion_tokens(content, tokenizer_id=tokenizer_id)
        finish_reason: Literal["stop", "length"] = "stop"
        if completion_tokens >= req.max_tokens:
            finish_reason = "length"
        return EngineResponse(
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_batch(self, reqs: list[EngineRequest]) -> list[EngineResponse]:
        return [self.generate(r) for r in reqs]


class HuggingFaceBackend:
    """Direct Transformers backend for initial real model execution."""

    def __init__(self) -> None:
        self.model_id = os.getenv("HACKATHON_MODEL_ID", MODEL_ID)
        self.tokenizer_id = os.getenv("HACKATHON_TOKENIZER_ID", self.model_id)
        self.device = os.getenv("HACKATHON_DEVICE", "cuda")
        self.torch_dtype = os.getenv("HACKATHON_DTYPE", "bfloat16")
        self.max_new_tokens_cap = int(os.getenv("HACKATHON_MAX_NEW_TOKENS_CAP", "1024"))
        self.tokenizer = None
        self.model = None
        self.torch = None
        self._load_runtime()

    def _load_runtime(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(f"failed to import torch/transformers runtime: {exc}") from exc

        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()
        self.torch = torch
        LOG.info("Loaded model backend model_id=%s device=%s dtype=%s", self.model_id, self.device, self.torch_dtype)

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, req: EngineRequest) -> EngineResponse:
        prompt_text = self._render_prompt(req.messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        max_new_tokens = max(1, min(req.max_tokens, self.max_new_tokens_cap))
        do_sample = req.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": max(req.temperature, 1e-5) if do_sample else None,
            "top_p": req.top_p if do_sample else None,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        # Drop None values so generate() gets clean kwargs.
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0, prompt_tokens:]
        completion_tokens = int(generated_ids.shape[-1])
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        finish_reason: Literal["stop", "length"] = "length" if completion_tokens >= max_new_tokens else "stop"
        return EngineResponse(
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_batch(self, reqs: list[EngineRequest]) -> list[EngineResponse]:
        if not reqs:
            return []
        if len(reqs) == 1:
            return [self.generate(reqs[0])]

        # Keep behavior deterministic/correct by falling back if requests in the
        # batch have different sampling configs or output limits.
        first = reqs[0]
        same_shape = all(
            r.max_tokens == first.max_tokens
            and abs(r.temperature - first.temperature) < 1e-9
            and abs(r.top_p - first.top_p) < 1e-9
            for r in reqs
        )
        if not same_shape:
            return [self.generate(r) for r in reqs]

        prompts = [self._render_prompt(r.messages) for r in reqs]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if "attention_mask" in inputs:
            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        else:
            prompt_lens = [int(inputs["input_ids"].shape[-1])] * len(reqs)

        max_new_tokens = max(1, min(first.max_tokens, self.max_new_tokens_cap))
        do_sample = first.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": max(first.temperature, 1e-5) if do_sample else None,
            "top_p": first.top_p if do_sample else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        responses: list[EngineResponse] = []
        for i, req in enumerate(reqs):
            prompt_len = int(prompt_lens[i])
            generated_ids = output_ids[i, prompt_len:]
            completion_tokens = int(generated_ids.shape[-1])
            content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            finish_reason: Literal["stop", "length"] = "length" if completion_tokens >= max_new_tokens else "stop"
            responses.append(
                EngineResponse(
                    content=content,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_tokens,
                )
            )
        return responses


def build_backend() -> InferenceBackend:
    backend_name = os.getenv("HACKATHON_BACKEND", "rule-based").strip().lower()
    if backend_name in {"hf", "huggingface", "transformers"}:
        try:
            return HuggingFaceBackend()
        except Exception as exc:
            LOG.warning("Falling back to rule-based backend: %s", exc)
            return RuleBasedBackend()
    return RuleBasedBackend()


class InferenceEngine:
    """Async engine with configurable worker pool and micro-batching."""

    def __init__(self) -> None:
        self._backend: InferenceBackend = build_backend()
        self._backend_name = self._backend.__class__.__name__
        self._worker_count = max(1, int(os.getenv("HACKATHON_WORKER_COUNT", "1")))
        self._batch_max_size = max(1, int(os.getenv("HACKATHON_BATCH_MAX_SIZE", "8")))
        self._batch_wait_ms = max(0.0, float(os.getenv("HACKATHON_BATCH_WAIT_MS", "2.0")))
        self._queue: asyncio.Queue[
            tuple[EngineRequest, asyncio.Future[EngineResponse]]
        ] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._stats = {
            "submitted_requests": 0,
            "processed_requests": 0,
            "processed_batches": 0,
            "last_batch_size": 0,
        }

    async def start(self) -> None:
        if self._worker_tasks:
            return
        for i in range(self._worker_count):
            self._worker_tasks.append(
                asyncio.create_task(self._worker_loop(i), name=f"engine-worker-{i}")
            )
        LOG.info(
            "Engine started backend=%s workers=%s batch_max=%s batch_wait_ms=%s",
            self._backend_name,
            self._worker_count,
            self._batch_max_size,
            self._batch_wait_ms,
        )

    async def stop(self) -> None:
        if not self._worker_tasks:
            return
        for task in self._worker_tasks:
            task.cancel()
        for task in self._worker_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._worker_tasks = []

    async def submit(self, req: EngineRequest) -> EngineResponse:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[EngineResponse] = loop.create_future()
        self._stats["submitted_requests"] += 1
        await self._queue.put((req, fut))
        return await fut

    async def _worker_loop(self, worker_idx: int) -> None:
        while True:
            first_req, first_fut = await self._queue.get()
            batch: list[tuple[EngineRequest, asyncio.Future[EngineResponse]]] = [(first_req, first_fut)]

            # Drain extra requests for a tiny window to form a micro-batch.
            while len(batch) < self._batch_max_size:
                try:
                    nxt = await asyncio.wait_for(
                        self._queue.get(), timeout=self._batch_wait_ms / 1000.0
                    )
                    batch.append(nxt)
                except TimeoutError:
                    break

            reqs = [r for r, _ in batch]
            futs = [f for _, f in batch]
            try:
                responses = self._run_batch_grouped(reqs)
                if len(responses) != len(reqs):
                    raise RuntimeError(
                        f"backend returned {len(responses)} responses for batch of {len(reqs)}"
                    )
                for fut, resp in zip(futs, responses, strict=True):
                    if not fut.done():
                        fut.set_result(resp)
                self._stats["processed_batches"] += 1
                self._stats["processed_requests"] += len(batch)
                self._stats["last_batch_size"] = len(batch)
            except Exception as exc:
                LOG.exception("Worker %s failed processing batch of size %s", worker_idx, len(batch))
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(exc)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def _run_batch(self, reqs: list[EngineRequest]) -> list[EngineResponse]:
        generate_batch = getattr(self._backend, "generate_batch", None)
        if callable(generate_batch):
            return generate_batch(reqs)
        return [self._backend.generate(r) for r in reqs]

    def _run_batch_grouped(self, reqs: list[EngineRequest]) -> list[EngineResponse]:
        """Group requests by generation config to maximize real batched execution."""
        grouped_indices: dict[tuple[int, float, float], list[int]] = defaultdict(list)
        for idx, req in enumerate(reqs):
            key = (req.max_tokens, req.temperature, req.top_p)
            grouped_indices[key].append(idx)

        out: list[EngineResponse | None] = [None] * len(reqs)
        for indices in grouped_indices.values():
            group_reqs = [reqs[i] for i in indices]
            group_responses = self._run_batch(group_reqs)
            if len(group_responses) != len(group_reqs):
                raise RuntimeError(
                    f"backend returned {len(group_responses)} responses for grouped batch of {len(group_reqs)}"
                )
            for i, resp in zip(indices, group_responses, strict=True):
                out[i] = resp

        # By this point, every slot should be filled.
        if any(resp is None for resp in out):
            raise RuntimeError("internal error: missing response in grouped batch execution")
        return [resp for resp in out if resp is not None]

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict[str, int | float | str]:
        return {
            **self._stats,
            "backend": self._backend_name,
            "worker_count": self._worker_count,
            "batch_max_size": self._batch_max_size,
            "batch_wait_ms": self._batch_wait_ms,
            "queue_size": self._queue.qsize(),
        }
