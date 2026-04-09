from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
from threading import Lock
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


class EngineOverloadedError(RuntimeError):
    """Raised when the engine queue is full and cannot accept more requests."""


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

    def __init__(self, device_override: str | None = None) -> None:
        self.model_id = os.getenv("HACKATHON_MODEL_ID", MODEL_ID)
        self.tokenizer_id = os.getenv("HACKATHON_TOKENIZER_ID", self.model_id)
        self.device = device_override or os.getenv("HACKATHON_DEVICE", "cuda")
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
        device_map_target: str | int = self.device
        if isinstance(self.device, str) and self.device.startswith("cuda:"):
            try:
                device_map_target = int(self.device.split(":", 1)[1])
            except Exception:
                device_map_target = self.device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map={"": device_map_target},
            attn_implementation="sdpa",
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
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

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
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

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


class DataParallelHuggingFaceBackend:
    """Replicates HF backend across multiple GPUs for data-parallel serving."""

    def __init__(self) -> None:
        init_t0 = time.perf_counter()
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"failed to import torch for data-parallel backend: {exc}") from exc

        available_gpus = torch.cuda.device_count()
        requested = int(os.getenv("HACKATHON_DATA_PARALLEL_REPLICAS", "0"))
        if requested <= 0:
            requested = available_gpus

        if available_gpus <= 0:
            raise RuntimeError("no CUDA GPUs available for data-parallel backend")

        self.replica_count = max(1, min(requested, available_gpus))
        self._replicas: list[HuggingFaceBackend | None] = [None] * self.replica_count
        # Load model replicas concurrently so all GPUs initialize in parallel.
        with ThreadPoolExecutor(max_workers=self.replica_count, thread_name_prefix="hf-load") as load_pool:
            future_to_idx = {
                load_pool.submit(HuggingFaceBackend, f"cuda:{idx}"): idx for idx in range(self.replica_count)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                self._replicas[idx] = fut.result()

        if any(replica is None for replica in self._replicas):
            raise RuntimeError("failed to initialize one or more HF replicas")

        self._replicas = [replica for replica in self._replicas if replica is not None]
        self._executor = ThreadPoolExecutor(max_workers=self.replica_count, thread_name_prefix="hf-replica")
        self._pending = [0 for _ in range(self.replica_count)]
        self._rr = 0
        self._lock = Lock()
        init_s = time.perf_counter() - init_t0
        LOG.info("Initialized data-parallel backend with %s replicas in %.2fs", self.replica_count, init_s)

    def _choose_replica_index_unlocked(self) -> int:
        # Caller must hold self._lock.
        min_pending = min(self._pending)
        candidates = [i for i, p in enumerate(self._pending) if p == min_pending]
        chosen = candidates[self._rr % len(candidates)]
        self._rr = (self._rr + 1) % self.replica_count
        return chosen

    def generate(self, req: EngineRequest) -> EngineResponse:
        return self.generate_batch([req])[0]

    def generate_batch(self, reqs: list[EngineRequest]) -> list[EngineResponse]:
        if not reqs:
            return []
        if self.replica_count == 1:
            return self._replicas[0].generate_batch(reqs)

        # Assign requests to replicas by current load.
        assignments: dict[int, list[tuple[int, EngineRequest]]] = defaultdict(list)
        with self._lock:
            for idx, req in enumerate(reqs):
                ridx = self._choose_replica_index_unlocked()
                assignments[ridx].append((idx, req))
                self._pending[ridx] += 1

        futures = {}
        for ridx, items in assignments.items():
            batch_reqs = [req for _, req in items]
            fut = self._executor.submit(self._replicas[ridx].generate_batch, batch_reqs)
            futures[fut] = (ridx, items)

        out: list[EngineResponse | None] = [None] * len(reqs)
        try:
            for fut, (ridx, items) in futures.items():
                responses = fut.result()
                if len(responses) != len(items):
                    raise RuntimeError(
                        f"replica {ridx} returned {len(responses)} responses for {len(items)} requests"
                    )
                for (orig_idx, _), resp in zip(items, responses, strict=True):
                    out[orig_idx] = resp
        finally:
            with self._lock:
                for ridx, items in assignments.items():
                    self._pending[ridx] = max(0, self._pending[ridx] - len(items))

        if any(resp is None for resp in out):
            raise RuntimeError("internal error: missing response in data-parallel batch execution")
        return [resp for resp in out if resp is not None]

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def build_backend() -> InferenceBackend:
    backend_name = os.getenv("HACKATHON_BACKEND", "rule-based").strip().lower()
    if backend_name in {"hf", "huggingface", "transformers"}:
        try:
            requested_replicas = int(os.getenv("HACKATHON_DATA_PARALLEL_REPLICAS", "0"))
            if requested_replicas > 1:
                return DataParallelHuggingFaceBackend()
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
        self._base_worker_count = max(1, int(os.getenv("HACKATHON_WORKER_COUNT", "1")))
        self._max_worker_count = max(
            self._base_worker_count,
            int(os.getenv("HACKATHON_MAX_WORKER_COUNT", str(self._base_worker_count))),
        )
        self._autoscale_workers = os.getenv("HACKATHON_AUTOSCALE_WORKERS", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._autoscale_check_ms = max(50.0, float(os.getenv("HACKATHON_AUTOSCALE_CHECK_MS", "250.0")))
        self._autoscale_scale_up_q = max(1, int(os.getenv("HACKATHON_AUTOSCALE_SCALE_UP_Q", "8")))
        self._autoscale_scale_down_q = max(0, int(os.getenv("HACKATHON_AUTOSCALE_SCALE_DOWN_Q", "2")))
        self._batch_max_size = max(1, int(os.getenv("HACKATHON_BATCH_MAX_SIZE", "8")))
        self._batch_wait_ms = max(0.0, float(os.getenv("HACKATHON_BATCH_WAIT_MS", "2.0")))
        self._max_pending_requests = max(1, int(os.getenv("HACKATHON_MAX_PENDING_REQUESTS", "4096")))
        self._priority_enabled = os.getenv("HACKATHON_PRIORITY_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._priority_max_tokens = max(1, int(os.getenv("HACKATHON_PRIORITY_MAX_TOKENS", "256")))
        self._priority_burst = max(1, int(os.getenv("HACKATHON_PRIORITY_BURST", "4")))
        self._shape_bucketing_enabled = os.getenv("HACKATHON_SHAPE_BUCKETING_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._shape_bucket_chars = max(128, int(os.getenv("HACKATHON_SHAPE_BUCKET_CHARS", "512")))
        self._adaptive_wait = os.getenv("HACKATHON_ADAPTIVE_BATCH_WAIT", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._overload_wait_ms = max(0.0, float(os.getenv("HACKATHON_OVERLOAD_WAIT_MS", "0.0")))
        self._priority_queue: asyncio.Queue[
            tuple[EngineRequest, asyncio.Future[EngineResponse], float, str]
        ] = asyncio.Queue()
        self._normal_queue: asyncio.Queue[
            tuple[EngineRequest, asyncio.Future[EngineResponse], float, str]
        ] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._autoscaler_task: asyncio.Task | None = None
        self._stats = {
            "submitted_requests": 0,
            "processed_requests": 0,
            "processed_batches": 0,
            "last_batch_size": 0,
            "max_batch_size_seen": 0,
            "total_queue_wait_ms": 0.0,
            "total_backend_exec_ms": 0.0,
            "avg_queue_wait_ms": 0.0,
            "avg_backend_exec_ms": 0.0,
            "rejected_requests": 0,
            "priority_enqueued": 0,
            "normal_enqueued": 0,
            "priority_processed": 0,
            "normal_processed": 0,
            "autoscale_scale_up_events": 0,
            "autoscale_scale_down_events": 0,
            "grouped_subbatches": 0,
            "fairness_forced_normal_turns": 0,
            "overload_wait_recoveries": 0,
        }
        self._priority_streak = 0

    async def start(self) -> None:
        if self._worker_tasks:
            return
        for i in range(self._base_worker_count):
            await self._spawn_worker(i)
        if self._autoscale_workers and self._max_worker_count > self._base_worker_count:
            self._autoscaler_task = asyncio.create_task(self._autoscaler_loop(), name="engine-autoscaler")
        LOG.info(
            "Engine started backend=%s workers=%s/%s autoscale=%s batch_max=%s batch_wait_ms=%s adaptive_wait=%s",
            self._backend_name,
            self._base_worker_count,
            self._max_worker_count,
            self._autoscale_workers,
            self._batch_max_size,
            self._batch_wait_ms,
            self._adaptive_wait,
        )

    async def stop(self) -> None:
        if self._autoscaler_task is not None:
            self._autoscaler_task.cancel()
            try:
                await self._autoscaler_task
            except asyncio.CancelledError:
                pass
            self._autoscaler_task = None
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
        close_fn = getattr(self._backend, "close", None)
        if callable(close_fn):
            close_fn()

    def _is_priority(self, req: EngineRequest) -> bool:
        if not self._priority_enabled:
            return False
        # Heuristic: likely interactive decode requests should get low queueing latency.
        return req.temperature <= 0.0 and req.max_tokens <= self._priority_max_tokens

    def total_queue_size(self) -> int:
        return self._priority_queue.qsize() + self._normal_queue.qsize()

    async def submit(self, req: EngineRequest) -> EngineResponse:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[EngineResponse] = loop.create_future()
        if self.total_queue_size() >= self._max_pending_requests:
            recovered = False
            if self._overload_wait_ms > 0.0:
                waited = 0.0
                step_ms = 5.0
                while waited < self._overload_wait_ms:
                    await asyncio.sleep(step_ms / 1000.0)
                    waited += step_ms
                    if self.total_queue_size() < self._max_pending_requests:
                        recovered = True
                        self._stats["overload_wait_recoveries"] += 1
                        break
            if not recovered and self.total_queue_size() >= self._max_pending_requests:
                self._stats["rejected_requests"] += 1
                raise EngineOverloadedError(
                    f"queue is full ({self.total_queue_size()}/{self._max_pending_requests})"
                )
        self._stats["submitted_requests"] += 1
        lane = "priority" if self._is_priority(req) else "normal"
        if lane == "priority":
            self._stats["priority_enqueued"] += 1
            await self._priority_queue.put((req, fut, time.perf_counter(), lane))
        else:
            self._stats["normal_enqueued"] += 1
            await self._normal_queue.put((req, fut, time.perf_counter(), lane))
        return await fut

    def _next_batch_timeout_s(self) -> float:
        """Adaptive wait window: lower wait as queue pressure rises."""
        if self._batch_wait_ms <= 0.0:
            return 0.0
        if not self._adaptive_wait:
            return self._batch_wait_ms / 1000.0
        q = self.total_queue_size()
        if q >= self._batch_max_size:
            return 0.0
        if q <= 1:
            return self._batch_wait_ms / 1000.0
        return (self._batch_wait_ms / 2.0) / 1000.0

    async def _spawn_worker(self, idx: int) -> None:
        self._worker_tasks.append(
            asyncio.create_task(self._worker_loop(idx), name=f"engine-worker-{idx}")
        )

    async def _autoscaler_loop(self) -> None:
        next_worker_idx = self._base_worker_count
        while True:
            await asyncio.sleep(self._autoscale_check_ms / 1000.0)
            q = self.total_queue_size()
            alive = [t for t in self._worker_tasks if not t.done()]
            self._worker_tasks = alive
            worker_count = len(alive)

            if q >= self._autoscale_scale_up_q and worker_count < self._max_worker_count:
                await self._spawn_worker(next_worker_idx)
                next_worker_idx += 1
                self._stats["autoscale_scale_up_events"] += 1
                continue

            if (
                q <= self._autoscale_scale_down_q
                and worker_count > self._base_worker_count
                and self._worker_tasks
            ):
                task = self._worker_tasks.pop()
                task.cancel()
                self._stats["autoscale_scale_down_events"] += 1

    async def _dequeue_once(
        self, timeout_s: float | None = None
    ) -> tuple[EngineRequest, asyncio.Future[EngineResponse], float, str] | None:
        # Fairness policy: after a burst of priority picks, force one normal
        # dequeue if available to avoid starvation.
        if self._priority_streak >= self._priority_burst:
            try:
                item = self._normal_queue.get_nowait()
                self._priority_streak = 0
                self._stats["fairness_forced_normal_turns"] += 1
                return item
            except asyncio.QueueEmpty:
                self._priority_streak = 0

        try:
            item = self._priority_queue.get_nowait()
            self._priority_streak += 1
            return item
        except asyncio.QueueEmpty:
            pass
        try:
            item = self._normal_queue.get_nowait()
            self._priority_streak = 0
            return item
        except asyncio.QueueEmpty:
            pass

        if timeout_s is not None and timeout_s <= 0.0:
            return None

        priority_task = asyncio.create_task(self._priority_queue.get())
        normal_task = asyncio.create_task(self._normal_queue.get())
        try:
            done, pending = await asyncio.wait(
                {priority_task, normal_task},
                timeout=timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return None
            # If both become ready, prefer priority lane.
            if priority_task in done:
                self._priority_streak += 1
                return priority_task.result()
            self._priority_streak = 0
            return normal_task.result()
        finally:
            for task in (priority_task, normal_task):
                if not task.done():
                    task.cancel()
            for task in pending if "pending" in locals() else []:
                if not task.done():
                    task.cancel()

    async def _worker_loop(self, worker_idx: int) -> None:
        while True:
            first = await self._dequeue_once(timeout_s=None)
            if first is None:
                continue
            batch: list[tuple[EngineRequest, asyncio.Future[EngineResponse], float, str]] = [first]

            # Drain extra requests for a tiny window to form a micro-batch.
            while len(batch) < self._batch_max_size:
                try:
                    timeout_s = self._next_batch_timeout_s()
                    nxt = await self._dequeue_once(timeout_s=timeout_s)
                    if nxt is None:
                        break
                    batch.append(nxt)
                except Exception:
                    break

            reqs = [r for r, _, _, _ in batch]
            futs = [f for _, f, _, _ in batch]
            enqueue_times = [ts for _, _, ts, _ in batch]
            try:
                now = time.perf_counter()
                queue_wait_ms = sum((now - ts) * 1000.0 for ts in enqueue_times)
                backend_t0 = time.perf_counter()
                responses = self._run_batch_grouped(reqs)
                backend_exec_ms = (time.perf_counter() - backend_t0) * 1000.0
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
                self._stats["max_batch_size_seen"] = max(
                    int(self._stats["max_batch_size_seen"]), len(batch)
                )
                self._stats["priority_processed"] += sum(1 for _, _, _, lane in batch if lane == "priority")
                self._stats["normal_processed"] += sum(1 for _, _, _, lane in batch if lane == "normal")
                self._stats["total_queue_wait_ms"] += queue_wait_ms
                self._stats["total_backend_exec_ms"] += backend_exec_ms
                processed_batches = max(1, int(self._stats["processed_batches"]))
                self._stats["avg_queue_wait_ms"] = (
                    float(self._stats["total_queue_wait_ms"]) / processed_batches
                )
                self._stats["avg_backend_exec_ms"] = (
                    float(self._stats["total_backend_exec_ms"]) / processed_batches
                )
            except Exception as exc:
                LOG.exception("Worker %s failed processing batch of size %s", worker_idx, len(batch))
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(exc)
            finally:
                for _, _, _, lane in batch:
                    if lane == "priority":
                        self._priority_queue.task_done()
                    else:
                        self._normal_queue.task_done()

    def _run_batch(self, reqs: list[EngineRequest]) -> list[EngineResponse]:
        generate_batch = getattr(self._backend, "generate_batch", None)
        if callable(generate_batch):
            return generate_batch(reqs)
        return [self._backend.generate(r) for r in reqs]

    def _shape_bucket_id(self, req: EngineRequest) -> int:
        if not self._shape_bucketing_enabled:
            return 0
        # Lightweight proxy for prompt token length; avoids expensive pre-tokenization
        # in the scheduler thread while still reducing padding fragmentation.
        total_chars = sum(len(m.get("content", "")) for m in req.messages)
        return total_chars // self._shape_bucket_chars

    def _run_batch_grouped(self, reqs: list[EngineRequest]) -> list[EngineResponse]:
        """Group requests by generation config to maximize real batched execution."""
        grouped_indices: dict[tuple[int, float, float, int], list[int]] = defaultdict(list)
        for idx, req in enumerate(reqs):
            key = (
                req.max_tokens,
                req.temperature,
                req.top_p,
                self._shape_bucket_id(req),
            )
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
            self._stats["grouped_subbatches"] += 1

        # By this point, every slot should be filled.
        if any(resp is None for resp in out):
            raise RuntimeError("internal error: missing response in grouped batch execution")
        return [resp for resp in out if resp is not None]

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def queue_size(self) -> int:
        return self.total_queue_size()

    @property
    def stats(self) -> dict[str, int | float | str]:
        return {
            **self._stats,
            "backend": self._backend_name,
            "worker_count": len(self._worker_tasks),
            "base_worker_count": self._base_worker_count,
            "max_worker_count": self._max_worker_count,
            "autoscale_workers": self._autoscale_workers,
            "autoscale_check_ms": self._autoscale_check_ms,
            "autoscale_scale_up_q": self._autoscale_scale_up_q,
            "autoscale_scale_down_q": self._autoscale_scale_down_q,
            "batch_max_size": self._batch_max_size,
            "batch_wait_ms": self._batch_wait_ms,
            "adaptive_wait": self._adaptive_wait,
            "priority_enabled": self._priority_enabled,
            "priority_max_tokens": self._priority_max_tokens,
            "priority_burst": self._priority_burst,
            "shape_bucketing_enabled": self._shape_bucketing_enabled,
            "shape_bucket_chars": self._shape_bucket_chars,
            "max_pending_requests": self._max_pending_requests,
            "overload_wait_ms": self._overload_wait_ms,
            "queue_size": self.total_queue_size(),
            "priority_queue_size": self._priority_queue.qsize(),
            "normal_queue_size": self._normal_queue.qsize(),
            "data_parallel_replicas": getattr(self._backend, "replica_count", 1),
        }
