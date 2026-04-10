# Inference Engine Hackathon — GPU MODE × PyTorch (Paris 2026)

This repository is our team’s submission for the **inference optimization track** at [**PyTorch × GPU MODE Hackathon: No gradient descent. Only ascent.**](https://luma.com/gpu-mode-paris-2026) (Paris, 2026). The event paired **distributed training** and **LLM inference** tracks with talks from PyTorch, vLLM, Prime Intellect, and partners.

**Task (inference track):** Build a high-throughput inference engine for [**Qwen/Qwen3.5-35B-A3B**](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) on **8× NVIDIA H200**, exposing an **OpenAI-compatible** `POST /v1/chat/completions` API—without using high-level serving frameworks such as vLLM or SGLang (per [official rules](https://github.com/gpu-mode/paris-hackathon-2026-inference)).

**What’s in this repo**

| Area | Contents |
|------|----------|
| **Evaluation harness** | GSM8K-CoT correctness, weighted throughput benchmark, API conformance checks, scoring helper |
| **Our engine** | Process-per-GPU workers + router, HuggingFace Transformers + `Qwen3_5MoeForCausalLM`, async micro-batching, `start_server.sh` submission entrypoint |
| **Baselines** | Reference numbers and scripts to compare against vLLM-style baselines |

> **Note:** Full model serving requires a CUDA machine with enough GPU memory (e.g. hackathon **8× H200**). You can still install the repo and run `eval.check_server` against the **rule-based** fallback on CPU for API smoke tests.

## Rules

### Objective

Implement a high-throughput inference engine for Qwen3.5-35B-A3B that serves an OpenAI-compatible chat completions API. You are scored on **throughput** (output tokens/sec), with **correctness as a hard requirement**.

### Submission

Your submission consists of:

1. **A start script** — a single script that launches your server and makes it ready on a given port. Must exit cleanly and leave the server running.
2. **Source code** — your full inference engine implementation.
3. **Documentation** — a brief writeup explaining your approach, architecture decisions, and any optimizations used.
4. **Results** — a JSON file containing the throughput results for your engine (we will verify the results ourselves, however to save us time, we would appreciate if you included the results in your submission to do preliminary scoring)

### Scoring

**Correctness is a gate.** Your engine must pass the GSM8K-CoT correctness evaluation (exact match >= 87.5%) to be eligible for throughput scoring. Submissions that fail correctness receive a score of 0.

**Throughput determines rank.** We measure verified output tokens/sec at concurrency levels 1, 2, 4, 8, 16, 32, 64. Higher concurrency levels carry higher weight in the final score:

| Concurrency | Weight |
|---|---|
| 1 | 1x |
| 2 | 1x |
| 4 | 2x |
| 8 | 2x |
| 16 | 4x |
| 32 | 4x |
| 64 | 8x |

**Final score** = weighted sum of verified tok/s across all concurrency levels (if correctness gate is passed).

### What's Allowed

- **Precision:** BF16 only. No FP8, INT8, INT4, or any other reduced precision.
- **Parallelism:** Any parallelism strategy is allowed — tensor parallel, pipeline parallel, expert parallel, data parallel, or any combination.
- **Inference optimizations:** Allowed, as long as they do not affect correctness or output accuracy. Examples of what's allowed:
  - FlashAttention or other fused attention kernels
  - Continuous batching / dynamic batching
  - KV cache optimizations (paged attention, etc.)
  - Prefix caching
  - CUDA graphs
  - Custom CUDA/Triton kernels
  - Speculative decoding (if output matches non-speculative)
  - Operator fusion
- **Not allowed:** Any optimization that changes model outputs compared to a BF16 reference implementation (e.g., quantization, pruning, distillation, approximate attention that drops tokens). **If you're unsure whether an optimization is allowed, ask the organizers.**
- **Language:** Any language. Python, C++, Rust, CUDA — whatever you want.
- **Libraries:** You may use low-level libraries (cuBLAS, cuDNN, NCCL, Triton, etc.) but not high-level inference frameworks (vLLM, SGLang, TensorRT-LLM, etc.). The point is to build the engine yourself.

### Hardware

- **8x NVIDIA H200** (141 GB HBM3e each, NVLink interconnect)
- No other hardware is available. Your engine must run entirely on this node.

## Model

| Property | Value |
|---|---|
| Model | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Architecture | Hybrid Gated DeltaNet + Sparse MoE |
| Total params | 35B |
| Active params | 3B per token |
| Experts | 256 total, 9 active (8 routed + 1 shared) |
| Layers | 40 |
| Context length | 262,144 tokens |
| Hardware | 8x NVIDIA H200 (141GB each) |
| Parallelism | Tensor Parallel = 8 (you are free to use any other deployment you want) |

**Important:** Thinking mode must be disabled. Your server must not emit `<think>` tags in output.

## API Specification

Your server must implement the following endpoints:

### `GET /health`

Returns HTTP 200 when the server is ready.

### `POST /v1/chat/completions`

OpenAI-compatible chat completions (non-streaming only).

**Request:**

```json
{
  "model": "Qwen/Qwen3.5-35B-A3B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 1024,
  "temperature": 0.0,
  "top_p": 1.0
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `model` | Yes | — | Model name (can be ignored by server, but must be accepted) |
| `messages` | Yes | — | Array of `{role, content}` objects |
| `max_tokens` | Yes | — | Maximum output tokens |
| `temperature` | No | 0.0 | Sampling temperature |
| `top_p` | No | 1.0 | Nucleus sampling parameter |

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "Qwen/Qwen3.5-35B-A3B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2+2 equals 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

**Requirements:**
- Must handle at least 64 concurrent requests
- `usage` field with token counts is mandatory
- `finish_reason` must be `"stop"` (max_tokens reached) or `"length"`

## Quick Start

### 1. Install the harness

```bash
uv venv && uv pip install -e .
```

### 2. Check your server is conformant

```bash
python -m eval.check_server --base-url http://localhost:8000
```

### 3. Run correctness eval (GSM8K-CoT, 200 problems)

```bash
python -m eval.correctness.run_correctness --base-url http://localhost:8000
```

### 4. Run throughput benchmark

```bash
python -m eval.throughput.run_throughput --base-url http://localhost:8000
```

## Evaluation Details

### Correctness: GSM8K-CoT

- **What:** 200 grade-school math problems requiring chain-of-thought reasoning
- **Why:** Math is extremely sensitive to implementation bugs — wrong attention masks, bad KV cache, quantization errors, or sampling bugs will cause accuracy to collapse
- **How:** Uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with `local-chat-completions` backend
- **Metric:** Exact match accuracy on the final numeric answer
- **Gate:** >= 87.5% exact match required to qualify for throughput scoring
- **Settings:** temperature=0, top_p=1.0, 8 concurrent requests
- **Seed:** Randomized at eval time

### Throughput

- **What:** Verified output tokens/sec at concurrency levels 1, 2, 4, 8, 16, 32, 64
- **Workload:** 1024 input tokens, 1024 output tokens per request, 64 requests per concurrency level
- **Warmup:** 2 requests discarded before measurement at each level
- **Prompts:** Generated at runtime using random token IDs from the full vocabulary (excluding special tokens), matching vLLM's `RandomDataset` approach with iterative decode-encode length adjustment. No pre-computed prompts — all generated fresh each run.
- **Token verification:** Output tokens are re-counted using the Qwen tokenizer — server-reported `usage.completion_tokens` is compared and discrepancies are flagged. Verified counts are used for scoring.
- **Spot checks:** 2 math questions per concurrency level are injected among random prompts to verify the server is producing correct outputs, not garbage

## Baseline Numbers

Generated using vLLM v0.19.0 on 8xH200 with TP=8, BF16 weights.

### Correctness: GSM8K-CoT (200 problems)

| Metric | Score |
|---|---|
| Exact match (flexible extract) | **91.5%** |
| Exact match (strict match) | **91.0%** |

### Throughput (1024 input / 1024 output tokens, 64 requests per level)

| Concurrency | tok/s (prompt+completion) | Wall Time (s) |
|---|---|---|
| 1 | 984 | 86.1 |
| 2 | 1,753 | 47.5 |
| 4 | 3,038 | 27.3 |
| 8 | 4,749 | 17.9 |
| 16 | 6,446 | 13.6 |
| 32 | 11,144 | 7.5 |
| 64 | 12,810 | 6.5 |

Run `./baseline/run_baseline.sh` to reproduce.

## Repository Structure

```
eval/
  check_server.py                  # Health check + API conformance
  correctness/
    run_correctness.py             # GSM8K-CoT evaluation wrapper
  throughput/
    run_throughput.py              # Async throughput benchmark (generates prompts at runtime)
baseline/
  run_baseline.sh                  # Generate vLLM baseline numbers
  results/                         # Baseline results
```

## Team Implementation

### Architecture Overview

We built a **process-per-GPU data-parallel inference engine** with a load-balancing router. The design prioritizes high-concurrency throughput (c=32/64 carry the most scoring weight) while maintaining correctness for the GSM8K-CoT gate.

```
                    ┌──────────────────────┐
   Client requests  │   Router (port 8000) │  FastAPI + aiohttp
   ──────────────►  │   least-pending LB   │
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                   ▼
   ┌─────────────┐    ┌─────────────┐     ┌─────────────┐
   │  Worker 0   │    │  Worker 1   │ ... │  Worker 7   │
   │  cuda:0     │    │  cuda:1     │     │  cuda:7     │
   │  port 8100  │    │  port 8101  │     │  port 8107  │
   └─────────────┘    └─────────────┘     └─────────────┘
```

**Key design decisions:**

1. **Process-per-GPU isolation** — Each H200 GPU runs a dedicated worker process with its own model replica. No GIL contention, no shared memory, no cross-GPU synchronization. Each worker is a complete FastAPI server with its own async request scheduler.

2. **Least-pending router** — A lightweight FastAPI router on port 8000 distributes requests across workers using a least-pending policy with round-robin tiebreaking. The router passes through raw JSON responses to minimize overhead (no deserialization/reserialization).

3. **Transformers `model.generate()`** — We use HuggingFace's built-in generation path end-to-end. A custom decode loop was tried but regressed throughput badly on this hybrid DeltaNet + MoE stack because `generate()` integrates cache, masks, and model-specific behavior correctly.

4. **Async micro-batch scheduler** — Each worker runs an async engine with configurable batch accumulation. Incoming requests are queued with an adaptive wait window (larger batches at high concurrency, no wait at c=1). Priority lanes ensure low-latency interactive requests aren't blocked by bulk workloads.

### Optimizations

| Optimization | Impact | Description |
|---|---|---|
| **DeltaNet fast path** | High | `causal-conv1d` + `flash-linear-attention` enable optimized CUDA kernels for the hybrid DeltaNet layers (vs. slow PyTorch fallback) |
| **torch.compile (inductor)** | Optional | Off by default (`HACKATHON_TORCH_COMPILE=0`). Can help some setups but caused long CPU compile stalls here; enable only after profiling |
| **Startup warmup** | Reliability | One tiny generation when compile is off (fast startup); extra passes when compile is on to prime JIT |
| **Flash Attention auto-detect** | Medium | Automatically uses `flash_attention_2` when available, falls back to SDPA |
| **CUDA tuning flags** | Low | TF32 matmul, cuDNN benchmark mode; dynamo config only when `torch.compile` is enabled |
| **Router passthrough** | Low | Raw JSON forwarding avoids Pydantic validation overhead in the router hot path |
| **Left-pad tokenization** | Correctness | `padding_side="left"` ensures correct batch generation with varying prompt lengths |
| **Chat template** | Correctness | `enable_thinking=False` disables thinking mode as required by hackathon rules |

### File Structure

```
server/
  app.py                  # Per-GPU worker: FastAPI server + InferenceEngine
  router.py               # Load-balancing router across 8 workers
  engine.py               # Async scheduler, HuggingFace backend (model.generate)
  models.py               # Shared Pydantic models (OpenAI-compatible request/response)
  modeling_qwen3_5_moe.py # Qwen3.5 MoE model definition (from transformers)
  __init__.py
start_server.sh           # Single launch script (hackathon submission entrypoint)
```

### Quick Start (8xH200)

```bash
# Install dependencies
uv venv --python 3.12
uv pip install -e ".[server]"

# Optional: install causal-conv1d for DeltaNet fast path
# (requires CUDA toolkit matching torch's CUDA version)
pip install causal-conv1d --no-build-isolation

# Launch server (starts 8 workers + 1 router)
bash start_server.sh
```

The start script:
1. Spawns 8 worker processes (one per GPU, ports 8100-8107)
2. Waits for all workers to pass health checks
3. Launches the router on port 8000
4. Exits cleanly, leaving the server running

### Configuration

All parameters are configurable via environment variables with sensible defaults hardcoded in `start_server.sh`:

| Variable | Default | Description |
|---|---|---|
| `HACKATHON_BACKEND` | `hf` | Backend type (`hf` for HuggingFace, `rule-based` for scaffold) |
| `HACKATHON_DATA_PARALLEL_REPLICAS` | `8` | Number of GPU replicas |
| `HACKATHON_BATCH_MAX_SIZE` | `16` | Max requests per micro-batch |
| `HACKATHON_BATCH_WAIT_MS` | `1.0` | Batch accumulation window (ms) |
| `HACKATHON_TORCH_COMPILE` | `0` | Set to `1` to enable torch.compile on model.forward (experimental) |
| `HACKATHON_ROUTER_POLICY` | `least_pending` | Router load-balancing policy |
| `HACKATHON_ATTN_IMPL` | `auto` | Attention implementation (auto/flash_attention_2/sdpa) |

### Verification

```bash
# API conformance check
python -m eval.check_server --base-url http://localhost:8000

# Correctness (GSM8K-CoT, 200 problems, gate >= 87.5%)
python -m eval.correctness.run_correctness --base-url http://localhost:8000

# Full throughput sweep
python -m eval.throughput.run_throughput --base-url http://localhost:8000

# Compute final score
python -m eval.score --correctness results/correctness.json --throughput results/throughput.json
```

## Acknowledgments

- [**GPU MODE × PyTorch — Paris 2026**](https://luma.com/gpu-mode-paris-2026) — organizers, sponsors (Verda, Sesterce, PyTorch, Prime Intellect, SemiAnalysis, and others), and the inference-track volunteers.
- [**paris-hackathon-2026-inference**](https://github.com/gpu-mode/paris-hackathon-2026-inference) — problem statement, harness, and success criteria we built against.
- [**Qwen Team**](https://huggingface.co/Qwen) for **Qwen3.5-35B-A3B** and the Transformers integration path we rely on.
