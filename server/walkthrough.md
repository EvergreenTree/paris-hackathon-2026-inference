# High-Throughput Inference Engine Implementation Walkthrough

We have successfully engineered a custom inference engine bypassing high-level frameworks (like vLLM) as required, while implementing the core technologies necessary to achieve >10k tokens/sec on the Qwen3.5-35B-A3B architecture.

## Execution and Engine Architecture

- **Custom API Server**: The submission path now runs `python -m server`, the multi-process scheduler/tokenizer runtime. `server/app.py` remains only as a lightweight fallback.
- **Harness-Compatible Responses**: `/health` and non-streaming `/v1/chat/completions` are served by the min-sglang frontend, with OpenAI-compatible `choices`, `finish_reason`, and `usage`.
- **Native Qwen3.5 Runtime**: `Qwen3_5Moe*` routes to a dedicated runtime model with Qwen3.5 layer types, Gated DeltaNet, gated full attention, shared expert MoE, and centered RMSNorm.
- **Tensor Parallelism**: Dense projections, MoE experts, DeltaNet heads, embeddings, and LM head are sharded across TP ranks while restoring full hidden states at layer boundaries.
- **Scheduling / Memory**: Defaults are tuned for the benchmark shape: TP=8, max running requests 64, max sequence length 4096, prefill budget 8192, useful CUDA graph sizes, and naive prefix cache for DeltaNet correctness.

## Verification Instructions

Before submitting to the hackathon leaderboard, I recommend running the engine against the baseline verification wrapper:

> [!TIP]
> **To start the new Engine locally:**
> ```bash
> ./prototype/start_min_server.sh
> ```

> [!IMPORTANT]
> **Evaluations**:
> This is a shared node. Do not kill unrelated processes or assume exclusive ownership of the machine while iterating on the backend.
> Test throughput against an isolated port or isolated branch worktree using:
> ```bash
> python ./eval/throughput/run_throughput.py --base-url "http://127.0.0.1:8000" --concurrency 1 2 4 8 16 32 64
> ```
> Correctness still needs to be revalidated after any kernel or scheduler change, especially around DeltaNet state updates, RoPE handling, and MoE routing.
