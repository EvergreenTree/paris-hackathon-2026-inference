# Prototype Launcher

Run this on the H200 node:

```bash
./prototype/start_min_server.sh
```

It will:

- install `uv` if needed
- create `.venv`
- install the local package plus missing `fastapi`, `uvicorn`, `transformers`, and `torch` deps
- launch `prototype/min_server.py` on the requested port
- verify `GET /health` and `POST /v1/chat/completions`

Useful overrides:

```bash
PORT=8001 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./prototype/start_min_server.sh
```

`prototype/start_vllm.sh` is still available as the old vLLM reference launcher, but the restored prototype flow now points at `start_min_server.sh`.

## Previous Metrics

These are the last remote measurements we collected on the H200 node before interrupting the throughput runs. They are partial smoke-run results from the harness with `1024` input tokens, `1024` output tokens, and `8` requests per tested concurrency.

### Initial Custom Server

| Concurrency | Throughput (tok/s) |
|-------------|--------------------|
| 1           | 74.22              |
| 2           | 62.50              |
| 4           | 73.91              |
| 8           | 70.64              |

### After Microbatching Pass

| Concurrency | Throughput (tok/s) |
|-------------|--------------------|
| 4           | 102.77             |
| 8           | 143.63             |
| 16          | 109.70             |

The optimized run was interrupted before `32` and `64` completed, so there are no reliable numbers for those levels yet.
