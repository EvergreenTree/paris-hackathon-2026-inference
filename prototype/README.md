# Prototype Launcher

Run this on the H200 node:

```bash
./prototype/start_vllm.sh
```

It will:

- install `uv` if needed
- create `.venv`
- install `vllm==0.19.0`
- launch `Qwen/Qwen3.5-35B-A3B` on port `8000`
- disable Qwen thinking mode
- verify `GET /health` and `POST /v1/chat/completions`

Useful overrides:

```bash
PORT=8001 TP_SIZE=8 MAX_MODEL_LEN=4096 ./prototype/start_vllm.sh
```
