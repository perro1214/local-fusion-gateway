# Local Fusion Gateway

[![CI](https://github.com/perro1214/local-fusion-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/perro1214/local-fusion-gateway/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

OpenAI-compatible local gateway for running a Fusion-style panel, judge, and synthesis flow against local LLM servers such as Ollama, LM Studio, and llama.cpp.

This is an experimental alpha project. It is not an official OpenRouter implementation, and it does not try to fully reproduce OpenRouter's hosted Fusion behavior. The goal is a practical local gateway that can be used by existing OpenAI-compatible clients by changing the base URL.

The implementation follows the local v1 policy in [docs/policy-ja.md](docs/policy-ja.md).

## Current Status

The current `main` branch has a working v1 Gateway:

- Plain proxy and Fusion requests are implemented.
- Gemini OpenAI-compatible smoke tests pass with `models/gemini-2.5-flash-lite`.
- Fusion judge calls request structured JSON with `response_format: json_schema`.
- Backends that reject `response_format` with `400` or `422` are retried once without it.
- Fusion debug metadata is available on request without exposing prompts or panel outputs.

## Features

- `GET /health`
- `POST /v1/chat/completions`
- Plain OpenAI-compatible proxy for configured local models
- Fusion execution when `model` is `openrouter/fusion`
- Fusion execution when `tool_choice` is `required` and `tools` contains `openrouter:fusion`
- Parallel panel calls with partial-failure tolerance
- Judge structured output via `response_format: json_schema`
- Judge JSON parsing with degraded fallback when the judge does not return valid schema JSON
- Non-streaming chat completions only
- `X-Request-ID` request tracking
- Optional Fusion debug metadata with `X-Local-Fusion-Debug: true`

v1 intentionally does not execute `openrouter:web_search` or `openrouter:web_fetch`. Those tools are accepted in Fusion parameters but ignored.

## Setup

```bash
uv sync --extra dev
cp config.example.yaml config.yaml
```

Edit `config.yaml` so each logical model points at a running OpenAI-compatible local backend.

```yaml
models:
  ollama-qwen:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    model: "qwen3:8b"
```

Environment variables in YAML values are expanded when the config is loaded, so API keys can be referenced as `${GEMINI_API_KEY}` instead of being written into the file.

## Run

```bash
LOCAL_FUSION_CONFIG=config.yaml uv run local-fusion-gateway
```

The server binds to `server.host` and `server.port` from `config.yaml`. Environment variables override those values:

```bash
LOCAL_FUSION_HOST=127.0.0.1 LOCAL_FUSION_PORT=8080 uv run local-fusion-gateway
```

## Health Check

```bash
curl http://127.0.0.1:8080/health
```

## Plain Proxy Request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "ollama-qwen",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

## Fusion Request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'X-Local-Fusion-Debug: true' \
  -d '{
    "model": "openrouter/fusion",
    "messages": [{"role": "user", "content": "Compare ridge, lasso, and elastic-net regression."}]
  }'
```

When debug is enabled, the response includes a top-level `local_fusion` object with request id, model names, latency, failed models, and degradation state. It does not include user prompt text or panel response text.

Example debug metadata shape:

```json
{
  "local_fusion": {
    "request_id": "3b4b1032669d4611addebf59219c30df",
    "panel_models": ["gemini-panel-a"],
    "judge_model": "gemini-judge",
    "failed_models": [],
    "analysis_present": true,
    "degraded_reason": null,
    "latency_ms": {
      "total": 3312.56,
      "panels": [
        {
          "model": "gemini-panel-a",
          "success": true,
          "latency_ms": 1063.68,
          "status_code": 200
        }
      ],
      "judge": {
        "model": "gemini-judge",
        "success": true,
        "latency_ms": 1233.56,
        "status_code": 200,
        "retry_without_response_format": false
      },
      "synthesis": {
        "model": "gemini-judge",
        "success": true,
        "latency_ms": 1014.71,
        "status_code": 200,
        "retry_without_response_format": false
      }
    }
  }
}
```

## Gemini API Smoke Test

Gemini's OpenAI-compatible endpoint can be used as a quick external backend test. This is optional and is not part of CI. The official base URL is:

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

Run a direct proxy smoke test through the Gateway code without starting a server:

```bash
export GEMINI_API_KEY="..."
uv run --extra dev python scripts/smoke_gemini.py
```

Run the Fusion path too:

```bash
uv run --extra dev python scripts/smoke_gemini.py --fusion
```

If the preliminary proxy request hits a transient Gemini `503 high demand`, skip it and run only the Fusion path:

```bash
uv run --extra dev python scripts/smoke_gemini.py --fusion-only
```

Show local Fusion debug metadata in the smoke response:

```bash
uv run --extra dev python scripts/smoke_gemini.py --fusion-only --debug
```

The sample config defaults to `models/gemini-2.5-flash-lite` because it is inexpensive and suitable for smoke tests. You can temporarily override the Gemini model used by all logical roles:

```bash
uv run --extra dev python scripts/smoke_gemini.py --fusion-only --model gemini-flash-latest
```

The Fusion smoke test uses one panel model plus one judge model, both configured in `config.gemini.example.yaml`. Because the same Gemini model is used for both logical roles, this is only a connectivity and pipeline test, not a quality benchmark.

## GSM8K Local Tiny Benchmark

Use this benchmark to compare single local Ollama models against the Fusion pipeline on
GSM8K CoT 8-shot strict-match. The first recommended run is limited to 128 questions.

The benchmark config uses three very small Ollama models:

- `qwen2.5:0.5b`
- `smollm2:135m`
- `gemma3:270m`

Install benchmark dependencies:

```bash
uv sync --extra dev --extra bench
```

Check the exact `lm_eval` commands without running them:

```bash
uv run --extra bench python scripts/benchmark_gsm8k_local.py --dry-run
```

Run a 2-question smoke benchmark and pull any missing Ollama models:

```bash
uv run --extra bench python scripts/benchmark_gsm8k_local.py --limit 2 --pull-missing
```

Run the planned 128-question comparison:

```bash
uv run --extra bench python scripts/benchmark_gsm8k_local.py \
  --limit 128 \
  --pull-missing \
  --request-timeout-seconds 1200
```

Results are written under `benchmark-results/gsm8k-local-tiny/<timestamp>/`, which is
gitignored. `summary.md` and `summary.json` report `strict-match` as the primary metric
and `flexible-extract` as a reference metric. The Fusion target uses `openrouter/fusion`
with all three tiny models as panel models and `qwen2.5:0.5b` as judge/synthesis model.

Fusion is much slower than the single-model targets because each benchmark item runs the
three panel models plus judge and synthesis calls. Long GSM8K prompts can exceed the
normal Gateway timeout, so the benchmark runner sets `LOCAL_FUSION_REQUEST_TIMEOUT_SECONDS`
for its child Gateway process. Use `--request-timeout-seconds` to tune that value for your
machine.

## Explicit Fusion Tool Request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "ollama-qwen",
    "tool_choice": "required",
    "messages": [{"role": "user", "content": "Compare ridge, lasso, and elastic-net regression."}],
    "tools": [
      {
        "type": "openrouter:fusion",
        "parameters": {
          "analysis_models": ["ollama-qwen", "lmstudio-llama"],
          "model": "ollama-qwen"
        }
      }
    ]
  }'
```

## OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local-fusion",
)

response = client.chat.completions.create(
    model="openrouter/fusion",
    messages=[
        {
            "role": "user",
            "content": "Compare ridge, lasso, and elastic-net regression.",
        }
    ],
)
print(response.choices[0].message.content)
```

## Development

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pytest
```

The test suite mocks all downstream LLM HTTP calls, so local LLM servers are not required for automated tests.

Current verification baseline:

- `uv run --extra dev pytest` passes with 19 tests.
- `uv run --extra dev ruff check .` passes.
- `uv run --extra dev ruff format --check .` passes.
