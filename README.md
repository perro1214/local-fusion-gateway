# Local Fusion Gateway

OpenAI-compatible local gateway for running a Fusion-style panel, judge, and synthesis flow against local LLM servers such as Ollama, LM Studio, and llama.cpp.

The implementation follows the local v1 policy in `../方針.md`.

## Features

- `GET /health`
- `POST /v1/chat/completions`
- Plain OpenAI-compatible proxy for configured local models
- Fusion execution when `model` is `openrouter/fusion`
- Fusion execution when `tool_choice` is `required` and `tools` contains `openrouter:fusion`
- Parallel panel calls with partial-failure tolerance
- Judge JSON parsing with degraded fallback when the judge does not return valid JSON
- Non-streaming chat completions only

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
  -d '{
    "model": "openrouter/fusion",
    "messages": [{"role": "user", "content": "Compare ridge, lasso, and elastic-net regression."}]
  }'
```

## Gemini API Smoke Test

Gemini's OpenAI-compatible endpoint can be used as a quick external backend test. The official base URL is:

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

You can also temporarily override the Gemini model used by all logical roles:

```bash
uv run --extra dev python scripts/smoke_gemini.py --fusion-only --model gemini-flash-latest
```

The Fusion smoke test uses one panel model plus one judge model, both configured in `config.gemini.example.yaml`. Because the same Gemini model is used for both logical roles, this is only a connectivity and pipeline test, not a quality benchmark.

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
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev ruff format .
```

The test suite mocks all downstream LLM HTTP calls, so local LLM servers are not required for automated tests.
