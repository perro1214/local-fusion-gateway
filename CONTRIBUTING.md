# Contributing

Thanks for taking a look at Local Fusion Gateway. This project is still alpha, so small, well-scoped changes are preferred.

## Development

Install dependencies:

```bash
uv sync --extra dev
```

Run the local checks before opening a pull request:

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pytest
```

The test suite mocks downstream LLM calls. Do not require real local LLM servers or external API keys for normal tests.

## Pull Requests

- Keep changes focused on one behavior or one documentation topic.
- Add or update tests for behavior changes.
- Do not commit `config.yaml`, `.env`, API keys, local benchmark outputs, or captured model responses with private prompts.
- Mention any manually verified backend, such as Ollama, LM Studio, llama.cpp, or Gemini's OpenAI-compatible endpoint.

## Commit Style

Use short imperative commit messages. The existing history uses prefixes such as:

- `feat:`
- `fix:`
- `test:`
- `docs:`
- `chore:`
