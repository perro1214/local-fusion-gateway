from __future__ import annotations

from pathlib import Path

from local_fusion_gateway.config import GatewayConfig, load_config


def test_config_model_validation() -> None:
    config = GatewayConfig.model_validate(
        {
            "models": {
                "gemini": {
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "api_key": "key",
                    "model": "gemini-3.5-flash",
                }
            }
        }
    )

    assert (
        config.models["gemini"].chat_completions_url
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


def test_load_config_expands_environment_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  gemini:
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-3.5-flash"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.models["gemini"].api_key == "test-gemini-key"
