from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    request_timeout_seconds: float = 120


class FusionConfig(BaseModel):
    default_analysis_models: list[str] = Field(default_factory=list)
    default_judge_model: str | None = None


class ModelConfig(BaseModel):
    base_url: str
    api_key: str = "local"
    model: str

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


class GatewayConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    def resolve_model(self, logical_name: str) -> ModelConfig:
        try:
            return self.models[logical_name]
        except KeyError as exc:
            raise UnknownModelError(logical_name) from exc


class UnknownModelError(ValueError):
    def __init__(self, model: str) -> None:
        super().__init__(f"Unknown local model: {model}")
        self.model = model


def load_config(path: str | os.PathLike[str]) -> GatewayConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    raw = _expand_env_vars(raw)
    return GatewayConfig.model_validate(raw)


def load_config_from_env() -> GatewayConfig:
    return load_config(os.environ.get("LOCAL_FUSION_CONFIG", "config.yaml"))


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value
