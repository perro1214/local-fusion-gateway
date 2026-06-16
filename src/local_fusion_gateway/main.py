from __future__ import annotations

import os

import uvicorn

from .app import create_app
from .config import GatewayConfig, load_config


def main() -> None:
    config_path = os.environ.get("LOCAL_FUSION_CONFIG", "config.yaml")
    config = _load_optional_config(config_path)
    if config is not None:
        _apply_env_overrides(config)
    host = os.environ.get("LOCAL_FUSION_HOST") or (config.server.host if config else "127.0.0.1")
    port = int(os.environ.get("LOCAL_FUSION_PORT") or (config.server.port if config else 8080))
    uvicorn.run(
        create_app(config),
        host=host,
        port=port,
        reload=False,
        log_level=os.environ.get("LOCAL_FUSION_LOG_LEVEL", "info"),
        lifespan="auto",
    )


def _load_optional_config(config_path: str) -> GatewayConfig | None:
    try:
        return load_config(config_path)
    except FileNotFoundError:
        return None


def _apply_env_overrides(config: GatewayConfig) -> None:
    timeout_seconds = os.environ.get("LOCAL_FUSION_REQUEST_TIMEOUT_SECONDS")
    if timeout_seconds is not None:
        config.server.request_timeout_seconds = float(timeout_seconds)


if __name__ == "__main__":
    main()
