from __future__ import annotations

import os

import uvicorn

from .app import create_app
from .config import GatewayConfig, load_config


def main() -> None:
    config_path = os.environ.get("LOCAL_FUSION_CONFIG", "config.yaml")
    config = _load_optional_config(config_path)
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


if __name__ == "__main__":
    main()
