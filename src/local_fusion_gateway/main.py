from __future__ import annotations

import os

import uvicorn


def main() -> None:
    config_path = os.environ.get("LOCAL_FUSION_CONFIG", "config.yaml")
    host = os.environ.get("LOCAL_FUSION_HOST", "127.0.0.1")
    port = int(os.environ.get("LOCAL_FUSION_PORT", "8080"))
    uvicorn.run(
        "local_fusion_gateway.app:create_app",
        host=host,
        port=port,
        factory=True,
        reload=False,
        env_file=None,
        app_dir=None,
        log_level=os.environ.get("LOCAL_FUSION_LOG_LEVEL", "info"),
        lifespan="auto",
        headers=[("x-local-fusion-config", config_path)],
    )


if __name__ == "__main__":
    main()

