from __future__ import annotations

import argparse
import os
import sys

from fastapi.testclient import TestClient

from local_fusion_gateway.app import create_app
from local_fusion_gateway.config import GatewayConfig, load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small Gemini API smoke test through the gateway."
    )
    parser.add_argument(
        "--config",
        default="config.gemini.example.yaml",
        help="Gateway YAML config to load.",
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Also run the Fusion path. This uses three Gemini calls.",
    )
    parser.add_argument(
        "--fusion-only",
        action="store_true",
        help="Run only the Fusion path, skipping the preliminary proxy call.",
    )
    parser.add_argument(
        "--model",
        help="Override every configured backend model name for this smoke test.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ask the gateway to include local_fusion debug metadata.",
    )
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set.", file=sys.stderr)
        print("Set it first: export GEMINI_API_KEY='...'", file=sys.stderr)
        return 2

    config = load_config(args.config)
    if args.model:
        _override_backend_model(config, args.model)

    client = TestClient(create_app(config))
    headers = {"X-Local-Fusion-Debug": "true"} if args.debug else {}
    if not args.fusion_only:
        proxy_response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "gemini-panel-a",
                "messages": [{"role": "user", "content": "Reply with exactly: gateway ok"}],
                "temperature": 0,
                "max_completion_tokens": 32,
            },
        )
        print("proxy status:", proxy_response.status_code)
        print(proxy_response.text)
        if proxy_response.status_code >= 400:
            return 1

    if not (args.fusion or args.fusion_only):
        return 0

    fusion_response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "openrouter/fusion",
            "messages": [
                {
                    "role": "user",
                    "content": "In one short sentence, compare JSON and YAML.",
                }
            ],
            "temperature": 0,
            "max_completion_tokens": 128,
        },
    )
    print("fusion status:", fusion_response.status_code)
    print(fusion_response.text)
    return 0 if fusion_response.status_code < 400 else 1


def _override_backend_model(config: GatewayConfig, model_name: str) -> None:
    for model in config.models.values():
        model.model = model_name


if __name__ == "__main__":
    raise SystemExit(main())
