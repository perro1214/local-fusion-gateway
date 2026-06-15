from __future__ import annotations

import argparse
import os
import sys

from fastapi.testclient import TestClient

from local_fusion_gateway.app import create_app
from local_fusion_gateway.config import load_config


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
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set.", file=sys.stderr)
        print("Set it first: export GEMINI_API_KEY='...'", file=sys.stderr)
        return 2

    client = TestClient(create_app(load_config(args.config)))
    proxy_response = client.post(
        "/v1/chat/completions",
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

    if not args.fusion:
        return 0

    fusion_response = client.post(
        "/v1/chat/completions",
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


if __name__ == "__main__":
    raise SystemExit(main())
