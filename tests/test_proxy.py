from __future__ import annotations

import json

import httpx
import respx
from fastapi.testclient import TestClient

from local_fusion_gateway.app import create_app
from local_fusion_gateway.config import GatewayConfig


def make_config() -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "server": {"request_timeout_seconds": 5},
            "models": {
                "local-a": {
                    "base_url": "http://backend-a.test/v1",
                    "api_key": "secret-a",
                    "model": "actual-a",
                }
            },
        }
    )


@respx.mock
def test_proxy_rewrites_logical_model_to_backend_model() -> None:
    route = respx.post("http://backend-a.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-proxy",
                "object": "chat.completion",
                "created": 1,
                "model": "actual-a",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "proxied"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    client = TestClient(create_app(make_config()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-a",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "proxied"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer secret-a"
    payload = json.loads(request.content)
    assert payload["model"] == "actual-a"
    assert payload["temperature"] == 0.2


def test_unknown_model_returns_404() -> None:
    client = TestClient(create_app(make_config()))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "missing", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["type"] == "unknown_model"


def test_missing_config_only_blocks_chat_endpoint() -> None:
    client = TestClient(create_app())

    health = client.get("/health")
    chat = client.post(
        "/v1/chat/completions",
        json={"model": "local-a", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert health.status_code == 200
    assert chat.status_code == 500
    assert chat.json()["detail"]["type"] == "configuration_error"


def test_streaming_request_returns_400() -> None:
    client = TestClient(create_app(make_config()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-a",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "streaming_unsupported"
