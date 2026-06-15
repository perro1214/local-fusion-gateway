from __future__ import annotations

import json

import httpx
import respx
from fastapi.testclient import TestClient

from local_fusion_gateway.app import create_app
from local_fusion_gateway.config import GatewayConfig


def make_fusion_config() -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "server": {"request_timeout_seconds": 5},
            "fusion": {
                "default_analysis_models": ["panel-a", "panel-b"],
                "default_judge_model": "judge",
            },
            "models": {
                "panel-a": {
                    "base_url": "http://panel-a.test/v1",
                    "api_key": "panel-a-key",
                    "model": "actual-panel-a",
                },
                "panel-b": {
                    "base_url": "http://panel-b.test/v1",
                    "api_key": "panel-b-key",
                    "model": "actual-panel-b",
                },
                "judge": {
                    "base_url": "http://judge.test/v1",
                    "api_key": "judge-key",
                    "model": "actual-judge",
                },
            },
        }
    )


def completion(content: str, model: str = "mock-model") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-{model}",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def valid_analysis() -> str:
    return json.dumps(
        {
            "consensus": ["shared"],
            "contradictions": [],
            "partial_coverage": [],
            "unique_insights": [],
            "blind_spots": [],
        }
    )


@respx.mock
def test_model_slug_runs_fusion_panel_judge_and_synthesis() -> None:
    panel_a = respx.post("http://panel-a.test/v1/chat/completions").mock(
        return_value=completion("panel a", "actual-panel-a")
    )
    panel_b = respx.post("http://panel-b.test/v1/chat/completions").mock(
        return_value=completion("panel b", "actual-panel-b")
    )
    judge = respx.post("http://judge.test/v1/chat/completions").mock(
        side_effect=[
            completion(valid_analysis(), "actual-judge"),
            completion("final", "actual-judge"),
        ]
    )
    client = TestClient(create_app(make_fusion_config()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openrouter/fusion",
            "messages": [{"role": "user", "content": "compare options"}],
            "temperature": 0.1,
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "openrouter/fusion"
    assert response.json()["choices"][0]["message"]["content"] == "final"
    assert panel_a.called
    assert panel_b.called
    assert judge.call_count == 2
    assert json.loads(panel_a.calls.last.request.content)["model"] == "actual-panel-a"
    assert json.loads(panel_a.calls.last.request.content)["stream"] is False


@respx.mock
def test_required_fusion_tool_uses_parameter_models() -> None:
    panel_a = respx.post("http://panel-a.test/v1/chat/completions").mock(
        return_value=completion("panel a", "actual-panel-a")
    )
    panel_b = respx.post("http://panel-b.test/v1/chat/completions").mock(
        return_value=completion("panel b", "actual-panel-b")
    )
    judge = respx.post("http://judge.test/v1/chat/completions").mock(
        side_effect=[
            completion(valid_analysis(), "actual-judge"),
            completion("final", "actual-judge"),
        ]
    )
    client = TestClient(create_app(make_fusion_config()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "panel-a",
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "compare options"}],
            "tools": [
                {
                    "type": "openrouter:fusion",
                    "parameters": {
                        "analysis_models": ["panel-a"],
                        "model": "judge",
                        "tools": [{"type": "openrouter:web_search"}],
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "panel-a"
    assert panel_a.called
    assert not panel_b.called
    assert judge.call_count == 2


@respx.mock
def test_partial_panel_failure_continues_with_successful_panels() -> None:
    respx.post("http://panel-a.test/v1/chat/completions").mock(
        return_value=completion("panel a", "actual-panel-a")
    )
    respx.post("http://panel-b.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="panel b failed")
    )
    judge = respx.post("http://judge.test/v1/chat/completions").mock(
        side_effect=[
            completion(valid_analysis(), "actual-judge"),
            completion("final", "actual-judge"),
        ]
    )
    client = TestClient(create_app(make_fusion_config()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openrouter/fusion",
            "messages": [{"role": "user", "content": "compare options"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "final"
    assert judge.call_count == 2


@respx.mock
def test_all_panel_failures_return_hard_failure() -> None:
    respx.post("http://panel-a.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="panel a failed")
    )
    respx.post("http://panel-b.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="panel b failed")
    )
    client = TestClient(create_app(make_fusion_config()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openrouter/fusion",
            "messages": [{"role": "user", "content": "compare options"}],
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"]["failure_reason"] == "all_panels_failed"
    assert len(response.json()["detail"]["failed_models"]) == 2


@respx.mock
def test_invalid_judge_json_degrades_to_panel_only_synthesis() -> None:
    respx.post("http://panel-a.test/v1/chat/completions").mock(
        return_value=completion("panel a", "actual-panel-a")
    )
    respx.post("http://panel-b.test/v1/chat/completions").mock(
        return_value=completion("panel b", "actual-panel-b")
    )
    judge = respx.post("http://judge.test/v1/chat/completions").mock(
        side_effect=[completion("not json", "actual-judge"), completion("final", "actual-judge")]
    )
    client = TestClient(create_app(make_fusion_config()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openrouter/fusion",
            "messages": [{"role": "user", "content": "compare options"}],
        },
    )

    assert response.status_code == 200
    synthesis_payload = json.loads(judge.calls.last.request.content)
    synthesis_user_payload = json.loads(synthesis_payload["messages"][1]["content"])
    assert synthesis_user_payload["analysis"] is None
    assert response.json()["choices"][0]["message"]["content"] == "final"
