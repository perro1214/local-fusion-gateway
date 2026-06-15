from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request

from .client import BackendClient
from .config import GatewayConfig, UnknownModelError, load_config_from_env
from .fusion import FusionError, FusionOrchestrator, is_fusion_request
from .schemas import ChatCompletionRequest


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    app = FastAPI(title="Local Fusion Gateway", version="0.1.0")
    app.state.config = config
    app.state.config_error = None

    if config is None:
        try:
            app.state.config = load_config_from_env()
        except FileNotFoundError as exc:
            app.state.config_error = exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        gateway_config = _require_config(raw_request.app)
        if is_fusion_request(request):
            orchestrator = FusionOrchestrator(gateway_config)
            try:
                fusion_result = await orchestrator.run(request)
            except FusionError as exc:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": str(exc),
                        "type": "fusion_error",
                        "failure_reason": exc.failure_reason,
                        "failed_models": [
                            {
                                "model": failed.model,
                                "error": failed.error,
                                "status_code": failed.status_code,
                            }
                            for failed in exc.failed_models
                        ],
                    },
                ) from exc
            return fusion_result.final_payload

        try:
            backend_model = gateway_config.resolve_model(request.model)
        except UnknownModelError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": str(exc),
                    "type": "unknown_model",
                    "model": exc.model,
                },
            ) from exc

        payload = request.to_backend_payload(backend_model.model)
        client = BackendClient(gateway_config.server.request_timeout_seconds)
        try:
            result = await client.chat_completion(request.model, backend_model, payload)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail={
                    "message": exc.response.text,
                    "type": "upstream_error",
                    "model": request.model,
                },
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": str(exc),
                    "type": "upstream_connection_error",
                    "model": request.model,
                },
            ) from exc
        return result.payload

    return app


def _require_config(app: FastAPI) -> GatewayConfig:
    if app.state.config is not None:
        return app.state.config
    raise HTTPException(
        status_code=500,
        detail={
            "message": f"Gateway config is not available: {app.state.config_error}",
            "type": "configuration_error",
        },
    )
