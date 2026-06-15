from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from .client import BackendClient
from .config import GatewayConfig, UnknownModelError, load_config_from_env
from .fusion import FusionError, FusionOrchestrator, is_fusion_request
from .schemas import ChatCompletionRequest

LOGGER = logging.getLogger(__name__)
REQUEST_ID_HEADER = "X-Request-ID"
DEBUG_HEADER = "X-Local-Fusion-Debug"


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
        response: Response,
    ) -> dict[str, Any]:
        request_id = _request_id(raw_request)
        response.headers[REQUEST_ID_HEADER] = request_id
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Streaming chat completions are not supported in v1.",
                    "type": "streaming_unsupported",
                },
                headers=_request_id_headers(request_id),
            )

        gateway_config = _require_config(raw_request.app, request_id)
        if is_fusion_request(request):
            orchestrator = FusionOrchestrator(gateway_config)
            try:
                fusion_result = await orchestrator.run(request, request_id=request_id)
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
                    headers=_request_id_headers(request_id),
                ) from exc
            payload = dict(fusion_result.final_payload)
            if _debug_enabled(raw_request):
                payload["local_fusion"] = fusion_result.trace.to_debug_metadata()
            return payload

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
                headers=_request_id_headers(request_id),
            ) from exc

        payload = request.to_backend_payload(backend_model.model)
        client = BackendClient(gateway_config.server.request_timeout_seconds)
        start = time.perf_counter()
        try:
            result = await client.chat_completion(request.model, backend_model, payload)
        except httpx.HTTPStatusError as exc:
            _log_proxy_event(
                request_id=request_id,
                logical_model=request.model,
                backend_model=backend_model.model,
                success=False,
                latency_ms=_elapsed_ms(start),
                status_code=exc.response.status_code,
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail={
                    "message": exc.response.text,
                    "type": "upstream_error",
                    "model": request.model,
                },
                headers=_request_id_headers(request_id),
            ) from exc
        except httpx.HTTPError as exc:
            _log_proxy_event(
                request_id=request_id,
                logical_model=request.model,
                backend_model=backend_model.model,
                success=False,
                latency_ms=_elapsed_ms(start),
                error=exc.__class__.__name__,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "message": str(exc),
                    "type": "upstream_connection_error",
                    "model": request.model,
                },
                headers=_request_id_headers(request_id),
            ) from exc
        _log_proxy_event(
            request_id=request_id,
            logical_model=request.model,
            backend_model=backend_model.model,
            success=True,
            latency_ms=_elapsed_ms(start),
            status_code=result.status_code,
        )
        return result.payload

    return app


def _require_config(app: FastAPI, request_id: str) -> GatewayConfig:
    if app.state.config is not None:
        return app.state.config
    raise HTTPException(
        status_code=500,
        detail={
            "message": f"Gateway config is not available: {app.state.config_error}",
            "type": "configuration_error",
        },
        headers=_request_id_headers(request_id),
    )


def _request_id(request: Request) -> str:
    return request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex


def _request_id_headers(request_id: str) -> dict[str, str]:
    return {REQUEST_ID_HEADER: request_id}


def _debug_enabled(request: Request) -> bool:
    return request.headers.get(DEBUG_HEADER, "").lower() in {"1", "true", "yes", "on"}


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _log_proxy_event(
    *,
    request_id: str,
    logical_model: str,
    backend_model: str,
    success: bool,
    latency_ms: float,
    status_code: int | None = None,
    error: str | None = None,
) -> None:
    LOGGER.info(
        "local_fusion_event %s",
        json.dumps(
            {
                "event": "proxy_completed",
                "request_id": request_id,
                "logical_model": logical_model,
                "backend_model": backend_model,
                "success": success,
                "latency_ms": round(latency_ms, 2),
                "status_code": status_code,
                "error": error,
            },
            sort_keys=True,
        ),
    )
