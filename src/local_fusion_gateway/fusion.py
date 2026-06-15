from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .client import BackendClient, BackendResult
from .config import GatewayConfig, ModelConfig, UnknownModelError
from .schemas import ChatCompletionRequest

LOGGER = logging.getLogger(__name__)
FUSION_MODEL = "openrouter/fusion"
FUSION_TOOL = "openrouter:fusion"
WEB_TOOL_PREFIX = "openrouter:web_"
MAX_ANALYSIS_MODELS = 8
FUSION_ANALYSIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "consensus": {"type": "array", "items": {"type": "string"}},
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "stances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string"},
                                "stance": {"type": "string"},
                            },
                            "required": ["model", "stance"],
                        },
                    },
                },
                "required": ["topic", "stances"],
            },
        },
        "partial_coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "models": {"type": "array", "items": {"type": "string"}},
                    "point": {"type": "string"},
                },
                "required": ["models", "point"],
            },
        },
        "unique_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "insight": {"type": "string"},
                },
                "required": ["model", "insight"],
            },
        },
        "blind_spots": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "consensus",
        "contradictions",
        "partial_coverage",
        "unique_insights",
        "blind_spots",
    ],
}


@dataclass(slots=True)
class FailedModel:
    model: str
    error: str
    status_code: int | None = None


@dataclass(slots=True)
class PanelResponse:
    model: str
    content: str


@dataclass(slots=True)
class FusionSettings:
    analysis_models: list[str]
    judge_model: str
    temperature: float | None = None
    max_completion_tokens: int | None = None
    max_tool_calls: int | None = None
    reasoning: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class FusionResult:
    final_payload: dict[str, Any]
    panel_responses: list[PanelResponse]
    failed_models: list[FailedModel]
    analysis: dict[str, Any] | None
    degraded_reason: str | None = None


class FusionError(RuntimeError):
    def __init__(
        self,
        message: str,
        failure_reason: str,
        failed_models: list[FailedModel] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_reason = failure_reason
        self.failed_models = failed_models or []


def is_fusion_request(request: ChatCompletionRequest) -> bool:
    if request.model == FUSION_MODEL:
        return True
    return request.tool_choice == "required" and _find_fusion_tool(request.tools) is not None


def resolve_fusion_settings(
    config: GatewayConfig, request: ChatCompletionRequest
) -> FusionSettings:
    fusion_tool = _find_fusion_tool(request.tools)
    parameters = fusion_tool.get("parameters", {}) if fusion_tool else {}
    if not isinstance(parameters, dict):
        parameters = {}

    analysis_models = parameters.get("analysis_models") or config.fusion.default_analysis_models
    if not isinstance(analysis_models, list) or not analysis_models:
        raise FusionError(
            "Fusion requires at least one analysis model.",
            failure_reason="missing_analysis_models",
        )
    if len(analysis_models) > MAX_ANALYSIS_MODELS:
        raise FusionError(
            "Fusion analysis_models is capped at 8 models.",
            failure_reason="too_many_analysis_models",
        )
    if not all(isinstance(model, str) and model for model in analysis_models):
        raise FusionError(
            "Fusion analysis_models must be non-empty strings.",
            failure_reason="invalid_analysis_models",
        )

    judge_model = parameters.get("model") or config.fusion.default_judge_model or analysis_models[0]
    if not isinstance(judge_model, str) or not judge_model:
        raise FusionError(
            "Fusion judge model must be a string.", failure_reason="invalid_judge_model"
        )
    try:
        config.resolve_model(judge_model)
    except UnknownModelError as exc:
        raise FusionError(str(exc), failure_reason="invalid_judge_model") from exc

    tools = parameters.get("tools") or []
    if not isinstance(tools, list):
        tools = []
    _log_unsupported_web_tools(tools)

    return FusionSettings(
        analysis_models=analysis_models,
        judge_model=judge_model,
        temperature=_first_float(parameters.get("temperature"), request.temperature),
        max_completion_tokens=_first_int(
            parameters.get("max_completion_tokens"),
            request.max_completion_tokens,
            request.max_tokens,
        ),
        max_tool_calls=_first_int(parameters.get("max_tool_calls")),
        reasoning=parameters.get("reasoning")
        if isinstance(parameters.get("reasoning"), dict)
        else None,
        tools=tools,
    )


class FusionOrchestrator:
    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._client = BackendClient(config.server.request_timeout_seconds)

    async def run(self, request: ChatCompletionRequest) -> FusionResult:
        settings = resolve_fusion_settings(self._config, request)
        panel_results, failed_models = await self._run_panel(request, settings)
        if not panel_results:
            raise FusionError(
                "All Fusion panel models failed.",
                failure_reason="all_panels_failed",
                failed_models=failed_models,
            )

        analysis: dict[str, Any] | None = None
        degraded_reason: str | None = None
        try:
            judge_result = await self._run_judge(request, settings, panel_results)
            analysis = parse_analysis_json(judge_result.content)
        except (httpx.HTTPError, UnknownModelError, ValueError) as exc:
            degraded_reason = _judge_failure_reason(exc)
            LOGGER.warning("Fusion judge degraded: %s", exc)

        final_result = await self._run_synthesis(request, settings, panel_results, analysis)
        final_payload = dict(final_result.payload)
        final_payload["model"] = request.model
        return FusionResult(
            final_payload=final_payload,
            panel_responses=panel_results,
            failed_models=failed_models,
            analysis=analysis,
            degraded_reason=degraded_reason,
        )

    async def _run_panel(
        self,
        request: ChatCompletionRequest,
        settings: FusionSettings,
    ) -> tuple[list[PanelResponse], list[FailedModel]]:
        tasks = [
            self._run_panel_model(model_name, request, settings)
            for model_name in settings.analysis_models
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        responses: list[PanelResponse] = []
        failed_models: list[FailedModel] = []
        for model_name, raw_result in zip(settings.analysis_models, raw_results, strict=True):
            if isinstance(raw_result, Exception):
                failed_models.append(_failed_model(model_name, raw_result))
                continue
            responses.append(PanelResponse(model=model_name, content=raw_result.content))
        return responses, failed_models

    async def _run_panel_model(
        self,
        model_name: str,
        request: ChatCompletionRequest,
        settings: FusionSettings,
    ) -> BackendResult:
        model = self._config.resolve_model(model_name)
        payload = build_inner_payload(request, model, settings)
        return await self._client.chat_completion(model_name, model, payload)

    async def _run_judge(
        self,
        request: ChatCompletionRequest,
        settings: FusionSettings,
        panel_responses: list[PanelResponse],
    ) -> BackendResult:
        judge = self._config.resolve_model(settings.judge_model)
        payload = build_judge_payload(
            request,
            judge,
            settings,
            panel_responses,
            use_response_format=True,
        )
        try:
            return await self._client.chat_completion(settings.judge_model, judge, payload)
        except httpx.HTTPStatusError as exc:
            if not _should_retry_without_response_format(exc):
                raise
            LOGGER.info(
                "Retrying Fusion judge without response_format after upstream %s.",
                exc.response.status_code,
            )
        fallback_payload = build_judge_payload(
            request,
            judge,
            settings,
            panel_responses,
            use_response_format=False,
        )
        return await self._client.chat_completion(
            settings.judge_model,
            judge,
            fallback_payload,
        )

    async def _run_synthesis(
        self,
        request: ChatCompletionRequest,
        settings: FusionSettings,
        panel_responses: list[PanelResponse],
        analysis: dict[str, Any] | None,
    ) -> BackendResult:
        judge = self._config.resolve_model(settings.judge_model)
        payload = build_synthesis_payload(request, judge, settings, panel_responses, analysis)
        return await self._client.chat_completion(settings.judge_model, judge, payload)


def build_inner_payload(
    request: ChatCompletionRequest,
    backend_model: ModelConfig,
    settings: FusionSettings,
) -> dict[str, Any]:
    payload = request.to_backend_payload(backend_model.model)
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload["stream"] = False
    _apply_inner_generation_settings(payload, settings)
    return payload


def build_judge_payload(
    request: ChatCompletionRequest,
    judge_model: ModelConfig,
    settings: FusionSettings,
    panel_responses: list[PanelResponse],
    *,
    use_response_format: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": judge_model.model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the judge in a local Fusion pipeline. Compare panel "
                    "answers and return only JSON matching the requested schema. "
                    "Do not include prose, Markdown, code fences, or commentary."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_messages": request.messages,
                        "panel_responses": [asdict(response) for response in panel_responses],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    if use_response_format:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "fusion_analysis",
                "schema": FUSION_ANALYSIS_JSON_SCHEMA,
            },
        }
    _apply_inner_generation_settings(payload, settings)
    return payload


def build_synthesis_payload(
    request: ChatCompletionRequest,
    judge_model: ModelConfig,
    settings: FusionSettings,
    panel_responses: list[PanelResponse],
    analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": judge_model.model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the final synthesis model in a local Fusion pipeline. "
                    "Answer the user directly using the panel responses and judge analysis. "
                    "If analysis is null, synthesize from the panel responses only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_messages": request.messages,
                        "analysis": analysis,
                        "panel_responses": [asdict(response) for response in panel_responses],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    _apply_inner_generation_settings(payload, settings)
    return payload


def parse_analysis_json(content: str) -> dict[str, Any]:
    candidates = [content, *_extract_json_code_blocks(content)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if _is_valid_analysis(parsed):
            return parsed
        raise ValueError("judge_schema_mismatch")
    raise ValueError("judge_not_valid_json")


def _apply_inner_generation_settings(payload: dict[str, Any], settings: FusionSettings) -> None:
    if settings.temperature is not None:
        payload["temperature"] = settings.temperature
    if settings.max_completion_tokens is not None:
        payload["max_completion_tokens"] = settings.max_completion_tokens
    if settings.reasoning is not None:
        payload["reasoning"] = settings.reasoning


def _find_fusion_tool(tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not tools:
        return None
    for tool in tools:
        if tool.get("type") == FUSION_TOOL:
            return tool
    return None


def _extract_json_code_blocks(content: str) -> list[str]:
    return re.findall(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)


def _is_valid_analysis(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    expected_list_keys = {
        "consensus",
        "contradictions",
        "partial_coverage",
        "unique_insights",
        "blind_spots",
    }
    return all(isinstance(parsed.get(key), list) for key in expected_list_keys)


def _first_float(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, int | float):
            return float(value)
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
    return None


def _failed_model(model_name: str, exc: Exception) -> FailedModel:
    if isinstance(exc, httpx.HTTPStatusError):
        return FailedModel(
            model=model_name, error=exc.response.text, status_code=exc.response.status_code
        )
    return FailedModel(model=model_name, error=str(exc))


def _judge_failure_reason(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        return "judge_upstream_error"
    return "judge_empty_completion"


def _should_retry_without_response_format(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code in {400, 422}


def _log_unsupported_web_tools(tools: list[dict[str, Any]]) -> None:
    unsupported = [
        tool.get("type") for tool in tools if str(tool.get("type", "")).startswith(WEB_TOOL_PREFIX)
    ]
    if unsupported:
        LOGGER.info("Ignoring unsupported v1 web tools: %s", unsupported)
