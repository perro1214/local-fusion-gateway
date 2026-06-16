from __future__ import annotations

import asyncio
import json
import logging
import re
import time
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
class PanelTrace:
    model: str
    success: bool
    latency_ms: float
    status_code: int | None = None
    error: str | None = None


@dataclass(slots=True)
class StepTrace:
    model: str
    success: bool
    latency_ms: float
    status_code: int | None = None
    error: str | None = None
    retry_without_response_format: bool = False


@dataclass(slots=True)
class FusionTrace:
    request_id: str | None
    panel_models: list[str]
    judge_model: str
    panels: list[PanelTrace] = field(default_factory=list)
    judge: StepTrace | None = None
    synthesis: StepTrace | None = None
    failed_models: list[FailedModel] = field(default_factory=list)
    analysis_present: bool = False
    degraded_reason: str | None = None
    total_latency_ms: float = 0

    def to_debug_metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "panel_models": self.panel_models,
            "judge_model": self.judge_model,
            "failed_models": [asdict(failed) for failed in self.failed_models],
            "analysis_present": self.analysis_present,
            "degraded_reason": self.degraded_reason,
            "latency_ms": {
                "total": _round_ms(self.total_latency_ms),
                "panels": [
                    {
                        "model": panel.model,
                        "success": panel.success,
                        "latency_ms": _round_ms(panel.latency_ms),
                        "status_code": panel.status_code,
                    }
                    for panel in self.panels
                ],
                "judge": _step_latency_metadata(self.judge),
                "synthesis": _step_latency_metadata(self.synthesis),
            },
        }


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
    trace: FusionTrace
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


class StepExecutionError(RuntimeError):
    def __init__(self, original: Exception, trace: StepTrace) -> None:
        super().__init__(str(original))
        self.original = original
        self.trace = trace


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

    async def run(
        self,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> FusionResult:
        total_start = time.perf_counter()
        settings = resolve_fusion_settings(self._config, request)
        trace = FusionTrace(
            request_id=request_id,
            panel_models=settings.analysis_models,
            judge_model=settings.judge_model,
        )
        panel_results, failed_models, panel_traces = await self._run_panel(request, settings)
        trace.panels = panel_traces
        trace.failed_models = failed_models
        if not panel_results:
            trace.total_latency_ms = _elapsed_ms(total_start)
            _log_fusion_trace("fusion_all_panels_failed", trace)
            raise FusionError(
                "All Fusion panel models failed.",
                failure_reason="all_panels_failed",
                failed_models=failed_models,
            )

        analysis: dict[str, Any] | None = None
        degraded_reason: str | None = None
        try:
            judge_result, judge_trace = await self._run_judge(request, settings, panel_results)
            trace.judge = judge_trace
            analysis = parse_analysis_json(judge_result.content)
        except StepExecutionError as exc:
            trace.judge = exc.trace
            degraded_reason = _judge_failure_reason(exc.original)
            trace.degraded_reason = degraded_reason
            LOGGER.warning("Fusion judge degraded: %s", exc.original)
        except (httpx.HTTPError, UnknownModelError, ValueError) as exc:
            degraded_reason = _judge_failure_reason(exc)
            trace.degraded_reason = degraded_reason
            LOGGER.warning("Fusion judge degraded: %s", exc)

        try:
            final_result, synthesis_trace = await self._run_synthesis(
                request,
                settings,
                panel_results,
                analysis,
            )
        except StepExecutionError as exc:
            trace.synthesis = exc.trace
            trace.degraded_reason = _synthesis_failure_reason(exc.original)
            trace.total_latency_ms = _elapsed_ms(total_start)
            _log_fusion_trace("fusion_synthesis_failed", trace)
            raise FusionError(
                "Fusion synthesis failed.",
                failure_reason=trace.degraded_reason,
                failed_models=[_failed_model(settings.judge_model, exc.original)],
            ) from exc.original
        trace.synthesis = synthesis_trace
        trace.analysis_present = analysis is not None
        trace.degraded_reason = degraded_reason
        trace.total_latency_ms = _elapsed_ms(total_start)
        _log_fusion_trace("fusion_completed", trace)
        final_payload = dict(final_result.payload)
        final_payload["model"] = request.model
        return FusionResult(
            final_payload=final_payload,
            panel_responses=panel_results,
            failed_models=failed_models,
            analysis=analysis,
            trace=trace,
            degraded_reason=degraded_reason,
        )

    async def _run_panel(
        self,
        request: ChatCompletionRequest,
        settings: FusionSettings,
    ) -> tuple[list[PanelResponse], list[FailedModel], list[PanelTrace]]:
        tasks = [
            self._run_panel_model(model_name, request, settings)
            for model_name in settings.analysis_models
        ]
        raw_results = await asyncio.gather(*tasks)
        responses: list[PanelResponse] = []
        failed_models: list[FailedModel] = []
        panel_traces: list[PanelTrace] = []
        for model_name, result, failed_model, panel_trace in raw_results:
            panel_traces.append(panel_trace)
            if failed_model is not None:
                failed_models.append(failed_model)
                continue
            if result is not None:
                responses.append(PanelResponse(model=model_name, content=result.content))
        return responses, failed_models, panel_traces

    async def _run_panel_model(
        self,
        model_name: str,
        request: ChatCompletionRequest,
        settings: FusionSettings,
    ) -> tuple[str, BackendResult | None, FailedModel | None, PanelTrace]:
        start = time.perf_counter()
        try:
            model = self._config.resolve_model(model_name)
            payload = build_inner_payload(request, model, settings)
            result = await self._client.chat_completion(model_name, model, payload)
            return (
                model_name,
                result,
                None,
                PanelTrace(
                    model=model_name,
                    success=True,
                    latency_ms=_elapsed_ms(start),
                    status_code=result.status_code,
                ),
            )
        except Exception as exc:
            failed_model = _failed_model(model_name, exc)
            return (
                model_name,
                None,
                failed_model,
                PanelTrace(
                    model=model_name,
                    success=False,
                    latency_ms=_elapsed_ms(start),
                    status_code=failed_model.status_code,
                    error=_safe_error(exc),
                ),
            )

    async def _run_judge(
        self,
        request: ChatCompletionRequest,
        settings: FusionSettings,
        panel_responses: list[PanelResponse],
    ) -> tuple[BackendResult, StepTrace]:
        judge = self._config.resolve_model(settings.judge_model)
        payload = build_judge_payload(
            request,
            judge,
            settings,
            panel_responses,
            use_response_format=True,
        )
        start = time.perf_counter()
        retry_without_response_format = False
        try:
            result = await self._client.chat_completion(settings.judge_model, judge, payload)
            return (
                result,
                StepTrace(
                    model=settings.judge_model,
                    success=True,
                    latency_ms=_elapsed_ms(start),
                    status_code=result.status_code,
                ),
            )
        except httpx.HTTPStatusError as exc:
            if not _should_retry_without_response_format(exc):
                raise StepExecutionError(
                    exc,
                    StepTrace(
                        model=settings.judge_model,
                        success=False,
                        latency_ms=_elapsed_ms(start),
                        status_code=exc.response.status_code,
                        error=_safe_error(exc),
                    ),
                ) from exc
            retry_without_response_format = True
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
        try:
            result = await self._client.chat_completion(
                settings.judge_model,
                judge,
                fallback_payload,
            )
        except httpx.HTTPError as exc:
            raise StepExecutionError(
                exc,
                StepTrace(
                    model=settings.judge_model,
                    success=False,
                    latency_ms=_elapsed_ms(start),
                    status_code=exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None,
                    error=_safe_error(exc),
                    retry_without_response_format=retry_without_response_format,
                ),
            ) from exc
        return (
            result,
            StepTrace(
                model=settings.judge_model,
                success=True,
                latency_ms=_elapsed_ms(start),
                status_code=result.status_code,
                retry_without_response_format=retry_without_response_format,
            ),
        )

    async def _run_synthesis(
        self,
        request: ChatCompletionRequest,
        settings: FusionSettings,
        panel_responses: list[PanelResponse],
        analysis: dict[str, Any] | None,
    ) -> tuple[BackendResult, StepTrace]:
        judge = self._config.resolve_model(settings.judge_model)
        payload = build_synthesis_payload(request, judge, settings, panel_responses, analysis)
        start = time.perf_counter()
        try:
            result = await self._client.chat_completion(settings.judge_model, judge, payload)
        except httpx.HTTPError as exc:
            raise StepExecutionError(
                exc,
                StepTrace(
                    model=settings.judge_model,
                    success=False,
                    latency_ms=_elapsed_ms(start),
                    status_code=exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None,
                    error=_safe_error(exc),
                ),
            ) from exc
        return (
            result,
            StepTrace(
                model=settings.judge_model,
                success=True,
                latency_ms=_elapsed_ms(start),
                status_code=result.status_code,
            ),
        )


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
    return FailedModel(model=model_name, error=str(exc) or exc.__class__.__name__)


def _judge_failure_reason(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, httpx.TimeoutException):
        return "judge_timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        return "judge_upstream_error"
    return "judge_empty_completion"


def _synthesis_failure_reason(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "synthesis_timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        return "synthesis_upstream_error"
    return "synthesis_connection_error"


def _should_retry_without_response_format(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code in {400, 422}


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _round_ms(value: float) -> float:
    return round(value, 2)


def _step_latency_metadata(step: StepTrace | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "model": step.model,
        "success": step.success,
        "latency_ms": _round_ms(step.latency_ms),
        "status_code": step.status_code,
        "retry_without_response_format": step.retry_without_response_format,
    }


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"upstream_status_{exc.response.status_code}"
    return exc.__class__.__name__


def _log_fusion_trace(event: str, trace: FusionTrace) -> None:
    LOGGER.info(
        "local_fusion_event %s",
        json.dumps(
            {
                "event": event,
                **trace.to_debug_metadata(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _log_unsupported_web_tools(tools: list[dict[str, Any]]) -> None:
    unsupported = [
        tool.get("type") for tool in tools if str(tool.get("type", "")).startswith(WEB_TOOL_PREFIX)
    ]
    if unsupported:
        LOGGER.info("Ignoring unsupported v1 web tools: %s", unsupported)
