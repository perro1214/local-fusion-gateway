from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import ModelConfig


@dataclass(slots=True)
class BackendResult:
    model: str
    payload: dict[str, Any]
    status_code: int = 200

    @property
    def content(self) -> str:
        choices = self.payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        return content if isinstance(content, str) else ""


class BackendClient:
    def __init__(self, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds

    async def chat_completion(
        self,
        logical_name: str,
        model: ModelConfig,
        payload: dict[str, Any],
    ) -> BackendResult:
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                model.chat_completions_url,
                headers=headers,
                json=payload,
            )
        response.raise_for_status()
        return BackendResult(
            model=logical_name,
            payload=response.json(),
            status_code=response.status_code,
        )
