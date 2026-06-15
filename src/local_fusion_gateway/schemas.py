from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    tool_choice: Any | None = None
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None

    def to_backend_payload(self, backend_model: str) -> dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        payload["model"] = backend_model
        return payload


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict[str, Any] | None = None
