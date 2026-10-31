"""Typed execution failures and payload-free error classification."""
from __future__ import annotations

import asyncio

import httpx
from langgraph.errors import GraphRecursionError


class RuntimeFailure(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 http_status: int = 500, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.http_status = http_status
        self.details = details or {}

    def event(self, run_id: str | None = None) -> dict:
        return {
            "type": "error", "code": self.code, "content": self.message,
            "retryable": self.retryable, **({"run_id": run_id} if run_id else {}),
        }


def transient_model_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.UnsupportedProtocol, httpx.LocalProtocolError)):
        return False
    # A quota 429 is permanent even though a request-rate 429 is transient.
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = str(error.get("code", "") or error.get("type", "")) if isinstance(error, dict) else ""
    if code.lower() in {"insufficient_quota", "quota_exceeded", "billing_hard_limit_reached"}:
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in {408, 429} or 500 <= status <= 599
    return isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError)) or type(exc).__name__ in {
        "APITimeoutError", "APIConnectionError",
    }


def public_failure(exc: Exception) -> RuntimeFailure:
    if isinstance(exc, RuntimeFailure):
        return exc
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return RuntimeFailure("run_timeout", "本次处理超时，请稍后重试。", retryable=True)
    if isinstance(exc, GraphRecursionError):
        return RuntimeFailure("step_limit", "本次处理已达到步骤上限，请补充问题后重试。")
    status = getattr(exc, "status_code", None)
    if status in {401, 403}:
        return RuntimeFailure("model_auth", "模型服务暂不可用，请联系服务维护人员。")
    if transient_model_error(exc):
        return RuntimeFailure("model_unavailable", "模型服务暂时不可用，请稍后重试。", retryable=True)
    return RuntimeFailure("execution_failed", "本次处理失败，请稍后重试。", retryable=True)
