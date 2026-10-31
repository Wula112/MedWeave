"""Validated, process-wide execution policy. No provider credentials live here."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

BACKEND_DIR = Path(__file__).resolve().parents[1]


class RuntimeSettings(BaseModel):
    data_dir: Path = BACKEND_DIR / ".runtime"
    run_timeout: float = Field(default=120, gt=0)
    vision_timeout: float = Field(default=180, gt=0)
    model_timeout: float = Field(default=30, gt=0)
    tool_timeout: float = Field(default=10, gt=0)
    retrieval_timeout: float = Field(default=10, gt=0)
    cleanup_timeout: float = Field(default=10, gt=0)
    lease_seconds: float = Field(default=15, ge=1)
    heartbeat_seconds: float = Field(default=3, gt=0)
    max_concurrent_runs: int = Field(default=8, ge=1)
    recursion_limit: int = Field(default=64, ge=8)
    agent_recursion_limit: int = Field(default=24, ge=6)
    max_tool_calls: int = Field(default=12, ge=1)
    repeat_limit: int = Field(default=3, ge=2)
    model_attempts: int = Field(default=3, ge=1, le=5)
    retry_base: float = Field(default=0.5, ge=0)
    retry_cap: float = Field(default=4, ge=0)
    retention_seconds: float = Field(default=86400, ge=60)
    event_buffer: int = Field(default=256, ge=8)

    @model_validator(mode="after")
    def validate_intervals(self) -> "RuntimeSettings":
        if self.heartbeat_seconds * 3 >= self.lease_seconds:
            raise ValueError("RUNTIME_LEASE_SECONDS must exceed three heartbeat intervals")
        return self

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(**{
            name: os.environ[key]
            for name in cls.model_fields
            if (key := f"RUNTIME_{name.upper()}") in os.environ
        })


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    return RuntimeSettings.from_env()
