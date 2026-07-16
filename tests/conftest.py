"""pytest fixtures — mock LLM/Ollama，避免依赖外部服务"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """返回 FastAPI TestClient，直接测试路由"""
    from src.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """确保测试环境使用正确的配置"""
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("PG_ENABLED", "false")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("ADJACENT_ENABLED", "false")
