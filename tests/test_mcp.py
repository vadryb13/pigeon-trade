"""Tests for MCP JSON-RPC server — dispatch and method handlers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from conftest import FakeSession


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Мокает TInvestAdapter, Embedder, async_session_factory."""
    from aqr.data import tinvest as tinvest_mod

    class _FakeAdapter:
        async def candles(self, *a, **kw):
            import pandas as pd
            rng = pd.date_range("2023-01-01", periods=5, freq="B")
            px = [100.0, 101.0, 102.0, 103.0, 104.0]
            return pd.DataFrame({
                "open": px, "high": px, "low": px, "close": px, "volume": [1000] * 5,
            }, index=rng)

        async def _resolve_figi(self, *a, **kw):
            return "BBG004730N88"

    tinvest_mod.TInvestAdapter = _FakeAdapter

    import sys
    import types

    fake_openai_mod = types.ModuleType("openai")
    class _FakeEmb:
        async def create(self, **kw):
            return MagicMock(data=[MagicMock(embedding=[0.1] * 768)])
    class _FakeClient:
        def __init__(self, **kw):
            self.embeddings = _FakeEmb()
    fake_openai_mod.AsyncOpenAI = _FakeClient
    sys.modules["openai"] = fake_openai_mod

    from aqr import session as db_mod

    db_mod.async_session_factory = lambda: FakeSession()


class TestMCPDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_success(self, mock_dependencies):
        """dispatch('get_candles', valid params) → success response."""
        from aqr.mcp.server import dispatch

        result = await dispatch("get_candles", {"ticker": "SBER", "interval": "D1",
                                                  "from_date": "2023-01-01", "to_date": "2024-01-01"})

        assert "result" in result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_dispatch_invalid_method(self):
        """dispatch с неизвестным методом → ошибка method_not_found."""
        from aqr.mcp.server import dispatch

        result = await dispatch("unknown_method", {})

        assert "error" in result
        assert "Method not found" in str(result["error"])

    @pytest.mark.asyncio
    async def test_dispatch_invalid_params_type(self):
        """params не dict → ошибка invalid_params."""
        from aqr.mcp.server import dispatch

        result = await dispatch("get_candles", "not_a_dict")

        assert "error" in result


class TestMCPRequestResponse:
    def test_request_to_dict_with_id(self):
        from aqr.mcp.protocol import MCPRequest

        req = MCPRequest(method="test", params={"a": 1}, id="req-1")
        d = req.to_dict()

        assert d["method"] == "test"
        assert d["id"] == "req-1"

    def test_response_to_dict(self):
        from aqr.mcp.protocol import MCPResponse

        resp = MCPResponse(result={"sharpe": 1.2}, id="req-1")
        d = resp.to_dict()

        assert d["result"]["sharpe"] == 1.2
        assert d["id"] == "req-1"

    def test_error_to_dict(self):
        from aqr.mcp.protocol import MCPError

        err = MCPError(code=-32601, message="Method not found")
        d = err.to_dict()

        assert d["code"] == -32601
        assert d["message"] == "Method not found"

    def test_success_response_helper(self):
        from aqr.mcp.protocol import success_response

        resp = success_response({"ok": True}, "req-1")

        assert resp["result"] == {"ok": True}
        assert resp["id"] == "req-1"

    def test_error_response_helper(self):
        from aqr.mcp.protocol import MCPError, error_response

        err = MCPError(code=-32600, message="Invalid Request")
        resp = error_response(err, "req-1")

        assert resp["error"]["code"] == -32600
        assert resp["error"]["message"] == "Invalid Request"
