"""JSON-RPC 2.0 protocol dataclasses.

Schemas for request/response/error objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPRequest:
    """JSON-RPC 2.0 request."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    jsonrpc: str = "2.0"
    id: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
        }
        if self.id is not None:
            d["id"] = self.id
        return d


@dataclass
class MCPResponse:
    """JSON-RPC 2.0 success response."""

    result: Any
    id: int | str | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "result": self.result,
        }
        if self.id is not None:
            d["id"] = self.id
        return d


@dataclass
class MCPError:
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


# Standard JSON-RPC error codes
METHOD_NOT_FOUND = MCPError(code=-32601, message="Method not found")
INVALID_PARAMS = MCPError(code=-32602, message="Invalid params")
INTERNAL_ERROR = MCPError(code=-32603, message="Internal error")


def error_response(error: MCPError, req_id: int | str | None = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response dict."""
    resp: dict[str, Any] = {
        "jsonrpc": "2.0",
        "error": error.to_dict(),
    }
    if req_id is not None:
        resp["id"] = req_id
    return resp


def success_response(result: Any, req_id: int | str | None = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response dict."""
    return MCPResponse(result=result, id=req_id).to_dict()
