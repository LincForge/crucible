"""Round-trip pin for ``crucible bellows`` over real OS pipes.

Spawns the stdio MCP server as a subprocess, speaks newline-delimited JSON-RPC on its
stdin and reads framed replies off its stdout: initialize, tools/list, repeated
tools/call of the device-free ``crucible_usage_stats``, a malformed line, an unknown
argument, hostile wire inputs, then EOF. Written against mcp 1.x first so it can serve
as the oracle for the 2.x port (LINC plan 2026-11-12-mcp-2x-fleet-migration, wave W2c).
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from linc_crucible import __version__

EXPECTED_TOOLS = {
    "crucible_assay",
    "crucible_score",
    "crucible_preflight",
    "crucible_extract_assets",
    "crucible_baseline",
    "crucible_bvt",
    "crucible_pour",
    "crucible_usage_stats",
}

_REPLY_DEADLINE_S = 10.0


class _Bellows:
    """A running ``crucible bellows`` child with a reader thread on stdout."""

    def __init__(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        env = {
            "HOME": str(home),
            "LINC_HOME": str(tmp_path / "linc_home"),
            "PATH": str(Path(sys.executable).parent),
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "from linc_crucible.cli import main; main()", "bellows"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._stderr: list[bytes] = []
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._next_id = 0
        self.init_result: dict[str, Any] = {}

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line)

    def stderr_tail(self) -> str:
        return b"".join(self._stderr[-20:]).decode("utf-8", "replace")

    def write_raw(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def send(self, payload: dict[str, Any]) -> None:
        self.write_raw((json.dumps(payload) + "\n").encode("utf-8"))

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one request and return the reply carrying its id (deadline-bounded)."""
        self._next_id += 1
        req_id = self._next_id
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        deadline = time.monotonic() + _REPLY_DEADLINE_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"no reply to {method} (id {req_id}) within {_REPLY_DEADLINE_S}s; "
                    f"stderr tail:\n{self.stderr_tail()}"
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise AssertionError(
                    f"bellows closed stdout before replying to {method}; "
                    f"stderr tail:\n{self.stderr_tail()}"
                )
            reply = json.loads(line)
            if reply.get("id") == req_id:
                return reply

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        reply = self.request("tools/call", {"name": name, "arguments": arguments})
        assert "result" in reply, reply
        return reply["result"]

    def close_and_wait(self, timeout: float = _REPLY_DEADLINE_S) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return self.proc.wait(timeout=timeout)

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)


@pytest.fixture
def bellows(tmp_path: Path) -> Iterator[_Bellows]:
    b = _Bellows(tmp_path)
    try:
        init = b.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "crucible-roundtrip-test", "version": "0"},
            },
        )
        assert "result" in init, init
        b.init_result = init["result"]
        b.notify("notifications/initialized")
        yield b
    finally:
        b.kill()


def _tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Return the tool's dict payload from a tools/call result (text content is JSON)."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and "total_calls" in structured:
        return structured
    texts = [c["text"] for c in result.get("content", []) if c.get("type") == "text"]
    assert texts, result
    return json.loads(texts[0])


def _error_text(result: dict[str, Any]) -> str:
    return " ".join(c.get("text", "") for c in result.get("content", []))


def test_bellows_roundtrip_initialize_list_call_malformed_unknown_arg_eof(
    bellows: _Bellows,
) -> None:
    """Pin: the full stdio session works end to end through ``crucible bellows``."""
    assert bellows.init_result["serverInfo"]["name"] == "linc-crucible"
    assert bellows.init_result["serverInfo"]["version"] == __version__

    listed = bellows.request("tools/list")
    names = {t["name"] for t in listed["result"]["tools"]}
    assert names == EXPECTED_TOOLS

    # Each call logs itself to $LINC_HOME/mcp-calls.jsonl after it returns, so the
    # n-th call sees exactly the n-1 earlier calls (the first sees an absent log).
    for n in range(3):
        result = bellows.call_tool("crucible_usage_stats", {})
        assert result.get("isError") is False, result
        assert _tool_payload(result)["total_calls"] == n

    bellows.write_raw(b"{this is not json\n")
    after_malformed = bellows.request("tools/list")
    assert {t["name"] for t in after_malformed["result"]["tools"]} == EXPECTED_TOOLS

    bad = bellows.call_tool("crucible_usage_stats", {"__unknown_arg__": 1})
    assert bad.get("isError") is True, bad
    assert "__unknown_arg__" in _error_text(bad)

    assert bellows.close_and_wait() == 0, bellows.stderr_tail()


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param(b"\n", id="empty-line"),
        pytest.param(b"x" * (1024 * 1024) + b"\n", id="one-megabyte-line"),
        pytest.param(b"y" * (17 * 1024 * 1024) + b"\n", id="over-line-cap"),
        pytest.param(b"\xff\xfe\xfd not utf-8\n", id="invalid-utf8"),
        pytest.param(b"[]\n", id="bare-json-array"),
    ],
)
def test_contract_hostile_wire_input_leaves_server_answering(
    bellows: _Bellows, hostile: bytes
) -> None:
    """[contract] A hostile stdin line never kills the reader: the next tools/list is answered."""
    bellows.write_raw(hostile)
    listed = bellows.request("tools/list")
    assert {t["name"] for t in listed["result"]["tools"]} == EXPECTED_TOOLS
    assert bellows.proc.poll() is None, bellows.stderr_tail()
