"""Tests for custom stdio transport (deadlock-free native asyncio).

The ``TestStdioServer`` class drives the real :func:`linc_crucible.stdio.stdio_server`
through a **subprocess**: a python child (``_STDIO_DRIVER``) enters the transport
with its *real* OS stdin/stdout, so ``asyncio.connect_read_pipe`` /
``connect_write_pipe`` attach to genuine pipes instead of mocked ``sys`` buffers.
The parent feeds JSON-RPC lines on the child's stdin and asserts framed responses
on its stdout. This replaces the previous mock-patched tests, which hung because
``asyncio.connect_read_pipe`` operates at the OS fd level and ignored
``sys.stdin/stdout`` patches (tracked: reactor improvement e44a7269).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys

import anyio
import pytest
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

VALID_JSONRPC = '{"jsonrpc":"2.0","method":"initialize","id":1}'

# Driver run inside a python child process. It enters the *real* stdio_server
# (attached to the child's genuine OS stdin/stdout), drains every message off
# read_stream, and emits one structured trace line per message on stderr so the
# parent can assert on read/parse, malformed->Exception, and EOF->clean close.
# `mode == "echo"` additionally pushes one framed JSON-RPC response back through
# write_stream so the parent can verify the stdout writer branch.
_STDIO_DRIVER = r"""
import asyncio
import sys

import anyio
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

from linc_crucible.stdio import stdio_server

MODE = sys.argv[1] if len(sys.argv) > 1 else "read"


def _trace(tag: str, detail: str = "") -> None:
    # stderr is reserved for the structured trace; stdout carries framed JSON-RPC.
    sys.stderr.write("TRACE %s %s\n" % (tag, detail))
    sys.stderr.flush()


async def _main() -> None:
    if MODE == "hold":
        # Deliberately never close write_stream ourselves — this exercises the
        # fix for issue #80: stdio_server() must shut down on its own once
        # stdin reaches EOF, even though the caller left write_stream open.
        # Wrapped in an inner fail_after so a regression reports as a "HANG"
        # trace line rather than actually blocking the subprocess for the
        # full outer test timeout.
        try:
            with anyio.fail_after(4):
                async with stdio_server() as (read_stream, write_stream):
                    response = JSONRPCMessage.model_validate(
                        {"jsonrpc": "2.0", "id": 1, "result": {"status": "ok"}}
                    )
                    await write_stream.send(SessionMessage(response))
                    try:
                        while True:
                            item = await read_stream.receive()
                            if isinstance(item, Exception):
                                _trace("EXC", type(item).__name__)
                            else:
                                _trace("MSG", item.message.root.method or "")
                    except anyio.EndOfStream:
                        _trace("EOF")
                    # No manual write_stream.aclose() here — falling through to
                    # `async with stdio_server()`'s own __aexit__ is the point.
            _trace("EXIT")
        except TimeoutError:
            _trace("HANG")
        return

    async with stdio_server() as (read_stream, write_stream):
        if MODE == "echo":
            response = JSONRPCMessage.model_validate(
                {"jsonrpc": "2.0", "id": 1, "result": {"status": "ok"}}
            )
            await write_stream.send(SessionMessage(response))
        try:
            while True:
                with anyio.fail_after(5):
                    item = await read_stream.receive()
                if isinstance(item, Exception):
                    _trace("EXC", type(item).__name__)
                else:
                    _trace("MSG", item.message.root.method or "")
        except anyio.EndOfStream:
            _trace("EOF")
        finally:
            # Close write_stream so stdout_writer (and the transport's task
            # group) drains and the context manager exits instead of hanging.
            await write_stream.aclose()
            await read_stream.aclose()


asyncio.run(_main())
"""


def _spawn_driver(mode: str = "read") -> subprocess.Popen:
    """Spawn the stdio driver child with real pipes on stdin/stdout/stderr."""
    return subprocess.Popen(
        [sys.executable, "-c", _STDIO_DRIVER, mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_driver(
    stdin_bytes: bytes, mode: str = "read", timeout: float = 15.0
) -> tuple[bytes, list[str]]:
    """Feed stdin_bytes to the driver, return (stdout_bytes, trace_lines).

    trace_lines are the ``TRACE <tag> <detail>`` lines emitted on stderr, one per
    message the transport delivered (plus a terminal ``EOF`` once stdin closes).
    """
    proc = _spawn_driver(mode)
    try:
        out, err = proc.communicate(input=stdin_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        raise
    traces = [
        line[len("TRACE ") :].rstrip()
        for line in err.decode("utf-8", "replace").splitlines()
        if line.startswith("TRACE ")
    ]
    return out, traces


class TestStdioServer:
    """Drive the real stdio_server in a subprocess over genuine OS pipes.

    Each test spawns a python child running ``stdio_server`` (see ``_STDIO_DRIVER``),
    writes JSON-RPC lines on its real stdin, and asserts on the framed stdout
    response and/or the per-message trace the child emits on stderr. Replaces the
    prior mock-patched tests that hung because connect_read_pipe ignores sys mocks
    (reactor improvement e44a7269). Subprocess startup is the only cost; each test
    completes in well under a second.
    """

    def test_reads_jsonrpc_from_stdin(self) -> None:
        """Valid JSON-RPC line on stdin is parsed into a SessionMessage."""
        _out, traces = _run_driver((VALID_JSONRPC + "\n").encode())
        # The valid line is parsed and delivered as a SessionMessage carrying
        # the `initialize` method, then stdin EOF closes the stream cleanly.
        assert "MSG initialize" in traces
        assert "EOF" in traces

    def test_writes_jsonrpc_to_stdout(self) -> None:
        """SessionMessage sent to write_stream appears framed on stdout."""
        # No stdin input — the child sends one response on write_stream itself.
        out, _traces = _run_driver(b"", mode="echo")
        decoded = out.decode("utf-8").strip()
        assert decoded, "expected a framed JSON-RPC response on stdout"
        assert '"jsonrpc"' in decoded
        assert '"result"' in decoded
        # The framing is one JSON object per newline-terminated line.
        first_line = decoded.splitlines()[0]
        JSONRPCMessage.model_validate_json(first_line)

    def test_malformed_input_sends_exception(self) -> None:
        """Non-JSON input is forwarded as an Exception through read_stream, not a crash."""
        _out, traces = _run_driver(b"not valid json\n")
        # Malformed line surfaces as an Exception item (the transport stays up),
        # then EOF closes the stream cleanly.
        assert any(t.startswith("EXC ") for t in traces), traces
        assert "EOF" in traces

    def test_eof_closes_read_stream(self) -> None:
        """When stdin closes with no input, read_stream closes cleanly (EndOfStream)."""
        _out, traces = _run_driver(b"")
        # No MSG/EXC items — just a clean EOF once the empty stdin closes.
        assert traces == ["EOF"], traces

    def test_stdin_eof_shuts_down_without_caller_closing_write_stream(self) -> None:
        """stdio_server() exits on stdin EOF even if write_stream is left open (issue #80).

        Regression guard: previously stdout_writer only returned once the
        caller closed write_stream, so `async with stdio_server()` hung
        forever at context exit whenever a caller forgot (or chose not) to
        close it. The driver's "hold" mode never calls write_stream.aclose()
        itself — the transport must do it on stdin EOF.
        """
        _out, traces = _run_driver(b"", mode="hold")
        assert "HANG" not in traces, traces
        assert "EOF" in traces, traces
        assert "EXIT" in traces, traces

    def test_stdin_eof_flushes_queued_write_before_shutdown(self) -> None:
        """A message queued on write_stream before EOF is still flushed to stdout (issue #80).

        The shutdown-on-EOF fix must close write_stream's *send* side without
        dropping anything already queued — stdout_writer's `async for` should
        drain the pending item before the loop (and the task group) exits.
        """
        out, traces = _run_driver(b"", mode="hold")
        assert "HANG" not in traces, traces
        decoded = out.decode("utf-8").strip()
        assert decoded, "expected the queued response to be flushed to stdout, not dropped"
        assert '"result"' in decoded
        JSONRPCMessage.model_validate_json(decoded.splitlines()[0])


class TestStdioServerRealPipes:
    """A real-pipe roundtrip that actually exercises stdio_server.

    The mocked tests above hang because they never close write_stream, so the
    transport's task group never exits (the documented "connect_read_pipe
    ignores mocks" reason is a misdiagnosis). Driving real OS pipes and closing
    the write stream on the way out lets the task group drain cleanly — and
    keeps stdio_server covered without the subprocess rewrite (improvement
    e44a7269). Bounded by fail_after so a regression fails fast, never hangs.
    """

    @pytest.mark.asyncio
    async def test_roundtrip_valid_malformed_and_stdout_write(self) -> None:
        import types
        from unittest.mock import patch

        stdin_r_fd, stdin_w_fd = os.pipe()
        stdout_r_fd, stdout_w_fd = os.pipe()
        stdin_r = os.fdopen(stdin_r_fd, "rb", buffering=0)
        stdin_w = os.fdopen(stdin_w_fd, "wb", buffering=0)
        stdout_r = os.fdopen(stdout_r_fd, "rb", buffering=0)
        stdout_w = os.fdopen(stdout_w_fd, "wb", buffering=0)

        # Queue a valid request and a malformed line, then EOF so the stdin
        # reader drains both branches and exits.
        stdin_w.write((VALID_JSONRPC + "\n").encode())
        stdin_w.write(b"not valid json\n")
        stdin_w.close()

        fake_sys = types.SimpleNamespace(
            stdin=types.SimpleNamespace(buffer=stdin_r),
            stdout=types.SimpleNamespace(buffer=stdout_w),
        )

        try:
            with patch("linc_crucible.stdio.sys", fake_sys):
                from linc_crucible.stdio import stdio_server

                with anyio.fail_after(5):
                    async with stdio_server() as (read_stream, write_stream):
                        # Valid line → SessionMessage.
                        first = await read_stream.receive()
                        assert isinstance(first, SessionMessage)
                        assert first.message.root.method == "initialize"
                        # Malformed line → Exception forwarded, not a crash.
                        second = await read_stream.receive()
                        assert isinstance(second, Exception)
                        # Exercise the stdout writer branch.
                        response = JSONRPCMessage.model_validate(
                            {"jsonrpc": "2.0", "id": 1, "result": {"status": "ok"}}
                        )
                        await write_stream.send(SessionMessage(response))
                        # Close write side so stdout_writer (and the task group)
                        # exits instead of hanging at context exit.
                        await write_stream.aclose()
            # stdout_w is still open here; the transport wrote the framed reply.
            stdout_w.close()
            out = stdout_r.read().decode("utf-8")
            assert '"result"' in out
        finally:
            for f in (stdin_r, stdout_r, stdout_w):
                with contextlib.suppress(OSError):
                    f.close()
