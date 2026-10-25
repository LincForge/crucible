"""[contract] REVIEW B8: a usage-log write failure never fails or masks an MCP tool call."""

from __future__ import annotations

import errno

import pytest
from _pytest.monkeypatch import MonkeyPatch


def _failing_log(*_args: object, **_kwargs: object) -> None:
    raise OSError(errno.ELOOP, "Too many levels of symbolic links")


def test_observed_sync_tool_survives_log_failure(monkeypatch: MonkeyPatch) -> None:
    """[contract] an OSError from log_mcp_call never fails the tool call; it is counted."""
    from linc_crucible import server_context

    monkeypatch.setattr(server_context, "log_mcp_call", _failing_log)
    before = server_context.usage_log_failures()

    @server_context._observed_tool
    def tool(x: int) -> int:
        return x + 1

    assert tool(x=1) == 2
    assert server_context.usage_log_failures() == before + 1


def test_observed_sync_tool_error_not_masked_by_log_failure(monkeypatch: MonkeyPatch) -> None:
    """[contract] the tool's real exception surfaces even when logging it raises OSError."""
    from linc_crucible import server_context

    monkeypatch.setattr(server_context, "log_mcp_call", _failing_log)

    @server_context._observed_tool
    def tool() -> None:
        raise ValueError("real failure")

    with pytest.raises(ValueError, match="real failure"):
        tool()


def test_observed_async_tool_survives_log_failure(monkeypatch: MonkeyPatch) -> None:
    """[contract] async tools: result returned and real exception surfaces past OSError."""
    import anyio

    from linc_crucible import server_context

    monkeypatch.setattr(server_context, "log_mcp_call", _failing_log)
    before = server_context.usage_log_failures()

    @server_context._observed_tool
    async def ok() -> str:
        return "done"

    @server_context._observed_tool
    async def boom() -> None:
        raise RuntimeError("real async failure")

    assert anyio.run(ok) == "done"
    with pytest.raises(RuntimeError, match="real async failure"):
        anyio.run(boom)
    assert server_context.usage_log_failures() == before + 2


class _Clock:
    """Deterministic stand-in for ``time``: monotonic() walks 10.0 → 10.5."""

    def __init__(self) -> None:
        self._ticks = iter([10.0, 10.5])

    def monotonic(self) -> float:
        return next(self._ticks)


class _Recorder:
    """Records positional/keyword calls (stands in for log_mcp_call and the logger)."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))

    def warning(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


def test_observed_tool_records_success_fields(monkeypatch: MonkeyPatch) -> None:
    """A successful call is logged with server, tool, params, exact duration and no error."""
    from linc_crucible import server_context

    recorder = _Recorder()
    monkeypatch.setattr(server_context, "log_mcp_call", recorder)
    monkeypatch.setattr(server_context, "_time", _Clock())

    @server_context._observed_tool
    def crucible_demo(x: int) -> int:
        return x

    crucible_demo(x=3)
    assert recorder.calls == [
        (("linc-crucible", "crucible_demo", {"x": 3}, 500.0, True, None), {})
    ]


def test_observed_tool_records_failure_fields(monkeypatch: MonkeyPatch) -> None:
    """A failing call is logged with success=False and the exception text."""
    from linc_crucible import server_context

    recorder = _Recorder()
    monkeypatch.setattr(server_context, "log_mcp_call", recorder)
    monkeypatch.setattr(server_context, "_time", _Clock())

    @server_context._observed_tool
    def crucible_demo() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        crucible_demo()
    assert recorder.calls == [(("linc-crucible", "crucible_demo", {}, 500.0, False, "boom"), {})]


def test_log_failure_emits_structured_warning(monkeypatch: MonkeyPatch) -> None:
    """A skipped usage-log write is reported once via structlog with tool and errno."""
    from linc_crucible import server_context

    warnings = _Recorder()
    monkeypatch.setattr(server_context, "log_mcp_call", _failing_log)
    monkeypatch.setattr(server_context, "logger", warnings)

    @server_context._observed_tool
    def crucible_demo() -> int:
        return 1

    crucible_demo()
    assert warnings.calls == [
        (("usage_log_write_failed",), {"tool": "crucible_demo", "errno": errno.ELOOP})
    ]


def test_observed_async_tool_records_failure(monkeypatch: MonkeyPatch) -> None:
    """An async tool's exception is awaited inside the wrapper and logged as a failure."""
    import anyio

    from linc_crucible import server_context

    recorder = _Recorder()
    monkeypatch.setattr(server_context, "log_mcp_call", recorder)
    monkeypatch.setattr(server_context, "_time", _Clock())

    @server_context._observed_tool
    async def crucible_async() -> None:
        raise RuntimeError("late")

    with pytest.raises(RuntimeError, match="late"):
        anyio.run(crucible_async)
    assert recorder.calls == [(("linc-crucible", "crucible_async", {}, 500.0, False, "late"), {})]


def test_usage_log_failure_counter_is_the_module_global(monkeypatch: MonkeyPatch) -> None:
    """[contract] usage_log_failures() reports the process-wide ``_usage_log_failures``."""
    from linc_crucible import server_context

    monkeypatch.setattr(server_context, "_usage_log_failures", 41)
    assert server_context.usage_log_failures() == 41
    monkeypatch.setattr(server_context, "log_mcp_call", _failing_log)
    server_context._log_call("crucible_pour", {}, 0.0)
    assert server_context._usage_log_failures == 42


def test_observed_tool_scrubs_outside_root_path_from_exception_in_redacted_mode(
    tmp_path, monkeypatch: MonkeyPatch
) -> None:
    """[contract] an exception path outside known roots raised by an @_observed_tool never
    appears in the written redacted usage log."""
    import json

    from linc_crucible import server_context

    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    monkeypatch.delenv("CRUCIBLE_USAGE_LOG", raising=False)
    secret_path = "/private/var/x/secret-dir/y.png"

    @server_context._observed_tool
    def crucible_demo(project: str, device_info: str) -> None:
        raise FileNotFoundError(f"missing '{secret_path}' for {device_info}")

    with pytest.raises(FileNotFoundError):
        crucible_demo(project="demo", device_info="Pixel R58N000TEST")

    raw = (tmp_path / "mcp-calls.jsonl").read_text()
    assert secret_path not in raw
    assert "secret-dir" not in raw
    assert "R58N000TEST" not in raw
    entry = json.loads(raw)
    assert entry["params"] == {"project": "demo", "device_info": "<redacted len=17>"}
    assert entry["error"] == "missing '<path>' for <redacted len=17>"
