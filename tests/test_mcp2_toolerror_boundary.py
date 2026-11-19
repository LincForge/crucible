"""[contract] Crucible tool failures reach the MCP client with their cause (mcp 2.1+).

mcp 2.1+ hides the text of any exception other than ``ToolError``: the client sees only
"Error executing tool <name>" (an ``UnexpectedToolError``). ``_observed_tool`` therefore
converts every other failure into the SDK ``ToolError`` carrying ``"<Class>: <message>"``
with the home directory shown as ``~`` and the text capped at 500 characters, while
``_log_call`` still records the original exception and the CLI keeps the original type.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest
from linc_cli_kit import SDKToolError
from mcp.server.mcpserver.exceptions import UnexpectedToolError

import linc_crucible.mcp_server  # noqa: F401  (registers every tool)
from linc_crucible import server_context as ctx

FAKE_HOME = "/srv/fakehomes/someone"


class _RaisingScorer:
    scores_dir = "/nonexistent"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def load_latest(self, project: str) -> Any:
        raise self.exc


class _RaisingDeployer:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def pour(self, **kwargs: Any) -> Any:
        raise self.exc


class _HostileStr(Exception):
    def __str__(self) -> str:
        raise RuntimeError("str() itself fails")


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, BaseException | None]]:
    """Fake the init and capture what _log_call records."""
    monkeypatch.setenv("HOME", FAKE_HOME)
    monkeypatch.setattr(ctx, "_ensure_initialized", lambda: None)
    records: list[tuple[str, BaseException | None]] = []

    def _capture(tool_name: str, kwargs: dict, t0: float, error: Exception | None = None) -> None:
        records.append((tool_name, error))

    monkeypatch.setattr(ctx, "_log_call", _capture)
    return records


def _score_raising(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    monkeypatch.setattr(ctx, "_scorer", _RaisingScorer(exc))


def _call(name: str, args: dict[str, Any]) -> Any:
    return anyio.run(ctx.server.call_tool, name, args)


def test_contract_sync_tool_failure_reaches_client_with_cause_and_no_home(
    monkeypatch: pytest.MonkeyPatch, logged: list[tuple[str, BaseException | None]]
) -> None:
    """[contract] crucible_score's OSError reaches the client as ToolError, home shown as ~."""
    original = OSError(f"no scores dir {FAKE_HOME}/.linc/x")
    _score_raising(monkeypatch, original)

    with pytest.raises(SDKToolError) as info:
        _call("crucible_score", {"project": "p"})

    assert not isinstance(info.value, UnexpectedToolError)
    text = str(info.value)
    assert "OSError" in text
    assert "~/.linc/x" in text
    assert FAKE_HOME not in text
    assert logged == [("crucible_score", original)]


def test_contract_async_tool_failure_reaches_client_with_cause(
    monkeypatch: pytest.MonkeyPatch, logged: list[tuple[str, BaseException | None]]
) -> None:
    """[contract] crucible_pour (async, thread-offloaded) behaves the same."""
    original = RuntimeError(f"gradle exploded in {FAKE_HOME}/projects/app")
    monkeypatch.setattr(ctx, "_deployer", _RaisingDeployer(original))

    with pytest.raises(SDKToolError) as info:
        _call("crucible_pour", {"project": "app"})

    assert not isinstance(info.value, UnexpectedToolError)
    text = str(info.value)
    assert "RuntimeError" in text
    assert "~/projects/app" in text
    assert FAKE_HOME not in text
    assert logged == [("crucible_pour", original)]


def test_contract_client_sees_is_error_with_message_over_the_wire(
    monkeypatch: pytest.MonkeyPatch, logged: list[tuple[str, BaseException | None]]
) -> None:
    """[contract] Through the low-level server the result is is_error with the cause text."""
    _score_raising(monkeypatch, ValueError("score store corrupt"))

    import mcp.types as types
    from mcp.client import Client

    async def _run() -> types.CallToolResult:
        async with Client(ctx.server) as client:
            return await client.call_tool("crucible_score", {"project": "p"})

    result = anyio.run(_run)
    assert result.is_error is True
    text = " ".join(c.text for c in result.content if isinstance(c, types.TextContent))
    assert "ValueError: score store corrupt" in text


def test_contract_long_message_is_truncated(
    monkeypatch: pytest.MonkeyPatch, logged: list[tuple[str, BaseException | None]]
) -> None:
    """[contract] A 10 000-char message reaches the client capped at 500 characters."""
    original = OSError("x" * 10_000)
    _score_raising(monkeypatch, original)

    with pytest.raises(SDKToolError) as info:
        _call("crucible_score", {"project": "p"})

    # The SDK prefixes "Error executing tool crucible_score: "; the cause is the cap.
    cause = info.value.__cause__
    assert isinstance(cause, SDKToolError)
    assert len(str(cause)) <= ctx.CLIENT_MESSAGE_MAX_CHARS
    assert str(cause).startswith("OSError: xxx")
    assert logged == [("crucible_score", original)]


def test_contract_unprintable_exception_is_reported_by_class_name(
    monkeypatch: pytest.MonkeyPatch, logged: list[tuple[str, BaseException | None]]
) -> None:
    """[contract] An exception whose str() raises is reported by its class name only."""
    original = _HostileStr()
    _score_raising(monkeypatch, original)

    with pytest.raises(SDKToolError) as info:
        _call("crucible_score", {"project": "p"})

    assert str(info.value.__cause__) == "_HostileStr"
    assert logged == [("crucible_score", original)]


def test_contract_tool_error_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, logged: list[tuple[str, BaseException | None]]
) -> None:
    """[contract] A ToolError raised inside a tool keeps its own text (no class prefix)."""
    original = SDKToolError("project p has no baseline yet")
    _score_raising(monkeypatch, original)

    with pytest.raises(SDKToolError) as info:
        _call("crucible_score", {"project": "p"})

    assert info.value.__cause__ is original
    assert "SDKToolError" not in str(info.value)
    assert "ToolError:" not in str(info.value)
    assert str(info.value).endswith("project p has no baseline yet")
    assert logged == [("crucible_score", original)]


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (OSError(f"{FAKE_HOME}/a"), "OSError: ~/a"),
        (OSError(f"{FAKE_HOME}"), "OSError: ~"),
        (OSError(f"{FAKE_HOME}else/a"), f"OSError: {FAKE_HOME}else/a"),
        (ValueError(""), "ValueError"),
        (_HostileStr(), "_HostileStr"),
    ],
)
def test_contract_client_message_is_pure_and_bounded(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException, expected: str
) -> None:
    """[contract] _client_message: home only on a whole component, empty text, hostile str."""
    monkeypatch.setenv("HOME", FAKE_HOME)
    assert ctx._client_message(exc) == expected


def test_contract_cli_path_keeps_the_original_exception(
    monkeypatch: pytest.MonkeyPatch, logged: list[tuple[str, BaseException | None]]
) -> None:
    """[contract] The CLI mount calls the logged-only layer, so it sees the original type."""
    from linc_crucible.cli import _local_fn

    original = OSError("disk gone")
    _score_raising(monkeypatch, original)
    spec = type("Spec", (), {})()
    spec.name = "crucible_score"
    spec.fn = ctx.server._tool_manager._tools["crucible_score"].fn

    with pytest.raises(OSError) as info:
        _local_fn(spec)(project="p")
    assert info.value is original
    assert logged == [("crucible_score", original)]
