"""All-tools invocability smoke tests for the linc-crucible FastMCP server.

These tests guard against the defect class shipped on 2026-05-28 in `crucible_bvt`
(commit 757ddbe fix): a wrapper/decorator layer that raised before the handler
body could ever execute, making the tool 100% broken via MCP. Per-tool tests in
`tests/test_mcp_server.py` already cover handler bodies, but **no test exercised
tool invocability** for every registered tool — so a broken wrapper on any new
or modified tool could ship silently.

What these tests assert:

1. Every `@server.tool()`-registered tool on the linc-crucible FastMCP server
   can be invoked without the `@_observed_tool` / `log_mcp_call` wrapper layer
   raising before the handler body is entered.
2. `crucible_bvt` is included (it previously had zero direct coverage).
3. The smoke assertion is itself proven to catch the defect class — a dummy
   tool that mis-uses `log_mcp_call` the pre-757ddbe way (as a context manager)
   is flagged by `_is_wrapper_layer_failure` as a wrapper-layer failure.

Tool enumeration uses the FastMCP private `server._tool_manager._tools` dict
guarded by `assert len(...) == EXPECTED_TOOL_COUNT` to catch drift if a tool
is added or removed.
"""

from __future__ import annotations

import traceback
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from linc_crucible.core.observability import log_mcp_call
from linc_crucible.mcp_server import _observed_tool, server

# Pinned: if FastMCP changes its internal tool registry layout, this test
# breaks loudly rather than silently skipping invocability coverage.
# revisit on FastMCP major bump
EXPECTED_TOOL_COUNT = 8

# Files whose frames in a traceback indicate the failure originated inside the
# observability wrapper layer (not the handler body).
_WRAPPER_FILENAME_MARKERS = ("observability.py",)
_WRAPPER_FUNCTION_NAMES = {
    "log_mcp_call",
    "_observed_tool",
    "async_wrapper",
    "sync_wrapper",
}
# Exception types the pre-757ddbe bug raised from the wrapper layer
# (TypeError because log_mcp_call wasn't a context manager;
#  AttributeError if a future regression swaps the helper for one that lacks
#  __enter__/__exit__ or similar wrapper-protocol attributes).
_WRAPPER_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (TypeError, AttributeError)


def _enumerated_tool_names() -> list[str]:
    """Enumerate registered tools via the FastMCP private registry.

    `list_tools()` is async and cannot be called at @pytest.mark.parametrize
    collection time, so we read `server._tool_manager._tools` directly.
    """
    tool_manager = server._tool_manager
    names = sorted(tool_manager._tools.keys())
    # Guard: catches drift if a tool is added/removed without updating tests.
    assert len(names) == EXPECTED_TOOL_COUNT, (
        f"Expected {EXPECTED_TOOL_COUNT} tools but found {len(names)}: {names}. "
        f"Update EXPECTED_TOOL_COUNT in tests/test_tool_smoke.py and add per-tool "
        f"coverage in tests/test_mcp_server.py for any new tool."
    )
    return names


def _is_wrapper_layer_failure(exc: BaseException, tool_name: str) -> bool:
    """Return True iff `exc` originates from the @_observed_tool / log_mcp_call
    wrapper layer rather than meaningful handler body execution.

    Detection (in order):

    1. **Source-line check:** If the LAST frame's source line invokes
       `log_mcp_call(` AND the exception is one of `_WRAPPER_EXCEPTION_TYPES`,
       it's a wrapper bug — the pre-757ddbe defect was `with log_mcp_call(...)`
       inside the handler, and the offending line shows up verbatim in
       `traceback.extract_tb(...).line`. This catches the pre-757ddbe shape
       even though a handler frame is present in the traceback (Python raises
       the TypeError/AttributeError at the `with` line *of the handler*).

    2. **Last-frame wrapper-file check:** If the LAST frame is inside the
       observability module or a wrapper function name, it's a wrapper bug.

    3. **No-handler-frame check:** If the exception type is one of
       `_WRAPPER_EXCEPTION_TYPES` and no frame named `tool_name` appears in
       the traceback at all, the exception fired before the body — wrapper
       bug.

    Otherwise: handler-internal failure (acceptable — body was entered).
    """
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        # No traceback context — be conservative and treat as wrapper failure
        # only if the exception type matches the wrapper-bug shape.
        return isinstance(exc, _WRAPPER_EXCEPTION_TYPES)

    last = frames[-1]
    last_filename = last.filename or ""
    last_line = (last.line or "").strip()

    # (1) Source-line check: the pre-757ddbe defect's offending line text.
    if isinstance(exc, _WRAPPER_EXCEPTION_TYPES) and "log_mcp_call(" in last_line:
        return True

    # (2) Last-frame is inside the observability wrapper.
    if any(last_filename.endswith(m) for m in _WRAPPER_FILENAME_MARKERS):
        return True
    if last.name in _WRAPPER_FUNCTION_NAMES:
        return True

    # (3) Wrapper-shape exception with no handler frame at all.
    if isinstance(exc, _WRAPPER_EXCEPTION_TYPES):
        return all(frame.name != tool_name for frame in frames)

    return False


# Minimal valid kwargs per tool — supply REQUIRED positional args so Python's
# signature-binding logic doesn't raise TypeError BEFORE the body runs. The brief
# accepts handler-internal failures (missing-arg, ValidationError) but in practice
# missing-arg TypeErrors raised at fn(**kwargs) time surface through the wrapper's
# frame and are indistinguishable from a true wrapper-layer bug. Supplying minimal
# valid kwargs makes the smoke check unambiguous: any TypeError/AttributeError
# with an observability frame after binding succeeds is a real wrapper defect.
_TOOL_MIN_KWARGS: dict[str, dict[str, Any]] = {
    "crucible_pour": {"project": "TestProject"},
    "crucible_assay": {"project": "TestProject"},
    "crucible_score": {"project": "TestProject"},
    "crucible_preflight": {"project": "TestProject"},
    "crucible_baseline": {"action": "list", "project": "TestProject"},
    "crucible_bvt": {"project": "TestProject", "device": "test-device"},
    "crucible_extract_assets": {"video_path": "/tmp/nonexistent.mp4"},
}


async def _invoke_tool(tool_name: str) -> tuple[bool, BaseException | None]:
    """Invoke a tool with all core service/repo deps mocked.

    Returns (entered_body, exc). `entered_body` is True if the handler body was
    entered (no wrapper-layer failure raised). `exc` is the exception that
    aborted invocation, or None if the call returned successfully.

    Handler-internal exceptions (validation error / file-not-found inside the
    body) are acceptable: they prove the body executed past the wrapper.
    """
    # Patch every module-level service/repo dep so the handler body short-circuits
    # to a simple mocked return value instead of hitting real I/O.
    patches = [
        patch("linc_crucible.server_context._ensure_initialized"),
        patch("linc_crucible.server_context._deployer", new=MagicMock()),
        patch("linc_crucible.server_context._assayer", new=MagicMock()),
        patch("linc_crucible.server_context._preflight_checker", new=MagicMock()),
        patch("linc_crucible.server_context._baseline_manager", new=MagicMock()),
        patch("linc_crucible.server_context._nerve_client", new=MagicMock()),
        patch("linc_crucible.server_context._forge_reader", new=MagicMock()),
        patch("linc_crucible.server_context._bvt_nerve_adapter", new=MagicMock()),
        patch("linc_crucible.server_context._scorer", new=MagicMock()),
    ]
    for p in patches:
        p.start()
    try:
        tool = server._tool_manager._tools[tool_name]
        fn = tool.fn
        kwargs = _TOOL_MIN_KWARGS.get(tool_name, {})
        # Smoke-call with minimal valid kwargs — the point is the handler body
        # must be ENTERED. Handler-internal failures (ValidationError, missing
        # optional file, etc.) are acceptable; only wrapper-layer failures count
        # as a bug.
        try:
            result = fn(**kwargs)
            # Coroutine? Drive it once.
            if hasattr(result, "__await__"):
                try:
                    await result
                except BaseException as exc:
                    return (not _is_wrapper_layer_failure(exc, tool_name), exc)
            return (True, None)
        except BaseException as exc:
            return (not _is_wrapper_layer_failure(exc, tool_name), exc)
    finally:
        for p in patches:
            p.stop()


# --- AC#1 + AC#2: parametrized smoke test over every registered tool ---


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _enumerated_tool_names())
async def test_tool_is_invocable(tool_name: str) -> None:
    """Every @server.tool()-registered tool must be invocable.

    The wrapper/decorator layer (@_observed_tool / log_mcp_call) must not raise
    before the handler body is entered. Handler-internal failures are allowed.

    This is the regression guard for the 2026-05-28 `crucible_bvt` defect: a
    broken `with log_mcp_call(...)` (pre-757ddbe) raised TypeError from the
    wrapper layer before the body ran, shipping a 100%-broken tool because no
    test exercised this layer.
    """
    entered_body, exc = await _invoke_tool(tool_name)
    assert entered_body, (
        f"Tool {tool_name!r} failed in the @_observed_tool / log_mcp_call "
        f"wrapper layer before the handler body could execute. "
        f"Exception: {type(exc).__name__}: {exc}"
    )


def test_crucible_bvt_is_in_smoke_coverage() -> None:
    """Explicit guard: AC#2 — `crucible_bvt` must be in the smoke set.

    Belt-and-braces against the smoke test being silently skipped or filtered
    in a way that drops bvt back to zero coverage (the pre-757ddbe state).
    """
    assert "crucible_bvt" in _enumerated_tool_names()


# --- AC#3: prove the smoke assertion catches the defect class ---


def _make_pre_757ddbe_broken_tool() -> Any:
    """Build a dummy tool that mis-uses log_mcp_call exactly the pre-757ddbe way:
    invokes it as a context manager. log_mcp_call is a plain logging function
    (5 required positional args) and is NOT a context manager — so this raises
    `TypeError` from the wrapper layer before the body runs, identical to the
    real defect.
    """

    @_observed_tool
    async def broken_dummy_tool(project: str = "x") -> str:
        # The bug: log_mcp_call used as a context manager. This is the exact
        # shape of the pre-757ddbe bug. Raises TypeError on __enter__ because
        # log_mcp_call is a plain function, not a CM.
        with log_mcp_call("linc-crucible", "broken_dummy_tool"):  # type: ignore[attr-defined]
            return "never reached"

    return broken_dummy_tool


@pytest.mark.asyncio
async def test_smoke_assertion_catches_pre_757ddbe_defect_class() -> None:
    """AC#3: the same `_is_wrapper_layer_failure` predicate used by the smoke
    test MUST flag a deliberately mis-wrapped dummy tool (no src/ revert needed).

    Builds a dummy tool that uses `log_mcp_call` as a context manager exactly
    the way the pre-757ddbe bug did, invokes it, and asserts the predicate
    classifies the resulting exception as a wrapper-layer failure.
    """
    broken_tool = _make_pre_757ddbe_broken_tool()

    raised: BaseException | None = None
    try:
        await broken_tool(project="x")
    except BaseException as exc:
        raised = exc

    assert raised is not None, (
        "Pre-757ddbe-shaped dummy tool unexpectedly did NOT raise. "
        "log_mcp_call must be a plain function (not a context manager) for "
        "this regression guard to be meaningful."
    )
    assert _is_wrapper_layer_failure(raised, "broken_dummy_tool"), (
        f"Smoke assertion failed to catch the pre-757ddbe defect class. "
        f"Got {type(raised).__name__}: {raised}. "
        f"The smoke test would have shipped a broken tool. "
        f"Inspect `_is_wrapper_layer_failure` for missing detection paths."
    )


def test_handler_internal_typeerror_is_NOT_flagged_as_wrapper_failure() -> None:
    """Counter-example for AC#3: a handler-internal `TypeError` (raised from
    inside the body, with no observability frame in the traceback) MUST NOT
    be flagged as a wrapper-layer failure.

    Without this counter-example, `_is_wrapper_layer_failure` could be a tautology
    that flags every `TypeError` as a wrapper failure, defeating the point of
    distinguishing wrapper-layer from handler-internal failures.
    """

    def handler_internal_raise() -> None:
        # Pure handler-body TypeError. No observability frame in tb.
        raise TypeError("handler-body error, not from wrapper")

    try:
        handler_internal_raise()
    except TypeError as exc:
        # Pass the actual function name so the predicate sees a handler frame.
        assert not _is_wrapper_layer_failure(exc, "handler_internal_raise"), (
            "_is_wrapper_layer_failure incorrectly flagged a handler-internal "
            "TypeError as a wrapper-layer failure. The predicate must allow "
            "exceptions raised after handler-body entry."
        )
