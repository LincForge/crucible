import asyncio
import os
import re
import sys
import threading
import time as _time
from functools import wraps
from types import ModuleType
from typing import Any

import anyio
import structlog
from linc_cli_kit import SDKToolError, strict_tools
from mcp.server.mcpserver import MCPServer

from linc_crucible import __version__
from linc_crucible.core.observability import log_mcp_call

logger = structlog.get_logger().bind(domain="build_pipeline")

server = MCPServer("linc-crucible", version=__version__)
strict_tools(server)

_thread_limiter: anyio.CapacityLimiter | None = None


def _get_thread_limiter() -> anyio.CapacityLimiter:
    global _thread_limiter
    if _thread_limiter is None:
        _thread_limiter = anyio.CapacityLimiter(4)
    return _thread_limiter


_cached_device_policy = None


def get_device_policy() -> Any:
    global _cached_device_policy
    if _cached_device_policy is None:
        from .core.device_policy import load_device_policy

        _cached_device_policy = load_device_policy()
    return _cached_device_policy


_get_device_policy = get_device_policy

_forge_reader = None
_deployer = None
_assayer = None
_scorer = None
_preflight_checker = None
_baseline_manager = None
_nerve_client = None
_tree_capture = None
_bvt_nerve_adapter = None
_initialized = False

# mcp 2.x runs the plain-``def`` tools (crucible_score, crucible_preflight,
# crucible_usage_stats) on anyio worker threads, so two first calls can reach
# _ensure_initialized() concurrently. The lock makes the singleton set build
# exactly once per process; `_initialized` is set only after every singleton exists.
_init_lock = threading.Lock()


def _ensure_initialized() -> None:
    """Build the lazy singleton set exactly once per process, from any thread.

    Double-checked under ``_init_lock``: the fast path reads ``_initialized``; the
    slow path builds every singleton into locals, publishes them, and sets the
    flag last. A constructor that raises propagates and leaves ``_initialized``
    False with no singleton published, so the next call retries from scratch.
    """
    global \
        _initialized, \
        _forge_reader, \
        _deployer, \
        _assayer, \
        _scorer, \
        _preflight_checker, \
        _baseline_manager, \
        _nerve_client, \
        _tree_capture, \
        _bvt_nerve_adapter

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        from .core.assayer import Assayer
        from .core.baseline import BaselineManager
        from .core.bvt.nerve_adapter import NerveAdapter
        from .core.deployer import Deployer
        from .core.forge_integration import ForgeProjectReader
        from .core.nerve_client import NerveClient
        from .core.preflight import PreflightChecker
        from .core.scorer import QualityScorer
        from .core.tree_capture import TreeCapture

        forge_reader = ForgeProjectReader()
        scorer = QualityScorer()
        nerve_client = NerveClient()
        tree_capture = TreeCapture()
        deployer = Deployer(
            forge_reader=forge_reader,
            nerve_client=nerve_client,
        )
        assayer = Assayer(
            forge_reader=forge_reader,
            scorer=scorer,
            mobsf_api_key=os.environ.get("CRUCIBLE_MOBSF_API_KEY", ""),
            nerve_client=nerve_client,
            tree_capture=tree_capture,
            device_policy=get_device_policy,
        )
        preflight_checker = PreflightChecker(
            forge_reader=forge_reader,
            scorer=scorer,
        )
        baseline_manager = BaselineManager()
        bvt_nerve_adapter = NerveAdapter(nerve=nerve_client)

        _forge_reader = forge_reader
        _scorer = scorer
        _nerve_client = nerve_client
        _tree_capture = tree_capture
        _deployer = deployer
        _assayer = assayer
        _preflight_checker = preflight_checker
        _baseline_manager = baseline_manager
        _bvt_nerve_adapter = bvt_nerve_adapter
        _initialized = True


_usage_log_failures = 0
_usage_log_failures_lock = threading.Lock()


def usage_log_failures() -> int:
    """Return how many usage-log writes failed (and were skipped) in this process."""
    return _usage_log_failures


def _log_call(tool_name: str, kwargs: dict, t0: float, error: Exception | None = None) -> None:
    """Record one tool call; a usage-log ``OSError`` is counted and never propagates.

    ELOOP (symlinked log), EACCES or ENOSPC must not fail the tool call or mask the
    tool's own exception.
    """
    global _usage_log_failures
    duration_ms = (_time.monotonic() - t0) * 1000
    message = None if error is None else str(error)
    try:
        log_mcp_call("linc-crucible", tool_name, kwargs, duration_ms, error is None, message)
    except OSError as exc:
        with _usage_log_failures_lock:
            _usage_log_failures += 1
        logger.warning("usage_log_write_failed", tool=tool_name, errno=exc.errno)


def _logged_tool(fn: Any) -> Any:
    """Wrap *fn* so every call is recorded by ``_log_call``; exceptions propagate unchanged."""
    tool_name = fn.__name__

    if asyncio.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(**kwargs: Any) -> Any:
            t0 = _time.monotonic()
            try:
                result = await fn(**kwargs)
            except Exception as exc:
                _log_call(tool_name, kwargs, t0, exc)
                raise
            _log_call(tool_name, kwargs, t0)
            return result

        return async_wrapper

    @wraps(fn)
    def sync_wrapper(**kwargs: Any) -> Any:
        t0 = _time.monotonic()
        try:
            result = fn(**kwargs)
        except Exception as exc:
            _log_call(tool_name, kwargs, t0, exc)
            raise
        _log_call(tool_name, kwargs, t0)
        return result

    return sync_wrapper


CLIENT_MESSAGE_MAX_CHARS = 500


def _client_message(exc: BaseException) -> str:
    """Return the model-facing text for a tool failure: ``"<Class>: <message>"``.

    The home directory is replaced by ``~`` and the text is capped at
    :data:`CLIENT_MESSAGE_MAX_CHARS`. An exception whose ``str()`` raises is
    reported by its class name only. Pure: never raises.
    """
    name = type(exc).__name__
    try:
        text = str(exc)
    except Exception:
        return name[:CLIENT_MESSAGE_MAX_CHARS]
    home = os.path.expanduser("~")
    if home and home not in ("~", "/"):
        # Only whole path components: a home ending in "al" must not rewrite "alice".
        text = re.sub(re.escape(home.rstrip("/")) + r"(?![\w.-])", "~", text)
    message = f"{name}: {text}" if text else name
    if len(message) > CLIENT_MESSAGE_MAX_CHARS:
        message = message[: CLIENT_MESSAGE_MAX_CHARS - 3] + "..."
    return message


def _observed_tool(fn: Any) -> Any:
    """Log every call of *fn*, and surface its failures to the MCP client as ``ToolError``.

    From mcp 2.1 on, only ``ToolError`` text reaches the client; any other exception is
    reported as a bare "Error executing tool <name>". A ``ToolError`` raised by the tool
    passes through unchanged; anything else becomes ``ToolError(_client_message(exc))``
    chained to the original, which ``_log_call`` still records as raised.

    The CLI mount calls the logged-only wrapper (``__crucible_logged__``) so its exit codes
    and error envelopes keep classifying the original exception type.
    """
    logged = _logged_tool(fn)

    if asyncio.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_boundary(**kwargs: Any) -> Any:
            try:
                return await logged(**kwargs)
            except SDKToolError:
                raise
            except Exception as exc:
                raise SDKToolError(_client_message(exc)) from exc

        async_boundary.__crucible_logged__ = logged  # type: ignore[attr-defined]
        return async_boundary

    @wraps(fn)
    def sync_boundary(**kwargs: Any) -> Any:
        try:
            return logged(**kwargs)
        except SDKToolError:
            raise
        except Exception as exc:
            raise SDKToolError(_client_message(exc)) from exc

    sync_boundary.__crucible_logged__ = logged  # type: ignore[attr-defined]
    return sync_boundary


class _ServerContextModule(ModuleType):
    @property
    def _device_policy(self) -> Any:
        return get_device_policy()

    @_device_policy.setter
    def _device_policy(self, value: Any) -> None:
        global _cached_device_policy
        _cached_device_policy = value

    @property
    def device_policy(self) -> Any:
        return get_device_policy()

    @device_policy.setter
    def device_policy(self, value: Any) -> None:
        global _cached_device_policy
        _cached_device_policy = value


sys.modules[__name__].__class__ = _ServerContextModule
