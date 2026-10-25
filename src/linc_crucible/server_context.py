import asyncio
import os
import sys
import time as _time
from functools import wraps
from types import ModuleType
from typing import Any

import anyio
import structlog
from linc_cli_kit import strict_tools
from mcp.server.fastmcp import FastMCP

from linc_crucible.core.observability import log_mcp_call

logger = structlog.get_logger().bind(domain="build_pipeline")

server = FastMCP("linc-crucible")
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


def _ensure_initialized() -> None:
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

    from .core.assayer import Assayer
    from .core.baseline import BaselineManager
    from .core.deployer import Deployer
    from .core.forge_integration import ForgeProjectReader
    from .core.nerve_client import NerveClient
    from .core.preflight import PreflightChecker
    from .core.scorer import QualityScorer
    from .core.tree_capture import TreeCapture

    _forge_reader = ForgeProjectReader()
    _scorer = QualityScorer()
    _nerve_client = NerveClient()
    _tree_capture = TreeCapture()
    _deployer = Deployer(
        forge_reader=_forge_reader,
        nerve_client=_nerve_client,
    )
    _assayer = Assayer(
        forge_reader=_forge_reader,
        scorer=_scorer,
        mobsf_api_key=os.environ.get("CRUCIBLE_MOBSF_API_KEY", ""),
        nerve_client=_nerve_client,
        tree_capture=_tree_capture,
        device_policy=get_device_policy,
    )
    _preflight_checker = PreflightChecker(
        forge_reader=_forge_reader,
        scorer=_scorer,
    )
    _baseline_manager = BaselineManager()

    from .core.bvt.nerve_adapter import NerveAdapter

    _bvt_nerve_adapter = NerveAdapter(nerve=_nerve_client)
    _initialized = True


_usage_log_failures = 0


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
        _usage_log_failures += 1
        logger.warning("usage_log_write_failed", tool=tool_name, errno=exc.errno)


def _observed_tool(fn: Any) -> Any:
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
