"""MCP tool call observability — JSONL logger.

Logs every MCP tool call with timing, params, and success/failure to
``$LINC_HOME/mcp-calls.jsonl`` (default ``~/.linc/mcp-calls.jsonl``).

``CRUCIBLE_USAGE_LOG`` selects what is recorded:

* ``redacted`` (default, and any unrecognised value) — only allowlisted
  parameter keys (``_UNREDACTED_KEYS``) keep their values; every other
  parameter value is replaced length-only (``<redacted len=N>``, no hash, which
  would be reversible for short values), and the ``error`` string is scrubbed
  of every redacted value, host root, and absolute path (``<path>``).
* ``off`` — nothing is written and no directory is created.
* ``full`` — parameters are logged as-is (bytes summarised, long strings
  truncated).

New state directories are created ``0700`` (pre-existing directories keep their
permissions); the log file is created ``0600`` and never opened through a
symlink. Non-finite floats are normalised to ``"NaN"``, ``"Infinity"``, or
``"-Infinity"`` before ``json.dumps(..., allow_nan=False)``.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import IO, Any

from .state_paths import ensure_private_dir

_DEFAULT_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
_MAX_STR_LEN = 500

# Allowlist of parameter keys kept verbatim in redacted mode (case-insensitive).
_UNREDACTED_KEYS = frozenset(
    {
        "project",
        "mode",
        "action",
        "platform",
        "device_type",
        "perf",
        "a11y",
        "a11y_tree",
        "history",
        "limit",
        "keep",
        "build_only",
        "skip_build",
        "launch",
        "build_type",
        "allow_personal",
        "allow_empty",
        "since_hours",
        "top_n",
        "max_assets",
        "screen_name",
    }
)
# Explicitly classified non-allowlisted tool parameters, device ids, host paths, credentials.
_REDACT_KEYS = frozenset(
    {
        "api_key",
        "app_path",
        "app_version",
        "budget_s",
        "compare_against_baseline",
        "device",
        "device_info",
        "device_serial",
        "devices",
        "exclude_tags",
        "image_path",
        "include_rotated",
        "launch_intent",
        "module",
        "output_dir",
        "password",
        "project_dir",
        "secret",
        "serial",
        "server_filter",
        "source_score",
        "suite_dir",
        "tags",
        "token",
        "udid",
        "video_path",
    }
)
# Any key ending in one of these names a host path.
_PATH_KEY_SUFFIXES = ("_path", "_dir", "file", "video")
# A session id must be a short, single-token label (no newline, slash or space).
_RUN_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
# Host-path roots whose prefix is scrubbed from redacted error text.
_HOST_PREFIX_VARS = ("LINC_HOME", "LINC_PROJECTS_DIR")
# Absolute POSIX (/...) or Windows drive (C:\..., C:/...) path tokens inside error text.
_ABS_PATH_RE = re.compile(r"(?<![\w.~>-])(?:/[^\s'\":,;)\]]+|[A-Za-z]:[\\/][^\s'\":,;)\]]*)")
# Owner read/write only (0600) — for both the create mode and the follow-up fchmod.
_LOG_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR
_DEFAULT_MODE = "redacted"
_EXPLICIT_MODES = ("off", "full")
_INF_STRINGS: dict[float, str] = {math.inf: "Infinity", -math.inf: "-Infinity"}


def usage_log_mode() -> str:
    """Return the usage log mode from ``CRUCIBLE_USAGE_LOG``.

    Returns:
        ``"off"`` or ``"full"`` when explicitly selected; ``"redacted"`` for an
        unset, empty or unrecognised value.
    """
    raw = os.getenv("CRUCIBLE_USAGE_LOG")
    if raw is None:
        return _DEFAULT_MODE
    val = raw.strip().lower()
    return val if val in _EXPLICIT_MODES else _DEFAULT_MODE


def _default_log_path() -> Path:
    """Return the default MCP call log path, honouring ``LINC_HOME``.

    Returns:
        Path to ``$LINC_HOME/mcp-calls.jsonl`` when ``LINC_HOME`` is set,
        otherwise ``~/.linc/mcp-calls.jsonl``.  Re-evaluated on every call
        so that ``os.environ`` monkeypatching in tests takes effect.
    """
    return Path(os.getenv("LINC_HOME", str(Path.home() / ".linc"))) / "mcp-calls.jsonl"


def _session_id() -> str:
    """Return ``CRUCIBLE_RUN_ID``, else the deprecated ``FORGE_RUN_ID``, else ``"manual"``.

    A value outside ``[A-Za-z0-9._-]{1,128}`` is untrusted input and becomes ``"manual"``.
    """
    run_id = os.getenv("CRUCIBLE_RUN_ID") or os.getenv("FORGE_RUN_ID") or ""
    return run_id if _RUN_ID_RE.fullmatch(run_id) else "manual"


def log_mcp_call(
    server: str,
    tool: str,
    params: dict,
    duration_ms: float,
    success: bool,
    error: str | None = None,
    log_path: Path | None = None,
    max_size_bytes: int = _DEFAULT_MAX_SIZE,
) -> None:
    """Append one MCP tool call record to the JSONL log.

    Raises:
        OSError: ``ELOOP`` when *log_path* is a symlink (never written through).
    """
    mode = usage_log_mode()
    if mode == "off":
        return
    if log_path is None:
        log_path = _default_log_path()
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": server,
        "tool": tool,
        "params": _safe_params(params, mode),
        "duration_ms": _duration_value(duration_ms),
        "success": success,
        "error": _scrub_error(error, params, mode),
        "session_id": _session_id(),
    }
    _write_log_entry(log_path, entry, max_size_bytes)


def _duration_value(duration_ms: float) -> float | str:
    """Round a finite duration to 0.1 ms; convert non-finite floats via :func:`_finite_float`."""
    return round(duration_ms, 1) if math.isfinite(duration_ms) else _finite_float(duration_ms)


def _finite_float(value: float) -> float | str:
    """Replace a non-finite float with its JSON-safe name; keep finite floats as-is."""
    if math.isnan(value):
        return "NaN"
    return _INF_STRINGS.get(value, value)


def _finite(value: Any) -> Any:
    """Recursively replace non-finite floats before ``json.dumps(..., allow_nan=False)``."""
    if isinstance(value, float):
        return _finite_float(value)
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _write_log_entry(log_path: Path, entry: dict, max_size_bytes: int) -> None:
    """Rotate *log_path* past *max_size_bytes*, then append *entry* as one JSON line."""
    ensure_private_dir(log_path.parent)
    if log_path.exists() and log_path.stat().st_size > max_size_bytes:
        rotated = log_path.with_suffix(f".{int(time.time())}.jsonl")
        log_path.rename(rotated)
    with _open_private_append(log_path) as f:
        f.write(json.dumps(_finite(entry), default=str, allow_nan=False) + "\n")


def _open_private_append(log_path: Path) -> IO[str]:
    """Open *log_path* for append, owner-only (0600), never following a symlink.

    ``fchmod`` also tightens a pre-existing world-readable log from older builds.
    """
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
    fd = os.open(log_path, flags, _LOG_FILE_MODE)
    os.fchmod(fd, _LOG_FILE_MODE)
    return os.fdopen(fd, "a")


def _safe_params(params: dict, mode: str) -> dict:
    """Sanitize *params* for the given usage log *mode* (``full`` or ``redacted``)."""
    if mode == "full":
        return {k: _full_value(v) for k, v in params.items()}
    return {k: _safe_value(k, v) for k, v in params.items()}


def _full_value(value: Any) -> Any:
    """Full mode (pre-B8 behaviour): summarise top-level bytes, truncate long strings."""
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str) and len(value) > _MAX_STR_LEN:
        return value[:_MAX_STR_LEN] + "..."
    return value


def _is_redacted_key(key: Any) -> bool:
    """True when every value under *key* must be redacted (case-insensitive allowlist)."""
    return str(key).lower() not in _UNREDACTED_KEYS


def _redacted(value: str | bytes) -> str:
    """Length-only placeholder — deliberately no digest (reversible for short values)."""
    return f"<redacted len={len(value)}>"


def _redact_all(value: Any) -> Any:
    """Redact every non-``None`` leaf of *value*, preserving dict/list shape and keys."""
    if isinstance(value, dict):
        return {k: _redact_all(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_all(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return _redacted(value)
    return _redacted(str(value))


def _safe_value(key: Any, value: Any) -> Any:
    """Redact a serial/path/credential key's value; otherwise sanitize it."""
    if _is_redacted_key(key):
        return _redact_all(value)
    return _sanitize(value)


def _sanitize(value: Any) -> Any:
    """Recurse into dicts/lists, summarise bytes, and truncate long strings."""
    if isinstance(value, dict):
        return {k: _safe_value(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return _full_value(value)


def _redacted_strings(value: Any) -> list[str]:
    """Collect every non-empty string leaf that redacted mode replaces in *value*."""
    if isinstance(value, dict):
        return [
            s
            for k, v in value.items()
            for s in (_string_leaves(v) if _is_redacted_key(k) else _redacted_strings(v))
        ]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _redacted_strings(v)]
    return []


def _string_leaves(value: Any) -> list[str]:
    """Return every non-empty string leaf of *value* (a value under a redacted key)."""
    if isinstance(value, dict):
        return [s for v in value.values() for s in _string_leaves(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _string_leaves(v)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _host_prefixes() -> list[str]:
    """Return the home, ``LINC_HOME`` and ``LINC_PROJECTS_DIR`` roots worth scrubbing.

    Empty and single-character values (``/``) are skipped so they never mangle text.
    """
    roots = [str(Path.home()), *filter(None, map(os.getenv, _HOST_PREFIX_VARS))]
    stripped = [root.rstrip(os.sep) for root in roots]
    return [root for root in stripped if len(root) > 1]


def _scrub_error(error: str | None, params: dict, mode: str) -> str | None:
    """In redacted mode, replace redacted parameter values, host roots and absolute paths."""
    if error is None or mode == "full":
        return error
    secrets = _redacted_strings(params) + _host_prefixes()
    for secret in sorted(secrets, key=len, reverse=True):
        error = error.replace(secret, _redacted(secret))
    return _ABS_PATH_RE.sub("<path>", error)
