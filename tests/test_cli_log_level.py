"""Regression: the CLI's structlog config honors `LINC_LOG_LEVEL` (default WARNING).

Import-time-bound structlog loggers (`server_context.py`, `core/forge_integration.py`, ...)
inherit whatever config was active when the CLI module imported them. The kit's
`configure_structlog_stderr()` only redirects the default logger factory to stderr — it sets
no level filter, so a `logger.debug(...)` call anywhere in the import chain prints a
`[debug]` line ahead of the JSON error envelope. `_logging.configure_cli_structlog()` adds
the missing filter, read from `LINC_LOG_LEVEL` (default WARNING) — NOT the kit's `--log-level`
CLI flag, which is parsed too late to re-level an already-bound logger.

Runs in a SUBPROCESS (not `CliRunner`) for the same reason as
tests/test_cli_stdout_hygiene.py: `PrintLoggerFactory` captures the stream object at
configure time, and only a real process proves which stream a log line reaches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _run_score_nosuchapp(tmp_path, env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "LINC_HOME": str(tmp_path / "linc_home")}
    env.pop("LINC_LOG_LEVEL", None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "linc_crucible.cli", "score", "NoSuchApp"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_default_level_is_warning_no_debug_leak(tmp_path) -> None:
    proc = _run_score_nosuchapp(tmp_path, {})

    assert "[debug" not in proc.stderr
    assert "forge_registry_not_found" not in proc.stderr

    stderr_lines = [line for line in proc.stderr.splitlines() if line.strip()]
    assert stderr_lines, f"expected an error envelope line, got stderr={proc.stderr!r}"
    envelope = json.loads(stderr_lines[-1])
    assert envelope["error"]["category"] == "not_found"


def test_linc_log_level_debug_is_honored(tmp_path) -> None:
    proc = _run_score_nosuchapp(tmp_path, {"LINC_LOG_LEVEL": "DEBUG"})

    assert "forge_registry_not_found" in proc.stderr
