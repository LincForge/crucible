"""Regression: `crucible usage-stats` must never put structlog output on stdout.

`linc_crucible/cli.py` calls `configure_structlog_stderr()` at MODULE IMPORT TIME, above
the `linc_crucible.server_context` import — `server_context.py` binds a structlog logger
at import time (`logger = structlog.get_logger().bind(domain="build_pipeline")`), and
structlog's unconfigured default logger factory writes to **stdout**. Configuring any
later than this (e.g. inside a command callback) leaves that already-bound logger on
stdout and breaks `crucible usage-stats | jq`.

Runs in a SUBPROCESS rather than through Click's `CliRunner`: `PrintLoggerFactory(file=...)`
captures the stream *object* at configure time, and a module-level logger bound before a
`CliRunner` swaps `sys.stdout`/`sys.stderr` is invisible to that swap — only a real process
proves which stream a log line actually reached.
"""

from __future__ import annotations

import json
import subprocess
import sys


def test_usage_stats_keeps_structlog_off_stdout(tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "linc_crucible.cli", "usage-stats"],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **__import__("os").environ,
            # crucible_usage_stats reads $LINC_HOME/mcp-calls.jsonl — isolate it so this
            # test never reads the real production log and its output stays deterministic.
            "LINC_HOME": str(tmp_path / "linc_home"),
        },
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"

    payload = json.loads(proc.stdout)
    assert "log_path" in payload
    assert "registered_count" in payload

    # The tool's own initialization logging (forge registry lookups, structlog's default
    # PrintLoggerFactory writes to stdout if unconfigured) must never appear on stdout.
    assert "forge_registry_not_found" not in proc.stdout
    assert "domain=build_pipeline" not in proc.stdout


def test_usage_stats_cli_module_runnable_as_script() -> None:
    """`__main__`-style invocation exists so the subprocess test above can target the module."""
    import linc_crucible.cli as cli_module

    assert hasattr(cli_module, "main")
