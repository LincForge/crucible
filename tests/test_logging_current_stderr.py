"""structlog follows the CURRENT ``sys.stderr``, never the stream that existed at configure time.

`cli.py` calls `_logging.configure_cli_structlog()` at import. linc-cli-kit's
`configure_structlog_stderr()` builds `PrintLoggerFactory(file=sys.stderr)`, which pins the
stream object of that moment. Under pytest that object is the first session's capture
stream, closed when the session ends, so a second `pytest.main` in the same process (the
mutmut clean run) died on "I/O operation on closed file" and pr-diff-mutation reported an
instrument error (PR #196).
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator

import pytest
import structlog

from linc_crucible import _logging


@pytest.fixture(autouse=True)
def _restore_structlog() -> Iterator[None]:
    saved = structlog.get_config()
    try:
        yield
    finally:
        structlog.configure(**saved)


def test_contract_log_after_stderr_swap_reaches_the_new_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[contract] Hostile: the stream present at configure time is closed and replaced; a
    later warning lands on the replacement and nothing raises."""
    first = io.StringIO()
    monkeypatch.setattr(sys, "stderr", first)
    _logging.configure_cli_structlog("WARNING")
    first.close()
    second = io.StringIO()
    monkeypatch.setattr(sys, "stderr", second)
    structlog.get_logger().warning("after_swap", note="")
    assert "after_swap" in second.getvalue()


def test_contract_import_time_bound_logger_follows_the_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[contract] A logger bound before the swap (the module-scope `.bind()` pattern) also
    writes to the current stream."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    _logging.configure_cli_structlog("WARNING")
    bound = structlog.get_logger().bind(domain="build_pipeline")
    sys.stderr.close()
    current = io.StringIO()
    monkeypatch.setattr(sys, "stderr", current)
    bound.warning("bound_after_swap")
    assert "bound_after_swap" in current.getvalue()


def test_level_filter_still_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    current = io.StringIO()
    monkeypatch.setattr(sys, "stderr", current)
    _logging.configure_cli_structlog("WARNING")
    structlog.get_logger().info("quiet_info")
    assert "quiet_info" not in current.getvalue()
