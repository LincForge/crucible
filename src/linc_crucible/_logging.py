"""Structured logging configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import structlog
from structlog.typing import EventDict, WrappedLogger

from ._redaction import redact_event

logging = sys.modules.get("logging") or __import__("logging")


def redact_home_processor(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Structlog processor rewriting ``Path.home()`` to ``~`` in every event value.

    Shares :func:`._redaction.redact_event` with ``fuelcheck``: tuples/sets, exception
    objects and a logged traceback (``exc_info``, formatted before the renderer sees it)
    are redacted too.
    """
    try:
        home = Path.home()
    except RuntimeError:
        return event_dict
    return redact_event(logger, method_name, event_dict, home)


class _CurrentStderr:
    """A file-like object that writes to whatever ``sys.stderr`` is at write time.

    `PrintLoggerFactory(file=sys.stderr)` pins the stream object of the moment it is built.
    Under pytest that is a capture stream closed at session end, so a second session in the
    same process (mutmut's clean run) hit "I/O operation on closed file".
    """

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def configure_cli_structlog(level: str | None = None) -> None:
    """Point the CLI's structlog config at stderr with a level filter, WARNING by default.

    linc-cli-kit's own `configure_structlog_stderr()` only repoints structlog's default
    logger factory at stderr — it applies no level filter, so an import-time-bound logger
    (`structlog.get_logger().bind(...)` at module scope, e.g. `server_context.py`,
    `core/forge_integration.py`) prints every `.debug()`/`.info()` call unfiltered, ahead of
    the CLI's JSON error envelope. This calls the kit's helper first, then layers a
    `structlog.make_filtering_bound_logger` gate on top via a second `structlog.configure()`
    call — `structlog.configure()` merges keyword arguments into the existing global config
    rather than resetting it, so the stderr factory set by the first call survives.

    The supported knob is the `LINC_LOG_LEVEL` env var — NOT the kit's `--log-level` CLI
    flag. Click flags are parsed only after `cli.py`'s module-level imports have already run
    (and every import-time `logger.bind(...)` call already happened), too late to re-level
    an already-bound logger. Reads `LINC_LOG_LEVEL`, defaulting to WARNING; a name the
    stdlib `logging` module doesn't recognize also falls back to WARNING.

    Args:
        level: Explicit level name, overriding `LINC_LOG_LEVEL`. Exposed for tests; CLI
            entrypoints should rely on the env var instead of passing this.
    """
    from linc_cli_kit import configure_structlog_stderr

    configure_structlog_stderr()
    chain = structlog.get_config()["processors"]
    if redact_home_processor not in chain:
        chain.insert(-1, redact_home_processor)
    name = (level or os.environ.get("LINC_LOG_LEVEL") or "WARNING").upper()
    log_level = getattr(logging, name, logging.WARNING)
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=_CurrentStderr()),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )


def configure(*, verbose: bool = False) -> None:
    """Configure structured logging. Call once at startup."""
    log_level = logging.DEBUG if verbose else logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            redact_home_processor,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
