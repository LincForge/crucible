"""Render the user's home dir as ``~`` in fuelcheck output and in what its probes log.

``crucible fuelcheck`` output is meant to be pasted into public issues, and that includes
stderr: the probes call real loaders (device policy, project discovery) whose warnings name
absolute paths. Those loggers are bound at import time, so they keep the processor list
object they were bound with; :func:`redacted_logs` therefore edits that live list in place
(as ``structlog.testing.capture_logs`` does) instead of reconfiguring structlog.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger


def redact_home(text: str, home: Path | None) -> str:
    """Render ``home`` (and paths under it) in ``text`` as ``~``.

    Args:
        text: Probe detail or log value that may embed absolute paths.
        home: The user's home dir. ``None``, ``/``, a one-character or a relative value leaves
            ``text`` untouched, so ``HOME=.`` can never turn ``e.g.`` into ``e~g~``.

    Returns:
        ``text`` with every whole-path occurrence of ``home`` replaced by ``~``.
    """
    root = str(home or "").rstrip("/")
    if len(root) < 2 or not os.path.isabs(root):
        return text
    return re.sub(re.escape(root) + r"(?![\w.-])", "~", text)


def _redact_value(value: Any, home: Path | None) -> Any:
    """Apply :func:`redact_home` to strings and paths, recursing into containers."""
    if isinstance(value, str | os.PathLike):
        return redact_home(str(value), home)
    if isinstance(value, dict):
        return {key: _redact_value(item, home) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_redact_value(item, home) for item in value]
    return value


def _home_redactor(home: Path | None) -> Processor:
    """A structlog processor rendering ``home`` as ``~`` in every event value.

    A logged exception is formatted first (its traceback names files under the home dir)
    and kept under ``traceback``, so the renderer never prints an unredacted one.
    """

    def redact(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        if event_dict.get("exc_info"):
            event_dict = structlog.processors.format_exc_info(logger, method_name, event_dict)
            event_dict["traceback"] = event_dict.pop("exception", "")
        return {key: _redact_value(value, home) for key, value in event_dict.items()}

    return redact


@contextmanager
def redacted_logs(home: Path | None) -> Iterator[None]:
    """Redact ``home`` from every structlog event emitted inside the block.

    Inserts :func:`_home_redactor` just before the renderer in structlog's live processor
    list and restores the list's contents (same object) on exit, even on error.
    """
    chain = structlog.get_config()["processors"]
    saved = list(chain)
    chain.insert(max(len(chain) - 1, 0), _home_redactor(home))
    try:
        yield
    finally:
        chain[:] = saved
