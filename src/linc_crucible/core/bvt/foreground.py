"""Pure helpers for the BVT launch/navigate foreground-wait window.

Split out of ``step_handlers.py`` so ``_wait_for_foreground``'s own branching
stays under its CRAP-ratchet complexity baseline — the message-building,
timeout-parsing, and error-bookkeeping decisions live here instead.

Deliberately has NO import of ``step_handlers.py`` (which imports THIS
module) — the local ``_coerce_duration`` below duplicates
``step_handlers._duration_seconds``'s small "12"/"12s"/"12000ms" parsing
convention rather than importing it, to keep this a one-directional, leaf
dependency.
"""

from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

# Parse "3s", "500ms", "1.5s" into seconds — same convention as
# step_handlers._duration_seconds (wait_for_element / dismiss_dialog / retry_backoff).
_DURATION_RE = re.compile(r"\A(\d+(?:\.\d+)?)[ \t]*(s|ms)\Z")


def _coerce_duration(raw: object, default: float) -> float:
    """Coerce a duration param to seconds; unparseable -> ``default`` + warning.

    Same convention/behavior as ``step_handlers._duration_seconds`` (kept as
    an independent copy here — see module docstring).
    """
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    m = _DURATION_RE.fullmatch(str(raw).strip())
    if m:
        value = float(m.group(1))
        return value / 1000.0 if m.group(2) == "ms" else value
    try:
        return float(raw)
    except (ValueError, TypeError):
        log.warning("bvt_bad_duration", what="foreground_timeout", value=raw, default=default)
        return default


def foreground_window_s(params: dict, default: float, maximum: float) -> float:
    """Per-step foreground-settle window from ``params['timeout']``.

    Same duration convention as ``wait_for_element``/``dismiss_dialog``
    (``12``, ``"12s"``, ``"12000ms"``). Absent or unparseable falls back to
    ``default`` (``_coerce_duration`` already warns ``bvt_bad_duration`` on
    the unparseable case). A parsed value that is zero or negative ALSO falls
    back to ``default`` — logged distinctly (``bvt_bad_foreground_timeout``)
    since it's a value-range problem, not a parse failure. Anything above
    ``maximum`` is clamped down to it (``bvt_foreground_timeout_clamped``) so
    a runaway ``timeout: 9999`` can't pin a device on an unattended runner.
    """
    raw = (params or {}).get("timeout")
    value = _coerce_duration(raw, default)
    if value <= 0:
        log.warning("bvt_bad_foreground_timeout", value=raw, default=default)
        return default
    if value > maximum:
        log.warning("bvt_foreground_timeout_clamped", requested=value, max=maximum)
        return maximum
    return value


def latest_nerve_error(new: str | None, previous: str | None) -> str | None:
    """Keep the most recent non-empty nerve error seen across polls.

    Extracted so ``_wait_for_foreground``'s own loop body has one fewer
    branch (a bare function call instead of an inline ``or``).
    """
    return new if new else previous


def foreground_default_s(tags: list[str], normal: float, cold_start: float) -> float:
    """Per-test foreground-wait default: a ``cold_start``-tagged test gets the
    longer window (first-install cold boot is slower than a warm relaunch)."""
    return cold_start if "cold_start" in tags else normal


def foreground_failure_message(
    device: str,
    target_package: str,
    window_s: float,
    polls: int,
    last_seen: str | None,
    last_error: str | None,
) -> str:
    """Build ``_wait_for_foreground``'s failure text.

    Two distinct failure shapes:
      * no readable foreground EVER (``last_seen`` is ``None``) — cites the
        last nerve error nerve itself reported, or says plainly that none
        was reported. Crucible never authors "daemon unreachable" here — that
        phrase can only appear if it's nerve's own error text.
      * a readable-but-wrong foreground — the normal "did not foreground".
    """
    if last_seen is None:
        if last_error:
            return (
                f"could not read foreground for {device} after {window_s}s "
                f"({polls} polls) — last nerve error: {last_error} "
                f"(launched {target_package})"
            )
        return (
            f"could not read foreground for {device} after {window_s}s "
            f"({polls} polls) — nerve_state returned no foreground and no "
            f"error (launched {target_package})"
        )
    return (
        f"app did not foreground: launched {target_package}, foreground is "
        f"{last_seen} after {window_s}s ({polls} polls)"
    )
