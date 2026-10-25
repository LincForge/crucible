"""BVT step handlers mixin."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from linc_crucible.core.bvt.assertions import (
    assert_clickable,
    assert_content_description,
    assert_element,
    assert_element_count,
    assert_element_not_present,
    assert_min_size,
    assert_screen_reader,
    assert_text,
    assert_text_not_present,
    assert_touch_target,
    min_touch_target_px,
)
from linc_crucible.core.bvt.foreground import (
    foreground_failure_message,
    foreground_window_s,
    latest_nerve_error,
)
from linc_crucible.core.bvt.nerve_adapter import DEFAULT_LONG_PRESS_MS
from linc_crucible.core.nerve_client import UiTreeUnavailableError

if TYPE_CHECKING:
    from linc_crucible.core.bvt.schema import BVTStep

log = structlog.get_logger()

# Parse "3s", "500ms", "1.5s" into seconds
_DURATION_RE = re.compile(r"\A(\d+(?:\.\d+)?)[ \t]*(s|ms)\Z")

# Foreground-settle window after launch/deeplink/navigate. `timeout` (params)
# overrides per-step; `_FOREGROUND_TIMEOUT_S` (normal) / `_COLD_START_FOREGROUND_TIMEOUT_S`
# (a `cold_start`-tagged test, whose first-install boot is slower) are the
# per-test default when no `timeout` param is given — see `foreground_default_s`
# / `foreground_window_s` in foreground.py. `_FOREGROUND_TIMEOUT_MAX_S` is a hard
# clamp so a runaway `timeout: 9999` can't pin a device on an unattended runner.
_FOREGROUND_TIMEOUT_S = 5.0
_COLD_START_FOREGROUND_TIMEOUT_S = 15.0
_FOREGROUND_TIMEOUT_MAX_S = 60.0
_FOREGROUND_POLL_INTERVAL_S = 0.5


def _parse_dimension(value: object, density_factor: float) -> int:
    """Parse a size dimension to pixels. ``"56dp"`` -> dp*factor; ``"154px"``/``"154"`` -> px.

    Tolerant of whitespace and floats (``"56.0 dp"``).
    """
    s = str(value).strip().lower().replace(" ", "")
    if s.endswith("dp"):
        return round(float(s[:-2]) * density_factor)
    if s.endswith("px"):
        return round(float(s[:-2]))
    return round(float(s))


def _parse_duration(s: str) -> float:
    """Parse a duration string into seconds."""
    m = _DURATION_RE.fullmatch(s.strip())
    if not m:
        # Default: try as seconds
        try:
            return float(s)
        except ValueError:
            return 1.0
    value = float(m.group(1))
    unit = m.group(2)
    return value / 1000.0 if unit == "ms" else value


# wait_for_element poll window (params override `timeout`); poll cadence matches
# dismiss_dialog. Per-step `retries` backoff + hard cap (autonomy runaway guard).
_WAIT_FOR_ELEMENT_TIMEOUT_S = 5.0
_WAIT_POLL_INTERVAL_S = 0.5
_DEFAULT_RETRY_BACKOFF_S = 0.5
_MAX_RETRIES = 10
_WAIT_MAX_S = 120.0
_WAIT_FOR_ELEMENT_MAX_S = 120.0
_DISMISS_TIMEOUT_MAX_S = 30.0

# Actions whose handler is NOT safe to blindly re-run: a partially-succeeded
# `type` re-types the whole string (foo -> foofoo); long_press may double-fire a
# context menu. retries on these warns (author override still allowed).
_NON_IDEMPOTENT_ACTIONS = frozenset({"type", "long_press"})


def _duration_seconds(
    raw: object,
    default: float,
    *,
    what: str,
    maximum: float | None = None,
) -> float:
    """Coerce a duration param to seconds. Never raises (an AI-authored typo
    must not crash a suite).

    ``None`` -> ``default``; ``int``/``float`` -> ``float`` (YAML writes
    ``timeout: 5`` as an int); ``"5s"``/``"500ms"``/``"1.5"`` parsed; anything
    unparseable -> ``default`` + a ``log.warning``.
    """
    if raw is None:
        value = default
    elif isinstance(raw, (int, float)):
        value = float(raw)
    else:
        m = _DURATION_RE.fullmatch(str(raw).strip())
        if m:
            val = float(m.group(1))
            value = val / 1000.0 if m.group(2) == "ms" else val
        else:
            try:
                value = float(raw)
            except (ValueError, TypeError):
                log.warning("bvt_bad_duration", what=what, value=raw, default=default)
                value = default

    if maximum is not None and value > maximum:
        log.warning(
            "bvt_duration_clamped",
            what=what,
            requested=value,
            max=maximum,
            maximum=maximum,
        )
        return maximum
    return value


def _parse_retries(params: dict) -> int:
    """Extra attempts after the first, from ``params['retries']``.

    Malformed (non-int) -> 0 + warning; negative/zero -> 0; clamped to
    ``_MAX_RETRIES`` (+ warning) so a runaway ``retries: 9999`` can't pin a
    device on an unattended runner. Never raises.
    """
    raw = params.get("retries", 0)
    try:
        n = int(raw)
    except (ValueError, TypeError):
        log.warning("bvt_bad_retries", value=raw)
        return 0
    if n <= 0:
        return 0
    if n > _MAX_RETRIES:
        log.warning("bvt_retries_clamped", requested=n, max=_MAX_RETRIES)
        return _MAX_RETRIES
    return n


@dataclass
class BVTStepResult:
    """Result of a single BVT step."""

    action: str
    passed: bool
    message: str = ""
    screenshot_path: str | None = None
    performance: dict | None = None
    # Screenshots/perf produced by an `if` step's branch sub-steps. Empty for
    # ordinary steps; surfaced into the test result by _run_test.
    extra_screenshots: list[str] = field(default_factory=list)
    extra_performance: list[dict] = field(default_factory=list)
    # True when this step failed because the UI tree itself could not be read
    # (UiTreeUnavailableError via `_assert_tree`) — a harness fault, never a
    # product failure. `passed` stays False either way; this only relabels it.
    harness_error: bool = False


@dataclass
class BVTTestResult:
    """Result of a single BVT test (collection of steps)."""

    name: str
    passed: bool
    package: str = ""
    skipped: bool = False
    skip_reason: str = ""
    steps_run: int = 0
    steps_passed: int = 0
    failure_message: str = ""
    screenshots: list[str] = field(default_factory=list)
    duration_ms: int = 0
    performance_snapshots: list[dict] = field(default_factory=list)
    # True when the test's fail-fast step was a harness fault (UI tree
    # unreadable), not a product failure — see BVTStepResult.harness_error.
    harness_error: bool = False


@dataclass
class BVTRunResult:
    """Result of running an entire BVT suite."""

    device: str
    package: str
    tests: list[BVTTestResult] = field(default_factory=list)
    duration_ms: int = 0
    # Non-empty when the whole suite was skipped by the device-readiness gate
    # (device not ready/absent). Lets callers (crucible_bvt) distinguish
    # "environment wasn't ready" from a run that actually executed.
    suite_skip_reason: str = ""

    @property
    def total_run(self) -> int:
        return len(self.tests)

    @property
    def _executed(self) -> list[BVTTestResult]:
        """Tests that actually ran — skipped tests are neither pass nor fail."""
        return [t for t in self.tests if not t.skipped]

    @property
    def total_passed(self) -> int:
        return sum(1 for t in self._executed if t.passed)

    @property
    def total_failed(self) -> int:
        return sum(1 for t in self._executed if not t.passed)

    @property
    def total_skipped(self) -> int:
        return sum(1 for t in self.tests if t.skipped)

    @property
    def total_harness_errors(self) -> int:
        """Count of executed (non-skipped) tests whose fail-fast step was a
        harness fault (UI tree unreadable) rather than a product failure."""
        return sum(1 for t in self._executed if t.harness_error)

    @property
    def passed(self) -> bool:
        executed = self._executed
        return all(t.passed for t in executed) and len(executed) > 0


def _safe_file_stem(text: str) -> str:
    """Sanitize a text stem for safe filename construction.

    Keep [A-Za-z0-9._-], collapse the rest to "_", and strip leading dots.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).lstrip(".")


def _split_tree(tree: list[dict] | BVTStepResult) -> tuple[list[dict], BVTStepResult | None]:
    """Split a ``_elements_or_harness`` read into ``(elements, harness_result)``.

    An unreadable tree becomes ``([], harness_result)`` so a poll loop can keep
    polling over "no elements" and only surface the harness result at its
    deadline; a readable tree is ``(elements, None)``.
    """
    if isinstance(tree, BVTStepResult):
        return [], tree
    return tree, None


def _dismiss_timeout_result(
    target: str, timeout: float, optional: bool, tapped: bool
) -> BVTStepResult:
    """Verdict for a ``dismiss_dialog`` whose window closed on a READABLE tree.

    Never tapped -> the dialog never appeared; tapped -> it is still on screen.
    ``optional`` forgives either (it never forgives an unreadable tree — the
    caller returns the harness result before reaching here).
    """
    what = "still present after" if tapped else "not found within"
    return BVTStepResult(
        action="dismiss_dialog",
        passed=optional,
        message=(
            f"Dialog '{target}' {what} {timeout}s" + (" (optional, ignored)" if optional else "")
        ),
    )


class BVTStepHandlersMixin:
    """Mixin containing step execution handlers for BVTRunner."""

    # Per-test foreground-wait default (`launch`/`navigate` window when the
    # step has no `timeout` param) — `_run_test` (runner.py) sets this
    # unconditionally before running a test's steps via `foreground_default_s`.
    # The class-level fallback below only guards a handler invoked outside
    # that path (e.g. a future direct unit test).
    _foreground_default_s: float = _FOREGROUND_TIMEOUT_S

    def _capture_metrics(self, device: str, package: str) -> dict | None:
        """Capture performance snapshot. Returns None on failure."""
        try:
            return self._nerve.metrics(device, package)
        except Exception as e:
            log.warning("bvt_metrics_capture_failed", error=str(e))
            return None

    def _execute_step(
        self, step: BVTStep, device: str, package: str, test_name: str
    ) -> BVTStepResult:
        """Dispatch a single step to the appropriate handler."""
        dispatch = {
            "launch": self._do_launch,
            "wait": self._do_wait,
            "wait_for_element": self._do_wait_for_element,
            "scroll_to": self._do_scroll_to,
            "tap": self._do_tap,
            "assert_element": self._do_assert_element,
            "assert_text": self._do_assert_text,
            "assert_content_description": self._do_assert_content_description,
            "assert_touch_target": self._do_assert_touch_target,
            "assert_screen_reader": self._do_assert_screen_reader,
            "screenshot": self._do_screenshot,
            "kill": self._do_kill,
            "grant_permission": self._do_grant_permission,
            "navigate": self._do_navigate,
            "deeplink": self._do_navigate,
            "dismiss_dialog": self._do_dismiss_dialog,
            "type": self._do_type,
            "long_press": self._do_long_press,
            "tap_xy": self._do_tap_xy,
            "tap_text": self._do_tap_text,
            "press_key": self._do_press_key,
            "ime_action": self._do_press_key,
            "enter": self._do_press_key,
            "assert_clickable": self._do_assert_clickable,
            "assert_min_size": self._do_assert_min_size,
            "assert_element_not_present": self._do_assert_element_not_present,
            "assert_text_not_present": self._do_assert_text_not_present,
            "assert_element_count": self._do_assert_element_count,
            "if": self._do_if,
        }

        handler = dispatch.get(step.action)
        if handler is None:
            return BVTStepResult(
                action=step.action,
                passed=False,
                message=f"Unknown action: {step.action}",
            )

        return self._guarded(handler, step, device, package, test_name)

    def _guarded(
        self,
        handler: Callable[..., BVTStepResult],
        step: BVTStep,
        device: str,
        package: str,
        test_name: str,
    ) -> BVTStepResult:
        """Wrap handler invocation so unexpected exceptions become harness errors."""
        try:
            return handler(step, device, package, test_name)
        except Exception as exc:
            return BVTStepResult(
                action=step.action,
                passed=False,
                message=f"step {step.action} raised {type(exc).__name__}: {exc}",
                harness_error=True,
            )

    def _execute_step_with_retries(
        self, step: BVTStep, device: str, package: str, test_name: str
    ) -> BVTStepResult:
        """Dispatch a step, retrying on a failed result (not on exception).

        ``retries: N`` (step.params) = N extra attempts after the first, with a
        ``retry_backoff``-second sleep between (default
        ``_DEFAULT_RETRY_BACKOFF_S``; ``"5s"``/``"500ms"`` accepted). Only
        ``passed=False`` retries — a raised nerve error propagates (a dead daemon
        is a hard failure, not a flake). Retrying re-runs the whole handler
        including side effects; ``type``/``long_press`` warn (re-run is not
        idempotent). Returns the final attempt's result.
        """
        retries = _parse_retries(step.params)
        if retries > 0 and step.action in _NON_IDEMPOTENT_ACTIONS:
            log.warning("bvt_retries_non_idempotent", action=step.action, retries=retries)
        backoff = _duration_seconds(
            step.params.get("retry_backoff"), _DEFAULT_RETRY_BACKOFF_S, what="retry_backoff"
        )
        result = self._execute_step(step, device, package, test_name)
        attempt = 0
        while not result.passed and attempt < retries:
            attempt += 1
            log.info("bvt_step_retry", action=step.action, attempt=attempt, of=retries)
            time.sleep(backoff)
            result = self._execute_step(step, device, package, test_name)
        return result

    def _eval_condition(self, condition: dict, device: str) -> bool | BVTStepResult:
        """Evaluate an if-condition against ONE fresh STRICT element dump.

        Exactly one of element/text is guaranteed by the parser; defaults to
        False if neither is present (defensive). An unreadable tree returns the
        harness ``BVTStepResult`` instead of a bool — it is no evidence either
        way, so neither branch may run on it.
        """
        elems = self._elements_or_harness(device, "if")
        if isinstance(elems, BVTStepResult):
            return elems
        if "element" in condition:
            return assert_element(elems, condition["element"]).passed
        if "text" in condition:
            return assert_text(elems, str(condition["text"])).passed
        return False

    def _run_branch(
        self,
        sub_steps: list[BVTStep],
        device: str,
        package: str,
        test_name: str,
        matched: bool,
        branch: str,
    ) -> BVTStepResult:
        """Run a then/else branch's sub-steps, fail-fast, aggregate into one result.

        Surfaces every sub-step screenshot/perf via extra_screenshots/extra_performance.
        On failure the message names the failing sub-step (branch + index + action:target)
        so a device-less CI log is debuggable, and the sub-step's ``harness_error``
        carries onto the ``if`` result (a harness fault stays a harness fault).
        """
        screenshots: list[str] = []
        perfs: list[dict] = []
        for i, s in enumerate(sub_steps, start=1):
            r = self._execute_step_with_retries(s, device, package, test_name)
            if r.screenshot_path:
                screenshots.append(r.screenshot_path)
            if r.performance:
                perfs.append(r.performance)
            if not r.passed:
                return BVTStepResult(
                    action="if",
                    passed=False,
                    message=(
                        f"if(matched={matched}) {branch}-step {i} "
                        f"'{s.action}:{s.target}' failed: {r.message}"
                    ),
                    extra_screenshots=screenshots,
                    extra_performance=perfs,
                    harness_error=r.harness_error,
                )
        return BVTStepResult(
            action="if",
            passed=True,
            message=f"if(matched={matched}): {len(sub_steps)} {branch}-step(s) passed",
            extra_screenshots=screenshots,
            extra_performance=perfs,
        )

    def _do_if(self, step: BVTStep, device: str, package: str, test_name: str) -> BVTStepResult:
        """Evaluate the condition (fresh dump) and run the matching branch.

        An unreadable condition tree is returned as the step's harness result;
        no branch runs.
        """
        condition = step.params.get("condition", {})
        matched = self._eval_condition(condition, device)
        if isinstance(matched, BVTStepResult):
            return matched
        branch = "then" if matched else "else"
        sub_steps = step.params.get(branch, [])
        return self._run_branch(sub_steps, device, package, test_name, matched, branch)

    def _ios_foreground_fallback(
        self, device: str, target_package: str, action: str, saw_valid: bool
    ) -> BVTStepResult | None:
        """Last-resort launch verification when NO foreground was ever read.

        On iOS simulators nerve's state carries no ``foreground`` key
        (structural — no WDA on the bench), so consult ``nerve_inspect``
        once: an iOS response with the target process running is a
        degraded-but-honest verification. Returns ``None`` (no fallback
        applies) when a foreground WAS seen at some point (``saw_valid``),
        or the inspect result isn't an iOS process-running one — Android
        deliberately never takes this path: foreground IS supported there,
        so empty reads mean daemon trouble, and a running process can
        legitimately be backgrounded.
        """
        if saw_valid:
            return None
        info = self._nerve.inspect(device, target_package)
        if info.get("platform") != "ios" or not (info.get("process") or {}).get("running"):
            return None
        return BVTStepResult(
            action=action,
            passed=True,
            message=(
                f"iOS: foreground detection unavailable; verified "
                f"{target_package} process running via nerve_inspect"
            ),
        )

    def _wait_for_foreground(
        self, device: str, target_package: str, action: str, window_s: float
    ) -> BVTStepResult:
        """Poll until ``target_package`` is foreground or ``window_s`` elapses.

        Uses ``NerveAdapter.foreground_probe`` (not the lossy ``foreground``)
        so a nerve-reported error or transport exception is preserved for the
        eventual failure message instead of being reduced to ``{}``. A wrong
        package keeps polling until the deadline, then fails-fast. Message
        branches live in ``foreground_failure_message`` (foreground.py) to
        keep this loop's own complexity low.
        """
        deadline = time.monotonic() + window_s
        saw_valid = False
        last_seen: str | None = None
        last_error: str | None = None
        polls = 0
        while True:
            polls += 1
            probe = self._nerve.foreground_probe(device)
            last_error = latest_nerve_error(probe.get("error"), last_error)
            fg = probe.get("foreground", {})
            pkg = fg.get("package")
            if pkg:
                saw_valid = True
                last_seen = f"{pkg}/{fg.get('activity')}"
                if pkg == target_package:
                    return BVTStepResult(action=action, passed=True, message="")
            if time.monotonic() < deadline:
                time.sleep(_FOREGROUND_POLL_INTERVAL_S)
                continue
            fallback = self._ios_foreground_fallback(device, target_package, action, saw_valid)
            if fallback is not None:
                return fallback
            message = foreground_failure_message(
                device, target_package, window_s, polls, last_seen, last_error
            )
            return BVTStepResult(action=action, passed=False, message=message)

    def _do_launch(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        target_package = step.target or package
        result = self._nerve.launch(device, target_package)
        if not result.get("success", False):
            return BVTStepResult(action="launch", passed=False, message=result.get("error", ""))
        window_s = foreground_window_s(
            step.params, self._foreground_default_s, _FOREGROUND_TIMEOUT_MAX_S
        )
        return self._wait_for_foreground(
            device, target_package, action="launch", window_s=window_s
        )

    def _do_kill(self, step: BVTStep, device: str, package: str, _test_name: str) -> BVTStepResult:
        target_package = step.target or package
        result = self._nerve.kill(device, target_package)
        return BVTStepResult(
            action="kill",
            passed=result.get("success", False),
            message=result.get("error", ""),
        )

    def _do_grant_permission(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        """Grant a runtime permission (Android ``pm grant``).

        ``grant_permission: android.permission.CAMERA`` — the target is the
        full permission string; the app is the test/suite package. Pair this
        with a ``cold_start`` tag to re-grant permissions that ``pm clear``
        revoked, "as part of the test's flow".
        """
        permission = step.target
        if not permission:
            return BVTStepResult(
                action="grant_permission",
                passed=False,
                message="grant_permission requires a permission string target",
            )
        result = self._nerve.grant_permission(device, package, permission)
        return BVTStepResult(
            action="grant_permission",
            passed=result.get("success", False),
            message=result.get("error", ""),
        )

    def _do_navigate(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        result = self._nerve.navigate(device, step.target)
        if not result.get("success", False):
            return BVTStepResult(action="navigate", passed=False, message=result.get("error", ""))
        window_s = foreground_window_s(
            step.params, self._foreground_default_s, _FOREGROUND_TIMEOUT_MAX_S
        )
        return self._wait_for_foreground(device, package, action="navigate", window_s=window_s)

    def _do_dismiss_dialog(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        """Dismiss a modal dialog by tapping a button matching text.

        Polls for the dialog button up to ``timeout`` seconds (default 3) and
        VERIFIES it actually closed: each iteration, if the target text is on
        screen it taps it (retrying), and only returns success once the target
        text is GONE after having been tapped. A tap can be injected without
        dismissing an animated/stateful modal, so confirming the dialog left the
        screen — not just that the tap call succeeded — is required.

        If ``optional`` is true, a dialog that never appears (or won't close) is
        not a failure. The tree is read STRICTLY: an unreadable tree is polled
        through like an absent button, never counts as "dismissed", and if the
        last read of the window is unreadable the step is a harness fault
        (``optional`` does not forgive that).
        """
        timeout = _duration_seconds(
            step.params.get("timeout"),
            3.0,
            what="dismiss_dialog.timeout",
            maximum=_DISMISS_TIMEOUT_MAX_S,
        )
        optional = bool(step.params.get("optional", False))
        target = step.target
        target_lc = target.lower()

        deadline = time.monotonic() + timeout
        tapped = False
        while True:
            elems, harness = _split_tree(self._elements_or_harness(device, "dismiss_dialog"))
            present = any(target_lc in (e.get("text", "") or "").lower() for e in elems)
            if present:
                self._nerve.tap_by_text(device, target)
                tapped = True
            elif tapped and harness is None:
                # Was present, tapped, and a READ tree shows it gone — dismissed.
                return BVTStepResult(
                    action="dismiss_dialog",
                    passed=True,
                    message=f"Dismissed dialog button '{target}'",
                )
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)

        return harness or _dismiss_timeout_result(target, timeout, optional, tapped)

    def _do_wait_for_element(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        """Poll the element tree until an element with the target id appears.

        Replaces the brittle ``wait: 3s`` + ``assert_element`` pair. Matches by
        resource id (exact or ``/suffix``) via ``assert_element``. ``timeout``
        comes from params (``"5s"``/``"500ms"``/int; default
        ``_WAIT_FOR_ELEMENT_TIMEOUT_S``). Deadline uses ``time.monotonic()`` so a
        wall-clock change mid-poll can't corrupt it. The tree is read STRICTLY
        (``_elements_or_harness``): an unreadable tree is polled through like an
        absent element until the deadline, and if the last read was unreadable
        the step is a harness fault (``harness_error=True``, ``"HARNESS: ..."``)
        rather than "did not appear (last saw 0 elements)".
        """
        target = step.target or str(step.params.get("target", ""))
        if not target:
            return BVTStepResult(
                action="wait_for_element",
                passed=False,
                message="wait_for_element requires a target element id",
            )
        timeout = _duration_seconds(
            step.params.get("timeout"),
            _WAIT_FOR_ELEMENT_TIMEOUT_S,
            what="wait_for_element.timeout",
            maximum=_WAIT_FOR_ELEMENT_MAX_S,
        )
        start = time.monotonic()
        deadline = start + timeout
        last_count = 0
        while True:
            elems, harness = _split_tree(self._elements_or_harness(device, "wait_for_element"))
            last_count = len(elems)
            if assert_element(elems, target).passed:
                elapsed = time.monotonic() - start
                return BVTStepResult(
                    action="wait_for_element",
                    passed=True,
                    message=f"element '{target}' appeared after {elapsed:.1f}s",
                )
            if time.monotonic() >= deadline:
                return harness or BVTStepResult(
                    action="wait_for_element",
                    passed=False,
                    message=(
                        f"element '{target}' did not appear within {timeout}s "
                        f"(last saw {last_count} elements)"
                    ),
                )
            time.sleep(_WAIT_POLL_INTERVAL_S)

    def _do_scroll_to(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        """Swipe until ``step.target`` is visible (scroll-until-visible; no tap).

        Delegates to ``NerveAdapter.scroll_to_element``, which shares the
        density/screen-size plumbing already used by ``assert_min_size``.
        ``params.max_swipes`` is optional (adapter default applies when absent
        or unparseable); does not tap — pair with a ``tap``/``assert_element``
        step.
        """
        target = step.target or str(step.params.get("target", ""))
        if not target:
            return BVTStepResult(
                action="scroll_to",
                passed=False,
                message="scroll_to requires a target element id",
            )

        kwargs: dict = {}
        if "max_swipes" in step.params:
            try:
                kwargs["max_swipes"] = int(step.params["max_swipes"])
            except (TypeError, ValueError):
                log.warning("bvt_bad_max_swipes", value=step.params["max_swipes"])

        result = self._nerve.scroll_to_element(device, target, **kwargs)
        if result.get("harness_error"):
            return BVTStepResult(
                action="scroll_to",
                passed=False,
                message=str(result.get("error")),
                harness_error=True,
            )
        swipes = result.get("swipes", 0)
        if result.get("found"):
            return BVTStepResult(
                action="scroll_to",
                passed=True,
                message=f"'{target}' visible after {swipes} swipes",
            )
        err = result.get("error")
        message = f"'{target}' not visible after {swipes} swipes"
        if err:
            message += f" ({err})"
        return BVTStepResult(action="scroll_to", passed=False, message=message)

    def _do_wait(
        self, step: BVTStep, _device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        raw = step.target or ((step.params or {}).get("duration") if step.params else None)
        duration = _duration_seconds(raw, 1.0, what="wait.duration", maximum=_WAIT_MAX_S)
        time.sleep(duration)
        return BVTStepResult(action="wait", passed=True, message=f"Waited {duration}s")

    def _do_tap(self, step: BVTStep, device: str, _package: str, _test_name: str) -> BVTStepResult:
        result = self._nerve.tap_by_id(device, step.target)
        return BVTStepResult(
            action="tap",
            passed=result.get("success", False),
            message=result.get("error", ""),
            harness_error=bool(result.get("harness_error", False)),
        )

    def _do_tap_xy(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        """Tap at an absolute (x, y) coordinate read from ``step.params``.

        For flows where no stable resource-id or visible text anchor exists. Both
        ``x`` and ``y`` are required; a missing coord fails-fast rather than
        guessing 0 (a silent tap at the origin would be a misleading no-op).
        """
        if "x" not in step.params or "y" not in step.params:
            return BVTStepResult(
                action="tap_xy",
                passed=False,
                message="tap_xy requires both 'x' and 'y' in params",
            )
        try:
            x = int(step.params["x"])
            y = int(step.params["y"])
        except (TypeError, ValueError):
            return BVTStepResult(
                action="tap_xy",
                passed=False,
                message=f"tap_xy x/y must be integers, got x={step.params['x']!r} "
                f"y={step.params['y']!r}",
            )
        # tap_xy is a literal-pixel escape hatch: tap the exact coordinate with
        # no synthesized down→up duration. The duration-primary fix is for
        # element-resolved taps (tap_by_id).
        result = self._nerve.tap(device, x, y)
        return BVTStepResult(
            action="tap_xy",
            passed=result.get("success", False),
            message=result.get("error", ""),
        )

    def _do_tap_text(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        """Tap a visible element by its on-screen text.

        Resolves the text to coordinates via the nerve element tree (NOT OCR) and
        taps the element center. Delegates to ``NerveAdapter.tap_by_text``.
        """
        result = self._nerve.tap_by_text(device, step.target)
        return BVTStepResult(
            action="tap_text",
            passed=result.get("success", False),
            message=result.get("error", ""),
            harness_error=bool(result.get("harness_error", False)),
        )

    def _do_press_key(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        """Press a hardware/IME key (default ENTER) to submit IME-"Done" forms.

        Dispatched from ``press_key`` / ``ime_action`` / ``enter``. The key name
        comes from ``step.target`` (e.g. ``press_key: BACK``) or
        ``step.params["key"]``; an empty target defaults to ENTER so the bare
        ``enter`` / ``ime_action`` shorthand submits the focused field.
        """
        key = step.target or step.params.get("key") or "ENTER"
        result = self._nerve.key(device, key)
        return BVTStepResult(
            action="press_key",
            passed=result.get("success", False),
            message=result.get("error", ""),
        )

    def _do_type(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        result = self._nerve.type(device, step.target)
        return BVTStepResult(
            action="type",
            passed=result.get("success", False),
            message=result.get("error", ""),
        )

    def _do_long_press(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        duration_ms = int(step.params.get("duration_ms", DEFAULT_LONG_PRESS_MS))
        result = self._nerve.long_press_by_id(device, step.target, duration_ms=duration_ms)
        return BVTStepResult(
            action="long_press",
            passed=result.get("success", False),
            message=result.get("error", ""),
            harness_error=bool(result.get("harness_error", False)),
        )

    def _assert_tree(self, device: str, action: str) -> list | BVTStepResult:
        """Read the element tree STRICTLY for an assert step.

        Returns the element list, or a harness-error ``BVTStepResult`` when
        the daemon's dump could not be read (``UiTreeUnavailableError`` from
        ``elements_strict`` — dead daemon, HTTP error, or a response missing/
        flagging an error). Callers must check ``isinstance(result,
        BVTStepResult)`` and return it as-is before asserting against it — a
        harness fault must never read as "0 elements" (a false product
        failure) or let a `*_not_present` assert vacuously pass.
        """
        return self._elements_or_harness(device, action)

    def _elements_or_harness(self, device: str, action: str) -> list[dict] | BVTStepResult:
        """THE strict tree read for every step that decides on the UI tree.

        Returns ``self._nerve.elements_strict(device)``, or — when the dump
        cannot be read (``UiTreeUnavailableError``) — a failed ``BVTStepResult``
        for ``action`` with ``harness_error=True`` and a ``"HARNESS: UI tree
        unavailable — <err>"`` message. Shared by the assert steps (via
        ``_assert_tree``), ``if`` conditions, ``dismiss_dialog`` and
        ``wait_for_element`` so an unread tree never reads as "0 elements".
        """
        try:
            return self._nerve.elements_strict(device)
        except UiTreeUnavailableError as exc:
            return BVTStepResult(
                action=action,
                passed=False,
                harness_error=True,
                message=f"HARNESS: UI tree unavailable — {exc}",
            )

    def _do_assert_clickable(
        self, step: BVTStep, device: str, _package: str, _test_name: str
    ) -> BVTStepResult:
        tree = self._assert_tree(device, "assert_clickable")
        if isinstance(tree, BVTStepResult):
            return tree
        result = assert_clickable(tree, step.target)
        return BVTStepResult(
            action="assert_clickable", passed=result.passed, message=result.message
        )

    def _get_density(self, device: str) -> dict | None:
        """Cached per-device density dict, or None if the daemon can't report it.

        THE single density source for every dp-based assertion (``assert_min_size``
        and ``assert_touch_target`` both go through here) — one ``nerve.density``
        call per device per run.
        """
        if device not in self._density_cache:
            d = self._nerve.density(device)
            factor = d.get("density_factor") if d else None
            if not factor:  # {} or missing/zero — unreadable
                return None
            self._density_cache[device] = d
        return self._density_cache[device]

    def _get_density_factor(self, device: str) -> float | None:
        """Cached per-device dp→px factor, or None if the daemon can't report it."""
        d = self._get_density(device)
        return float(d["density_factor"]) if d else None

    def _do_assert_min_size(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        factor = self._get_density_factor(device)
        if factor is None:
            return BVTStepResult(
                action="assert_min_size",
                passed=False,
                message="density unavailable — cannot evaluate dp assertion",
            )
        tree = self._assert_tree(device, "assert_min_size")
        if isinstance(tree, BVTStepResult):
            return tree
        params = step.params
        min_w = _parse_dimension(params["width"], factor)
        min_h = _parse_dimension(params["height"], factor)
        result = assert_min_size(tree, params["element"], min_w, min_h)
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_min_size",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_assert_element(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        elems = self._assert_tree(device, "assert_element")
        if isinstance(elems, BVTStepResult):
            return elems
        result = assert_element(elems, step.target)
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_element",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_assert_text(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        elems = self._assert_tree(device, "assert_text")
        if isinstance(elems, BVTStepResult):
            return elems
        result = assert_text(elems, step.target)
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_text",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_assert_content_description(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        elems = self._assert_tree(device, "assert_content_description")
        if isinstance(elems, BVTStepResult):
            return elems
        result = assert_content_description(elems, step.target)
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_content_description",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_assert_touch_target(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        density = self._get_density(device)
        if density is None:
            return BVTStepResult(
                action="assert_touch_target",
                passed=False,
                message="density unavailable — cannot evaluate touch-target assertion",
            )
        factor = float(density["density_factor"])
        override = step.params.get("min_size_px")
        if override is None:
            min_size_px = min_touch_target_px(factor, density.get("platform", "android"))
        else:
            min_size_px = _parse_dimension(override, factor)
        elems = self._assert_tree(device, "assert_touch_target")
        if isinstance(elems, BVTStepResult):
            return elems
        result = assert_touch_target(elems, step.target, min_size_px=min_size_px)
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_touch_target",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_assert_screen_reader(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        elems = self._assert_tree(device, "assert_screen_reader")
        if isinstance(elems, BVTStepResult):
            return elems
        result = assert_screen_reader(elems, step.target)
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_screen_reader",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_assert_element_not_present(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        elems = self._assert_tree(device, "assert_element_not_present")
        if isinstance(elems, BVTStepResult):
            return elems
        result = assert_element_not_present(elems, step.target)
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_element_not_present",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_assert_text_not_present(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        elems = self._assert_tree(device, "assert_text_not_present")
        if isinstance(elems, BVTStepResult):
            return elems
        result = assert_text_not_present(elems, step.target)
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_text_not_present",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_assert_element_count(
        self, step: BVTStep, device: str, package: str, _test_name: str
    ) -> BVTStepResult:
        elems = self._assert_tree(device, "assert_element_count")
        if isinstance(elems, BVTStepResult):
            return elems
        p = step.params
        try:
            result = assert_element_count(
                elems,
                p["element"],
                expected=p.get("expected"),
                min_count=p.get("min"),
                max_count=p.get("max"),
            )
        except (ValueError, KeyError) as e:
            # Malformed bounds / missing element key: a fail, not a suite crash.
            return BVTStepResult(action="assert_element_count", passed=False, message=str(e))
        perf = self._capture_metrics(device, package) if result.passed else None
        return BVTStepResult(
            action="assert_element_count",
            passed=result.passed,
            message=result.message,
            performance=perf,
        )

    def _do_screenshot(
        self, step: BVTStep, device: str, _package: str, test_name: str
    ) -> BVTStepResult:
        label = step.target or "screenshot"
        safe_test = _safe_file_stem(test_name)
        safe_label = _safe_file_stem(label)
        filename = f"{safe_test}_{safe_label}.png"
        filepath = self._screenshots_dir / filename

        try:
            if not filepath.resolve().is_relative_to(self._screenshots_dir.resolve()):
                return BVTStepResult(
                    action="screenshot",
                    passed=False,
                    message=f"Screenshot path escapes screenshots directory: {filepath}",
                    harness_error=True,
                )
        except (ValueError, RuntimeError):
            return BVTStepResult(
                action="screenshot",
                passed=False,
                message=f"Screenshot path escapes screenshots directory: {filepath}",
                harness_error=True,
            )

        try:
            png_bytes = self._nerve.screenshot(device)
            if not _png_is_valid(png_bytes):
                return BVTStepResult(
                    action="screenshot",
                    passed=False,
                    harness_error=True,
                    message=f"HARNESS: screenshot returned {len(png_bytes)} bytes, not a PNG",
                )
            filepath.write_bytes(png_bytes)
            return BVTStepResult(
                action="screenshot",
                passed=True,
                message=f"Saved {filepath}",
                screenshot_path=str(filepath),
            )
        except Exception as e:
            return BVTStepResult(
                action="screenshot",
                passed=False,
                message=f"Screenshot failed: {e}",
            )


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_is_valid(data: bytes) -> bool:
    """Validate PNG data has valid magic signature and minimum length."""
    return bool(data) and len(data) >= 8 and data.startswith(_PNG_SIGNATURE)
