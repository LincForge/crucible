"""BVT Runner — executes test suites on real devices.

Pure Python, no AI tokens. Uses NerveAdapter for device interaction.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from linc_crucible.core.bvt import step_handlers
from linc_crucible.core.bvt.foreground import foreground_default_s
from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.step_handlers import (
    BVTRunResult,
    BVTStepHandlersMixin,
    BVTStepResult,
    BVTTestResult,
    _duration_seconds,
    _parse_dimension,
    _parse_duration,
    _parse_retries,
)

__all__ = [
    "_SUITE_BUDGET_DEFAULT_S",
    "BVTBudgetedRunResult",
    "BVTRunResult",
    "BVTRunner",
    "BVTStepResult",
    "BVTTestResult",
    "_budget_exhausted",
    "_duration_seconds",
    "_parse_dimension",
    "_parse_duration",
    "_parse_retries",
    "_readiness_reason",
]

if TYPE_CHECKING:
    from linc_crucible.core.bvt.schema import BVTSuite, BVTTest

log = structlog.get_logger()

# Wall-clock budget for a whole suite. Checked BEFORE each test —
# it never preempts a running step (per-step foreground windows and duration
# caps bound those); once spent, every remaining test is a harness skip.
_SUITE_BUDGET_DEFAULT_S = 1800.0


def _budget_exhausted(started: float, budget_s: float | None, now: float) -> bool:
    """True once ``now - started`` has reached ``budget_s``; never when it is None."""
    if budget_s is None:
        return False
    return now - started >= budget_s


def _fmt_seconds(value: float) -> str:
    """Render seconds compactly for a skip reason: 1000.0 -> "1000", 12.34 -> "12.3"."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


@dataclass
class BVTBudgetedRunResult(BVTRunResult):
    """A BVTRunResult that also records whether the suite budget cut it short.

    Budget-skipped tests are skipped (neither passed nor failed), so the base
    ``passed`` alone would call a suite green on the tests that happened to fit
    in the budget. An exhausted budget is a harness failure: never green.
    """

    budget_exhausted: bool = False

    @property
    def passed(self) -> bool:
        return not self.budget_exhausted and super().passed


class _SuiteBudget:
    """Tracks one suite's wall-clock budget against an injectable monotonic clock."""

    def __init__(self, budget_s: float | None, clock: Callable[[], float]) -> None:
        self.budget_s = budget_s
        self._clock = clock
        self._started = clock()
        self.exhausted = False

    def skip_reason(self) -> str | None:
        """The harness skip reason once the budget is spent, else None."""
        now = self._clock()
        if not _budget_exhausted(self._started, self.budget_s, now):
            return None
        self.exhausted = True
        return (
            f"HARNESS: suite budget {_fmt_seconds(self.budget_s or 0.0)}s exhausted "
            f"after {_fmt_seconds(now - self._started)}s"
        )


def _readiness_reason(readiness: dict, device: str) -> str:
    """Build the "device not ready" skip detail from a nerve_ensure_ready result.

    nerve's device-resolution failure shape carries ``error`` (no ``reason``), e.g.
    ``{"error": "Device '<serial>' not found. Run nerve_discover to list devices."}``,
    while other readiness failures (asleep, locked, low battery) carry ``reason``.
    Prefer ``reason``, then ``error``, then a fixed fallback — never surface a bare
    ``None`` when neither key holds a non-empty string.
    """
    detail = (
        readiness.get("reason")
        or readiness.get("error")
        or "nerve_ensure_ready returned no reason"
    )
    return f"device not ready: {detail} (device {device})"


class BVTRunner(BVTStepHandlersMixin):
    """Executes BVT test suites against devices via NerveAdapter."""

    def __init__(self, nerve: NerveAdapter, output_dir: Path) -> None:
        self._nerve = nerve
        self._output_dir = output_dir
        self._screenshots_dir = output_dir / "bvt_screenshots"
        # Full nerve density dict per device (platform + density_factor), populated on
        # the first dp/touch-target assertion. Stable per run; THE single density source.
        self._density_cache: dict[str, dict] = {}

    def run_suite(
        self,
        suite: BVTSuite,
        device: str,
        app_path: str | None = None,
        budget_s: float | None = _SUITE_BUDGET_DEFAULT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> BVTRunResult:
        """Run all tests in a suite on the given device.

        Tests run sequentially. Within a test, steps are fail-fast
        (remaining steps skipped on first failure). But the suite
        continues to the next test even if one fails.

        ``app_path`` is the built artifact (.app/.apk) used to reinstall the
        app when a cold_start clear reports ``needs_reinstall`` (iOS
        simulators, where nerve's clear is uninstall-based). Optional —
        Android cold_start (``pm clear``) never needs it.

        ``budget_s`` is the suite's wall-clock budget (default
        ``_SUITE_BUDGET_DEFAULT_S``; ``None`` = unlimited), measured on
        ``clock``. It is checked before each test; once spent, every remaining
        test is recorded skipped with a "HARNESS: suite budget ..." reason and
        the suite reports ``passed=False``. A running step is never preempted.
        """
        start = time.monotonic()
        budget = _SuiteBudget(budget_s, clock)
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Device-readiness gate: a locked/asleep/low device silently corrupts
        # artifacts (lock-screen screenshots, no-op taps, 0-byte recordings).
        # Skip the whole suite with a reason rather than run blind.
        skipped = self._skipped_suite(suite, device, start)
        if skipped is not None:
            return skipped

        return self._run_all_tests(suite, device, start, app_path, budget)

    def _skipped_suite(self, suite: BVTSuite, device: str, start: float) -> BVTRunResult | None:
        """If the device isn't ready, return an all-skipped result; else None."""
        readiness = self._nerve.ensure_ready(device)
        if readiness.get("ready"):
            return None
        reason = _readiness_reason(readiness, device)
        log.warning("bvt_suite_skipped_device_unready", device=device, reason=reason)
        return BVTRunResult(
            device=device,
            package=suite.package,
            tests=[
                BVTTestResult(
                    name=t.name,
                    passed=False,
                    package=getattr(t, "package", None) or suite.package,
                    skipped=True,
                    skip_reason=reason,
                )
                for t in suite.tests
            ],
            duration_ms=int((time.monotonic() - start) * 1000),
            suite_skip_reason=reason,
        )

    def _run_all_tests(
        self,
        suite: BVTSuite,
        device: str,
        start: float,
        app_path: str | None,
        budget: _SuiteBudget,
    ) -> BVTBudgetedRunResult:
        """Run every test in the suite sequentially (suite continues past failures).

        Before each test the suite ``budget`` is checked; once it is spent every
        remaining test is recorded skipped (see ``_run_or_budget_skip``).
        """
        results: list[BVTTestResult] = []
        for test in suite.tests:
            result = self._run_or_budget_skip(test, suite, device, app_path, budget)
            results.append(result)
            log.info(
                "bvt_test_complete",
                test=test.name,
                passed=result.passed,
                steps=f"{result.steps_passed}/{result.steps_run}",
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return BVTBudgetedRunResult(
            device=device,
            package=suite.package,
            tests=results,
            duration_ms=elapsed_ms,
            budget_exhausted=budget.exhausted,
        )

    def _run_or_budget_skip(
        self,
        test: BVTTest,
        suite: BVTSuite,
        device: str,
        app_path: str | None,
        budget: _SuiteBudget,
    ) -> BVTTestResult:
        """Run ``test`` — or, when the suite budget is spent, record it skipped."""
        package = test.package or suite.package
        reason = budget.skip_reason()
        if reason is None:
            return self._run_test(test, device, package, app_path)
        log.warning("bvt_test_skipped_budget_exhausted", test=test.name, reason=reason)
        return BVTTestResult(
            name=test.name,
            passed=False,
            package=package,
            skipped=True,
            skip_reason=reason,
        )

    def _run_test(
        self, test: BVTTest, device: str, package: str, app_path: str | None = None
    ) -> BVTTestResult:
        """Run a single test. Fail-fast on first step failure."""
        start = time.monotonic()

        # Reset app for clean state between tests. A test tagged `cold_start`
        # gets a full `pm clear` (first-install state) so consent/onboarding
        # screens reappear; all others get the cheaper kill+relaunch.
        reset_error = self._reset_app(
            device,
            package,
            test.name,
            cold_start="cold_start" in test.tags,
            app_path=app_path,
        )
        if reset_error is not None:
            # Fatal reset failure (e.g. iOS-sim cold_start with no artifact to
            # reinstall) — record a failed test rather than run steps against
            # a device that no longer has the app installed.
            return BVTTestResult(
                name=test.name,
                passed=False,
                package=package,
                failure_message=reset_error,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        self._foreground_default_s = foreground_default_s(
            test.tags,
            step_handlers._FOREGROUND_TIMEOUT_S,
            step_handlers._COLD_START_FOREGROUND_TIMEOUT_S,
        )
        return self._run_steps(test, device, package, start)

    def _run_steps(self, test: BVTTest, device: str, package: str, start: float) -> BVTTestResult:
        """Execute a test's steps fail-fast and build its result.

        Extracted verbatim from ``_run_test`` (CRAP ratchet) — behavior
        unchanged: steps run sequentially, the first failure records a
        step-attributed message and stops the test.
        """
        steps_run = 0
        steps_passed = 0
        failure_message = ""
        harness_error = False
        screenshots: list[str] = []
        perf_snapshots: list[dict] = []

        for step in test.steps:
            steps_run += 1
            step_result = self._execute_step_with_retries(step, device, package, test.name)

            if step_result.screenshot_path:
                screenshots.append(step_result.screenshot_path)
            screenshots.extend(step_result.extra_screenshots)

            if step_result.performance:
                perf_snapshots.append(step_result.performance)
            perf_snapshots.extend(step_result.extra_performance)

            if step_result.passed:
                steps_passed += 1
            else:
                harness_error = step_result.harness_error
                # A harness fault's message is already the self-contained
                # "HARNESS: ..." label — wrapping it in "Step N '...' failed:"
                # would bury that label and re-add step-attributed noise a
                # harness fault (as opposed to a product failure) doesn't need.
                failure_message = (
                    step_result.message
                    if harness_error
                    else f"Step {steps_run} '{step.action}:{step.target}' "
                    f"failed: {step_result.message}"
                )
                break  # Fail-fast within test

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return BVTTestResult(
            name=test.name,
            passed=steps_run == steps_passed and steps_run > 0,
            package=package,
            steps_run=steps_run,
            steps_passed=steps_passed,
            failure_message=failure_message,
            screenshots=screenshots,
            duration_ms=elapsed_ms,
            performance_snapshots=perf_snapshots,
            harness_error=harness_error,
        )

    def _reset_app(
        self,
        device: str,
        package: str,
        test_name: str,
        cold_start: bool = False,
        app_path: str | None = None,
    ) -> str | None:
        """Reset the app before a test for clean state.

        ``cold_start=True`` runs ``pm clear`` first (full first-install state —
        wipes app-level consent/onboarding) before the usual kill+relaunch;
        otherwise just kill+relaunch (which preserves data/permissions).

        On iOS simulators nerve's clear is uninstall-based and its response
        carries ``needs_reinstall: true`` (absent/false on Android — older
        daemons omit the key). When set, the app is reinstalled from
        ``app_path`` before the relaunch; with no ``app_path`` to reinstall
        from, the test cannot proceed and the returned failure message fails it.

        Returns a failure message on a fatal reset problem (test must not
        run — this includes a cold_start clear that fails or raises, not
        only the needs_reinstall branch), else None. kill/launch failures
        stay non-fatal: logged warnings that do not abort the test. NOTE:
        ``pm clear`` also revokes runtime
        permissions and AndroidKeystore aliases — a cold_start test that needs
        a permission pre-granted must re-grant it via a ``grant_permission``
        step (see the schema docs).
        """
        # Cold start: wipe data before kill+launch so onboarding reappears.
        if cold_start:
            fatal = self._cold_start_wipe(device, package, test_name, app_path)
            if fatal is not None:
                return fatal

        self._safe_reset_op(
            lambda: self._nerve.kill(device, package),
            test_name,
            package,
            "bvt_reset_kill_failed",
            "bvt_reset_kill_error",
        )
        self._safe_reset_op(
            lambda: self._nerve.launch(device, package),
            test_name,
            package,
            "bvt_reset_launch_failed",
            "bvt_reset_launch_error",
        )

        # Brief wait for app initialization
        time.sleep(1.5)
        return None

    def _cold_start_wipe(
        self, device: str, package: str, test_name: str, app_path: str | None
    ) -> str | None:
        """Cold-start clear, honoring nerve's ``needs_reinstall`` contract.

        Clears app data; when the clear response carries ``needs_reinstall``
        (iOS simulators — uninstall-based clear), reinstalls the app from
        ``app_path`` so the subsequent relaunch has an app to launch.

        Returns a fatal failure message when the device is left with no app
        installed (no ``app_path`` to reinstall from, or the reinstall itself
        failed) OR when the clear itself failed/raised — a cold_start test's
        whole point is a wiped device, so a clear that didn't happen must not
        silently downgrade to a warm run reporting PASS (for example, when a
        missing ``confirm=True`` makes every clear a no-op or when a permission
        or platform error blocks the clear). Else None.
        """
        clear_result = self._safe_reset_op(
            lambda: self._nerve.clear_app_data(device, package),
            test_name,
            package,
            "bvt_reset_clear_failed",
            "bvt_reset_clear_error",
        )
        if clear_result is None:
            log.warning(
                "bvt_reset_clear_raised_fatal",
                test=test_name,
                package=package,
                device=device,
            )
            return "cold_start clear_app_data raised — device state is unknown, cannot proceed"
        if not clear_result.get("success", False):
            error = clear_result.get("error", "unknown")
            log.warning(
                "bvt_reset_clear_failed_fatal",
                test=test_name,
                package=package,
                device=device,
                error=error,
            )
            return (
                "cold_start clear_app_data failed — device state is "
                f"unknown, cannot proceed: {error}"
            )
        if not clear_result.get("needs_reinstall"):
            return None
        if app_path is None:
            log.warning(
                "bvt_reset_reinstall_no_app_path",
                test=test_name,
                package=package,
                device=device,
            )
            return (
                "iOS cold_start requires app_path (uninstall-based clear "
                "removed the app) — pass the built artifact path"
            )
        install_result = self._safe_reset_op(
            lambda: self._nerve.install(device, app_path),
            test_name,
            package,
            "bvt_reset_install_failed",
            "bvt_reset_install_error",
        )
        if install_result is None or not install_result.get("success", False):
            # Fatal, symmetric with the no-app_path branch: the clear
            # uninstalled the app and the reinstall didn't land, so
            # kill/launch would run against a device with no app.
            error = (install_result or {}).get("error", "unknown")
            log.warning(
                "bvt_reset_reinstall_failed_fatal",
                test=test_name,
                package=package,
                device=device,
                error=error,
            )
            return (
                "iOS cold_start reinstall failed after uninstall-based "
                f"clear — device has no app installed: {error}"
            )
        return None

    def _safe_reset_op(
        self,
        op: Callable[[], dict],
        test_name: str,
        package: str,
        fail_event: str,
        error_event: str,
    ) -> dict | None:
        """Run a single reset op (clear/kill/launch/install); log on failure,
        never raise.

        Reset is best-effort: a failed or exception-raising op logs a warning
        under the given event names but does not abort the test (mirrors the
        original per-op kill/launch handling). Returns the op's result dict
        (so callers can read response flags like ``needs_reinstall``), or None
        when the op raised.
        """
        try:
            result = op()
            if not result.get("success", False):
                log.warning(
                    fail_event,
                    test=test_name,
                    package=package,
                    error=result.get("error", "unknown"),
                )
            return result
        except Exception as e:
            log.warning(error_event, test=test_name, error=str(e))
            return None
