"""Per-step bounded foreground timeout, longer cold-start default, and an
evidence-carrying failure message (reactor 5c609e12, 0caecb2f).

Before this fix ``_wait_for_foreground`` polled a fixed 5s window regardless
of how slow a particular launch/navigate legitimately is, and its failure
message guessed "nerve daemon unreachable" whenever nothing foregrounded —
even when nerve had reported a concrete error (or raised one) that the
adapter's lossy ``foreground()`` reduced to ``{}``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from linc_crucible.core.bvt import step_handlers as step_handlers_mod
from linc_crucible.core.bvt.foreground import (
    foreground_default_s,
    foreground_window_s,
)
from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.runner import BVTRunner
from linc_crucible.core.bvt.schema import BVTStep, BVTSuite, BVTTest
from tests.conftest import make_mock_nerve_client


@pytest.fixture(autouse=True)
def _capture_structlog() -> Iterator[None]:
    """Record structlog events instead of printing them to an import-time-bound stream.

    Importing ``linc_crucible.cli`` points structlog at the then-current ``sys.stderr``,
    a pytest capture stream. mutmut re-runs ``pytest.main()`` in one process; the first
    session closes that stream, so any later log call raised "I/O operation on closed
    file". Mirrors ``captured_logs`` in ``tests/test_crucible_bvt_mcp.py``.
    """
    with structlog.testing.capture_logs():
        yield


TARGET_PKG = "com.app"
OTHER_PKG = "com.other"


def _runner(mock, tmp_path: Path) -> BVTRunner:
    return BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)


def _suite(step: BVTStep, tags: list[str] | None = None) -> BVTSuite:
    return BVTSuite(
        package=TARGET_PKG,
        tests=[BVTTest(name="t", package=TARGET_PKG, steps=[step], tags=tags or [])],
    )


# --- foreground_window_s (pure unit tests) ----------------------------------


class TestForegroundWindowS:
    def test_int_timeout(self) -> None:
        assert foreground_window_s({"timeout": 12}, default=5.0, maximum=60.0) == 12.0

    def test_seconds_suffix(self) -> None:
        assert foreground_window_s({"timeout": "12s"}, default=5.0, maximum=60.0) == 12.0

    def test_millis_suffix(self) -> None:
        assert foreground_window_s({"timeout": "12000ms"}, default=5.0, maximum=60.0) == 12.0

    def test_absent_uses_default(self) -> None:
        assert foreground_window_s({}, default=5.0, maximum=60.0) == 5.0

    def test_zero_uses_default(self) -> None:
        assert foreground_window_s({"timeout": 0}, default=5.0, maximum=60.0) == 5.0

    def test_negative_uses_default(self) -> None:
        assert foreground_window_s({"timeout": -3}, default=5.0, maximum=60.0) == 5.0

    def test_unparseable_uses_default(self) -> None:
        assert foreground_window_s({"timeout": "soon"}, default=5.0, maximum=60.0) == 5.0

    def test_above_maximum_is_clamped(self) -> None:
        assert foreground_window_s({"timeout": 999}, default=5.0, maximum=60.0) == 60.0


class TestForegroundDefaultS:
    def test_untagged_uses_normal(self) -> None:
        assert foreground_default_s([], normal=5.0, cold_start=15.0) == 5.0

    def test_cold_start_tag_uses_cold_start_default(self) -> None:
        assert foreground_default_s(["cold_start"], normal=5.0, cold_start=15.0) == 15.0


# --- NerveAdapter.foreground_probe -------------------------------------------


class TestForegroundProbe:
    def test_foreground_present_has_no_error(self) -> None:
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.app", "activity": ".M"}}]
        )
        probe = NerveAdapter(nerve=mock).foreground_probe("DEV")
        assert probe == {"foreground": {"package": "com.app", "activity": ".M"}, "error": None}

    def test_error_key_surfaces_and_foreground_is_empty(self) -> None:
        mock = make_mock_nerve_client(screen_state_responses=[{"error": "x"}])
        probe = NerveAdapter(nerve=mock).foreground_probe("DEV")
        assert probe == {"foreground": {}, "error": "x"}

    def test_raising_call_names_the_exception_class(self) -> None:
        mock = make_mock_nerve_client(
            screen_state_responses=[ConnectionError("Nerve unreachable: refused")]
        )
        probe = NerveAdapter(nerve=mock).foreground_probe("DEV")
        assert probe["foreground"] == {}
        assert probe["error"] == "ConnectionError: Nerve unreachable: refused"


# --- fake-clock timeline behavior through the runner -------------------------


class _FakeClock:
    """A monotonic clock a test fully controls: sleep() advances it, no real
    wall-clock time passes — so a 12s or 15s simulated window runs instantly.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(step_handlers_mod, "time", clock)
    return clock


def test_explicit_timeout_passes_once_foreground_settles_within_window(
    fake_clock: _FakeClock, tmp_path: Path
) -> None:
    """launch(timeout=12): wrong app until t=8, then the target — PASSES.

    Pre-fix this fails at t=5 ("did not foreground") because the window was
    hardcoded to 5s regardless of the step's own ``timeout`` param.
    """
    mock = make_mock_nerve_client()

    def _screen_state(device: str) -> dict:
        pkg = OTHER_PKG if fake_clock.t < 8 else TARGET_PKG
        return {"foreground": {"package": pkg, "activity": ".M"}}

    mock.screen_state.side_effect = _screen_state
    step = BVTStep(action="launch", target=TARGET_PKG, params={"timeout": 12})
    res = _runner(mock, tmp_path).run_suite(_suite(step), device="DEV")
    t = res.tests[0]
    assert t.passed, t.failure_message


def test_no_param_untagged_test_still_fails_at_unchanged_default(
    fake_clock: _FakeClock, tmp_path: Path
) -> None:
    """No ``timeout`` param, no ``cold_start`` tag: still fails around t=5 —
    pins the unchanged normal default."""
    mock = make_mock_nerve_client()
    mock.screen_state.side_effect = lambda device: {
        "foreground": {"package": OTHER_PKG, "activity": ".M"}
    }
    step = BVTStep(action="launch", target=TARGET_PKG)
    res = _runner(mock, tmp_path).run_suite(_suite(step), device="DEV")
    t = res.tests[0]
    assert not t.passed
    assert "app did not foreground" in t.failure_message
    assert fake_clock.t == pytest.approx(5.0, abs=0.01)


def test_cold_start_tag_gets_longer_default_window(fake_clock: _FakeClock, tmp_path: Path) -> None:
    """No ``timeout`` param, ``cold_start`` tag: foreground arrives at t=12,
    inside the 15s cold-start default — PASSES.

    Pre-fix this fails: the window was a flat 5s regardless of tags.
    """
    mock = make_mock_nerve_client()

    def _screen_state(device: str) -> dict:
        pkg = OTHER_PKG if fake_clock.t < 12 else TARGET_PKG
        return {"foreground": {"package": pkg, "activity": ".M"}}

    mock.screen_state.side_effect = _screen_state
    step = BVTStep(action="launch", target=TARGET_PKG)
    res = _runner(mock, tmp_path).run_suite(_suite(step, tags=["cold_start"]), device="DEV")
    t = res.tests[0]
    assert t.passed, t.failure_message


# --- failure-message evidence (real-time fast poll) --------------------------


@pytest.fixture
def _fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step_handlers_mod, "_FOREGROUND_TIMEOUT_S", 0.05)
    monkeypatch.setattr(step_handlers_mod, "_FOREGROUND_POLL_INTERVAL_S", 0.01)


def test_nerve_error_surfaces_in_failure_message(_fast_poll, tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[{"error": "uiautomator dump timed out"}])
    step = BVTStep(action="launch", target=TARGET_PKG)
    res = _runner(mock, tmp_path).run_suite(_suite(step), device="DEV")
    msg = res.tests[0].failure_message
    assert "could not read foreground" in msg
    assert "uiautomator dump timed out" in msg
    assert "polls" in msg
    assert "daemon unreachable" not in msg


def test_transport_exception_names_the_exception_class(_fast_poll, tmp_path: Path) -> None:
    mock = make_mock_nerve_client(
        screen_state_responses=[
            ConnectionError("Nerve unreachable at http://localhost:8111: refused")
        ]
    )
    step = BVTStep(action="launch", target=TARGET_PKG)
    res = _runner(mock, tmp_path).run_suite(_suite(step), device="DEV")
    msg = res.tests[0].failure_message
    assert "ConnectionError: Nerve unreachable" in msg


def test_empty_state_reports_no_foreground_and_no_error(_fast_poll, tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[{}])
    step = BVTStep(action="launch", target=TARGET_PKG)
    res = _runner(mock, tmp_path).run_suite(_suite(step), device="DEV")
    msg = res.tests[0].failure_message
    assert "nerve_state returned no foreground and no error" in msg


def test_wrong_app_message_names_the_other_package_and_poll_count(
    _fast_poll, tmp_path: Path
) -> None:
    mock = make_mock_nerve_client(
        screen_state_responses=[{"foreground": {"package": OTHER_PKG, "activity": ".X"}}]
    )
    step = BVTStep(action="launch", target=TARGET_PKG)
    res = _runner(mock, tmp_path).run_suite(_suite(step), device="DEV")
    msg = res.tests[0].failure_message
    assert "app did not foreground" in msg
    assert OTHER_PKG in msg
    assert "polls" in msg
