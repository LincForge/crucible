"""Fix B: launch/deeplink foreground verification + per-test package launch.

The runner must launch each test's own package and fail-fast when the app
never reaches the foreground, instead of asserting against a stale screen
(root cause 8f09da13, masking e46e1112).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

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


FG_OK = {"foreground": {"package": "com.app.watch", "activity": ".WatchActivity"}}


def _runner(mock, tmp_path: Path) -> BVTRunner:
    return BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)


def test_runner_launches_test_package_not_suite_package(tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[FG_OK])
    suite = BVTSuite(
        package="com.app",  # suite default = phone
        tests=[
            BVTTest(
                name="watch_home",
                package="com.app.watch",  # this test's package
                steps=[BVTStep(action="launch", target="com.app.watch")],
            )
        ],
    )
    runner = _runner(mock, tmp_path)
    runner.run_suite(suite, device="DEV")
    # _reset_app + _do_launch must use the watch package
    assert ("DEV", "com.app.watch") in mock.launch_calls
    assert ("DEV", "com.app") not in mock.launch_calls


# --- Task 4: NerveAdapter.foreground_probe ---


def test_foreground_returns_package_and_activity(tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[FG_OK])
    adapter = NerveAdapter(nerve=mock)
    probe = adapter.foreground_probe("DEV")
    fg = probe["foreground"]
    assert fg["package"] == "com.app.watch"
    assert fg["activity"] == ".WatchActivity"
    assert probe["error"] is None


def test_foreground_empty_state_returns_empty(tmp_path: Path) -> None:
    # Empty/unavailable state -> {} (never raises); the poll loop tolerates it.
    mock = make_mock_nerve_client(screen_state_responses=[{}])
    adapter = NerveAdapter(nerve=mock)
    probe = adapter.foreground_probe("DEV")
    assert probe["foreground"] == {}
    assert probe["error"] is None


def test_foreground_swallows_transient_exception(tmp_path: Path) -> None:
    # A transient transport error is swallowed to {} so one flake can't abort a run.
    mock = make_mock_nerve_client(screen_state_responses=[ConnectionError("nerve hiccup")])
    adapter = NerveAdapter(nerve=mock)
    probe = adapter.foreground_probe("DEV")
    assert probe["foreground"] == {}
    assert "ConnectionError" in (probe["error"] or "")



# --- Task 5: foreground gate wired into launch / deeplink ---

from linc_crucible.core.bvt import step_handlers as step_handlers_mod  # noqa: E402


@pytest.fixture
def _fast_poll(monkeypatch):
    # keep production constants; make the poll fast in tests
    monkeypatch.setattr(step_handlers_mod, "_FOREGROUND_TIMEOUT_S", 0.2)
    monkeypatch.setattr(step_handlers_mod, "_FOREGROUND_POLL_INTERVAL_S", 0.01)


def _launch_test() -> BVTSuite:
    return BVTSuite(
        package="com.app.watch",
        tests=[
            BVTTest(
                name="t",
                package="com.app.watch",
                steps=[BVTStep(action="launch", target="com.app.watch")],
            )
        ],
    )


def test_launch_passes_when_foreground_settles(_fast_poll, tmp_path: Path) -> None:
    wrong = {"foreground": {"package": "com.other", "activity": ".X"}}
    mock = make_mock_nerve_client(screen_state_responses=[wrong, wrong, FG_OK])
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="DEV")
    assert res.tests[0].passed


def test_launch_fails_when_foreground_never_matches(_fast_poll, tmp_path: Path) -> None:
    wrong = {"foreground": {"package": "com.other", "activity": ".X"}}
    mock = make_mock_nerve_client(screen_state_responses=[wrong])
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="DEV")
    t = res.tests[0]
    assert not t.passed
    assert "did not foreground" in t.failure_message
    assert "com.other" in t.failure_message


def test_launch_failure_skips_foreground_poll(_fast_poll, tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    mock.launch.side_effect = lambda device, package: {"success": False, "error": "boom"}
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="DEV")
    t = res.tests[0]
    assert not t.passed
    assert "boom" in t.failure_message
    assert "did not foreground" not in t.failure_message
    mock.screen_state.assert_not_called()


def test_unavailable_foreground_reports_distinct_failure(_fast_poll, tmp_path: Path) -> None:
    # No readable foreground for the whole window -> distinct daemon-unreachable
    # failure (NOT masked as a normal "did not foreground"). One transient flake
    # would simply be retried; here every read is empty.
    mock = make_mock_nerve_client(screen_state_responses=[{}])
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="DEV")
    t = res.tests[0]
    assert not t.passed
    assert "could not read foreground" in t.failure_message
    assert "did not foreground" not in t.failure_message


def test_transient_empty_then_settles_passes(_fast_poll, tmp_path: Path) -> None:
    # A couple of transient empty reads must not abort: poll recovers and passes.
    mock = make_mock_nerve_client(screen_state_responses=[{}, {}, FG_OK])
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="DEV")
    assert res.tests[0].passed


def test_already_foregrounded_passes_immediately(_fast_poll, tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[FG_OK])
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="DEV")
    assert res.tests[0].passed


# --- iOS process-running fallback (foreground structurally unavailable) -------
#
# nerve_state on iOS simulators carries no `foreground` key (no WDA on the
# bench), so the poll's saw_valid stays False forever and every iOS launch
# verb failed as "could not read foreground" — even when the cold_start
# cycle (clear → reinstall → launch) provably worked and the app process was
# running. When the whole window yields no readable foreground, the runner
# now consults nerve_inspect once: an iOS-platform response with the target
# process running is accepted as a degraded-but-honest launch verification.
# Android is deliberately excluded — foreground IS supported there, so empty
# reads mean daemon trouble, and process-running would mask a backgrounded app.


def test_launch_ios_falls_back_to_process_running(_fast_poll, tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[{"platform": "ios"}])
    mock.inspect.return_value = {
        "platform": "ios",
        "process": {"pid": 80907, "running": True, "package": "com.app.watch"},
    }
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="SIM")
    t = res.tests[0]
    assert t.passed
    mock.inspect.assert_called_once_with(device="SIM", package="com.app.watch")


def test_launch_ios_fallback_fails_when_process_not_running(
    _fast_poll, tmp_path: Path
) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[{"platform": "ios"}])
    mock.inspect.return_value = {"platform": "ios", "process": {"running": False}}
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="SIM")
    t = res.tests[0]
    assert not t.passed
    assert "could not read foreground" in t.failure_message


def test_launch_android_empty_foreground_never_uses_process_fallback(
    _fast_poll, tmp_path: Path
) -> None:
    """On Android an unreadable foreground means daemon trouble — a running
    process is NOT evidence the app foregrounded (it can sit backgrounded)."""
    mock = make_mock_nerve_client(screen_state_responses=[{}])
    mock.inspect.return_value = {
        "platform": "android",
        "process": {"pid": 1, "running": True},
    }
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="DEV")
    t = res.tests[0]
    assert not t.passed
    assert "could not read foreground" in t.failure_message


def test_launch_ios_fallback_survives_inspect_error(_fast_poll, tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[{}])
    mock.inspect.side_effect = ConnectionError("daemon gone")
    res = _runner(mock, tmp_path).run_suite(_launch_test(), device="SIM")
    t = res.tests[0]
    assert not t.passed
    assert "could not read foreground" in t.failure_message
