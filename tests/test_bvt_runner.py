"""Tests for BVT runner with mock NerveAdapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

from linc_crucible.core.bvt.nerve_adapter import NerveAdapter, _parse_bounds_center
from linc_crucible.core.bvt.runner import (
    _SUITE_BUDGET_DEFAULT_S,
    BVTRunner,
    BVTRunResult,
    BVTTestResult,
    _budget_exhausted,
    _parse_duration,
)
from linc_crucible.core.bvt.schema import BVTStep, BVTSuite, BVTTest
from linc_crucible.core.nerve_client import UiTreeUnavailableError
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


def _mock_with_launch_error(error_msg: str, elements_response: list[dict] | None = None):
    """Helper: realistic mock whose launch() raises RuntimeError."""
    mock = make_mock_nerve_client(elements_response=elements_response)

    def _failing_launch(device: str, package: str) -> dict:
        mock.launch_calls.append((device, package))
        raise RuntimeError(error_msg)

    mock.launch.side_effect = _failing_launch
    return mock


SAMPLE_ELEMENTS = [
    {"id": "level_screen", "text": "", "bounds": "[0,0][1080,2400]"},
    {"id": "pitch_readout", "text": "0.0\u00b0", "bounds": "[100,200][300,400]"},
    {"id": "settings_button", "text": "Settings", "bounds": "[900,50][1050,150]"},
]


# --- Helper tests ---


class TestParseDuration:
    def test_seconds(self) -> None:
        assert _parse_duration("3s") == 3.0

    def test_milliseconds(self) -> None:
        assert _parse_duration("500ms") == 0.5

    def test_float_seconds(self) -> None:
        assert _parse_duration("1.5s") == 1.5

    def test_bare_number(self) -> None:
        assert _parse_duration("2") == 2.0

    def test_invalid(self) -> None:
        assert _parse_duration("fast") == 1.0

    def test_whitespace_before_unit_accepted(self) -> None:
        from linc_crucible.core.bvt.foreground import foreground_window_s
        from linc_crucible.core.bvt.step_handlers import _duration_seconds

        assert _parse_duration("5s") == 5.0
        assert _parse_duration("250 ms") == 0.25
        assert _duration_seconds("5s", 9.0, what="t") == 5.0
        assert _duration_seconds("250 ms", 9.0, what="t") == 0.25
        assert foreground_window_s({"timeout": "5s"}, default=9.0, maximum=60.0) == 5.0
        assert foreground_window_s({"timeout": "250 ms"}, default=9.0, maximum=60.0) == 0.25

    def test_trailing_junk_and_newline_rejected(self) -> None:
        from linc_crucible.core.bvt import foreground, step_handlers
        from linc_crucible.core.bvt.foreground import foreground_window_s
        from linc_crucible.core.bvt.step_handlers import _duration_seconds

        assert step_handlers._DURATION_RE.pattern == r"\A(\d+(?:\.\d+)?)[ \t]*(s|ms)\Z"
        assert foreground._DURATION_RE.pattern == r"\A(\d+(?:\.\d+)?)[ \t]*(s|ms)\Z"
        assert step_handlers._DURATION_RE.fullmatch("5s\n") is None
        assert foreground._DURATION_RE.fullmatch("5s\n") is None
        assert _parse_duration("5s x") == 1.0
        assert _parse_duration("5s\n x") == 1.0
        assert _duration_seconds("5s x", 9.0, what="t") == 9.0
        assert _duration_seconds("5s\n x", 9.0, what="t") == 9.0
        assert foreground_window_s({"timeout": "5s x"}, default=9.0, maximum=60.0) == 9.0
        assert foreground_window_s({"timeout": "5s\n x"}, default=9.0, maximum=60.0) == 9.0

    def test_newline_between_number_and_unit_rejected(self) -> None:
        """[contract] only spaces/tabs may separate number and unit — never a newline."""
        from linc_crucible.core.bvt import foreground, step_handlers
        from linc_crucible.core.bvt.foreground import foreground_window_s
        from linc_crucible.core.bvt.step_handlers import _duration_seconds

        for regex in (step_handlers._DURATION_RE, foreground._DURATION_RE):
            assert regex.fullmatch("5\ns") is None
            assert regex.fullmatch("5\r\nms") is None
            assert regex.fullmatch("5 \t s") is not None
        assert _parse_duration("5\ns") == 1.0
        assert _duration_seconds("5\ns", 9.0, what="t") == 9.0
        assert foreground_window_s({"timeout": "5\ns"}, default=9.0, maximum=60.0) == 9.0


class TestParseBoundsCenter:
    def test_valid(self) -> None:
        assert _parse_bounds_center("[100,200][300,400]") == (200, 300)

    def test_origin(self) -> None:
        assert _parse_bounds_center("[0,0][100,100]") == (50, 50)

    def test_invalid(self) -> None:
        assert _parse_bounds_center("invalid") is None

    def test_empty(self) -> None:
        assert _parse_bounds_center("") is None


# --- NerveAdapter HTTP-only tests ---
# (Deleted: these tested the now-removed HTTPNerveClient.call_tool delegation
#  pattern. The new NerveAdapter calls typed NerveClient methods directly;
#  tests/test_bvt_nerve_adapter.py covers delegation under the new API.)


# --- BVTRunner tests ---


def _make_suite(
    steps: list[BVTStep],
    test_name: str = "test1",
    package: str = "com.test",
) -> BVTSuite:
    return BVTSuite(
        package=package,
        tests=[BVTTest(name=test_name, steps=steps)],
    )


class TestBVTRunner:
    def test_launch_and_screenshot(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(
            elements_response=SAMPLE_ELEMENTS,
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="launch", target="com.test"),
                BVTStep(action="screenshot", target="launched"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert result.total_run == 1
        assert result.total_passed == 1
        assert result.tests[0].steps_run == 2
        assert result.tests[0].steps_passed == 2
        assert len(result.tests[0].screenshots) == 1
        # Verify screenshot file exists
        ss_path = result.tests[0].screenshots[0]
        assert Path(ss_path).exists()
        # Reset launches app before test, then explicit launch step
        assert mock.launch_calls == [("DEV1", "com.test"), ("DEV1", "com.test")]
        # Reset also kills app before test
        assert mock.kill_calls == [("DEV1", "com.test")]

    @pytest.mark.parametrize("bad_bytes", [b"", b"not a png"])
    def test_screenshot_invalid_png_fails_with_harness_error(
        self, tmp_path: Path, bad_bytes: bytes
    ) -> None:
        mock = make_mock_nerve_client(
            screenshot_bytes=bad_bytes,
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
        suite = _make_suite(
            [
                BVTStep(action="launch", target="com.test"),
                BVTStep(action="screenshot", target="snap"),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed
        assert result.tests[0].harness_error is True
        expected = f"HARNESS: screenshot returned {len(bad_bytes)} bytes, not a PNG"
        assert expected in result.tests[0].failure_message
        assert len(result.tests[0].screenshots) == 0
        screenshot_dir = tmp_path / "screenshots"
        written_files = list(screenshot_dir.glob("*.png")) if screenshot_dir.exists() else []
        assert written_files == []

    def test_screenshot_valid_png_passes_and_written(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
        suite = _make_suite(
            [
                BVTStep(action="launch", target="com.test"),
                BVTStep(action="screenshot", target="snap"),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert result.tests[0].harness_error is False
        assert len(result.tests[0].screenshots) == 1
        assert Path(result.tests[0].screenshots[0]).exists()

    def test_assert_element_passes(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="assert_element", target="level_screen"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed

    def test_assert_element_fails(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="assert_element", target="nonexistent_element"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed
        assert result.total_failed == 1
        assert "not found" in result.tests[0].failure_message

    def test_assert_text_passes(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="assert_text", target="Settings"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed

    def test_fail_fast_within_test(self, tmp_path: Path) -> None:
        """After first failure, remaining steps are skipped."""
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="assert_element", target="nonexistent"),  # fails
                BVTStep(action="screenshot", target="should_not_run"),  # skipped
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed
        assert result.tests[0].steps_run == 1  # Only first step ran
        assert result.tests[0].steps_passed == 0

    def test_continue_across_tests(self, tmp_path: Path) -> None:
        """Suite continues to next test even if one fails."""
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = BVTSuite(
            package="com.test",
            tests=[
                BVTTest(
                    name="failing",
                    steps=[
                        BVTStep(action="assert_element", target="nonexistent"),
                    ],
                ),
                BVTTest(
                    name="passing",
                    steps=[
                        BVTStep(action="assert_element", target="level_screen"),
                    ],
                ),
            ],
        )

        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed  # Suite fails (one test failed)
        assert result.total_run == 2
        assert result.total_passed == 1
        assert result.total_failed == 1
        assert not result.tests[0].passed
        assert result.tests[1].passed
        # Each test gets a kill+launch reset cycle
        assert len(mock.kill_calls) == 2
        assert len(mock.launch_calls) == 2

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_wait_step(self, mock_sleep, tmp_path: Path) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="wait", target="3s"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        # sleep called twice: 1.5s for app reset + 3.0s for the wait step
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.5)
        mock_sleep.assert_any_call(3.0)

    def test_tap_step(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="tap", target="settings_button"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        # tap_by_id now synthesizes a ~50ms down→up via long_press (combinedClickable fix)
        mock.long_press.assert_called_once_with("DEV1", 975, 100, 50)
        assert mock.tap_calls == []

    def test_unknown_action(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="unknown_action", target="foo"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed
        assert "Unknown action" in result.tests[0].failure_message

    def test_launch_failure_fails_test(self, tmp_path: Path) -> None:
        mock = _mock_with_launch_error("Activity not found")
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="launch", target="com.test"),
                BVTStep(action="screenshot", target="should_not_run"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed
        assert result.tests[0].steps_run == 1  # Fail-fast after launch

    def test_kill_step(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="kill", target="com.test"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        # kill called twice: once for app reset, once for the explicit kill step
        assert mock.kill_calls == [("DEV1", "com.test"), ("DEV1", "com.test")]

    def test_navigate_step_calls_nerve_navigate(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}]
        )
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="navigate", target="app://settings"),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        assert result.tests[0].steps_passed == 1
        # Verify navigate called nerve.navigate with uri
        assert mock.navigate_calls == [("DEV1", "app://settings")]

    def test_deeplink_alias_dispatches_to_navigate(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}]
        )
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="deeplink", target="precisionlevel://screen/settings"),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        assert result.tests[0].steps_passed == 1
        assert mock.navigate_calls == [("DEV1", "precisionlevel://screen/settings")]

    def test_navigate_step_failure(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client()

        def _failing_navigate(device: str, uri: str) -> dict:
            mock.navigate_calls.append((device, uri))
            raise ConnectionError("daemon down")

        mock.navigate.side_effect = _failing_navigate
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="navigate", target="app://settings"),
                BVTStep(action="screenshot", target="should_not_run"),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        assert result.tests[0].steps_run == 1  # Fail-fast

    def test_run_result_properties(self, tmp_path: Path) -> None:
        result = BVTRunResult(device="DEV1", package="com.test", tests=[])
        assert result.total_run == 0
        assert result.total_passed == 0
        assert result.total_failed == 0
        assert not result.passed  # Empty suite = not passed

    def test_duration_tracked(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="assert_element", target="level_screen"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.duration_ms >= 0
        assert result.tests[0].duration_ms >= 0

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_full_bvt_flow(self, mock_sleep, tmp_path: Path) -> None:
        """End-to-end: launch → wait → assert → tap → assert → screenshot."""
        mock = make_mock_nerve_client(
            elements_response=SAMPLE_ELEMENTS,
            screen_state_responses=[
                {"foreground": {"package": "com.lincmobile.level", "activity": ".M"}}
            ],
        )
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="launch", target="com.lincmobile.level"),
                BVTStep(action="wait", target="100ms"),
                BVTStep(action="assert_element", target="level_screen"),
                BVTStep(action="assert_element", target="pitch_readout"),
                BVTStep(action="tap", target="settings_button"),
                BVTStep(action="assert_text", target="Settings"),
                BVTStep(action="screenshot", target="settings_open"),
            ],
            test_name="full_flow",
            package="com.lincmobile.level",
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert result.tests[0].steps_run == 7
        assert result.tests[0].steps_passed == 7
        assert len(result.tests[0].screenshots) == 1
        # Reset launches app before test, then explicit launch step in the flow
        assert mock.launch_calls == [
            ("DEV1", "com.lincmobile.level"),
            ("DEV1", "com.lincmobile.level"),
        ]
        assert mock.kill_calls == [("DEV1", "com.lincmobile.level")]
        # tap step now routes through long_press with the default ~50ms duration
        mock.long_press.assert_called_once_with("DEV1", 975, 100, 50)
        assert mock.tap_calls == []

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_mixed_package_suite(self, mock_sleep, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = BVTSuite(
            package="com.default",
            tests=[
                BVTTest(
                    name="default_pkg",
                    steps=[BVTStep(action="assert_element", target="level_screen")],
                ),
                BVTTest(
                    name="override_pkg",
                    package="com.override",
                    steps=[BVTStep(action="assert_element", target="level_screen")],
                ),
            ],
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert result.tests[0].package == "com.default"
        assert result.tests[1].package == "com.override"

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_app_reset_between_tests(self, mock_sleep, tmp_path: Path) -> None:
        """Each test gets a kill+launch reset cycle for clean state (IMP-046)."""
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = BVTSuite(
            package="com.test",
            tests=[
                BVTTest(
                    name="test_a",
                    steps=[
                        BVTStep(action="assert_element", target="level_screen"),
                    ],
                ),
                BVTTest(
                    name="test_b",
                    steps=[
                        BVTStep(action="assert_element", target="level_screen"),
                    ],
                ),
                BVTTest(
                    name="test_c",
                    steps=[
                        BVTStep(action="assert_element", target="level_screen"),
                    ],
                ),
            ],
        )

        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert result.total_run == 3
        # 3 resets = 3 kills + 3 launches
        assert len(mock.kill_calls) == 3
        assert len(mock.launch_calls) == 3
        # Each reset waits 1.5s for app init
        assert mock_sleep.call_count == 3
        for call in mock_sleep.call_args_list:
            assert call.args[0] == 1.5

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_app_reset_nonfatal_on_kill_failure(self, mock_sleep, tmp_path: Path) -> None:
        """Reset is non-fatal: test proceeds even if kill raises an exception."""
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)

        # Make kill raise an exception
        def failing_kill(serial: str, package: str) -> None:
            raise RuntimeError("kill failed")

        mock.kill = failing_kill

        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
        suite = _make_suite(
            [
                BVTStep(action="assert_element", target="level_screen"),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        # Test still passes despite kill failure in reset
        assert result.passed

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_failure_message_includes_step_number(self, mock_sleep, tmp_path: Path) -> None:
        """Failed assertion message includes the step index for diagnostics."""
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="assert_element", target="level_screen"),
                BVTStep(action="assert_element", target="nonexistent"),  # fails at step 2
            ]
        )

        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed
        assert "Step 2" in result.tests[0].failure_message
        assert "nonexistent" in result.tests[0].failure_message


# --- tap_by_text tests ---


class TestTapByText:
    def test_tap_by_text_success(self) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        result = adapter.tap_by_text("DEV1", "Settings")
        assert result["success"]
        # Center of [900,50][1050,150] = (975, 100); tap_by_text uses the legacy
        # raw tap (the duration fix is scoped to tap_by_id).
        assert mock.tap_calls == [("DEV1", 975, 100)]

    def test_tap_by_text_not_found(self) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        result = adapter.tap_by_text("DEV1", "Nonexistent")
        assert not result["success"]
        assert "Nonexistent" in result["error"]

    def test_tap_by_text_case_insensitive(self) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        result = adapter.tap_by_text("DEV1", "settings")
        assert result["success"]


# --- dismiss_dialog tests ---


class TestDismissDialog:
    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_dismiss_dialog_success(self, mock_sleep, tmp_path: Path) -> None:
        # Verify-gone semantics: the dialog button is present until tapped, then
        # gone — dismiss_dialog only passes once the target has left the screen.
        state = {"tapped": False}
        settings_el = [{"text": "Settings", "bounds": "[845,123][967,245]"}]
        mock = make_mock_nerve_client()
        mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: (
            [] if state["tapped"] else settings_el
        )

        def _tap(device, x, y):
            state["tapped"] = True
            return {"success": True, "action": "tap", "x": x, "y": y}

        mock.tap.side_effect = _tap
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="dismiss_dialog", target="Settings", params={"timeout": 3}),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert result.tests[0].steps_passed == 1
        assert state["tapped"]

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    @patch("linc_crucible.core.bvt.runner.time.monotonic")
    def test_dismiss_dialog_not_found(self, mock_mono, mock_sleep, tmp_path: Path) -> None:
        # monotonic calls: run_suite start(0), _run_test start(0),
        # dismiss_dialog deadline(0), dismiss_dialog check(4),
        # _run_test end(4), run_suite end(4)
        mock_mono.side_effect = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]
        mock = make_mock_nerve_client(elements_response=[])
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="dismiss_dialog", target="Accept", params={"timeout": 3}),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed
        assert "not found" in result.tests[0].failure_message

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    @patch("linc_crucible.core.bvt.runner.time.monotonic")
    def test_dismiss_dialog_optional_not_found(
        self, mock_mono, mock_sleep, tmp_path: Path
    ) -> None:
        mock_mono.side_effect = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]
        mock = make_mock_nerve_client(elements_response=[])
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(
                    action="dismiss_dialog",
                    target="Accept",
                    params={"timeout": 3, "optional": True},
                ),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")
        assert result.passed

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    @patch("linc_crucible.core.bvt.runner.time.monotonic")
    def test_dismiss_dialog_timeout_zero(self, mock_mono, mock_sleep, tmp_path: Path) -> None:
        """With timeout=0, tries once then stops."""
        # deadline = 0+0 = 0, check: 0 >= 0 is True, so breaks immediately
        mock_mono.side_effect = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock = make_mock_nerve_client(elements_response=[])
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(action="dismiss_dialog", target="OK", params={"timeout": 0}),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed


class TestNewVerbHandlers:
    """type / long_press / assert_clickable / assert_min_size dispatch + dp conversion."""

    def _runner(self, tmp_path: Path):
        from unittest.mock import MagicMock

        adapter = MagicMock(spec=NerveAdapter)
        adapter.metrics.return_value = None
        # Mirror the lenient elements() stub onto the STRICT read the assert
        # handlers now use — tests below only ever configure `adapter.elements`.
        adapter.elements_strict.side_effect = lambda device: adapter.elements(device)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
        return runner, adapter

    def test_do_type_calls_adapter(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.type.return_value = {"success": True}
        result = runner._execute_step(BVTStep(action="type", target="06"), "DEV1", "pkg", "t")
        assert result.passed
        adapter.type.assert_called_once_with("DEV1", "06")

    def test_do_type_failure(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.type.return_value = {"success": False, "error": "boom"}
        result = runner._execute_step(BVTStep(action="type", target="x"), "DEV1", "pkg", "t")
        assert not result.passed
        assert "boom" in result.message

    def test_do_long_press_forwards_int_duration(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.long_press_by_id.return_value = {"success": True}
        step = BVTStep(action="long_press", target="info_button", params={"duration_ms": "800"})
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        adapter.long_press_by_id.assert_called_once_with("DEV1", "info_button", duration_ms=800)

    def test_do_assert_clickable(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.elements.return_value = [{"id": "info_button", "clickable": True}]
        result = runner._execute_step(
            BVTStep(action="assert_clickable", target="info_button"), "DEV1", "pkg", "t"
        )
        assert result.passed

    def test_assert_min_size_dp_conversion_pass(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {"density_factor": 2.625}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][147,147]"}]
        step = BVTStep(
            action="assert_min_size",
            target="",
            params={"element": "btn", "width": "56dp", "height": "56dp"},
        )
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed  # 56dp * 2.625 = 147px, element is 147px

    def test_assert_min_size_px_passthrough_and_float_space(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {"density_factor": 2.625}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][100,100]"}]
        step = BVTStep(
            action="assert_min_size",
            target="",
            params={"element": "btn", "width": "100px", "height": "100.0 px"},
        )
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed

    def test_assert_min_size_hard_fails_when_density_unreadable(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {}  # daemon down / no nerve_density route
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][9999,9999]"}]
        step = BVTStep(
            action="assert_min_size",
            target="",
            params={"element": "btn", "width": "56dp", "height": "56dp"},
        )
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert not result.passed
        assert "density unavailable" in result.message

    def test_assert_min_size_density_cached_per_run(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {"density_factor": 2.625}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][147,147]"}]
        step = BVTStep(
            action="assert_min_size",
            target="",
            params={"element": "btn", "width": "56dp", "height": "56dp"},
        )
        runner._execute_step(step, "DEV1", "pkg", "t")
        runner._execute_step(step, "DEV1", "pkg", "t")
        assert adapter.density.call_count == 1

    # --- assert_touch_target: density-derived threshold (issue #54) ---

    def test_touch_target_uses_density_not_the_old_132_default(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        # SM-S921U (Galaxy S24): 480dpi -> 3.0x, so 48dp = 144px, stricter than 132.
        adapter.density.return_value = {"platform": "android", "density_factor": 3.0}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][140,140]"}]
        step = BVTStep(action="assert_touch_target", target="btn")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert not result.passed  # would have PASSED under the retired 132px default
        assert "144px" in result.message

    def test_touch_target_low_dpi_threshold_is_looser(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        # Pixel 8 Pro: 360dpi -> 2.25x, so 48dp = 108px.
        adapter.density.return_value = {"platform": "android", "density_factor": 2.25}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][110,110]"}]
        step = BVTStep(action="assert_touch_target", target="btn")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        assert ">= 108px" in result.message

    def test_touch_target_ios_uses_44pt(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {"platform": "ios", "density_factor": 3.0}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][132,132]"}]
        step = BVTStep(action="assert_touch_target", target="btn")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed  # 44pt * 3.0 = 132px

    def test_touch_target_hard_fails_when_density_unreadable(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][9999,9999]"}]
        step = BVTStep(action="assert_touch_target", target="btn")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert not result.passed
        assert "density unavailable" in result.message

    def test_touch_target_explicit_override_still_honoured(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {"platform": "android", "density_factor": 3.0}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][100,100]"}]
        step = BVTStep(action="assert_touch_target", target="btn", params={"min_size_px": 96})
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed

    def test_touch_target_override_accepts_dp(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {"platform": "android", "density_factor": 3.0}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][180,180]"}]
        step = BVTStep(action="assert_touch_target", target="btn", params={"min_size_px": "60dp"})
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed  # 60dp * 3.0 = 180px

    def test_touch_target_shares_the_density_cache_with_min_size(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.density.return_value = {"platform": "android", "density_factor": 2.625}
        adapter.elements.return_value = [{"id": "btn", "bounds": "[0,0][147,147]"}]
        runner._execute_step(
            BVTStep(
                action="assert_min_size",
                target="",
                params={"element": "btn", "width": "56dp", "height": "56dp"},
            ),
            "DEV1",
            "pkg",
            "t",
        )
        runner._execute_step(
            BVTStep(action="assert_touch_target", target="btn"), "DEV1", "pkg", "t"
        )
        assert adapter.density.call_count == 1  # one density source, shared

    # --- tap_xy ---

    def test_do_tap_xy_taps_at_coords(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.tap.return_value = {"success": True}
        step = BVTStep(action="tap_xy", target="", params={"x": 120, "y": 340})
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        # tap_xy taps the literal pixel with no synthesized duration (raw-coord escape hatch)
        adapter.tap.assert_called_once_with("DEV1", 120, 340)

    def test_do_tap_xy_coerces_string_coords(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.tap.return_value = {"success": True}
        step = BVTStep(action="tap_xy", target="", params={"x": "120", "y": "340"})
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        adapter.tap.assert_called_once_with("DEV1", 120, 340)

    def test_do_tap_xy_missing_coords_fails(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        step = BVTStep(action="tap_xy", target="", params={"x": 120})
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert not result.passed
        assert "x" in result.message and "y" in result.message
        adapter.tap.assert_not_called()

    def test_do_tap_xy_propagates_failure(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.tap.return_value = {"success": False, "error": "boom"}
        step = BVTStep(action="tap_xy", target="", params={"x": 1, "y": 2})
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert not result.passed
        assert "boom" in result.message

    # --- tap_text ---

    def test_do_tap_text_resolves_and_taps(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.tap_by_text.return_value = {"success": True}
        step = BVTStep(action="tap_text", target="Settings")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        adapter.tap_by_text.assert_called_once_with("DEV1", "Settings")

    def test_do_tap_text_not_found_fails(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.tap_by_text.return_value = {
            "success": False,
            "error": "No element with text 'Nope'",
        }
        step = BVTStep(action="tap_text", target="Nope")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert not result.passed
        assert "Nope" in result.message

    # --- press_key / ime_action / enter ---

    def test_do_press_key_defaults_to_enter(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.key.return_value = {"success": True}
        # Bare `press_key` with no target/params submits the IME-Done form.
        step = BVTStep(action="press_key")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        adapter.key.assert_called_once_with("DEV1", "ENTER")

    def test_do_press_key_target_key_name(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.key.return_value = {"success": True}
        step = BVTStep(action="press_key", target="BACK")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        adapter.key.assert_called_once_with("DEV1", "BACK")

    def test_do_press_key_params_key_name(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.key.return_value = {"success": True}
        step = BVTStep(action="press_key", target="", params={"key": "TAB"})
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        adapter.key.assert_called_once_with("DEV1", "TAB")

    def test_ime_action_alias_dispatches_to_press_key(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.key.return_value = {"success": True}
        step = BVTStep(action="ime_action")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        adapter.key.assert_called_once_with("DEV1", "ENTER")

    def test_enter_alias_dispatches_to_press_key(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.key.return_value = {"success": True}
        step = BVTStep(action="enter")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert result.passed
        adapter.key.assert_called_once_with("DEV1", "ENTER")

    def test_do_press_key_propagates_failure(self, tmp_path: Path) -> None:
        runner, adapter = self._runner(tmp_path)
        adapter.key.return_value = {"success": False, "error": "boom"}
        step = BVTStep(action="press_key", target="ENTER")
        result = runner._execute_step(step, "DEV1", "pkg", "t")
        assert not result.passed
        assert "boom" in result.message


class TestTapVerbIntegration:
    """End-to-end through run_suite to prove the verbs are wired into dispatch."""

    def test_tap_xy_step_through_run_suite(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
        suite = _make_suite([BVTStep(action="tap_xy", target="", params={"x": 50, "y": 60})])
        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert ("DEV1", 50, 60) in mock.tap_calls

    def test_tap_text_step_through_run_suite(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
        suite = _make_suite([BVTStep(action="tap_text", target="Settings")])
        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        # Center of Settings element [900,50][1050,150] = (975, 100); tap_text
        # uses the legacy raw tap (duration fix is scoped to tap_by_id).
        assert ("DEV1", 975, 100) in mock.tap_calls

    def test_press_key_step_through_run_suite(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
        suite = _make_suite([BVTStep(action="press_key", target="ENTER")])
        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert ("DEV1", "ENTER") in mock.key_calls


def test_bvt_test_result_supports_skip():
    r = BVTTestResult(
        name="t",
        passed=True,
        package="com.test",
        skipped=True,
        skip_reason="device not ready: asleep",
    )
    assert r.skipped is True and r.skip_reason == "device not ready: asleep"


def test_bvt_test_result_skip_defaults_false():
    # package is additive (defaults to ""): construction without it must still work.
    assert BVTTestResult(name="t", passed=True).skipped is False
    assert BVTTestResult(name="t", passed=True).package == ""


def test_run_suite_skips_all_tests_when_device_not_ready(tmp_path: Path) -> None:
    """A not-ready device must SKIP every test (no execution, no corrupt artifacts)."""
    mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
    adapter = NerveAdapter(nerve=mock)
    adapter.ensure_ready = lambda device: {"ready": False, "reason": "asleep (wake failed)"}
    runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
    suite = BVTSuite(
        package="com.test",
        tests=[
            BVTTest(name="t1", steps=[BVTStep(action="assert_element", target="level_screen")]),
            BVTTest(name="t2", steps=[BVTStep(action="assert_element", target="level_screen")]),
        ],
    )

    result = runner.run_suite(suite, device="DEV1")

    assert all(t.skipped for t in result.tests)
    assert len(result.tests) == 2
    assert "asleep" in result.tests[0].skip_reason
    # No test executed -> no app reset (kill/launch) happened.
    assert mock.kill_calls == []
    assert mock.launch_calls == []


def test_skipped_suite_populates_per_test_package(tmp_path: Path) -> None:
    """AC3: even on the all-skipped (device-not-ready) path, each skipped test carries
    its resolved package — the test's own package overrides, else the suite package."""
    mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
    adapter = NerveAdapter(nerve=mock)
    adapter.ensure_ready = lambda device: {"ready": False, "reason": "asleep (wake failed)"}
    runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
    suite = BVTSuite(
        package="com.default",
        tests=[
            BVTTest(
                name="uses_suite_pkg",
                steps=[BVTStep(action="assert_element", target="level_screen")],
            ),
            BVTTest(
                name="overrides_pkg",
                package="com.override",
                steps=[BVTStep(action="assert_element", target="level_screen")],
            ),
        ],
    )

    result = runner.run_suite(suite, device="DEV1")

    assert all(t.skipped for t in result.tests)
    assert result.tests[0].package == "com.default"
    assert result.tests[1].package == "com.override"


def test_all_skipped_suite_is_not_passed(tmp_path: Path) -> None:
    """Skipped tests are not passes: an all-skipped run must not report green."""
    mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
    adapter = NerveAdapter(nerve=mock)
    adapter.ensure_ready = lambda device: {"ready": False, "reason": "asleep (wake failed)"}
    runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
    suite = BVTSuite(
        package="com.test",
        tests=[
            BVTTest(name="t1", steps=[BVTStep(action="assert_element", target="level_screen")]),
            BVTTest(name="t2", steps=[BVTStep(action="assert_element", target="level_screen")]),
        ],
    )

    result = runner.run_suite(suite, device="DEV1")

    assert result.passed is False
    assert result.total_passed == 0
    assert result.total_failed == 0  # skipped is neither passed nor failed
    assert result.total_skipped == 2


def test_run_result_aggregates_exclude_skipped() -> None:
    """Mixed results: skipped tests count in neither passed nor failed."""
    result = BVTRunResult(
        device="DEV1",
        package="com.test",
        tests=[
            BVTTestResult(name="ok", passed=True, package="com.test"),
            BVTTestResult(
                name="skip",
                passed=False,
                package="com.test",
                skipped=True,
                skip_reason="not ready",
            ),
            BVTTestResult(name="bad", passed=False, package="com.test"),
        ],
    )
    assert result.total_passed == 1
    assert result.total_failed == 1
    assert result.total_skipped == 1
    assert result.passed is False


def test_run_result_passed_when_executed_all_green_despite_skips() -> None:
    """A skip must not veto an otherwise-green run — only exclude itself."""
    result = BVTRunResult(
        device="DEV1",
        package="com.test",
        tests=[
            BVTTestResult(name="ok", passed=True, package="com.test"),
            BVTTestResult(
                name="skip",
                passed=False,
                package="com.test",
                skipped=True,
                skip_reason="not ready",
            ),
        ],
    )
    assert result.passed is True
    assert result.total_passed == 1


# --- T5: negative + count assertion handlers ---


def _runner_for(mock, tmp_path):
    return BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)


_NEG_COUNT_ELEMENTS = [
    {"id": "row", "text": "A", "bounds": "[0,0][10,10]"},
    {"id": "row", "text": "B", "bounds": "[0,10][10,20]"},
    {"id": "banner", "text": "Error happened", "bounds": "[0,20][10,30]"},
]


class TestNegativeAndCountHandlers:
    def test_assert_element_not_present_pass(self, tmp_path) -> None:
        mock = make_mock_nerve_client(elements_response=_NEG_COUNT_ELEMENTS)
        r = _runner_for(mock, tmp_path)._execute_step(
            BVTStep(action="assert_element_not_present", target="ghost"), "dev", "pkg", "t"
        )
        assert r.passed

    def test_assert_element_not_present_fail(self, tmp_path) -> None:
        mock = make_mock_nerve_client(elements_response=_NEG_COUNT_ELEMENTS)
        r = _runner_for(mock, tmp_path)._execute_step(
            BVTStep(action="assert_element_not_present", target="row"), "dev", "pkg", "t"
        )
        assert not r.passed

    def test_assert_text_not_present_fail(self, tmp_path) -> None:
        mock = make_mock_nerve_client(elements_response=_NEG_COUNT_ELEMENTS)
        r = _runner_for(mock, tmp_path)._execute_step(
            BVTStep(action="assert_text_not_present", target="error"), "dev", "pkg", "t"
        )
        assert not r.passed

    def test_assert_element_count_pass(self, tmp_path) -> None:
        mock = make_mock_nerve_client(elements_response=_NEG_COUNT_ELEMENTS)
        r = _runner_for(mock, tmp_path)._execute_step(
            BVTStep(action="assert_element_count", params={"element": "row", "expected": 2}),
            "dev",
            "pkg",
            "t",
        )
        assert r.passed

    def test_assert_element_count_bad_bounds_fails_gracefully(self, tmp_path) -> None:
        mock = make_mock_nerve_client(elements_response=_NEG_COUNT_ELEMENTS)
        r = _runner_for(mock, tmp_path)._execute_step(
            BVTStep(action="assert_element_count", params={"element": "row"}),  # no bound
            "dev",
            "pkg",
            "t",
        )
        assert not r.passed  # ValueError caught -> failed step, not a crash
        assert "required" in r.message.lower() or "bound" in r.message.lower()


# --- Harness-error UI tree unavailable (reactor 6a4164a5) ---


def _mock_tree_unavailable(detail: str = "could not get idle state"):
    """Realistic mock whose STRICT elements read raises UiTreeUnavailableError
    while the lenient elements() stays on the default [] happy path."""
    mock = make_mock_nerve_client()
    mock.elements_strict.side_effect = UiTreeUnavailableError(detail)
    return mock


class TestHarnessErrorUiTreeUnavailable:
    """A failed UI-tree dump must surface as a HARNESS fault — never as a
    false product failure ('not found in 0 elements') and never as a
    vacuous not-present pass on a tree that was never read."""

    def test_assert_text_reports_harness_error(self, tmp_path: Path) -> None:
        mock = _mock_tree_unavailable()
        r = _runner_for(mock, tmp_path)._execute_step(
            BVTStep(action="assert_text", target="Settings"), "dev", "pkg", "t"
        )
        assert not r.passed
        assert r.harness_error
        assert r.message.startswith("HARNESS:")
        assert "0 elements" not in r.message

    def test_assert_element_not_present_reports_harness_error(self, tmp_path: Path) -> None:
        mock = _mock_tree_unavailable()
        r = _runner_for(mock, tmp_path)._execute_step(
            BVTStep(action="assert_element_not_present", target="ghost"), "dev", "pkg", "t"
        )
        assert not r.passed
        assert r.harness_error

    def test_assert_text_not_present_reports_harness_error(self, tmp_path: Path) -> None:
        mock = _mock_tree_unavailable()
        r = _runner_for(mock, tmp_path)._execute_step(
            BVTStep(action="assert_text_not_present", target="error"), "dev", "pkg", "t"
        )
        assert not r.passed
        assert r.harness_error

    def test_run_suite_marks_test_and_run_result_harness_error(self, tmp_path: Path) -> None:
        mock = _mock_tree_unavailable("could not get idle state")
        runner = _runner_for(mock, tmp_path)
        suite = _make_suite([BVTStep(action="assert_text", target="Settings")])

        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        assert result.total_harness_errors == 1
        test_result = result.tests[0]
        assert test_result.harness_error
        assert test_result.failure_message.startswith("HARNESS:")

    def test_wait_for_element_read_empty_tree_is_product_timeout(self, tmp_path: Path) -> None:
        """Pin: a tree that IS read (empty) and lacks the element is a normal
        timeout, no harness flag — only an UNREADABLE tree is a harness fault."""
        mock = make_mock_nerve_client(elements_response=[])
        step = BVTStep(action="wait_for_element", target="home_btn", params={"timeout": 0})
        res = _runner_for(mock, tmp_path)._do_wait_for_element(step, "DEV", "pkg", "t")
        assert not res.passed
        assert not res.harness_error
        assert "did not appear" in res.message

    @pytest.mark.parametrize(
        "step",
        [
            BVTStep(action="tap", target="settings_button"),
            BVTStep(action="tap_text", target="Settings"),
            BVTStep(action="long_press", target="settings_button"),
        ],
        ids=["tap", "tap_text", "long_press"],
    )
    def test_element_verb_on_unavailable_tree_is_harness_error(
        self, step: BVTStep, tmp_path: Path
    ) -> None:
        """audit F7: tap/tap_text/long_press resolve the element over a STRICT
        read; an unreadable tree is HARNESS, not "No element ... found"."""
        mock = _mock_tree_unavailable("could not get idle state")
        r = _runner_for(mock, tmp_path)._execute_step(step, "dev", "pkg", "t")
        assert not r.passed
        assert r.harness_error is True
        assert r.message.startswith("HARNESS:")
        assert "not found" not in r.message
        mock.long_press.assert_not_called()
        assert mock.tap_calls == []

    @pytest.mark.parametrize(
        "step",
        [
            BVTStep(action="tap", target="ghost"),
            BVTStep(action="tap_text", target="Ghost"),
            BVTStep(action="long_press", target="ghost"),
        ],
        ids=["tap", "tap_text", "long_press"],
    )
    def test_element_verb_on_read_tree_without_element_is_product_fault(
        self, step: BVTStep, tmp_path: Path
    ) -> None:
        """Pin: a tree that IS read but lacks the element is a product fault."""
        mock = make_mock_nerve_client(elements_response=SAMPLE_ELEMENTS)
        r = _runner_for(mock, tmp_path)._execute_step(step, "dev", "pkg", "t")
        assert not r.passed
        assert r.harness_error is False
        assert not r.message.startswith("HARNESS:")

    def test_suite_whose_only_failure_is_unreadable_tap_reports_harness_in_dict(
        self, tmp_path: Path
    ) -> None:
        """A suite failing ONLY on a tap over an unreadable tree is failed, and
        the test's harness_error is visible in the serialized result dict."""
        from dataclasses import asdict

        mock = _mock_tree_unavailable("could not get idle state")
        runner = _runner_for(mock, tmp_path)
        suite = BVTSuite(
            package="com.test",
            tests=[
                BVTTest(name="test_ok", steps=[BVTStep(action="wait", target="0s")]),
                BVTTest(name="test_tap", steps=[BVTStep(action="tap", target="settings_button")]),
            ],
        )

        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        assert result.total_failed == 1
        assert result.total_harness_errors == 1
        tests = {t["name"]: t for t in asdict(result)["tests"]}
        assert tests["test_ok"]["passed"] is True
        assert tests["test_ok"]["harness_error"] is False
        assert tests["test_tap"]["passed"] is False
        assert tests["test_tap"]["harness_error"] is True
        assert tests["test_tap"]["failure_message"].startswith("HARNESS:")

    def test_handler_raising_unexpected_exception_fails_step_as_harness_error_and_suite_continues(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        mock = make_mock_nerve_client()
        runner = _runner_for(mock, tmp_path)

        def _bad_tap(step, device, package, test_name):
            raise KeyError("missing_key")

        monkeypatch.setattr(runner, "_do_tap", _bad_tap)

        step_res = runner._execute_step(BVTStep(action="tap", target="btn"), "dev", "pkg", "t")
        assert not step_res.passed
        assert step_res.harness_error
        assert step_res.message == "step tap raised KeyError: 'missing_key'"

        suite = BVTSuite(
            package="com.test",
            tests=[
                BVTTest(
                    name="test_failing",
                    steps=[BVTStep(action="tap", target="btn")],
                ),
                BVTTest(
                    name="test_passing",
                    steps=[BVTStep(action="wait", target="0.1s")],
                ),
            ],
        )

        result = runner.run_suite(suite, device="DEV1")
        assert not result.passed
        assert result.total_run == 2
        assert result.total_passed == 1
        assert result.total_failed == 1
        assert result.total_harness_errors == 1
        assert result.tests[0].harness_error
        assert "raised KeyError" in result.tests[0].failure_message
        assert result.tests[1].passed


# --- Cold-start gate + grant_permission verb ---


def _make_tagged_suite(
    steps: list[BVTStep],
    tags: list[str],
    test_name: str = "test1",
    package: str = "com.test",
) -> BVTSuite:
    return BVTSuite(
        package=package,
        tests=[BVTTest(name=test_name, steps=steps, tags=tags)],
    )


class TestColdStartGate:
    def test_cold_start_tag_triggers_pm_clear(self, tmp_path: Path) -> None:
        """A test tagged `cold_start` is reset with pm clear before kill+launch."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        # pm clear ran exactly once, for this package
        assert mock.clear_app_data_calls == [("DEV1", "com.test")]
        # and the usual kill+relaunch still happened after
        assert mock.kill_calls == [("DEV1", "com.test")]

    def test_untagged_test_does_not_pm_clear(self, tmp_path: Path) -> None:
        """Untagged tests keep the cheap kill+relaunch — no pm clear."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite([BVTStep(action="launch", target="com.test")], tags=["smoke"])

        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        assert mock.clear_app_data_calls == []
        assert mock.kill_calls == [("DEV1", "com.test")]

    def test_cold_start_clear_failure_is_fatal(self, tmp_path: Path) -> None:
        """A failed pm clear is fatal — a cold_start test must not silently
        proceed warm and report PASS when the wipe itself failed. This is
        the vacuous-gate class the missing-confirm bug (#129) hid behind:
        the clear result's success flag must actually gate the test."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )

        def _failing_clear(device: str, package: str) -> dict:
            mock.clear_app_data_calls.append((device, package))
            return {"success": False, "error": "pm clear failed"}

        mock.clear_app_data.side_effect = _failing_clear
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        test_result = result.tests[0]
        assert not test_result.passed
        assert not test_result.skipped
        # Pin the production-owned prefix, not just the mock's own error
        # string — a regression to bare error passthrough (dropping the
        # "cold_start clear_app_data failed" wording) would still satisfy
        # a substring check on "pm clear failed" alone.
        assert "cold_start clear_app_data failed" in test_result.failure_message
        assert "pm clear failed" in test_result.failure_message
        assert mock.clear_app_data_calls == [("DEV1", "com.test")]
        # Device state is unknown after a failed clear — must not proceed
        # to kill/launch (that would run steps against an unclean device
        # while still reporting the failure, or worse, mask it entirely).
        assert mock.kill_calls == []
        assert mock.launch_calls == []

    def test_cold_start_clear_nerve_error_without_success_is_fatal(self, tmp_path: Path) -> None:
        """A cold_start clear whose nerve clear_app_data returns {"error": ...}
        WITHOUT a success key is fatal — the test is reported failed with the
        harness attribution #130 uses, and the runner never proceeds to launch
        (fails pre-fix: the test runs warm and passes) (audit F2)."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )

        def _error_clear(device: str, package: str) -> dict:
            mock.clear_app_data_calls.append((device, package))
            return {"error": "Clear app data failed: device offline"}

        mock.clear_app_data.side_effect = _error_clear
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        test_result = result.tests[0]
        assert not test_result.passed
        assert not test_result.skipped
        assert "cold_start clear_app_data failed" in test_result.failure_message
        assert "Clear app data failed: device offline" in test_result.failure_message
        assert mock.clear_app_data_calls == [("DEV1", "com.test")]
        assert mock.kill_calls == []
        assert mock.launch_calls == []

    def test_cold_start_clear_exception_is_fatal(self, tmp_path: Path) -> None:
        """A cold_start clear that raises past the adapter boundary is
        equally fatal — no exception escapes the runner, and the test fails
        loud rather than silently proceeding warm. (NerveAdapter.clear_app_data
        normally converts transport exceptions to a `{"success": False}` dict
        itself, which the sibling test above covers; this pins the runner's
        OWN defensive `_safe_reset_op` boundary in case that adapter contract
        is ever violated — mirrors the raising-install fatal path already
        pinned for the needs_reinstall branch.)"""
        from unittest.mock import MagicMock

        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        runner._nerve.clear_app_data = MagicMock(side_effect=RuntimeError("daemon down"))
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        test_result = result.tests[0]
        assert not test_result.passed
        assert not test_result.skipped
        assert "clear_app_data raised" in test_result.failure_message
        assert mock.kill_calls == []
        assert mock.launch_calls == []


class TestColdStartNeedsReinstall:
    """nerve's `needs_reinstall` contract: iOS-sim clear_app_data is
    uninstall-based, so the app must be reinstalled from the built artifact
    before the relaunch. Android (`pm clear`) responses carry
    needs_reinstall=False (or omit the key on older daemons) — behavior there
    must stay byte-identical to today."""

    @staticmethod
    def _mock_with_clear_response(clear_response: dict):
        """Mock whose clear_app_data returns the given dict and whose install
        is logged into op_log so cross-op ordering can be asserted."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )

        def _clear(device: str, package: str) -> dict:
            mock.clear_app_data_calls.append((device, package))
            mock.op_log.append(("clear_app_data", device, package))
            return dict(clear_response)

        def _install(device: str, path: str) -> dict:
            mock.install_calls.append((device, path))
            mock.op_log.append(("install", device, path))
            return {"success": True}

        mock.clear_app_data.side_effect = _clear
        mock.install.side_effect = _install
        return mock

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_needs_reinstall_with_app_path_installs_before_relaunch(
        self, _sleep, tmp_path: Path
    ) -> None:
        """needs_reinstall=True + app_path → install(device, app_path) runs
        after the clear and BEFORE the relaunch; relaunch still happens."""
        mock = self._mock_with_clear_response({"success": True, "needs_reinstall": True})
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="SIM1", app_path="/builds/App.app")

        assert result.passed
        assert mock.install_calls == [("SIM1", "/builds/App.app")]
        # Ordering: clear → install → launch (relaunch still happens, after install)
        ops = [op[0] for op in mock.op_log]
        assert ops.index("clear_app_data") < ops.index("install")
        assert ops.index("install") < ops.index("launch")
        # Reset relaunch + the test's own launch step
        assert mock.launch_calls == [("SIM1", "com.test"), ("SIM1", "com.test")]

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_needs_reinstall_matches_real_nerve_adapter_contract_shape(
        self, _sleep, tmp_path: Path
    ) -> None:
        """Regression pin for the fatal-on-failed-clear gate: feed
        `_cold_start_wipe` a shape modeled on `NerveAdapter.clear_app_data`'s
        own normalization contract (nerve_adapter.py:127:
        ``return result if "success" in result else {**result, "success": True}``)
        — extra `action`/`package` keys plus `needs_reinstall`, not just the
        trimmed `{"success": True, "needs_reinstall": True}` shape used
        elsewhere in this class. (This is a shape assumption modeled on the
        adapter's own code, not a verified capture of the underlying nerve
        daemon's wire payload — that lives outside this repo.) The `success`
        fatal-check must not be tripped by the extra `action`/`package` keys
        a real response carries — execution must still reach install +
        relaunch."""
        mock = self._mock_with_clear_response(
            {
                "success": True,
                "action": "clear_app_data",
                "package": "com.test",
                "needs_reinstall": True,
            }
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="SIM1", app_path="/builds/App.app")

        assert result.passed
        assert mock.install_calls == [("SIM1", "/builds/App.app")]
        ops = [op[0] for op in mock.op_log]
        assert ops.index("clear_app_data") < ops.index("install") < ops.index("launch")

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_needs_reinstall_missing_success_key_still_proceeds(
        self, _sleep, tmp_path: Path
    ) -> None:
        """Regression pin for round-1 review finding 1: a raw nerve response
        that OMITS `success` entirely (older daemon) must not trip the
        fatal-on-failed-clear gate, because `NerveAdapter.clear_app_data`
        normalizes it BEFORE the runner ever sees it — see nerve_adapter.py:127
        (``return result if "success" in result else {**result, "success": True}``).
        This drives the RAW mock (the one `NerveAdapter` wraps, matching
        production's `BVTRunner(nerve=NerveAdapter(...))` wiring — see
        bvt/__init__.py:10 and bvt_tools.py:200) rather than pinning a
        hand-built post-adapter dict, so it exercises the real adapter
        boundary instead of assuming its output."""
        mock = self._mock_with_clear_response({"needs_reinstall": True})
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="SIM1", app_path="/builds/App.app")

        assert result.passed
        assert mock.install_calls == [("SIM1", "/builds/App.app")]
        ops = [op[0] for op in mock.op_log]
        assert ops.index("clear_app_data") < ops.index("install") < ops.index("launch")

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_needs_reinstall_without_app_path_fails_test_actionably(
        self, _sleep, tmp_path: Path
    ) -> None:
        """needs_reinstall=True + no app_path → the TEST fails with an
        actionable message; no install attempted; no exception escapes."""
        mock = self._mock_with_clear_response({"success": True, "needs_reinstall": True})
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="SIM1")

        assert not result.passed
        assert result.total_failed == 1
        test_result = result.tests[0]
        assert not test_result.passed
        assert not test_result.skipped
        assert (
            "iOS cold_start requires app_path "
            "(uninstall-based clear removed the app) — pass the built artifact path"
        ) in test_result.failure_message
        assert mock.install_calls == []

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_failed_reinstall_is_fatal_no_relaunch(self, _sleep, tmp_path: Path) -> None:
        """Reinstall failure after an uninstall-based clear is fatal —
        symmetric with the no-app_path branch: the device has no app, so
        kill/launch must not run and the test fails with the reinstall error."""
        mock = self._mock_with_clear_response({"success": True, "needs_reinstall": True})

        def _failing_install(device: str, path: str) -> dict:
            mock.install_calls.append((device, path))
            mock.op_log.append(("install", device, path))
            return {"success": False, "error": "boom"}

        mock.install.side_effect = _failing_install
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="SIM1", app_path="/builds/App.app")

        assert not result.passed
        test_result = result.tests[0]
        assert not test_result.passed
        assert not test_result.skipped
        assert (
            "iOS cold_start reinstall failed after uninstall-based clear — "
            "device has no app installed: boom"
        ) in test_result.failure_message
        # No launch ops after the failed install — device has no app.
        assert [op[0] for op in mock.op_log] == ["clear_app_data", "install"]

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_raising_reinstall_is_fatal_no_relaunch(self, _sleep, tmp_path: Path) -> None:
        """An install that raises (transport error) is equally fatal — no
        exception escapes, no kill/launch after the install attempt."""
        mock = self._mock_with_clear_response({"success": True, "needs_reinstall": True})

        def _raising_install(device: str, path: str) -> dict:
            mock.op_log.append(("install", device, path))
            raise RuntimeError("daemon down")

        mock.install.side_effect = _raising_install
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="SIM1", app_path="/builds/App.app")

        assert not result.passed
        test_result = result.tests[0]
        assert not test_result.passed
        assert (
            "iOS cold_start reinstall failed after uninstall-based clear — "
            "device has no app installed"
        ) in test_result.failure_message
        assert [op[0] for op in mock.op_log] == ["clear_app_data", "install"]

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_needs_reinstall_absent_keeps_current_behavior(self, _sleep, tmp_path: Path) -> None:
        """Key absent (older nerve daemon) → no install call, flow identical
        to today: clear → kill → launch and the test passes."""
        mock = self._mock_with_clear_response({"success": True})
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="DEV1", app_path="/builds/app.apk")

        assert result.passed
        assert mock.install_calls == []
        assert [op[0] for op in mock.op_log] == ["clear_app_data", "kill", "launch", "launch"]

    @patch("linc_crucible.core.bvt.runner.time.sleep")
    def test_needs_reinstall_false_keeps_current_behavior(self, _sleep, tmp_path: Path) -> None:
        """Explicit needs_reinstall=False (Android) → same as absent."""
        mock = self._mock_with_clear_response({"success": True, "needs_reinstall": False})
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_tagged_suite(
            [BVTStep(action="launch", target="com.test")], tags=["cold_start"]
        )

        result = runner.run_suite(suite, device="DEV1", app_path="/builds/app.apk")

        assert result.passed
        assert mock.install_calls == []
        assert [op[0] for op in mock.op_log] == ["clear_app_data", "kill", "launch", "launch"]


class TestColdStartOrderIndependence:
    """AC3 + council ship-blocker: `pm clear` is destructive across the suite
    (it revokes runtime permissions and wipes app state), so the suite MUST be
    order-independent — each test resets itself and a `cold_start` test must
    re-establish its own preconditions (re-grant permissions) as part of its
    own flow. These tests assert that property is enforced by the runner, not
    by luck of test ordering.
    """

    @staticmethod
    def _multi_test_suite(tests: list[BVTTest], package: str = "com.test") -> BVTSuite:
        return BVTSuite(package=package, tests=tests)

    @staticmethod
    def _op_log_segments(op_log: list[tuple]) -> list[list[tuple]]:
        """Split a flat op_log into per-test segments, keyed on the reset
        boundary. Each test's reset begins with either a ``clear_app_data``
        (cold_start) or a ``kill`` (untagged) op — the first reset op of a
        test. We segment on those reset boundaries so each segment is exactly
        one test's reset ops followed by that test's own step ops. This lets a
        caller attribute the clear (and any step op) to a *specific* test and
        assert ordering WITHIN that test, not just across the whole suite.

        NOTE: this segmenter assumes the suites it processes have no
        step-level ``kill`` or ``clear_app_data`` ops — only reset-level ones.
        A ``kill`` step (a valid BVTStep action) not preceded by a
        ``clear_app_data`` would be mis-read as a new test boundary. The five
        suites here use no such step ops, so segmentation is sound as applied.
        """
        segments: list[list[tuple]] = []
        current: list[tuple] = []
        for i, entry in enumerate(op_log):
            op = entry[0]
            # A new test begins at a clear_app_data, or at a kill that is NOT
            # immediately preceded by a clear_app_data (untagged reset). A
            # cold_start reset is clear -> kill -> launch, so its kill should
            # NOT start a new segment.
            starts_test = op == "clear_app_data" or (
                op == "kill" and (i == 0 or op_log[i - 1][0] != "clear_app_data")
            )
            if starts_test and current:
                segments.append(current)
                current = []
            current.append(entry)
        if current:
            segments.append(current)
        return segments

    def test_cold_start_then_untagged_each_resets_independently(self, tmp_path: Path) -> None:
        """A cold_start test followed by an untagged test: the cold_start test
        gets its own `pm clear` (before its own steps), and the untagged test
        that runs AFTER it gets a fresh kill + launch with NO pm clear in its
        own segment. Asserting per-test segments (not the aggregate footprint)
        proves the clear is attributed to the cold_start test specifically and
        the untagged test never sees a clear — even though both share the same
        package."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        # Distinct step-level anchor ops per test (tap coords differ) so each
        # test's OWN steps are identifiable in the ordered op_log.
        suite = self._multi_test_suite(
            [
                BVTTest(
                    name="fresh_consent",
                    steps=[BVTStep(action="grant_permission", target="android.permission.CAMERA")],
                    tags=["cold_start"],
                ),
                BVTTest(
                    name="main_screen",
                    steps=[BVTStep(action="grant_permission", target="android.permission.MIC")],
                    tags=["smoke"],
                ),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        segments = self._op_log_segments(mock.op_log)
        assert len(segments) == 2, f"expected one segment per test, got {segments}"

        cold_seg, untagged_seg = segments
        cold_ops = [e[0] for e in cold_seg]
        untagged_ops = [e[0] for e in untagged_seg]

        # The cold_start test's segment: clear FIRST (before its own step),
        # and the clear precedes that test's CAMERA grant step.
        assert cold_ops[0] == "clear_app_data"
        clear_idx = cold_ops.index("clear_app_data")
        cam_idx = cold_seg.index(
            ("grant_permission", "DEV1", "com.test", "android.permission.CAMERA")
        )
        assert clear_idx < cam_idx, "clear must precede the cold_start test's own steps"

        # The untagged test's segment: NO clear at all; its own MIC grant is
        # present and reset is kill+launch only.
        assert "clear_app_data" not in untagged_ops, "untagged test must never be cleared"
        assert ("grant_permission", "DEV1", "com.test", "android.permission.MIC") in untagged_seg
        assert untagged_ops[:2] == ["kill", "launch"]

        # Whole-suite: clear ran exactly once total.
        assert mock.clear_app_data_calls == [("DEV1", "com.test")]

    def test_untagged_then_cold_start_does_not_leak_into_earlier_test(
        self, tmp_path: Path
    ) -> None:
        """Reverse order: an untagged test running BEFORE a cold_start test must
        NOT be affected by the later test's `pm clear`. Asserting on the ordered
        op_log proves the untagged test's segment contains no clear and its own
        step ran before any clear in the whole run — so suite order does not
        change the untagged test's preconditions."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = self._multi_test_suite(
            [
                BVTTest(
                    name="main_screen",
                    steps=[BVTStep(action="grant_permission", target="android.permission.MIC")],
                    tags=["smoke"],
                ),
                BVTTest(
                    name="fresh_consent",
                    steps=[BVTStep(action="grant_permission", target="android.permission.CAMERA")],
                    tags=["cold_start"],
                ),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        segments = self._op_log_segments(mock.op_log)
        assert len(segments) == 2, f"expected one segment per test, got {segments}"

        untagged_seg, cold_seg = segments
        untagged_ops = [e[0] for e in untagged_seg]

        # Untagged test ran FIRST and its segment has no clear; its MIC grant
        # (its own precondition) is present in its own segment.
        assert "clear_app_data" not in untagged_ops
        assert ("grant_permission", "DEV1", "com.test", "android.permission.MIC") in untagged_seg

        # The untagged test's own step occurred BEFORE the only clear in the run.
        flat_ops = [e[0] for e in mock.op_log]
        mic_pos = mock.op_log.index(
            ("grant_permission", "DEV1", "com.test", "android.permission.MIC")
        )
        clear_pos = flat_ops.index("clear_app_data")
        assert mic_pos < clear_pos, "untagged test's step must precede the later clear"

        # The clear belongs to the cold_start test's segment.
        assert cold_seg[0][0] == "clear_app_data"
        assert mock.clear_app_data_calls == [("DEV1", "com.test")]

    def test_untagged_only_suite_never_clears(self, tmp_path: Path) -> None:
        """Negative test (AC2): a suite with NO cold_start tests must produce
        ZERO pm clear calls — every test resets via kill+relaunch only. This
        directly asserts the untagged path is never cleared, independent of any
        cold_start test sharing the suite."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = self._multi_test_suite(
            [
                BVTTest(
                    name="a",
                    steps=[BVTStep(action="grant_permission", target="android.permission.MIC")],
                    tags=["smoke"],
                ),
                BVTTest(
                    name="b",
                    steps=[BVTStep(action="grant_permission", target="android.permission.CAMERA")],
                    tags=["stateful"],
                ),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        assert mock.clear_app_data_calls == []
        # Each test still got its own kill+relaunch reset (order-independent).
        assert mock.kill_calls == [("DEV1", "com.test"), ("DEV1", "com.test")]
        assert "clear_app_data" not in [e[0] for e in mock.op_log]

    def test_cold_start_test_regrants_its_own_permissions(self, tmp_path: Path) -> None:
        """Council hazard: `pm clear` revokes runtime permissions. A cold_start
        test that needs a permission must re-grant it as part of its own flow.
        Here the cold_start test's steps include a grant_permission, and the
        grant happens AFTER the wipe (clear → re-grant in that order) so the
        test arrives at its required precondition without relying on leftover
        grants from any prior test."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = self._multi_test_suite(
            [
                BVTTest(
                    name="camera_after_cold_start",
                    steps=[
                        BVTStep(
                            action="grant_permission",
                            target="android.permission.CAMERA",
                        ),
                        BVTStep(action="launch", target="com.test"),
                    ],
                    tags=["cold_start"],
                ),
            ]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        # The wipe happened (cold_start) ...
        assert mock.clear_app_data_calls == [("DEV1", "com.test")]
        # ... and the test re-granted its own permission as part of its flow.
        assert mock.grant_permission_calls == [("DEV1", "com.test", "android.permission.CAMERA")]
        # Ordering: the clear fired BEFORE the re-grant (clear → re-grant), so
        # the grant restores the precondition the wipe revoked.
        flat_ops = [e[0] for e in mock.op_log]
        assert flat_ops.index("clear_app_data") < flat_ops.index("grant_permission")

    def test_repeated_invocation_is_idempotent_per_test(self, tmp_path: Path) -> None:
        """Running the SAME suite twice yields identical per-test reset sequences:
        a suite is order-independent AND repeatable — no cross-invocation state
        carries over (each run does its own resets from scratch)."""
        suite = self._multi_test_suite(
            [
                BVTTest(
                    name="fresh_consent",
                    steps=[BVTStep(action="launch", target="com.test")],
                    tags=["cold_start"],
                ),
                BVTTest(
                    name="main_screen",
                    steps=[BVTStep(action="launch", target="com.test")],
                    tags=["smoke"],
                ),
            ]
        )

        first = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        BVTRunner(nerve=NerveAdapter(nerve=first), output_dir=tmp_path).run_suite(
            suite, device="DEV1"
        )

        second = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        BVTRunner(nerve=NerveAdapter(nerve=second), output_dir=tmp_path).run_suite(
            suite, device="DEV1"
        )

        # Identical reset footprint across both invocations.
        assert first.clear_app_data_calls == second.clear_app_data_calls == [("DEV1", "com.test")]
        assert (
            first.kill_calls
            == second.kill_calls
            == [
                ("DEV1", "com.test"),
                ("DEV1", "com.test"),
            ]
        )


class TestGrantPermissionStep:
    def test_grant_permission_step(self, tmp_path: Path) -> None:
        """grant_permission step calls nerve with (device, package, permission)."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_suite(
            [BVTStep(action="grant_permission", target="android.permission.CAMERA")]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        assert mock.grant_permission_calls == [("DEV1", "com.test", "android.permission.CAMERA")]

    def test_grant_permission_requires_target(self, tmp_path: Path) -> None:
        """grant_permission with no permission string is a clean step failure."""
        mock = make_mock_nerve_client()
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_suite([BVTStep(action="grant_permission", target="")])

        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        assert mock.grant_permission_calls == []

    def test_grant_permission_nerve_error_fails_step(self, tmp_path: Path) -> None:
        """A grant_permission step whose nerve call returns {"error": ...} without
        success is a failed step (fails pre-fix: passes) (audit F2)."""
        mock = make_mock_nerve_client(
            screen_state_responses=[{"foreground": {"package": "com.test", "activity": ".M"}}],
        )

        def _error_grant(device: str, package: str, permission: str) -> dict:
            mock.grant_permission_calls.append((device, package, permission))
            return {"error": "Permission denial: cannot grant"}

        mock.grant_permission.side_effect = _error_grant
        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        suite = _make_suite(
            [BVTStep(action="grant_permission", target="android.permission.CAMERA")]
        )

        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        test_result = result.tests[0]
        assert not test_result.passed
        assert "Permission denial: cannot grant" in test_result.failure_message
        assert mock.grant_permission_calls == [("DEV1", "com.test", "android.permission.CAMERA")]


class TestScreenshotFilenameSanitization:
    def test_safe_file_stem_has_no_path_separators(self) -> None:
        from linc_crucible.core.bvt.step_handlers import _safe_file_stem

        stem = _safe_file_stem("../../etc/passwd")
        assert "/" not in stem
        assert "\\" not in stem

    def test_safe_file_stem_pin_normal_name(self) -> None:
        from linc_crucible.core.bvt.step_handlers import _safe_file_stem

        assert _safe_file_stem("test_login") == "test_login"
        assert _safe_file_stem("screenshot-1") == "screenshot-1"
        assert _safe_file_stem("step.1") == "step.1"

    def test_screenshot_traversal_written_under_screenshots_dir(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [BVTStep(action="screenshot", target="snap")],
            test_name="../x",
        )
        result = runner.run_suite(suite, device="DEV1")
        assert result.passed
        assert len(result.tests[0].screenshots) == 1
        ss_path = Path(result.tests[0].screenshots[0]).resolve()
        assert ss_path.is_relative_to(runner._screenshots_dir.resolve())
        assert ss_path.exists()

    def test_screenshot_escape_fails_with_harness_error(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)
        step = BVTStep(action="screenshot", target="snap")
        with patch(
            "linc_crucible.core.bvt.step_handlers._safe_file_stem",
            return_value="../../escape",
        ):
            result = runner._do_screenshot(step, "DEV1", "com.test", "test1")
            assert not result.passed
            assert result.harness_error is True


# --- Suite wall-clock budget (audit F11) ---


def _budget_suite(n: int) -> BVTSuite:
    """An n-test suite; the steps never run (``_run_test`` is faked below)."""
    return BVTSuite(
        package="com.test",
        tests=[
            BVTTest(name=f"t{i}", steps=[BVTStep(action="wait", target="1ms")])
            for i in range(1, n + 1)
        ],
    )


def _budget_harness(tmp_path: Path, per_test_s: float) -> tuple[BVTRunner, list[str], object]:
    """A runner whose ``_run_test`` advances a fake monotonic clock ``per_test_s``.

    Returns (runner, names-of-executed-tests, clock). The clock only moves when
    a test executes, so the budget check before each test sees exactly the
    elapsed suite time the acceptance describes.
    """
    now = [0.0]
    ran: list[str] = []

    def _clock() -> float:
        return now[0]

    def _fake_run_test(test, device, package, app_path=None) -> BVTTestResult:
        ran.append(test.name)
        now[0] += per_test_s
        return BVTTestResult(
            name=test.name, passed=True, package=package, steps_run=1, steps_passed=1
        )

    runner = BVTRunner(nerve=NerveAdapter(nerve=make_mock_nerve_client()), output_dir=tmp_path)
    runner._run_test = _fake_run_test  # type: ignore[method-assign]
    return runner, ran, _clock


class TestSuiteBudget:
    def test_budget_exhausted_skips_remaining_tests(self, tmp_path: Path) -> None:
        """700s/test with a 1000s budget: t1+t2 run, t3 is a harness-reasoned skip."""
        runner, ran, clock = _budget_harness(tmp_path, per_test_s=700.0)

        result = runner.run_suite(_budget_suite(3), device="DEV1", budget_s=1000, clock=clock)

        assert ran == ["t1", "t2"]
        assert [t.name for t in result.tests] == ["t1", "t2", "t3"]
        skipped = result.tests[2]
        assert skipped.skipped is True
        assert skipped.passed is False
        assert skipped.package == "com.test"
        assert skipped.skip_reason == "HARNESS: suite budget 1000s exhausted after 1400s"
        assert result.total_passed == 2
        assert result.total_skipped == 1
        # Two green tests do NOT make a green suite when the budget cut it short.
        assert result.passed is False
        assert result.budget_exhausted is True

    def test_budget_none_never_skips(self, tmp_path: Path) -> None:
        """Pin: budget_s=None disables the budget — every test runs, suite stays green."""
        runner, ran, clock = _budget_harness(tmp_path, per_test_s=700.0)

        result = runner.run_suite(_budget_suite(3), device="DEV1", budget_s=None, clock=clock)

        assert ran == ["t1", "t2", "t3"]
        assert result.total_skipped == 0
        assert result.passed is True
        assert result.budget_exhausted is False

    def test_default_budget_is_module_constant(self, tmp_path: Path) -> None:
        """No budget_s → _SUITE_BUDGET_DEFAULT_S (1800s): the 4th 700s test is skipped."""
        assert _SUITE_BUDGET_DEFAULT_S == 1800.0
        runner, ran, clock = _budget_harness(tmp_path, per_test_s=700.0)

        result = runner.run_suite(_budget_suite(4), device="DEV1", clock=clock)

        assert ran == ["t1", "t2", "t3"]
        assert result.tests[3].skip_reason == ("HARNESS: suite budget 1800s exhausted after 2100s")
        assert result.passed is False

    def test_every_remaining_test_is_skipped_once_exhausted(self, tmp_path: Path) -> None:
        """Once the budget is spent, ALL remaining tests are skipped, not just the next."""
        runner, ran, clock = _budget_harness(tmp_path, per_test_s=700.0)

        result = runner.run_suite(_budget_suite(5), device="DEV1", budget_s=500, clock=clock)

        assert ran == ["t1"]
        assert [t.skipped for t in result.tests] == [False, True, True, True, True]
        assert all(
            t.skip_reason.startswith("HARNESS: suite budget 500s") for t in result.tests[1:]
        )

    def test_within_budget_suite_unchanged(self, tmp_path: Path) -> None:
        """A suite that finishes inside its budget is green and carries no skips."""
        runner, ran, clock = _budget_harness(tmp_path, per_test_s=1.0)

        result = runner.run_suite(_budget_suite(3), device="DEV1", budget_s=1000, clock=clock)

        assert ran == ["t1", "t2", "t3"]
        assert result.passed is True
        assert result.budget_exhausted is False


class TestBudgetExhausted:
    def test_none_budget_is_never_exhausted(self) -> None:
        assert _budget_exhausted(0.0, None, 1e9) is False

    def test_under_budget(self) -> None:
        assert _budget_exhausted(10.0, 100.0, 109.9) is False

    def test_exactly_at_budget_is_exhausted(self) -> None:
        assert _budget_exhausted(10.0, 100.0, 110.0) is True

    def test_over_budget(self) -> None:
        assert _budget_exhausted(10.0, 100.0, 500.0) is True
