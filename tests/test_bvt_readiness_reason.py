"""Tests for BVT device-readiness skip-reason formatting (reactor 01e1622e).

Pins that a not-ready device's skip reason always carries nerve's real
error/reason text and the target serial — never the literal "None" that
resulted from blindly formatting a missing ``reason`` key.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from linc_crucible.core.bvt.runner import BVTRunner, _readiness_reason
from linc_crucible.core.bvt.schema import BVTStep, BVTSuite, BVTTest


def _make_suite() -> BVTSuite:
    return BVTSuite(
        package="com.test",
        tests=[
            BVTTest(name="t1", steps=[BVTStep(action="assert_element", target="level_screen")]),
            BVTTest(name="t2", steps=[BVTStep(action="assert_element", target="level_screen")]),
        ],
    )


class TestRunSuiteReadinessReason:
    def test_nerve_not_found_error_surfaces_verbatim(self, tmp_path: Path) -> None:
        mock_adapter = MagicMock()
        mock_adapter.ensure_ready.return_value = {
            "error": "Device '9397270F' not found. Run nerve_discover to list devices."
        }
        runner = BVTRunner(nerve=mock_adapter, output_dir=tmp_path)
        suite = _make_suite()

        result = runner.run_suite(suite, device="9397270F")

        reason = result.suite_skip_reason
        assert reason is not None
        assert "Device '9397270F' not found" in reason
        assert "(device 9397270F)" in reason
        assert "None" not in reason
        assert reason.startswith("device not ready: ")
        for test in result.tests:
            assert test.skip_reason == reason

    def test_reason_wins_over_error_when_both_present(self, tmp_path: Path) -> None:
        mock_adapter = MagicMock()
        mock_adapter.ensure_ready.return_value = {
            "ready": False,
            "reason": "asleep (wake failed)",
            "error": "some other error text",
        }
        runner = BVTRunner(nerve=mock_adapter, output_dir=tmp_path)
        suite = _make_suite()

        result = runner.run_suite(suite, device="DEV1")

        assert result.suite_skip_reason is not None
        assert result.suite_skip_reason.startswith("device not ready: asleep (wake failed)")

    def test_empty_readiness_dict_falls_back_without_none(self, tmp_path: Path) -> None:
        mock_adapter = MagicMock()
        mock_adapter.ensure_ready.return_value = {}
        runner = BVTRunner(nerve=mock_adapter, output_dir=tmp_path)
        suite = _make_suite()

        result = runner.run_suite(suite, device="DEV2")

        assert result.suite_skip_reason is not None
        assert "nerve_ensure_ready returned no reason" in result.suite_skip_reason
        assert "DEV2" in result.suite_skip_reason
        assert "None" not in result.suite_skip_reason

    def test_none_reason_and_empty_error_falls_back_without_none(self, tmp_path: Path) -> None:
        mock_adapter = MagicMock()
        mock_adapter.ensure_ready.return_value = {"ready": False, "reason": None, "error": ""}
        runner = BVTRunner(nerve=mock_adapter, output_dir=tmp_path)
        suite = _make_suite()

        result = runner.run_suite(suite, device="DEV3")

        assert result.suite_skip_reason is not None
        assert "nerve_ensure_ready returned no reason" in result.suite_skip_reason
        assert "DEV3" in result.suite_skip_reason
        assert "None" not in result.suite_skip_reason


class TestReadinessReasonHelper:
    def test_reason_key_wins(self) -> None:
        reason = _readiness_reason({"reason": "asleep (wake failed)", "error": "x"}, "DEV1")
        assert reason == "device not ready: asleep (wake failed) (device DEV1)"

    def test_falls_back_to_error(self) -> None:
        reason = _readiness_reason(
            {"error": "Device '9397270F' not found. Run nerve_discover to list devices."},
            "9397270F",
        )
        assert reason == (
            "device not ready: Device '9397270F' not found. "
            "Run nerve_discover to list devices. (device 9397270F)"
        )

    def test_falls_back_to_default_message(self) -> None:
        assert _readiness_reason({}, "DEV2") == (
            "device not ready: nerve_ensure_ready returned no reason (device DEV2)"
        )
        assert _readiness_reason({"reason": None, "error": ""}, "DEV3") == (
            "device not ready: nerve_ensure_ready returned no reason (device DEV3)"
        )
