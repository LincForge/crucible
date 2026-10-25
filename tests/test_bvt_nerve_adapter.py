"""Smoke tests for the refactored NerveAdapter — confirms each method
delegates to NerveClient with the expected args + handles empty-return
hardening for the Hotz silent-failure risk."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog

from linc_crucible.core.bvt.nerve_adapter import (
    NerveAdapter,
    _normalize_nerve_result,
    _parse_bounds_center,
)
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


class TestBoundsParse:
    def test_parses_standard_bounds(self) -> None:
        assert _parse_bounds_center("[10,20][110,120]") == (60, 70)

    def test_returns_none_on_bad_input(self) -> None:
        assert _parse_bounds_center("not bounds") is None

    def test_parses_unit_bounds_and_rejects_trailing_junk(self) -> None:
        from linc_crucible.core.bvt.nerve_adapter import _BOUNDS_RE

        assert _BOUNDS_RE.pattern == r"\A\[(\d+),(\d+)\]\[(\d+),(\d+)\]\Z"
        assert _parse_bounds_center("[0,0][1,1]") == (0, 0)
        assert _parse_bounds_center("[0,0][1,1]\n") is None
        assert _parse_bounds_center("[0,0][1,1]extra") is None


class TestAdapterDelegation:
    def test_discover_returns_devices(self) -> None:
        adapter = NerveAdapter(nerve=make_mock_nerve_client())
        devices = adapter.discover()
        assert len(devices) == 1
        assert devices[0]["serial"] == "FAKE123"

    def test_launch_returns_success(self) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        result = adapter.launch("DEV1", "com.test")
        assert result["success"] is True
        assert mock.launch_calls == [("DEV1", "com.test")]

    def test_elements_is_the_lenient_passthrough(self) -> None:
        # elements() stays the lenient read for callers outside the strict
        # element-resolving verbs (audit F7 moved those to elements_strict).
        tree = [{"id": "a", "bounds": "[0,0][1,1]"}]
        mock = make_mock_nerve_client(elements_response=tree)
        assert NerveAdapter(nerve=mock).elements("DEV1") == tree
        mock.elements.assert_called_once_with(device="DEV1")

    def test_kill_returns_success(self) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        result = adapter.kill("DEV1", "com.test")
        assert result["success"] is True
        assert mock.kill_calls == [("DEV1", "com.test")]

    def test_tap_returns_success(self) -> None:
        # Bare adapter.tap defaults to duration_ms=0 → raw nerve_tap (legacy
        # literal-coordinate path). The duration-primary fix is opt-in and is
        # exercised by tap_by_id — see TestTapDurationPrimary.
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        result = adapter.tap("DEV1", 100, 200)
        assert result["success"] is True
        assert mock.tap_calls == [("DEV1", 100, 200)]

    def test_navigate_returns_success(self) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        result = adapter.navigate("DEV1", "app://settings")
        assert result["success"] is True
        assert mock.navigate_calls == [("DEV1", "app://settings")]

    def test_clear_app_data_returns_success(self) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        result = adapter.clear_app_data("DEV1", "com.test")
        assert result["success"] is True
        assert mock.clear_app_data_calls == [("DEV1", "com.test")]

    def test_clear_app_data_passes_needs_reinstall_true_through(self) -> None:
        """nerve's needs_reinstall flag (iOS-sim uninstall-based clear) must
        survive the adapter untouched so the runner can act on it."""
        mock = make_mock_nerve_client()

        def _ios_clear(device: str, package: str) -> dict:
            mock.clear_app_data_calls.append((device, package))
            return {"success": True, "needs_reinstall": True}

        mock.clear_app_data.side_effect = _ios_clear
        adapter = NerveAdapter(nerve=mock)
        result = adapter.clear_app_data("SIM1", "com.test")
        assert result["success"] is True
        assert result["needs_reinstall"] is True

    def test_clear_app_data_passes_needs_reinstall_false_through(self) -> None:
        mock = make_mock_nerve_client()

        def _android_clear(device: str, package: str) -> dict:
            mock.clear_app_data_calls.append((device, package))
            return {"success": True, "needs_reinstall": False}

        mock.clear_app_data.side_effect = _android_clear
        adapter = NerveAdapter(nerve=mock)
        result = adapter.clear_app_data("DEV1", "com.test")
        assert result["success"] is True
        assert result["needs_reinstall"] is False

    def test_clear_app_data_missing_needs_reinstall_stays_missing(self) -> None:
        """Older nerve daemon (key absent) — adapter must not invent the key."""
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        result = adapter.clear_app_data("DEV1", "com.test")
        assert result["success"] is True
        assert "needs_reinstall" not in result

    def test_grant_permission_returns_success(self) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        result = adapter.grant_permission("DEV1", "com.test", "android.permission.CAMERA")
        assert result["success"] is True
        assert mock.grant_permission_calls == [("DEV1", "com.test", "android.permission.CAMERA")]

    def test_clear_app_data_error_returns_failure_dict(self) -> None:
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        nerve.clear_app_data.side_effect = RuntimeError("daemon down")
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.clear_app_data("DEV1", "com.test")
        assert result["success"] is False
        assert "daemon down" in result["error"]

    def test_screenshot_returns_png(self) -> None:
        adapter = NerveAdapter(nerve=make_mock_nerve_client())
        data = adapter.screenshot("DEV1")
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_metrics_returns_none_by_default(self) -> None:
        adapter = NerveAdapter(nerve=make_mock_nerve_client())
        assert adapter.metrics("DEV1", "com.test") is None


class TestEmptyReturnHardening:
    """Hotz finding — NerveClient may return {} where forge HTTPNerveClient
    raised. Adapter must convert empty returns to explicit failure dicts."""

    def test_launch_returning_empty_dict_treated_as_failure(self) -> None:
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        nerve.launch.return_value = {}  # NerveClient sometimes returns this on partial errors
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.launch("DEV1", "com.test")
        assert result["success"] is False
        assert "empty" in result["error"]

    def test_kill_returning_empty_dict_treated_as_failure(self) -> None:
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        nerve.kill.return_value = {}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.kill("DEV1", "com.test")
        assert result["success"] is False

    def test_navigate_returning_empty_dict_treated_as_failure(self) -> None:
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        nerve.navigate.return_value = {}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.navigate("DEV1", "app://x")
        assert result["success"] is False


class TestNewVerbAdapters:
    """type / long_press_by_id / density (BVT verb-gap circuit)."""

    def _client(self):
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        return MagicMock(spec=NerveClient)

    def test_type_returns_success(self) -> None:
        nerve = self._client()
        nerve.type_text.return_value = {"success": True, "action": "type"}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.type("DEV1", "06")
        assert result["success"] is True
        nerve.type_text.assert_called_once_with(device="DEV1", text="06")

    def test_type_catches_exception(self) -> None:
        nerve = self._client()
        nerve.type_text.side_effect = ConnectionError("daemon down")
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.type("DEV1", "06")
        assert result["success"] is False
        assert "daemon down" in result["error"]

    def test_long_press_by_id_resolves_center_and_duration(self) -> None:
        nerve = self._client()
        nerve.elements.return_value = nerve.elements_strict.return_value = [
            {"id": "info_button", "bounds": "[0,0][100,100]"}
        ]
        nerve.long_press.return_value = {"success": True}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.long_press_by_id("DEV1", "info_button", duration_ms=1000)
        assert result["success"] is True
        nerve.long_press.assert_called_once_with("DEV1", 50, 50, 1000)

    def test_long_press_by_id_not_found(self) -> None:
        nerve = self._client()
        nerve.elements.return_value = nerve.elements_strict.return_value = []
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.long_press_by_id("DEV1", "info_button")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_density_passthrough(self) -> None:
        nerve = self._client()
        nerve.density.return_value = {"density_dpi": 420, "density_factor": 2.625}
        adapter = NerveAdapter(nerve=nerve)
        assert adapter.density("DEV1") == {"density_dpi": 420, "density_factor": 2.625}

    def test_density_returns_empty_on_exception(self) -> None:
        nerve = self._client()
        nerve.density.side_effect = ConnectionError("down")
        adapter = NerveAdapter(nerve=nerve)
        assert adapter.density("DEV1") == {}

    def test_key_sends_keyevent_via_nerve(self) -> None:
        nerve = self._client()
        nerve._call_tool.return_value = {"success": True, "action": "key"}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.key("DEV1", "ENTER")
        assert result["success"] is True
        nerve._call_tool.assert_called_once_with("nerve_key", {"device": "DEV1", "key": "ENTER"})

    def test_key_defaults_to_enter(self) -> None:
        nerve = self._client()
        nerve._call_tool.return_value = {"success": True}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.key("DEV1")
        assert result["success"] is True
        nerve._call_tool.assert_called_once_with("nerve_key", {"device": "DEV1", "key": "ENTER"})

    def test_key_catches_exception(self) -> None:
        nerve = self._client()
        nerve._call_tool.side_effect = ConnectionError("daemon down")
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.key("DEV1", "ENTER")
        assert result["success"] is False
        assert "daemon down" in result["error"]

    def test_key_empty_result_is_failure(self) -> None:
        nerve = self._client()
        nerve._call_tool.return_value = {}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.key("DEV1", "ENTER")
        assert result["success"] is False


class TestTapDurationPrimary:
    """AC1 — tap_by_id uses a configurable down/up duration defaulting to ~50ms,
    matching ``adb input tap``. A near-zero-duration synthetic tap is discarded
    by Compose ``combinedClickable`` (council carmack/beck/humble, 2026-06-03),
    so the duration is synthesized via ``nerve_long_press``.
    """

    def _client(self):
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        return MagicMock(spec=NerveClient)

    def test_tap_with_duration_uses_long_press(self) -> None:
        # _tap_with_duration synthesizes the down→up gap via nerve_long_press.
        nerve = self._client()
        nerve.long_press.return_value = {"success": True}
        adapter = NerveAdapter(nerve=nerve)
        adapter._tap_with_duration("DEV1", 10, 20, 120)
        nerve.long_press.assert_called_once_with("DEV1", 10, 20, 120)
        nerve.tap.assert_not_called()

    def test_tap_with_zero_duration_falls_back_to_raw_tap(self) -> None:
        nerve = self._client()
        nerve.tap.return_value = {"success": True, "action": "tap"}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter._tap_with_duration("DEV1", 5, 6, 0)
        assert result["success"] is True
        nerve.tap.assert_called_once_with(device="DEV1", x=5, y=6)
        nerve.long_press.assert_not_called()

    def test_bare_tap_is_raw_zero_duration(self) -> None:
        # Bare adapter.tap stays on the legacy raw nerve_tap; literal-coordinate
        # callers (tap_by_text, tap_xy) are unaffected by the duration fix.
        nerve = self._client()
        nerve.tap.return_value = {"success": True, "action": "tap"}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.tap("DEV1", 5, 6)
        assert result["success"] is True
        nerve.tap.assert_called_once_with(device="DEV1", x=5, y=6)
        nerve.long_press.assert_not_called()

    def test_tap_by_id_uses_duration_by_default(self) -> None:
        nerve = self._client()
        nerve.elements.return_value = nerve.elements_strict.return_value = [
            {"id": "info_button", "bounds": "[0,0][100,100]"}
        ]
        nerve.long_press.return_value = {"success": True}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.tap_by_id("DEV1", "info_button")
        assert result["success"] is True
        # center of [0,0][100,100] is (50, 50); default 50ms duration.
        nerve.long_press.assert_called_once_with("DEV1", 50, 50, 50)


class TestTapByIdVerifyAndRetry:
    """AC2/AC3/AC4 — verify-and-retry is a SECONDARY backstop after the
    duration fix. It re-dumps after a tap and re-taps when foreground +
    element-set are unchanged, up to a max; a successful nav short-circuits.
    """

    def _client(self):
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        return MagicMock(spec=NerveClient)

    def _info_button(self):
        return [{"id": "info_button", "bounds": "[0,0][100,100]"}]

    def test_retries_when_state_unchanged(self) -> None:
        # AC2 — foreground + element-set never change → re-tap up to max_retries.
        nerve = self._client()
        nerve.elements.return_value = nerve.elements_strict.return_value = self._info_button()
        nerve.long_press.return_value = {"success": True}
        # screen_state is stable (same foreground every dump) → considered "unchanged".
        nerve.screen_state.return_value = {"foreground": {"package": "com.test", "activity": ".A"}}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.tap_by_id("DEV1", "info_button", max_retries=2)
        assert result["success"] is True
        # 1 initial tap + 2 retries = 3 long_press calls.
        assert nerve.long_press.call_count == 3

    def test_successful_nav_short_circuits_retry(self) -> None:
        # AC3 — when the post-tap dump shows a CHANGED foreground (nav happened),
        # the retry loop stops immediately. Only the initial tap fires.
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        nerve.elements.return_value = nerve.elements_strict.return_value = self._info_button()
        nerve.long_press.return_value = {"success": True}
        # First dump: screen A. After the tap: screen B (nav succeeded).
        nerve.screen_state.side_effect = [
            {"foreground": {"package": "com.test", "activity": ".A"}},
            {"foreground": {"package": "com.test", "activity": ".B"}},
        ]
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.tap_by_id("DEV1", "info_button", max_retries=3)
        assert result["success"] is True
        # Only the initial tap — nav detected, no retries.
        assert nerve.long_press.call_count == 1

    def test_stale_frame_regression_retries_then_succeeds(self) -> None:
        # AC4 — mocked-nerve stale-frame case: the first post-tap dump is a stale
        # frame (foreground unchanged), the retry's dump shows nav landed.
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        nerve.elements.return_value = nerve.elements_strict.return_value = self._info_button()
        nerve.long_press.return_value = {"success": True}
        nerve.screen_state.side_effect = [
            {"foreground": {"package": "com.test", "activity": ".A"}},  # pre-tap baseline
            {"foreground": {"package": "com.test", "activity": ".A"}},  # stale frame → retry
            {"foreground": {"package": "com.test", "activity": ".B"}},  # nav landed
        ]
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.tap_by_id("DEV1", "info_button", max_retries=3)
        assert result["success"] is True
        # initial tap + exactly one retry (then nav detected).
        assert nerve.long_press.call_count == 2

    def test_no_retry_when_duration_zero(self) -> None:
        # Retry must NOT fire on a zero-duration tap (re-tapping a 0ms tap just
        # re-discards it). duration_ms=0 disables the backstop entirely.
        nerve = self._client()
        nerve.elements.return_value = nerve.elements_strict.return_value = self._info_button()
        nerve.tap.return_value = {"success": True}
        nerve.screen_state.return_value = {"foreground": {"package": "com.test", "activity": ".A"}}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.tap_by_id("DEV1", "info_button", duration_ms=0, max_retries=3)
        assert result["success"] is True
        nerve.tap.assert_called_once_with(device="DEV1", x=50, y=50)
        nerve.long_press.assert_not_called()

    def test_retry_disabled_by_default(self) -> None:
        # Backwards-compat: default max_retries=0 means a single tap, no dumps —
        # callers that don't opt in keep the cheap one-shot behavior.
        nerve = self._client()
        nerve.elements.return_value = nerve.elements_strict.return_value = self._info_button()
        nerve.long_press.return_value = {"success": True}
        adapter = NerveAdapter(nerve=nerve)
        result = adapter.tap_by_id("DEV1", "info_button")
        assert result["success"] is True
        assert nerve.long_press.call_count == 1
        nerve.screen_state.assert_not_called()


class TestEnsureReady:
    def test_ensure_ready_passthrough(self) -> None:
        mock = make_mock_nerve_client()
        mock._call_tool.return_value = {"ready": True, "reason": "ready"}
        adapter = NerveAdapter(nerve=mock)
        assert adapter.ensure_ready("S")["ready"] is True
        mock._call_tool.assert_called_once_with("nerve_ensure_ready", {"device": "S"})

    def test_ensure_ready_transport_error_is_not_ready(self) -> None:
        mock = make_mock_nerve_client()
        mock._call_tool.side_effect = RuntimeError("boom")
        adapter = NerveAdapter(nerve=mock)
        out = adapter.ensure_ready("S")
        assert out["ready"] is False and "boom" in out["reason"]


class TestNormalizeNerveResult:
    """Direct unit tests per branch of _normalize_nerve_result."""

    def test_none_returns_empty_failure(self) -> None:
        res = _normalize_nerve_result(None, "launch")
        assert res == {"success": False, "error": "nerve launch returned empty"}

    def test_empty_dict_returns_empty_failure(self) -> None:
        res = _normalize_nerve_result({}, "kill")
        assert res == {"success": False, "error": "nerve kill returned empty"}

    def test_success_key_present_unchanged_true(self) -> None:
        payload = {"success": True, "data": 123}
        res = _normalize_nerve_result(payload, "launch")
        assert res == payload
        assert res is payload

    def test_success_key_present_unchanged_false(self) -> None:
        payload = {"success": False, "error": "boom"}
        res = _normalize_nerve_result(payload, "launch")
        assert res == payload
        assert res is payload

    def test_success_key_present_with_error_unchanged(self) -> None:
        # nerve's word wins: explicit success: True with error remains unchanged
        payload = {"success": True, "error": "warning"}
        res = _normalize_nerve_result(payload, "type")
        assert res == payload
        assert res is payload

    def test_error_key_without_success_stamps_success_false(self) -> None:
        payload = {"error": "Kill failed: device offline", "detail": 42}
        res = _normalize_nerve_result(payload, "kill")
        assert res["success"] is False
        assert res["error"] == "Kill failed: device offline"
        assert res["detail"] == 42

    def test_elements_error_key_without_success_stamps_success_false(self) -> None:
        payload = {"elements_error": "daemon unmapped", "code": 500}
        res = _normalize_nerve_result(payload, "launch")
        assert res["success"] is False
        assert res["elements_error"] == "daemon unmapped"
        assert res["code"] == 500

    def test_other_payload_stamps_success_true(self) -> None:
        payload = {"device": "DEV1", "action": "typed"}
        res = _normalize_nerve_result(payload, "type")
        assert res["success"] is True
        assert res["device"] == "DEV1"
        assert res["action"] == "typed"


class TestNerveAdapterErrorHandling:
    """audit F2: NerveAdapter treats nerve's {"error": ...} payload as success False
    on every verb (launch, kill, clear_app_data, grant_permission, type, key, navigate)."""

    @pytest.mark.parametrize(
        ("verb", "caller", "mock_method"),
        [
            ("launch", lambda a: a.launch("DEV1", "com.test"), "launch"),
            ("kill", lambda a: a.kill("DEV1", "com.test"), "kill"),
            ("clear_app_data", lambda a: a.clear_app_data("DEV1", "com.test"), "clear_app_data"),
            (
                "grant_permission",
                lambda a: a.grant_permission("DEV1", "com.test", "android.permission.CAMERA"),
                "grant_permission",
            ),
            ("type", lambda a: a.type("DEV1", "hello"), "type_text"),
            ("key", lambda a: a.key("DEV1", "ENTER"), "_call_tool"),
            ("navigate", lambda a: a.navigate("DEV1", "app://settings"), "navigate"),
        ],
    )
    def test_verb_returns_error_yields_success_false_with_error_preserved(
        self, verb: str, caller, mock_method: str
    ) -> None:
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        getattr(nerve, mock_method).return_value = {"error": "Kill failed: device offline"}
        adapter = NerveAdapter(nerve=nerve)
        result = caller(adapter)
        assert result["success"] is False
        assert result["error"] == "Kill failed: device offline"

    @pytest.mark.parametrize(
        ("verb", "caller", "mock_method"),
        [
            ("launch", lambda a: a.launch("DEV1", "com.test"), "launch"),
            ("kill", lambda a: a.kill("DEV1", "com.test"), "kill"),
            ("clear_app_data", lambda a: a.clear_app_data("DEV1", "com.test"), "clear_app_data"),
            (
                "grant_permission",
                lambda a: a.grant_permission("DEV1", "com.test", "android.permission.CAMERA"),
                "grant_permission",
            ),
            ("type", lambda a: a.type("DEV1", "hello"), "type_text"),
            ("key", lambda a: a.key("DEV1", "ENTER"), "_call_tool"),
            ("navigate", lambda a: a.navigate("DEV1", "app://settings"), "navigate"),
        ],
    )
    def test_verb_returns_error_with_explicit_success_true_remains_true(
        self, verb: str, caller, mock_method: str
    ) -> None:
        """{"error": ...} with an explicit success: True is left unchanged
        (nerve's word wins; pin)."""
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        getattr(nerve, mock_method).return_value = {
            "error": "Non-fatal warning from daemon",
            "success": True,
        }
        adapter = NerveAdapter(nerve=nerve)
        result = caller(adapter)
        assert result["success"] is True
        assert result["error"] == "Non-fatal warning from daemon"

    @pytest.mark.parametrize(
        ("verb", "caller", "mock_method"),
        [
            ("clear_app_data", lambda a: a.clear_app_data("DEV1", "com.test"), "clear_app_data"),
            (
                "grant_permission",
                lambda a: a.grant_permission("DEV1", "com.test", "android.permission.CAMERA"),
                "grant_permission",
            ),
            ("type", lambda a: a.type("DEV1", "hello"), "type_text"),
            ("key", lambda a: a.key("DEV1", "ENTER"), "_call_tool"),
        ],
    )
    def test_verb_empty_and_exception_yields_success_false(
        self, verb: str, caller, mock_method: str
    ) -> None:
        """{} and an exception still yield success False (pin, from TestEmptyReturnHardening)."""
        from unittest.mock import MagicMock

        from linc_crucible.core.nerve_client import NerveClient

        # Empty dict
        nerve = MagicMock(spec=NerveClient)
        getattr(nerve, mock_method).return_value = {}
        adapter = NerveAdapter(nerve=nerve)
        res_empty = caller(adapter)
        assert res_empty["success"] is False
        assert "empty" in res_empty["error"]

        # Exception
        nerve_exc = MagicMock(spec=NerveClient)
        getattr(nerve_exc, mock_method).side_effect = RuntimeError("transport crash")
        adapter_exc = NerveAdapter(nerve=nerve_exc)
        res_exc = caller(adapter_exc)
        assert res_exc["success"] is False
        assert "transport crash" in res_exc["error"]


def _tree_client(elements: list[dict] | None = None, unavailable: str | None = None):
    """MagicMock(spec=NerveClient) whose two tree reads agree, like the real client.

    ``unavailable`` set -> the STRICT read raises UiTreeUnavailableError while
    the lenient ``elements()`` swallows the same fault to ``[]`` (NerveClient's
    real lenient contract). Otherwise both return ``elements``.
    """
    from unittest.mock import MagicMock

    from linc_crucible.core.nerve_client import NerveClient, UiTreeUnavailableError

    nerve = MagicMock(spec=NerveClient)
    tree = elements or []
    nerve.elements.return_value = [] if unavailable else tree
    if unavailable:
        nerve.elements_strict.side_effect = UiTreeUnavailableError(unavailable)
    else:
        nerve.elements_strict.return_value = tree
    nerve.long_press.return_value = {"success": True}
    nerve.tap.return_value = {"success": True}
    nerve.swipe.return_value = {"success": True}
    nerve.screen_state.return_value = {"screen_size": {"width": 1080, "height": 2280}}
    return nerve


_TREE_VERBS = [
    ("tap_by_id", lambda a: a.tap_by_id("DEV1", "info_button")),
    ("tap_by_text", lambda a: a.tap_by_text("DEV1", "Info")),
    ("long_press_by_id", lambda a: a.long_press_by_id("DEV1", "info_button")),
    ("scroll_to_element", lambda a: a.scroll_to_element("DEV1", "info_button")),
]


class TestTreeVerbsStrictRead:
    """audit F7: element-resolving verbs read the tree STRICTLY. An unreadable
    tree is a HARNESS fault, never a product "No element ... found"."""

    @pytest.mark.parametrize(("verb", "caller"), _TREE_VERBS)
    def test_unavailable_tree_is_harness_error(self, verb: str, caller) -> None:
        nerve = _tree_client(unavailable="could not get idle state")
        result = caller(NerveAdapter(nerve=nerve))
        assert result["success"] is False
        assert result["harness_error"] is True
        assert result["error"].startswith("HARNESS:")
        assert "could not get idle state" in result["error"]
        nerve.long_press.assert_not_called()
        nerve.tap.assert_not_called()
        nerve.swipe.assert_not_called()

    def test_scroll_unavailable_tree_reports_not_found(self) -> None:
        nerve = _tree_client(unavailable="dump failed")
        result = NerveAdapter(nerve=nerve).scroll_to_element("DEV1", "info_button")
        assert result["found"] is False
        assert result["swipes"] == 0

    @pytest.mark.parametrize(("verb", "caller"), _TREE_VERBS[:3])
    def test_read_tree_without_element_is_product_fault(self, verb: str, caller) -> None:
        """Pin: a tree that IS read but lacks the element stays a product fault."""
        nerve = _tree_client(elements=[{"id": "other", "text": "Other", "bounds": "[0,0][9,9]"}])
        result = caller(NerveAdapter(nerve=nerve))
        assert result["success"] is False
        assert not result.get("harness_error", False)
        assert not result["error"].startswith("HARNESS:")

    def test_scroll_read_tree_without_element_is_not_harness(self) -> None:
        """Pin: scroll_to_element over a readable tree gives up on swipes, not harness."""
        nerve = _tree_client(elements=[{"id": "other", "bounds": "[0,0][9,9]"}])
        result = NerveAdapter(nerve=nerve).scroll_to_element("DEV1", "info_button", max_swipes=1)
        assert result == {"found": False, "swipes": 1}

    def test_unparseable_bounds_still_reported(self) -> None:
        nerve = _tree_client(elements=[{"id": "info_button", "bounds": "garbage"}])
        result = NerveAdapter(nerve=nerve).long_press_by_id("DEV1", "info_button")
        assert result["success"] is False
        assert "Cannot parse bounds" in result["error"]
        assert not result.get("harness_error", False)
