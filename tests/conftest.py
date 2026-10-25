"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from linc_crucible.core.nerve_client import NerveClient


@pytest.fixture(autouse=True)
def _isolate_linc_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect LINC_HOME to a per-test tmp dir so nothing touches real ~/.linc.

    Two failure modes this guards against:
      1. Sentinel-heartbeat's launchd plist sets LINC_HOME process-wide, which
         leaks into pytest subprocesses and masks default-path failures.
      2. Any tool exercised through ``_observed_tool`` calls ``log_mcp_call``
         with the *default* path. The old fixture only ``delenv``'d LINC_HOME,
         so those records leaked into the production ``~/.linc/mcp-calls.jsonl``
         (391 junk fixture records found in the telemetry audit, imp 01606251).
    Pointing LINC_HOME at tmp_path fixes both: launchd's value is overridden AND
    the default path resolves under tmp. Tests that assert specific LINC_HOME
    behaviour set their own value locally (later setenv wins).
    """
    monkeypatch.setenv("LINC_HOME", str(tmp_path / "linc_home"))


def make_mock_nerve_client(
    elements_response: list[dict] | None = None,
    discover_devices: list[dict] | None = None,
    screenshot_bytes: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
    metrics_response: dict | None = None,
    screen_state_responses: list | None = None,
) -> MagicMock:
    """Build a realistic-shape mock of NerveClient.

    This exists because a naive `MagicMock(spec=NerveClient)` returns
    `MagicMock` objects for every method call — and code that does
    `result["elements"]` or `result.get("success")` against those mocks
    silently passes when production would have failed.

    See Hotz finding in spec 0351fdd7: NerveClient swallows errors and
    returns {} on some paths, so mocks must return realistic dict shapes
    per call to surface KeyError/TypeError problems at test time.

    Tracks calls for assertion: mock.launch_calls, mock.tap_calls,
    mock.kill_calls, mock.navigate_calls (lists of arg tuples).
    """
    mock = MagicMock(spec=NerveClient)
    mock.available = True
    mock.launch_calls = []
    mock.tap_calls = []
    mock.kill_calls = []
    mock.navigate_calls = []
    mock.install_calls = []
    mock.key_calls = []
    mock.clear_app_data_calls = []
    mock.grant_permission_calls = []
    # Unified, ordered event log of every nerve op in real invocation order.
    # Each entry is a tuple (op_name, *args). Lets a test assert the *order*
    # of operations across reset + step boundaries (e.g. that a cold_start
    # test's `pm clear` fires before that test's own steps), not just the
    # aggregate per-op footprint. See cold-start order-independence tests.
    mock.op_log: list[tuple] = []

    elements = elements_response or []
    devices = discover_devices or [
        {"serial": "FAKE123", "platform": "android", "model": "Pixel 8 Pro"}
    ]

    def _discover() -> list[dict]:
        return devices

    def _launch(device: str, package: str) -> dict:
        mock.launch_calls.append((device, package))
        mock.op_log.append(("launch", device, package))
        return {"success": True, "action": "launch", "package": package}

    def _kill(device: str, package: str) -> dict:
        mock.kill_calls.append((device, package))
        mock.op_log.append(("kill", device, package))
        return {"success": True, "action": "kill"}

    def _clear_app_data(device: str, package: str) -> dict:
        mock.clear_app_data_calls.append((device, package))
        mock.op_log.append(("clear_app_data", device, package))
        return {"success": True, "action": "clear_app_data", "package": package}

    def _grant_permission(device: str, package: str, permission: str) -> dict:
        mock.grant_permission_calls.append((device, package, permission))
        mock.op_log.append(("grant_permission", device, package, permission))
        return {
            "success": True,
            "action": "grant_permission",
            "package": package,
            "permission": permission,
        }

    def _tap(device: str, x: int, y: int) -> dict:
        mock.tap_calls.append((device, x, y))
        return {"success": True, "action": "tap", "x": x, "y": y}

    def _elements(device: str) -> list[dict]:
        return elements

    def _elements_strict(device: str) -> list[dict]:
        # Default: mirrors the lenient elements() happy path. Tests exercising
        # a HARNESS fault override this directly:
        # `mock.elements_strict.side_effect = UiTreeUnavailableError(...)`.
        return elements

    def _screenshot(device: str) -> bytes:
        return screenshot_bytes

    def _navigate(device: str, uri: str) -> dict:
        mock.navigate_calls.append((device, uri))
        return {"success": True, "action": "navigate", "uri": uri}

    def _metrics(device: str, package: str) -> dict | None:
        return metrics_response

    def _install(device: str, path: str) -> dict:
        mock.install_calls.append((device, path))
        return {"success": True}

    def _health() -> dict:
        return {"status": "ok"}

    def _call_tool(tool: str, params: dict, timeout: int = 30) -> dict:
        # NerveAdapter.key reaches through to the raw tool call. Track key
        # presses with a realistic success shape so adapter result.get("success")
        # is a real bool, not a truthy MagicMock.
        if tool == "nerve_key":
            mock.key_calls.append((params.get("device"), params.get("key")))
            return {"success": True, "action": "key", "key": params.get("key")}
        if tool == "nerve_ensure_ready":
            # Default: device is ready, so the readiness gate doesn't skip the run.
            # Per-test override the verdict to exercise the SKIP path.
            return {"ready": True, "reason": "ready", "awake": True, "unlocked": True}
        return {"success": True}

    # screen_state: successive responses; each item is either a dict (returned)
    # or an Exception instance (raised). The last item repeats once exhausted so
    # steady-state polling stays stable.
    _state_seq = list(screen_state_responses) if screen_state_responses else [{}]

    def _screen_state(device: str) -> dict:
        item = _state_seq[0] if len(_state_seq) == 1 else _state_seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    mock.discover.side_effect = _discover
    mock.launch.side_effect = _launch
    mock.kill.side_effect = _kill
    mock.clear_app_data.side_effect = _clear_app_data
    mock.grant_permission.side_effect = _grant_permission
    mock.tap.side_effect = _tap
    mock.elements.side_effect = _elements
    mock.elements_strict.side_effect = _elements_strict
    mock.screenshot.side_effect = _screenshot
    mock.navigate.side_effect = _navigate
    mock.metrics.side_effect = _metrics
    mock.install.side_effect = _install
    mock.health.side_effect = _health
    mock.screen_state.side_effect = _screen_state
    mock._call_tool.side_effect = _call_tool
    return mock


@pytest.fixture
def mock_nerve_client() -> MagicMock:
    """Default-configured NerveClient mock with realistic return shapes."""
    return make_mock_nerve_client()
