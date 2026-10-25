"""Tests for NerveClient — HTTP client to nerve's device API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from linc_crucible.core.nerve_client import NerveClient, find_reservation


@pytest.fixture
def client() -> NerveClient:
    return NerveClient(base_url="http://localhost:8101")


# --- discover ---


def test_discover_returns_device_list(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {
            "devices": [
                {"serial": "abc123", "platform": "android"},
                {"serial": "def456", "platform": "ios"},
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.discover()

    assert result == [
        {"serial": "abc123", "platform": "android"},
        {"serial": "def456", "platform": "ios"},
    ]
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_discover",
        json={},
        timeout=30,
    )


def test_discover_returns_empty_on_http_error(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with patch.object(client._http, "post", return_value=mock_response):
        result = client.discover()

    assert result == []


def test_discover_returns_empty_on_connection_error(client: NerveClient) -> None:
    with patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")):
        result = client.discover()

    assert result == []


# --- install ---


def test_install_calls_correct_endpoint(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "ok"}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.install(device="abc123", path="/tmp/app.apk")

    assert result == {"status": "ok"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_install",
        json={"device": "abc123", "path": "/tmp/app.apk"},
        timeout=120,
    )


def test_install_raises_on_connection_error(client: NerveClient) -> None:
    with (
        patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")),
        pytest.raises(ConnectionError, match="Nerve unreachable"),
    ):
        client.install(device="abc123", path="/tmp/app.apk")


# --- pool ---


def test_pool_calls_correct_endpoint(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {
            "reservations": [
                {
                    "device_serial": "R58N000TEST",
                    "session_id": "other-sess",
                    "purpose": "bench-grid",
                }
            ],
            "count": 1,
            "session_id": "my-sess",
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.pool()

    assert result == {
        "reservations": [
            {
                "device_serial": "R58N000TEST",
                "session_id": "other-sess",
                "purpose": "bench-grid",
            }
        ],
        "count": 1,
        "session_id": "my-sess",
    }
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_pool",
        json={},
        timeout=30,
    )


def test_pool_raises_on_connection_error(client: NerveClient) -> None:
    with (
        patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")),
        pytest.raises(ConnectionError, match="Nerve unreachable"),
    ):
        client.pool()


# --- find_reservation ---


def test_find_reservation_matches_by_serial() -> None:
    reservations = [
        {"device_serial": "OTHER", "session_id": "sess-a"},
        {"device_serial": "R58N000TEST", "session_id": "other-sess", "purpose": "bench-grid"},
    ]
    result = find_reservation(reservations, "R58N000TEST")
    assert result == {
        "device_serial": "R58N000TEST",
        "session_id": "other-sess",
        "purpose": "bench-grid",
    }


def test_find_reservation_returns_none_when_no_match() -> None:
    reservations = [{"device_serial": "OTHER", "session_id": "sess-a"}]
    assert find_reservation(reservations, "R58N000TEST") is None


def test_find_reservation_handles_empty_list() -> None:
    assert find_reservation([], "R58N000TEST") is None


# --- launch ---


def test_launch_calls_correct_endpoint(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "launched"}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.launch(device="abc123", package="com.example.app")

    assert result == {"status": "launched"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_launch",
        json={"device": "abc123", "package": "com.example.app"},
        timeout=30,
    )


def test_launch_sends_uri_when_non_empty(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "launched"}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.launch(
            device="abc123",
            package="com.example.app",
            uri="myapp://settings",
        )

    assert result == {"status": "launched"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_launch",
        json={"device": "abc123", "package": "com.example.app", "uri": "myapp://settings"},
        timeout=30,
    )


def test_launch_omits_uri_when_empty(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "launched"}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.launch(device="abc123", package="com.example.app", uri="")

    assert result == {"status": "launched"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_launch",
        json={"device": "abc123", "package": "com.example.app"},
        timeout=30,
    )


def test_launch_raises_on_connection_error(client: NerveClient) -> None:
    with (
        patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")),
        pytest.raises(ConnectionError, match="Nerve unreachable"),
    ):
        client.launch(device="abc123", package="com.example.app")


# --- kill ---


def test_kill_calls_correct_endpoint(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "killed"}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.kill(device="abc123", package="com.example.app")

    assert result == {"status": "killed"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_kill",
        json={"device": "abc123", "package": "com.example.app"},
        timeout=30,
    )


# --- clear_app_data / grant_permission ---


def test_clear_app_data_calls_correct_endpoint(client: NerveClient) -> None:
    """Must send confirm=True: nerve#95's destructive-op gate rejects an
    unconfirmed clear with {"success": False, "error": "confirmation_required"}.
    Without it every cold_start "wipe" silently degraded to a warm
    kill+relaunch while the test still reported PASS."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"success": True}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.clear_app_data(device="abc123", package="com.example.app")

    assert result == {"success": True}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_clear_app_data",
        json={"device": "abc123", "package": "com.example.app", "confirm": True},
        timeout=30,
    )


def test_grant_permission_calls_correct_endpoint(client: NerveClient) -> None:
    """Must send confirm=True: nerve#95's gate covers grant_permission too, and
    an unconfirmed grant is rejected with confirmation_required. Unlike the
    clear_app_data case this failure is NOT silent — the grant step fails the
    BVT outright (ZenBalance#60, nightly red since the mini's nerve daemon
    picked up the gate)."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"success": True}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.grant_permission(
            device="abc123",
            package="com.example.app",
            permission="android.permission.CAMERA",
        )

    assert result == {"success": True}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_grant_permission",
        json={
            "device": "abc123",
            "package": "com.example.app",
            "permission": "android.permission.CAMERA",
            "confirm": True,
        },
        timeout=30,
    )


# --- _call_tool ---


def test_call_tool_posts_to_correct_url(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"screenshot": "/tmp/shot.png"}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client._call_tool("nerve_screenshot", {"device": "abc123"}, timeout=60)

    assert result == {"screenshot": "/tmp/shot.png"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_screenshot",
        json={"device": "abc123"},
        timeout=60,
    )


# --- response envelope unwrapping ---


class TestNerveClientResponseFormat:
    def test_discover_unwraps_result_envelope(self) -> None:
        """NerveClient must unwrap nerve's {"result": ...} envelope."""
        client = NerveClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        # This is what nerve ACTUALLY returns
        mock_response.json.return_value = {
            "result": {
                "devices": [
                    {
                        "serial": "ABC",
                        "platform": "android",
                        "device_type": "phone",
                        "model": "Pixel",
                    },
                ],
                "count": 1,
            }
        }
        with patch.object(client._http, "post", return_value=mock_response):
            devices = client.discover()
        assert len(devices) == 1, f"Expected 1 device but got {len(devices)}"
        assert devices[0]["serial"] == "ABC"

    def test_install_unwraps_result_envelope(self) -> None:
        client = NerveClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {"success": True, "action": "install", "path": "/app.apk"}
        }
        with patch.object(client._http, "post", return_value=mock_response):
            result = client.install("ABC", "/app.apk")
        assert result.get("success") is True

    def test_call_tool_unwraps_result(self) -> None:
        client = NerveClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": {"foo": "bar"}}
        with patch.object(client._http, "post", return_value=mock_response):
            data = client._call_tool("test_tool", {})
        assert data == {"foo": "bar"}


class TestNerveClientDefaultPort:
    def test_default_port_is_8111(self) -> None:
        """Default port must be 8111 (nerve daemon), not 8101 (forge-srv)."""
        client = NerveClient()
        assert "8111" in client._base_url


# --- zero-bridge contract ---


# --- elements ---


def test_elements_returns_element_list(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {
            "elements": [
                {"id": "btn", "text": "OK", "class": "android.widget.Button"},
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.elements(device="abc123")

    assert result == [{"id": "btn", "text": "OK", "class": "android.widget.Button"}]
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_elements",
        json={"device": "abc123"},
        timeout=30,
    )


def test_elements_returns_empty_on_connection_error(client: NerveClient) -> None:
    with patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")):
        result = client.elements(device="abc123")

    assert result == []


def test_elements_returns_empty_on_http_error(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with patch.object(client._http, "post", return_value=mock_response):
        result = client.elements(device="abc123")

    assert result == []


def test_elements_returns_empty_list_when_key_missing(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response):
        result = client.elements(device="abc123")

    assert result == []


def test_elements_returns_empty_on_error_dict_pins_lenient_path(client: NerveClient) -> None:
    """`elements()` stays lenient even for the same error-shaped dict that
    makes `elements_strict()` raise — the two methods must diverge only in
    the error case, never in the happy path."""
    with patch.object(
        client,
        "_call_tool",
        return_value={"error": "Element dump failed: could not get idle state"},
    ):
        result = client.elements(device="abc123")

    assert result == []


# --- elements_strict ---


def test_elements_strict_returns_element_list(client: NerveClient) -> None:
    with patch.object(client, "_call_tool", return_value={"elements": []}):
        result = client.elements_strict(device="abc123")

    assert result == []


def test_elements_strict_returns_nonempty_element_list(client: NerveClient) -> None:
    elements = [{"id": "btn", "text": "OK"}]
    with patch.object(client, "_call_tool", return_value={"elements": elements}):
        result = client.elements_strict(device="abc123")

    assert result == elements


def test_elements_strict_raises_on_error_key(client: NerveClient) -> None:
    from linc_crucible.core.nerve_client import UiTreeUnavailableError

    with patch.object(
        client,
        "_call_tool",
        return_value={"error": "Element dump failed: could not get idle state"},
    ), pytest.raises(UiTreeUnavailableError, match="could not get idle state"):
        client.elements_strict(device="abc123")


def test_elements_strict_raises_on_elements_error_key(client: NerveClient) -> None:
    from linc_crucible.core.nerve_client import UiTreeUnavailableError

    with patch.object(
        client,
        "_call_tool",
        return_value={"elements_error": "uiautomator dump timed out"},
    ), pytest.raises(UiTreeUnavailableError, match="uiautomator dump timed out"):
        client.elements_strict(device="abc123")


def test_elements_strict_raises_on_missing_elements_key(client: NerveClient) -> None:
    from linc_crucible.core.nerve_client import UiTreeUnavailableError

    with (
        patch.object(client, "_call_tool", return_value={}),
        pytest.raises(UiTreeUnavailableError),
    ):
        client.elements_strict(device="abc123")


def test_elements_strict_raises_on_connection_error(client: NerveClient) -> None:
    from linc_crucible.core.nerve_client import UiTreeUnavailableError

    with (
        patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")),
        pytest.raises(UiTreeUnavailableError),
    ):
        client.elements_strict(device="abc123")


def test_elements_strict_raises_on_http_error(client: NerveClient) -> None:
    from linc_crucible.core.nerve_client import UiTreeUnavailableError

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with (
        patch.object(client._http, "post", return_value=mock_response),
        pytest.raises(UiTreeUnavailableError),
    ):
        client.elements_strict(device="abc123")


# --- screen_state ---


def test_screen_state_returns_dict(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {"package": "com.example.app", "activity": ".MainActivity"}
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.screen_state(device="abc123")

    assert result == {"package": "com.example.app", "activity": ".MainActivity"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_state",
        json={"device": "abc123"},
        timeout=30,
    )


def test_screen_state_returns_empty_on_connection_error(client: NerveClient) -> None:
    with patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")):
        result = client.screen_state(device="abc123")

    assert result == {}


def test_screen_state_returns_empty_on_http_error(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with patch.object(client._http, "post", return_value=mock_response):
        result = client.screen_state(device="abc123")

    assert result == {}


class TestZeroBridgeImports:

    def test_no_bridge_imports_in_source(self) -> None:
        """After Phase 3, crucible must not import ADBBridge or IOSBridge."""
        import ast
        from pathlib import Path

        src_dir = Path(__file__).parent.parent / "src" / "linc_crucible"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            if py_file.name in ("adb_bridge.py", "ios_bridge.py"):
                continue  # Files being deleted
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and ("adb_bridge" in node.module or "ios_bridge" in node.module)
                ):
                    violations.append(f"{py_file.name}:{node.lineno}: from {node.module}")
        assert violations == [], "Found bridge imports:\n" + "\n".join(violations)


# --- type_text / long_press / swipe / density ---


def test_type_text_calls_correct_endpoint(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"success": True, "action": "type"}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.type_text(device="abc123", text="06")

    assert result == {"success": True, "action": "type"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_type",
        json={"device": "abc123", "text": "06"},
        timeout=30,
    )


def test_long_press_calls_correct_endpoint(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"success": True, "action": "long_press"}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.long_press(device="abc123", x=50, y=75, duration_ms=1000)

    assert result == {"success": True, "action": "long_press"}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_long_press",
        json={"device": "abc123", "x": 50, "y": 75, "duration_ms": 1000},
        timeout=30,
    )


def test_swipe_calls_correct_endpoint(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {"success": True, "action": "swipe", "from": [100, 800], "to": [100, 300]}
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.swipe(device="abc123", x1=100, y1=800, x2=100, y2=300, duration_ms=250)

    assert result == {"success": True, "action": "swipe", "from": [100, 800], "to": [100, 300]}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_swipe",
        json={
            "device": "abc123",
            "start_x": 100,
            "start_y": 800,
            "end_x": 100,
            "end_y": 300,
            "duration_ms": 250,
        },
        timeout=30,
    )


def test_swipe_default_duration_ms(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"success": True}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        client.swipe(device="abc123", x1=1, y1=2, x2=3, y2=4)

    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_swipe",
        json={
            "device": "abc123",
            "start_x": 1,
            "start_y": 2,
            "end_x": 3,
            "end_y": 4,
            "duration_ms": 300,
        },
        timeout=30,
    )


def test_density_calls_correct_endpoint(client: NerveClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"density_dpi": 420, "density_factor": 2.625}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._http, "post", return_value=mock_response) as mock_post:
        result = client.density(device="abc123")

    assert result == {"density_dpi": 420, "density_factor": 2.625}
    mock_post.assert_called_once_with(
        "http://localhost:8101/tools/nerve_density",
        json={"device": "abc123"},
        timeout=30,
    )


def test_density_returns_empty_on_error(client: NerveClient) -> None:
    with patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")):
        assert client.density(device="abc123") == {}


# --- cold-connection retry (T9) ---


class TestColdConnectionRetry:
    """Retry transient cold-connection resets on health + first launch.

    The nerve daemon intermittently drops the first keep-alive connection in
    a burst (Errno 54, reset by peer), which aborted an otherwise-healthy BVT
    run. Health probe + initial launch now retry a few times with backoff.
    """

    def _client(self) -> NerveClient:
        return NerveClient(base_url="http://localhost:8101", max_retries=3, retry_backoff=0.01)

    def _ok_response(self, payload: dict) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        return resp

    def test_health_retries_on_reset_then_succeeds(self) -> None:
        client = self._client()
        ok = self._ok_response({"status": "ok"})
        with (
            patch.object(
                client._http,
                "get",
                side_effect=[ConnectionResetError(54, "reset by peer"), ok],
            ) as mock_get,
            patch("linc_crucible.core.nerve_client.time.sleep") as mock_sleep,
        ):
            result = client.health()

        assert result == {"status": "ok"}
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()  # backed off before the retry

    def test_health_retries_on_httpx_read_error(self) -> None:
        client = self._client()
        ok = self._ok_response({"status": "ok"})
        with (
            patch.object(
                client._http,
                "get",
                side_effect=[httpx.ReadError("peer reset"), ok],
            ) as mock_get,
            patch("linc_crucible.core.nerve_client.time.sleep"),
        ):
            assert client.health() == {"status": "ok"}
        assert mock_get.call_count == 2

    def test_launch_retries_on_reset_then_succeeds(self) -> None:
        client = self._client()
        ok = self._ok_response({"result": {"status": "launched"}})
        with (
            patch.object(
                client._http,
                "post",
                side_effect=[ConnectionResetError(54, "reset by peer"), ok],
            ) as mock_post,
            patch("linc_crucible.core.nerve_client.time.sleep"),
        ):
            result = client.launch(device="abc123", package="com.example.app")

        assert result == {"status": "launched"}
        assert mock_post.call_count == 2

    def test_genuinely_down_daemon_raises_after_bounded_retries(self) -> None:
        client = self._client()
        with (
            patch.object(
                client._http, "get", side_effect=httpx.ConnectError("refused")
            ) as mock_get,
            patch("linc_crucible.core.nerve_client.time.sleep"),
            pytest.raises(ConnectionError, match="Nerve unreachable"),
        ):
            client.health()
        # Bounded: exactly max_retries attempts, then give up.
        assert mock_get.call_count == 3

    def test_retry_is_logged_not_silent(self) -> None:
        client = self._client()
        ok = self._ok_response({"status": "ok"})
        with (
            patch.object(
                client._http,
                "get",
                side_effect=[ConnectionResetError(54, "reset"), ok],
            ),
            patch("linc_crucible.core.nerve_client.time.sleep"),
            patch("linc_crucible.core.nerve_client.logger") as mock_logger,
        ):
            client.health()
        assert mock_logger.warning.called

    def test_no_retry_on_first_try_success(self) -> None:
        client = self._client()
        ok = self._ok_response({"status": "ok"})
        with (
            patch.object(client._http, "get", return_value=ok) as mock_get,
            patch("linc_crucible.core.nerve_client.time.sleep") as mock_sleep,
        ):
            client.health()
        mock_get.assert_called_once()
        mock_sleep.assert_not_called()

    def test_available_true_after_cold_retry(self) -> None:
        client = self._client()
        ok = self._ok_response({"status": "ok"})
        with (
            patch.object(
                client._http,
                "get",
                side_effect=[ConnectionResetError(54, "reset"), ok],
            ),
            patch("linc_crucible.core.nerve_client.time.sleep"),
        ):
            assert client.available is True


# --- loopback proxy isolation (linc-crucible-dq9, R4 Medium 4) ---


class TestNerveClientLoopbackProxyIsolation:
    """[contract] NerveClient ignores HTTP(S)_PROXY / ALL_PROXY env vars."""

    def test_nerve_client_disables_trust_env_and_never_contacts_proxy_listener(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import contextlib
        import socket
        import threading
        from collections.abc import Callable

        def _start_listener(response_body: bytes) -> tuple[int, list[bytes], Callable[[], None]]:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen(5)
            srv.settimeout(0.05)
            port = srv.getsockname()[1]
            captured: list[bytes] = []
            stop = threading.Event()

            def _serve() -> None:
                while not stop.is_set():
                    try:
                        conn, _ = srv.accept()
                    except TimeoutError:
                        continue
                    except OSError:
                        break
                    with conn:
                        conn.settimeout(0.5)
                        with contextlib.suppress(OSError):
                            data = conn.recv(4096)
                            if data:
                                captured.append(data)
                            reply = (
                                b"HTTP/1.1 200 OK\r\n"
                                b"Content-Type: application/json\r\n"
                                + f"Content-Length: {len(response_body)}\r\n".encode("ascii")
                                + b"Connection: close\r\n\r\n"
                                + response_body
                            )
                            conn.sendall(reply)

            thread = threading.Thread(target=_serve, daemon=True)
            thread.start()

            def _close() -> None:
                stop.set()
                with contextlib.suppress(OSError):
                    srv.close()
                thread.join(timeout=1.0)

            return port, captured, _close

        proxy_port, proxy_captured, close_proxy = _start_listener(b'{"result":{"devices":[]}}')
        target_port, target_captured, close_target = _start_listener(b'{"result":{"devices":[]}}')
        try:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            for var in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ):
                monkeypatch.setenv(var, proxy_url)
            for no_proxy_var in ("NO_PROXY", "no_proxy"):
                monkeypatch.delenv(no_proxy_var, raising=False)

            nc = NerveClient(base_url=f"http://127.0.0.1:{target_port}", max_retries=1)
            assert nc.discover() == []
        finally:
            close_proxy()
            close_target()

        proxy_log = [
            chunk.decode("latin-1", errors="replace").splitlines()[0]
            for chunk in proxy_captured
        ]
        assert proxy_captured == [], f"Proxy listener received nerve traffic: {proxy_log}"
        assert len(target_captured) == 1
        assert nc._http._trust_env is False

    def test_launch_default_uri_omitted_from_payload(self) -> None:
        nc = NerveClient(max_retries=1)
        with patch.object(nc, "_call_tool", return_value={"success": True}) as mock_call:
            assert nc.launch("dev1", "com.example.app") == {"success": True}
        mock_call.assert_called_once_with(
            "nerve_launch", {"device": "dev1", "package": "com.example.app"}
        )


