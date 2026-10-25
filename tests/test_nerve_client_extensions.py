"""Tests for NerveClient methods added to support BVT migration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from linc_crucible.core.nerve_client import NerveClient


@pytest.fixture
def client() -> NerveClient:
    return NerveClient(base_url="http://localhost:8111")


class TestScreenshot:
    def test_returns_bytes_when_daemon_returns_path(
        self, client: NerveClient, tmp_path: Path
    ) -> None:
        png = tmp_path / "shot.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\nFAKE_PAYLOAD")

        with patch.object(client, "_call_tool", return_value={"path": str(png)}):
            data = client.screenshot(device="DEV1")

        assert data == b"\x89PNG\r\n\x1a\nFAKE_PAYLOAD"

    def test_returns_bytes_when_daemon_returns_screenshot_alias(
        self, client: NerveClient, tmp_path: Path
    ) -> None:
        png = tmp_path / "shot.png"
        png.write_bytes(b"PNGBYTES")

        with patch.object(client, "_call_tool", return_value={"screenshot": str(png)}):
            data = client.screenshot(device="DEV1")

        assert data == b"PNGBYTES"

    def test_raises_when_no_path_in_response(self, client: NerveClient) -> None:
        with (
            patch.object(client, "_call_tool", return_value={}),
            pytest.raises(RuntimeError, match="returned no path"),
        ):
            client.screenshot(device="DEV1")


class TestTap:
    def test_passes_coordinates_to_call_tool(self, client: NerveClient) -> None:
        with patch.object(client, "_call_tool", return_value={"success": True}) as ct:
            result = client.tap(device="DEV1", x=120, y=240)

        ct.assert_called_once_with("nerve_tap", {"device": "DEV1", "x": 120, "y": 240})
        assert result == {"success": True}


class TestNavigate:
    def test_calls_nerve_launch_with_uri_and_empty_package(self, client: NerveClient) -> None:
        with patch.object(client, "_call_tool", return_value={"success": True}) as ct:
            result = client.navigate(device="DEV1", uri="precisionlevel://screen/settings")

        ct.assert_called_once_with(
            "nerve_launch",
            {"device": "DEV1", "package": "", "uri": "precisionlevel://screen/settings"},
        )
        assert result == {"success": True}


class TestMetrics:
    def test_returns_dict_on_success(self, client: NerveClient) -> None:
        sample = {"memory_mb": 87, "cpu_pct": 4.2, "fps_avg": 59.7}
        with patch.object(client, "_call_tool", return_value=sample):
            result = client.metrics(device="DEV1", package="com.x")

        assert result == sample

    def test_returns_none_on_error(self, client: NerveClient) -> None:
        # Metrics is best-effort — should not propagate errors (callers treat
        # missing metrics as "skip perf snapshot").
        with patch.object(client, "_call_tool", side_effect=ConnectionError("daemon down")):
            result = client.metrics(device="DEV1", package="com.x")

        assert result is None


class TestHealthAndAvailable:
    def test_health_calls_GET_health_endpoint(self, client: NerveClient) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"status": "ok"})
        with patch.object(client._http, "get", return_value=mock_resp) as get:
            result = client.health()

        get.assert_called_once_with("http://localhost:8111/health", timeout=5)
        assert result == {"status": "ok"}

    def test_health_raises_connection_error_when_http_get_connect_errors(self) -> None:
        # Daemon-independent: patch the HTTP layer to raise httpx.ConnectError
        # and assert health() translates it to a builtin ConnectionError. This
        # covers the `except httpx.ConnectError -> raise ConnectionError` branch
        # in health() without a live nerve daemon on :8111. max_retries=1 so the
        # transient-retry wrapper raises on the first attempt (no backoff sleep,
        # deterministic and fast).
        offline = NerveClient(base_url="http://localhost:8111", max_retries=1)
        with (
            patch.object(
                offline._http,
                "get",
                side_effect=httpx.ConnectError("connection refused"),
            ),
            pytest.raises(ConnectionError, match="Nerve unreachable"),
        ):
            offline.health()

    def test_available_true_when_health_succeeds(self, client: NerveClient) -> None:
        with patch.object(client, "health", return_value={"status": "ok"}):
            assert client.available is True

    def test_available_false_when_health_raises_connection_error(
        self, client: NerveClient
    ) -> None:
        with patch.object(client, "health", side_effect=ConnectionError("daemon down")):
            assert client.available is False

    def test_available_false_when_health_raises_httpx_connect_error(
        self, client: NerveClient
    ) -> None:
        with patch.object(client, "health", side_effect=httpx.ConnectError("nope")):
            assert client.available is False
