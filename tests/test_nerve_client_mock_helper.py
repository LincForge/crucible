"""Sanity tests for make_mock_nerve_client — proves the realistic-shape
contract from the Hotz finding (spec 0351fdd7)."""

from __future__ import annotations

from tests.conftest import make_mock_nerve_client


def test_discover_returns_at_least_one_device_by_default() -> None:
    mock = make_mock_nerve_client()
    devices = mock.discover()
    assert len(devices) >= 1
    assert "serial" in devices[0]
    assert "platform" in devices[0]


def test_launch_returns_success_dict_not_empty() -> None:
    mock = make_mock_nerve_client()
    result = mock.launch("DEV1", "com.test")
    # Critical: result["success"] must work, not raise KeyError
    assert result["success"] is True
    assert result["package"] == "com.test"


def test_launch_calls_tracked() -> None:
    mock = make_mock_nerve_client()
    mock.launch("DEV1", "com.test")
    mock.launch("DEV2", "com.other")
    assert mock.launch_calls == [("DEV1", "com.test"), ("DEV2", "com.other")]


def test_tap_returns_success_with_coords() -> None:
    mock = make_mock_nerve_client()
    result = mock.tap("DEV1", x=100, y=200)
    assert result["success"] is True
    assert result["x"] == 100
    assert result["y"] == 200


def test_screenshot_returns_png_magic_bytes() -> None:
    mock = make_mock_nerve_client()
    data = mock.screenshot("DEV1")
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_elements_defaults_to_empty_list_not_none() -> None:
    mock = make_mock_nerve_client()
    elems = mock.elements("DEV1")
    assert elems == []  # Empty list, NOT None — supports `for el in elems: ...`


def test_elements_returns_configured_response() -> None:
    sample = [{"id": "btn_ok", "text": "OK", "bounds": "[0,0][100,100]"}]
    mock = make_mock_nerve_client(elements_response=sample)
    assert mock.elements("DEV1") == sample


def test_navigate_returns_success_with_uri() -> None:
    mock = make_mock_nerve_client()
    result = mock.navigate("DEV1", "app://settings")
    assert result["success"] is True
    assert result["uri"] == "app://settings"


def test_metrics_defaults_to_none() -> None:
    mock = make_mock_nerve_client()
    assert mock.metrics("DEV1", "com.test") is None


def test_metrics_returns_configured_response() -> None:
    sample = {"memory_mb": 87, "cpu_pct": 4.2}
    mock = make_mock_nerve_client(metrics_response=sample)
    assert mock.metrics("DEV1", "com.test") == sample
