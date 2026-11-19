"""[contract] Tests for NerveClient lease owner validation and forwarding."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from linc_crucible.core.nerve_client import (
    _OWNER_RE,
    _RESERVATION_EXEMPT_TOOLS,
    NerveClient,
    _tool_body,
    validate_owner,
)


def _attach_recorder(
    client: NerveClient,
    shot_path: Path | None = None,
) -> list[tuple[str, dict]]:
    """Attach an httpx.MockTransport to ``client`` that records (path, json_body)."""
    recorded: list[tuple[str, dict]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        recorded.append((request.url.path, body))
        if request.url.path.endswith("/nerve_screenshot") and shot_path is not None:
            return httpx.Response(200, json={"result": {"path": str(shot_path)}})
        if request.url.path.endswith("/nerve_elements"):
            return httpx.Response(200, json={"result": {"elements": [{"id": "ok"}]}})
        if request.url.path.endswith("/nerve_discover"):
            return httpx.Response(200, json={"result": {"devices": [{"serial": "dev-1"}]}})
        if request.url.path.endswith("/nerve_pool"):
            return httpx.Response(200, json={"result": {"reservations": [], "count": 0}})
        return httpx.Response(200, json={"result": {"success": True}})

    client._http = httpx.Client(transport=httpx.MockTransport(_handler), trust_env=False)
    return recorded


def test_nerve_client_with_owner_sends_owner_on_device_tools_and_omits_on_exempt_tools(
    tmp_path: Path,
) -> None:
    """[contract] NerveClient(owner="seat-a") sends owner on tap/launch/elements/screenshot
    and omits owner on pool/discover and all exempt tools."""
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = NerveClient(owner="seat-a")
    recorded = _attach_recorder(client, shot_path=shot)

    client.tap("dev-1", 10, 20)
    client.launch("dev-1", "com.example.app")
    assert client.elements("dev-1") == [{"id": "ok"}]
    assert client.screenshot("dev-1") == b"\x89PNG\r\n\x1a\n"

    assert recorded == [
        ("/tools/nerve_tap", {"device": "dev-1", "x": 10, "y": 20, "owner": "seat-a"}),
        (
            "/tools/nerve_launch",
            {"device": "dev-1", "package": "com.example.app", "owner": "seat-a"},
        ),
        ("/tools/nerve_elements", {"device": "dev-1", "owner": "seat-a"}),
        ("/tools/nerve_screenshot", {"device": "dev-1", "owner": "seat-a"}),
    ]

    recorded.clear()
    client.pool()
    client.discover()
    expected_exempt = {
        "nerve_reserve",
        "nerve_release",
        "nerve_pool",
        "nerve_discover",
        "nerve_record_stop",
        "nerve_stream_stop",
    }
    assert set(_RESERVATION_EXEMPT_TOOLS) == expected_exempt
    for exempt_tool in sorted(_RESERVATION_EXEMPT_TOOLS):
        client._call_tool(exempt_tool, {"device": "dev-1"})
        assert _tool_body(exempt_tool, {"device": "dev-1"}, "seat-a") == {"device": "dev-1"}

    for path, body in recorded:
        assert "owner" not in body, f"exempt call {path} leaked owner in body: {body}"


def test_nerve_client_default_owner_bodies_identical_to_today(tmp_path: Path) -> None:
    """[contract] pin: NerveClient() default owner is empty and sends today's owner-free bodies."""
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = NerveClient()
    recorded = _attach_recorder(client, shot_path=shot)

    client.tap("dev-1", 10, 20)
    client.launch("dev-1", "com.example.app")
    client.elements("dev-1")
    client.screenshot("dev-1")
    client.pool()
    client.discover()

    assert recorded == [
        ("/tools/nerve_tap", {"device": "dev-1", "x": 10, "y": 20}),
        ("/tools/nerve_launch", {"device": "dev-1", "package": "com.example.app"}),
        ("/tools/nerve_elements", {"device": "dev-1"}),
        ("/tools/nerve_screenshot", {"device": "dev-1"}),
        ("/tools/nerve_pool", {}),
        ("/tools/nerve_discover", {}),
    ]


def test_with_owner_shares_http_client_and_leaves_original_owner_free() -> None:
    """[contract] with_owner("seat-a") shares httpx.Client, base_url and retry settings
    and leaves the original client's bodies owner-free."""
    base = NerveClient(base_url="http://127.0.0.1:8199", max_retries=5, retry_backoff=0.5)
    recorded = _attach_recorder(base)

    owned = base.with_owner("seat-a")
    assert owned is not base
    assert owned._http is base._http
    assert owned._base_url == "http://127.0.0.1:8199"
    assert owned._max_retries == 5
    assert owned._retry_backoff == 0.5
    assert owned._owner == "seat-a"
    assert base._owner == ""

    owned.tap("dev-1", 1, 2)
    base.tap("dev-1", 1, 2)
    cleared = owned.with_owner("")
    cleared.tap("dev-1", 1, 2)

    assert recorded == [
        ("/tools/nerve_tap", {"device": "dev-1", "x": 1, "y": 2, "owner": "seat-a"}),
        ("/tools/nerve_tap", {"device": "dev-1", "x": 1, "y": 2}),
        ("/tools/nerve_tap", {"device": "dev-1", "x": 1, "y": 2}),
    ]


def test_nerve_client_strips_owner_and_preserves_caller_params_dict() -> None:
    """[contract] NerveClient strips surrounding spaces from owner, accepts 128-char owners,
    and _call_tool never mutates the caller's params dict."""
    client = NerveClient(owner="  seat-a  ")
    assert client._owner == "seat-a"
    assert validate_owner("  seat-a  ") == "seat-a"
    assert validate_owner("") == ""

    max_owner = "a" * 128
    assert _OWNER_RE.fullmatch(max_owner) is not None
    assert NerveClient(owner=max_owner)._owner == max_owner

    recorded = _attach_recorder(client)
    caller_params = {"device": "dev-1", "x": 3, "y": 4}
    client._call_tool("nerve_tap", caller_params)

    assert caller_params == {"device": "dev-1", "x": 3, "y": 4}
    assert "owner" not in caller_params
    assert recorded == [
        ("/tools/nerve_tap", {"device": "dev-1", "x": 3, "y": 4, "owner": "seat-a"}),
    ]


@pytest.mark.parametrize(
    "bad_owner",
    [
        "seat a",
        "a\nb",
        "\x00",
        "a" * 129,
        123,
        None,
    ],
    ids=[
        "space",
        "newline",
        "nul",
        "too_long_129",
        "int_123",
        "none",
    ],
)
def test_hostile_owner_raises_value_error(bad_owner: object) -> None:
    """[contract] Hostile owner values raise ValueError in NerveClient and with_owner."""
    pattern = (
        r"\Aowner must be (a string|1\.\.128 printable ASCII characters without whitespace)\Z"
    )
    with pytest.raises(ValueError, match=pattern):
        NerveClient(owner=bad_owner)  # type: ignore[arg-type]

    base = NerveClient()
    with pytest.raises(ValueError, match=pattern):
        base.with_owner(bad_owner)  # type: ignore[arg-type]
