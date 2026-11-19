"""[contract] Tests for crucible_bvt lease owner guard and nerve call forwarding."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import structlog
from click.testing import CliRunner

import linc_crucible.server_context as ctx
from linc_crucible.bvt_tools import (
    _attach_reservation_owner,
    _bvt_adapter,
    _device_guard,
    _reservation_refusal,
    _resolve_owner,
    crucible_bvt,
)
from linc_crucible.cli import cli
from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.device_policy import DevicePolicy
from linc_crucible.core.nerve_client import _RESERVATION_EXEMPT_TOOLS, NerveClient
from linc_crucible.core.tool_args import _BVT_SPECS, InvalidToolArg, bvt_args_error, check_owner

_DEVICE = "R3CFAKE00001"
_PACKAGE = "com.example.myapp"
_EXPECTED_RESERVED_DETAIL = "R3CFAKE00001 reserved by session sess-1 (purpose: bvt, age 12s)"


@pytest.fixture(autouse=True)
def _clean_owner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ambient NERVE_OWNER never leaks into tests."""
    monkeypatch.delenv("NERVE_OWNER", raising=False)
    monkeypatch.setattr(
        ctx,
        "_cached_device_policy",
        DevicePolicy(personal_serials=frozenset(), loaded=True, source="test-fleet.yaml"),
    )


@pytest.fixture(autouse=True)
def captured_logs() -> Iterator[list[dict[str, Any]]]:
    """Capture structlog events so mutmut in-process re-runs never hit a closed stderr."""
    with structlog.testing.capture_logs() as logs:
        yield logs


def _setup_bvt_project(tmp_path: Path) -> Path:
    """Create a minimal project with a 1-test .bvt suite and return its root path."""
    proj_dir = tmp_path / "myapp"
    bvt_dir = proj_dir / ".bvt"
    bvt_dir.mkdir(parents=True)
    (proj_dir / "linc-manifest.yaml").write_text(
        f"schema_version: 1\nidentity:\n  name: myapp\n  package: {_PACKAGE}\n",
        encoding="utf-8",
    )
    (bvt_dir / "smoke.yaml").write_text(
        f"package: {_PACKAGE}\n"
        "tests:\n"
        "  - name: smoke_home\n"
        "    tags: [smoke]\n"
        "    steps:\n"
        f"      - launch: {_PACKAGE}\n"
        "      - assert_element: home_root\n",
        encoding="utf-8",
    )
    return proj_dir


def _make_fake_nerve_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    reservations: list[dict[str, Any]] | None = None,
) -> tuple[NerveClient, NerveAdapter, list[tuple[str, dict[str, Any]]]]:
    """Wire ctx with a real NerveClient + NerveAdapter over an httpx.MockTransport."""
    proj_dir = _setup_bvt_project(tmp_path)
    pool_reservations = (
        [
            {
                "device_serial": _DEVICE,
                "owner": "seat-a",
                "session_id": "sess-1",
                "purpose": "bvt",
                "age_seconds": 12,
            }
        ]
        if reservations is None
        else reservations
    )
    recorded: list[tuple[str, dict[str, Any]]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        recorded.append((request.url.path, body))
        tool = request.url.path.rsplit("/", 1)[-1]
        if tool == "nerve_pool":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "reservations": pool_reservations,
                        "count": len(pool_reservations),
                    }
                },
            )
        if tool == "nerve_discover":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "devices": [
                            {
                                "serial": _DEVICE,
                                "platform": "android",
                                "state": "online",
                            }
                        ]
                    }
                },
            )
        if tool == "nerve_ensure_ready":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "ready": True,
                        "reason": "ready",
                        "awake": True,
                        "unlocked": True,
                    }
                },
            )
        if tool == "nerve_state":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "foreground": {
                            "package": _PACKAGE,
                            "activity": ".MainActivity",
                        }
                    }
                },
            )
        if tool == "nerve_elements":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "elements": [
                            {
                                "id": f"{_PACKAGE}:id/home_root",
                                "text": "Home",
                                "bounds": "[0,0][100,100]",
                                "clickable": True,
                            }
                        ]
                    }
                },
            )
        if tool == "nerve_metrics":
            return httpx.Response(200, json={"result": {"memory_mb": 42.0}})
        return httpx.Response(200, json={"result": {"success": True}})

    client = NerveClient(max_retries=1, retry_backoff=0.0)
    client._http = httpx.Client(transport=httpx.MockTransport(_handler), trust_env=False)
    singleton_adapter = NerveAdapter(nerve=client)

    fake_reader = SimpleNamespace(get=lambda name: SimpleNamespace(path=str(proj_dir)))
    monkeypatch.setattr(ctx, "_ensure_initialized", lambda: None)
    monkeypatch.setattr(ctx, "_forge_reader", fake_reader)
    monkeypatch.setattr(ctx, "_nerve_client", client)
    monkeypatch.setattr(ctx, "_bvt_nerve_adapter", singleton_adapter)
    return client, singleton_adapter, recorded


@pytest.mark.asyncio
async def test_bvt_own_lease_passes_guard_and_forwards_owner_on_every_device_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """[contract] crucible_bvt(device="R3CFAKE00001", owner="seat-a") passes the reservation
    guard, forwards owner="seat-a" on every non-exempt nerve device call, and stamps
    reservation_owner="seat-a" on the result JSON."""
    _, singleton_adapter, recorded = _make_fake_nerve_env(monkeypatch, tmp_path)

    out = json.loads(
        await crucible_bvt(
            project="myapp",
            device=_DEVICE,
            output_dir="out",
            owner="seat-a",
        )
    )

    assert out["ran"] is True
    assert out["passed"] is True
    assert out["reservation_owner"] == "seat-a"
    assert singleton_adapter._nerve._owner == ""

    device_calls = [
        (path, body)
        for path, body in recorded
        if path.rsplit("/", 1)[-1] not in _RESERVATION_EXEMPT_TOOLS
    ]
    exempt_calls = [
        (path, body)
        for path, body in recorded
        if path.rsplit("/", 1)[-1] in _RESERVATION_EXEMPT_TOOLS
    ]
    assert len(device_calls) >= 4
    assert len(exempt_calls) >= 2
    for path, body in device_calls:
        assert body.get("owner") == "seat-a", f"missing owner on {path}: {body}"
    for path, body in exempt_calls:
        assert "owner" not in body, f"exempt call {path} carried owner: {body}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller_owner",
    ["seat-b", ""],
    ids=["foreign_owner_seat_b", "empty_owner"],
)
async def test_bvt_foreign_or_empty_owner_refuses_reserved_device_with_pinned_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caller_owner: str,
) -> None:
    """[contract] pin: owner="seat-b" and owner="" both return {ran: false,
    reason: "device-reserved"} with today's exact detail text and never run the suite."""
    _, _, recorded = _make_fake_nerve_env(monkeypatch, tmp_path)

    out = json.loads(
        await crucible_bvt(
            project="myapp",
            device=_DEVICE,
            output_dir="out",
            owner=caller_owner,
        )
    )

    assert out == {
        "ran": False,
        "reason": "device-reserved",
        "detail": _EXPECTED_RESERVED_DETAIL,
    }
    called_tools = [path.rsplit("/", 1)[-1] for path, _ in recorded]
    assert called_tools == ["nerve_discover", "nerve_pool"]


@pytest.mark.asyncio
async def test_bvt_nerve_owner_env_behaves_like_owner_arg_and_explicit_arg_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """[contract] NERVE_OWNER=seat-a with no owner argument behaves like owner="seat-a";
    an explicit non-empty owner argument overrides NERVE_OWNER."""
    _, _, recorded = _make_fake_nerve_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NERVE_OWNER", "  seat-a  ")

    out = json.loads(
        await crucible_bvt(
            project="myapp",
            device=_DEVICE,
            output_dir="out",
        )
    )

    assert out["ran"] is True
    assert out["passed"] is True
    assert out["reservation_owner"] == "seat-a"
    device_calls = [
        (path, body)
        for path, body in recorded
        if path.rsplit("/", 1)[-1] not in _RESERVATION_EXEMPT_TOOLS
    ]
    assert device_calls
    for path, body in device_calls:
        assert body.get("owner") == "seat-a", f"missing owner on {path}: {body}"

    recorded.clear()
    refused = json.loads(
        await crucible_bvt(
            project="myapp",
            device=_DEVICE,
            output_dir="out2",
            owner="seat-b",
        )
    )
    assert refused == {
        "ran": False,
        "reason": "device-reserved",
        "detail": _EXPECTED_RESERVED_DETAIL,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_owner",
    [
        "a b",
        "a" * 129,
        "a\nb",
        "\x00",
        123,
        None,
    ],
    ids=[
        "space",
        "too_long_129",
        "newline",
        "nul",
        "int_123",
        "none",
    ],
)
async def test_bvt_invalid_owner_returns_validation_error_and_makes_zero_nerve_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_owner: object,
) -> None:
    """[contract] Invalid owner argument returns tool validation error JSON with
    zero nerve calls."""
    _, _, recorded = _make_fake_nerve_env(monkeypatch, tmp_path)

    raw = await crucible_bvt(
        project="myapp",
        device=_DEVICE,
        owner=bad_owner,  # type: ignore[arg-type]
    )
    out = json.loads(raw)

    assert out["validation_error"] is True
    assert out["error_category"] == "validation"
    assert "owner" in out["error"]
    assert recorded == []


@pytest.mark.asyncio
async def test_bvt_invalid_nerve_owner_env_returns_validation_error_with_zero_nerve_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """[contract] Invalid NERVE_OWNER in environment returns tool validation error JSON with
    zero nerve calls when owner argument is omitted."""
    _, _, recorded = _make_fake_nerve_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NERVE_OWNER", "a b")

    out = json.loads(await crucible_bvt(project="myapp", device=_DEVICE))

    assert out["validation_error"] is True
    assert out["error_category"] == "validation"
    assert "owner" in out["error"]
    assert recorded == []


@pytest.mark.asyncio
async def test_bvt_unreserved_device_with_no_owner_uses_singleton_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """[test] pin: An unreserved device with no owner runs through the singleton adapter,
    sends no owner field on nerve calls, and omits reservation_owner from the result JSON."""
    _, singleton_adapter, recorded = _make_fake_nerve_env(monkeypatch, tmp_path, reservations=[])
    seen_adapters: list[object] = []
    orig_discover = singleton_adapter.discover

    def _tracking_discover() -> list[dict[str, Any]]:
        seen_adapters.append(singleton_adapter)
        return orig_discover()

    monkeypatch.setattr(singleton_adapter, "discover", _tracking_discover)

    out = json.loads(
        await crucible_bvt(
            project="myapp",
            device=_DEVICE,
            output_dir="out",
        )
    )

    assert out["ran"] is True
    assert out["passed"] is True
    assert "reservation_owner" not in out
    assert seen_adapters == [singleton_adapter]
    for path, body in recorded:
        assert "owner" not in body, f"unexpected owner on {path}: {body}"


@pytest.mark.asyncio
async def test_bvt_personal_device_refused_before_pool_lookup_even_with_matching_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """[test] pin: A personal device is refused before nerve_pool even when owner matches
    the active reservation."""
    _, _, recorded = _make_fake_nerve_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ctx,
        "_cached_device_policy",
        DevicePolicy(
            personal_serials=frozenset({_DEVICE}),
            loaded=True,
            source="test-fleet.yaml",
        ),
    )

    out = json.loads(
        await crucible_bvt(
            project="myapp",
            device=_DEVICE,
            output_dir="out",
            owner="seat-a",
        )
    )

    assert out["ran"] is False
    assert out["success"] is False
    assert out["error_category"] == "personal_device"
    called_tools = [path.rsplit("/", 1)[-1] for path, _ in recorded]
    assert "nerve_pool" not in called_tools
    assert called_tools == ["nerve_discover"]


def test_reservation_refusal_empty_owner_never_matches_ownerless_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """[contract] A reservation entry with empty/missing owner is still refused when the
    caller's effective owner is empty (only a non-empty matching owner passes)."""
    _make_fake_nerve_env(
        monkeypatch,
        tmp_path,
        reservations=[
            {
                "device_serial": _DEVICE,
                "owner": "",
                "session_id": "sess-empty",
                "purpose": "bvt",
                "age_seconds": 5,
            }
        ],
    )

    refused_default = _reservation_refusal(_DEVICE)
    assert refused_default is not None
    assert json.loads(refused_default)["reason"] == "device-reserved"

    refused_explicit_empty = _device_guard(_DEVICE, False, "")
    assert refused_explicit_empty is not None
    assert json.loads(refused_explicit_empty)["reason"] == "device-reserved"


def test_bvt_args_error_validates_owner_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """[contract] bvt_args_error and owner helpers validate boundaries and defaults."""
    _, singleton_adapter, _ = _make_fake_nerve_env(monkeypatch, tmp_path)

    for fn in (_reservation_refusal, _device_guard, crucible_bvt, bvt_args_error):
        assert inspect.signature(fn).parameters["owner"].default == ""
    assert inspect.signature(check_owner).parameters["field"].default == "owner"
    assert ("owner", check_owner, ()) in _BVT_SPECS

    assert _resolve_owner("") == ""
    monkeypatch.setenv("NERVE_OWNER", "env-seat")
    assert _resolve_owner("") == "env-seat"
    assert _resolve_owner("arg-seat") == "arg-seat"
    monkeypatch.delenv("NERVE_OWNER", raising=False)

    assert _bvt_adapter("") is singleton_adapter
    owned_adapter = _bvt_adapter("seat-a")
    assert owned_adapter is not singleton_adapter
    assert owned_adapter._nerve._owner == "seat-a"

    payload_empty: dict[str, Any] = {}
    _attach_reservation_owner(payload_empty, "")
    assert payload_empty == {}
    payload_owned: dict[str, Any] = {}
    _attach_reservation_owner(payload_owned, "seat-a")
    assert payload_owned == {"reservation_owner": "seat-a"}

    assert check_owner("") == ""
    assert check_owner("  seat-a  ") == "seat-a"
    assert check_owner("a" * 128) == "a" * 128

    with pytest.raises(InvalidToolArg) as exc_default_field:
        check_owner("a b")
    assert exc_default_field.value.field == "owner"
    assert (
        exc_default_field.value.reason
        == "owner must be 1..128 printable ASCII characters without whitespace"
    )
    assert str(exc_default_field.value) == (
        "invalid owner: owner must be 1..128 printable ASCII characters "
        "without whitespace (got 'a b')"
    )

    with pytest.raises(InvalidToolArg) as exc_info:
        check_owner("a b", "custom_owner")
    assert exc_info.value.field == "custom_owner"
    assert "printable ASCII" in exc_info.value.reason

    assert (
        bvt_args_error("myapp", _DEVICE, ".bvt", None, None, None, False, None, 1800.0, False)
        is None
    )
    assert (
        bvt_args_error(
            "myapp", _DEVICE, ".bvt", None, None, None, False, None, 1800.0, False, "seat-a"
        )
        is None
    )
    err_raw = bvt_args_error(
        "myapp", _DEVICE, ".bvt", None, None, None, False, None, 1800.0, False, "a b"
    )
    assert err_raw is not None
    err = json.loads(err_raw)
    assert err["validation_error"] is True
    assert err["error_category"] == "validation"
    assert "invalid owner:" in err["error"]


def test_generated_cli_for_crucible_bvt_lists_owner_option() -> None:
    """[test] The cli-kit-generated CLI for crucible_bvt exposes --owner in --help and schema."""
    runner = CliRunner()
    help_res = runner.invoke(cli, ["bvt", "--help"])
    assert help_res.exit_code == 0
    assert "--owner" in help_res.output

    schema_res = runner.invoke(cli, ["schema", "--json"])
    assert schema_res.exit_code == 0
    schema = json.loads(schema_res.output)
    owner_opt = schema["commands"]["bvt"]["options"]["owner"]
    assert owner_opt["flag"] == "--owner"
    assert owner_opt["required"] is False
    assert owner_opt["default"] == ""
