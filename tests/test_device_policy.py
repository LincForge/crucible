from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

import linc_crucible.core.device_policy as device_policy
import linc_crucible.server_context as server_context
from linc_crucible.core.device_policy import DevicePolicy, load_device_policy


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


def test_temp_fleet_yaml_personal_and_bench_device(tmp_path: Path) -> None:
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text(
        """
version: 1
devices:
  - id: watch-1
    serial: "PERSONALWATCH01"
    personal: true
  - id: bench-phone
    serial: "BENCH_DEVICE_123"
    personal: false
"""
    )
    policy = load_device_policy(fleet_file)
    assert policy.state == "loaded"
    assert policy.loaded is True
    assert policy.is_personal("PERSONALWATCH01") is True
    assert policy.is_personal("BENCH_DEVICE_123") is False
    assert policy.refusal("BENCH_DEVICE_123") is None
    expected_refusal = (
        f"device PERSONALWATCH01 is registered personal: true in {fleet_file}; "
        "pass allow_personal=True to override"
    )
    assert policy.refusal("PERSONALWATCH01") == expected_refusal


def test_personal_entry_carrying_hardware_id_matched_by_either_key(tmp_path: Path) -> None:
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text(
        """
version: 1
ios:
  devices:
    - id: personal-ios
      serial: "SERIAL_XYZ"
      hardware_id: "HW_XYZ"
      personal: true
"""
    )
    policy = load_device_policy(fleet_file)
    assert policy.state == "loaded"
    assert policy.loaded is True
    assert policy.is_personal("SERIAL_XYZ") is True
    assert policy.is_personal("HW_XYZ") is True
    assert policy.refusal("SERIAL_XYZ") == (
        f"device SERIAL_XYZ is registered personal: true in {fleet_file}; "
        "pass allow_personal=True to override"
    )
    assert policy.refusal("HW_XYZ") == (
        f"device HW_XYZ is registered personal: true in {fleet_file}; "
        "pass allow_personal=True to override"
    )


def test_missing_path_fails_open_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] Missing fleet file -> state='absent', fails open, and warns once."""
    missing_path = tmp_path / "non_existent_fleet.yaml"
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(missing_path))
    monkeypatch.setattr(device_policy, "_absent_warned", False, raising=False)

    with structlog.testing.capture_logs() as logs:
        policy = load_device_policy()
        second = load_device_policy(missing_path)

    assert policy.state == "absent"
    assert policy.loaded is False
    assert policy.personal_serials == frozenset()
    assert policy.error == f"file not found: {missing_path}"
    assert policy.is_personal("ANYSERIAL") is False
    assert policy.is_personal("") is False
    assert policy.refusal("ANYSERIAL") is None
    assert second.state == "absent"

    absent_warnings = [e for e in logs if e.get("event") == "device_fleet_absent"]
    assert len(absent_warnings) == 1
    assert absent_warnings[0]["log_level"] == "warning"
    assert absent_warnings[0]["path"] == str(missing_path)
    assert absent_warnings[0]["hint"] == "set LINC_DEVICE_FLEET_FILE to register personal devices"


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses chmod 000 permission checks")
def test_chmod_000_file_fails_closed(tmp_path: Path) -> None:
    """[contract] Unreadable (chmod 000) fleet file -> state='unreadable', fails closed."""
    locked_file = tmp_path / "locked_fleet.yaml"
    locked_file.write_text("version: 1\ndevices: []\n")
    locked_file.chmod(0o000)
    try:
        policy = load_device_policy(locked_file)
    finally:
        locked_file.chmod(0o600)

    assert policy.state == "unreadable"
    assert policy.loaded is False
    assert policy.is_personal("ANYSERIAL") is True
    refusal = policy.refusal("ANYSERIAL")
    assert refusal is not None
    assert str(locked_file) in refusal
    assert policy.error in refusal


def test_unparseable_yaml_fails_closed(tmp_path: Path) -> None:
    """[contract] Malformed YAML fleet file -> state='unreadable', fails closed."""
    bad_file = tmp_path / "bad_fleet.yaml"
    bad_file.write_text("::: unparseable yaml ::: [ {")

    policy = load_device_policy(bad_file)
    assert policy.state == "unreadable"
    assert policy.loaded is False
    assert policy.is_personal("ANYSERIAL") is True

    refusal = policy.refusal("ANYSERIAL")
    assert refusal is not None
    assert str(bad_file) in refusal
    assert policy.error in refusal
    assert (
        refusal == f"device registry {bad_file} could not be read ({policy.error}); "
        "refusing device operations without an explicit allow_personal=True"
    )


def test_non_mapping_yaml_fails_closed(tmp_path: Path) -> None:
    """[contract] Non-mapping YAML fleet file -> state='unreadable', fails closed."""
    bad_file = tmp_path / "list_fleet.yaml"
    bad_file.write_text("- item1\n- item2\n")

    policy = load_device_policy(bad_file)
    assert policy.state == "unreadable"
    assert policy.loaded is False
    assert policy.is_personal("ANYSERIAL") is True

    refusal = policy.refusal("ANYSERIAL")
    assert refusal is not None
    assert str(bad_file) in refusal
    assert policy.error in refusal
    assert (
        refusal == f"device registry {bad_file} could not be read ({policy.error}); "
        "refusing device operations without an explicit allow_personal=True"
    )


def test_loaded_property_parity(tmp_path: Path) -> None:
    """`loaded` property matches `state == 'loaded'` across all states and init forms."""
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text("version: 1\ndevices: []\n")
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("- not-a-mapping\n")
    missing_file = tmp_path / "missing.yaml"

    loaded_policy = load_device_policy(fleet_file)
    absent_policy = load_device_policy(missing_file)
    unreadable_policy = load_device_policy(bad_file)

    assert (loaded_policy.state, loaded_policy.loaded) == ("loaded", True)
    assert (absent_policy.state, absent_policy.loaded) == ("absent", False)
    assert (unreadable_policy.state, unreadable_policy.loaded) == ("unreadable", False)
    assert loaded_policy() is loaded_policy
    assert loaded_policy.is_personal("") is False

    compat_true = DevicePolicy(personal_serials=frozenset(), loaded=True, source="x")
    compat_false = DevicePolicy(personal_serials=frozenset(), loaded=False, source="x")
    assert (compat_true.state, compat_true.loaded) == ("loaded", True)
    assert (compat_false.state, compat_false.loaded) == ("unreadable", False)


def test_ios_signing_read_fleet_team_missing_file_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] Pin: ios_signing._read_fleet_team returns None for a missing fleet file."""
    from linc_crucible.core import ios_signing

    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(tmp_path / "absent.yaml"))
    assert ios_signing._read_fleet_team() is None


def test_real_fleet_registry_when_present() -> None:
    # The operator's real registry location is host-specific, so it comes from the env.
    raw_path = os.environ.get("CRUCIBLE_E2E_FLEET_FILE")
    if not raw_path:
        pytest.skip("CRUCIBLE_E2E_FLEET_FILE not set")
    real_path = Path(raw_path)
    if not real_path.is_file():
        pytest.skip(f"Real device fleet file not found at {real_path}")

    # Real serials are host-specific, so they come from the environment.
    personal = os.environ.get("CRUCIBLE_E2E_PERSONAL_SERIAL")
    bench = os.environ.get("CRUCIBLE_E2E_SERIAL")
    if not personal or not bench:
        pytest.skip("CRUCIBLE_E2E_PERSONAL_SERIAL / CRUCIBLE_E2E_SERIAL not set")

    policy = load_device_policy(real_path)
    assert policy.loaded is True
    assert policy.is_personal(personal) is True
    assert policy.is_personal(bench) is False
    assert policy.refusal(bench) is None


def test_server_context_exposes_policy_and_loads_at_most_once(tmp_path: Path) -> None:
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text(
        """
version: 1
devices:
  - id: watch-1
    serial: "PERSONALWATCH01"
    personal: true
"""
    )
    # Reset cached policy on server_context
    server_context._device_policy = None

    read_count = 0
    real_read_capped = device_policy.read_capped

    def counting_read_capped(path: Path, cap: int) -> bytes:
        nonlocal read_count
        if str(path) == str(fleet_file):
            read_count += 1
        return real_read_capped(path, cap)

    with (
        patch.object(device_policy, "read_capped", side_effect=counting_read_capped),
        patch(
            "linc_crucible.core.device_policy._fleet_registry_path",
            return_value=fleet_file,
        ),
    ):
        # Access or call twice
        p1 = server_context.get_device_policy()
        p2 = server_context._device_policy

        assert p1 is p2
        assert read_count == 1
        assert p1.is_personal("PERSONALWATCH01") is True

        # Call again
        p3 = server_context.get_device_policy()
        assert p3 is p1
        assert read_count == 1


@pytest.mark.parametrize(
    "query",
    [
        "r58n000test",
        " R58N000TEST ",
        "R58N000TEST\n",
        " r58n000test\n",
    ],
)
def test_is_personal_casefold_and_whitespace_variants(tmp_path: Path, query: str) -> None:
    """[contract] Casefold and whitespace/newline variants of a personal serial are refused."""
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text(
        """
version: 1
devices:
  - id: personal-phone
    serial: "R58N000TEST"
    personal: true
"""
    )
    policy = load_device_policy(fleet_file)
    assert policy.is_personal(query) is True
    refusal = policy.refusal(query)
    assert refusal is not None
    assert "allow_personal=True" in refusal


def test_hardware_id_only_personal_entry_case_insensitive(tmp_path: Path) -> None:
    """[contract] A personal entry with only hardware_id (no serial) matches case-insensitively."""
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text(
        """
version: 1
ios:
  devices:
    - id: personal-watch
      hardware_id: "00008110-00AABBCCDDEEFF00"
      personal: true
"""
    )
    policy = load_device_policy(fleet_file)
    assert policy.is_personal("00008110-00aabbccddeeff00") is True
    assert policy.is_personal(" 00008110-00AABBCCDDEEFF00\n") is True
    assert policy.refusal("00008110-00aabbccddeeff00") is not None


def test_ensure_initialized_wires_device_policy_into_assayer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linc_crucible import server_context

    monkeypatch.setattr(server_context, "_initialized", False)
    server_context._ensure_initialized()
    assert server_context._assayer is not None
    assert server_context._assayer._device_policy is server_context.get_device_policy


def test_recursive_alias_registry_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] Recursive-alias registry -> 'unreadable', pour/bvt refused, no exception."""
    import json
    from unittest.mock import MagicMock

    from linc_crucible import bvt_tools
    from linc_crucible.core.deployer import Deployer

    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text("devices: &a\n  - *a\n")

    policy = load_device_policy(fleet_file)
    assert policy.state == "unreadable"
    assert policy.loaded is False
    assert policy.error == "recursive YAML alias 'a' is not allowed"
    assert policy.is_personal("BENCH_DEVICE_123") is True
    refusal = policy.refusal("BENCH_DEVICE_123")
    assert refusal is not None
    assert "allow_personal=True" in refusal

    # pour refused before any build or nerve call
    deployer = Deployer(forge_reader=MagicMock(), nerve_client=MagicMock())
    pour_out = deployer.pour("MyApp", device="BENCH_DEVICE_123", policy=policy).to_dict()
    assert pour_out["success"] is False
    assert pour_out["error_category"] == "personal_device"
    assert str(fleet_file) in pour_out["error"]

    # bvt refused before any runner call
    monkeypatch.setattr(server_context, "_device_policy", policy)
    monkeypatch.setattr(server_context, "_nerve_client", None)
    bvt_raw = bvt_tools._personal_device_refusal("BENCH_DEVICE_123", allow_personal=False)
    assert bvt_raw is not None
    bvt_out = json.loads(bvt_raw)
    assert bvt_out["ran"] is False
    assert bvt_out["error_category"] == "personal_device"
    assert str(fleet_file) in bvt_out["error"]


def test_alias_chain_bomb_fails_closed_quickly_and_within_memory_budget(
    tmp_path: Path,
) -> None:
    """[contract] 30-level alias-chain bomb returns 'unreadable' within 2 s and < 50 MB peak."""
    import time
    import tracemalloc

    lines = ["version: 1", 'a0: &a0 ["DEV_A", "DEV_B"]']
    for i in range(1, 31):
        lines.append(f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}]")
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text("\n".join(lines) + "\n")

    tracemalloc.start()
    try:
        t0 = time.monotonic()
        policy = load_device_policy(fleet_file)
        elapsed = time.monotonic() - t0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert policy.state == "unreadable"
    assert policy.loaded is False
    assert elapsed < 2.0
    assert peak < 50 * 1024 * 1024
    assert policy.is_personal("DEV_A") is True
    assert policy.refusal("DEV_A") is not None


@pytest.mark.parametrize("size", [1024 * 1024 + 1, 2 * 1024 * 1024])
def test_oversized_registry_fails_closed_with_too_large_error(tmp_path: Path, size: int) -> None:
    """[contract] Registry over the 1 MiB cap -> 'unreadable' with 'is too large' in error."""
    fleet_file = tmp_path / "device-fleet.yaml"
    header = b"version: 1\ndevices: []\n"
    fleet_file.write_bytes(header + b"#" * (size - len(header)))

    policy = load_device_policy(fleet_file)
    assert policy.state == "unreadable"
    assert policy.loaded is False
    assert "is too large" in policy.error
    refusal = policy.refusal("BENCH_DEVICE_123")
    assert refusal is not None
    assert "is too large" in refusal


def test_deeply_nested_flow_list_fails_closed(tmp_path: Path) -> None:
    """[contract] 10,000-level deeply nested flow list -> 'unreadable', no RecursionError."""
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text("devices: " + "[" * 10_000 + "]" * 10_000 + "\n")

    policy = load_device_policy(fleet_file)
    assert policy.state == "unreadable"
    assert policy.loaded is False
    assert policy.is_personal("BENCH_DEVICE_123") is True
    assert policy.refusal("BENCH_DEVICE_123") is not None


def test_personal_entry_inside_list_of_dicts_is_found(tmp_path: Path) -> None:
    """[test] Valid file with a personal entry inside a list of dicts still finds it."""
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text(
        """
version: 1
clusters:
  - name: bench-cluster
    nodes:
      - serial: "BENCH_01"
        personal: false
      - serial: "NESTED_PERSONAL_01"
        personal: true
"""
    )
    policy = load_device_policy(fleet_file)
    assert policy.state == "loaded"
    assert policy.is_personal("NESTED_PERSONAL_01") is True
    assert policy.is_personal("BENCH_01") is False


def test_shared_non_recursive_alias_used_twice_loads(tmp_path: Path) -> None:
    """[test] Shared (non-recursive) alias used twice is legal and loads as before."""
    fleet_file = tmp_path / "device-fleet.yaml"
    fleet_file.write_text(
        """
version: 1
shared_personal: &shared_dev
  serial: "SHARED_PERSONAL_99"
  personal: true
primary:
  - *shared_dev
secondary:
  - *shared_dev
  - serial: "BENCH_99"
    personal: false
"""
    )
    policy = load_device_policy(fleet_file)
    assert policy.state == "loaded"
    assert policy.loaded is True
    assert policy.is_personal("SHARED_PERSONAL_99") is True
    assert policy.is_personal("BENCH_99") is False


def test_registry_at_exact_size_cap_loads(tmp_path: Path) -> None:
    """[test] A valid registry of exactly _MAX_REGISTRY_BYTES loads normally."""
    fleet_file = tmp_path / "device-fleet.yaml"
    header = b'version: 1\ndevices:\n  - serial: "CAP_PERSONAL"\n    personal: true\n'
    fleet_file.write_bytes(header + b"#" * (device_policy._MAX_REGISTRY_BYTES - len(header)))

    policy = load_device_policy(fleet_file)
    assert policy.state == "loaded"
    assert policy.is_personal("CAP_PERSONAL") is True


def test_alias_limit_boundary(tmp_path: Path) -> None:
    """[test] 100 aliases is allowed; 101 aliases is rejected as unreadable."""
    base = 'base: &b\n  serial: "P1"\n  personal: true\ndevices:\n'
    ok_file = tmp_path / "ok.yaml"
    ok_file.write_text(base + "".join("  - *b\n" for _ in range(100)))
    ok_policy = load_device_policy(ok_file)
    assert ok_policy.state == "loaded"
    assert ok_policy.is_personal("P1") is True

    over_file = tmp_path / "over.yaml"
    over_file.write_text(base + "".join("  - *b\n" for _ in range(101)))
    over_policy = load_device_policy(over_file)
    assert over_policy.state == "unreadable"
    assert "alias count exceeded limit of 100" in over_policy.error


def test_yaml_node_limit_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """[test] _BoundedSafeLoader enforces _MAX_YAML_NODES at the exact boundary."""
    monkeypatch.setattr(device_policy, "_MAX_YAML_NODES", 5)
    # mapping (1) + key 'a' (1) + list (1) + 2 items (2) = 5 nodes -> allowed
    ok_file = tmp_path / "ok.yaml"
    ok_file.write_text("a: [1, 2]\n")
    assert load_device_policy(ok_file).state == "loaded"

    # mapping (1) + key 'a' (1) + list (1) + 3 items (3) = 6 nodes -> rejected
    over_file = tmp_path / "over.yaml"
    over_file.write_text("a: [1, 2, 3]\n")
    over_policy = load_device_policy(over_file)
    assert over_policy.state == "unreadable"
    assert "node count exceeded limit of 5" in over_policy.error

    # Verify alias weight calculation (before = _node_count - 1; weight = _node_count - before):
    # mapping(1) + key 'a'(1) + list &x(1) + 2 items(2) + key 'b'(1) + alias *x(weight 3) = 9 nodes
    alias_file = tmp_path / "alias_weight.yaml"
    alias_file.write_text("a: &x [1, 2]\nb: *x\n")
    monkeypatch.setattr(device_policy, "_MAX_YAML_NODES", 9)
    assert load_device_policy(alias_file).state == "loaded"
    monkeypatch.setattr(device_policy, "_MAX_YAML_NODES", 8)
    assert load_device_policy(alias_file).state == "unreadable"

    # Non-mapping YAML reports exact error string
    non_map_file = tmp_path / "non_map.yaml"
    non_map_file.write_text("- item1\n- item2\n")
    non_map_policy = load_device_policy(non_map_file)
    assert non_map_policy.state == "unreadable"
    assert non_map_policy.personal_serials == frozenset()
    assert non_map_policy.error == "device registry is not a YAML mapping"


def test_extract_personal_serials_iterative_visited_and_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[test] _extract_personal_serials handles cyclic containers and enforces node_budget."""
    from typing import Any

    cyclic_dict: dict[str, Any] = {"serial": "CYC_DICT", "hardware_id": "   ", "personal": True}
    cyclic_dict["self"] = cyclic_dict
    cyclic_list: list[Any] = [cyclic_dict]
    cyclic_list.append(cyclic_list)
    root = {"items": cyclic_list}

    assert device_policy._extract_personal_serials(root) == {"CYC_DICT"}
    assert device_policy._extract_personal_serials({"a": [1, 2]}, node_budget=4) == set()

    with pytest.raises(device_policy.RegistryNodeBudgetError, match="exceeded 3-node budget"):
        device_policy._extract_personal_serials({"a": [1, 2]}, node_budget=3)
    assert issubclass(device_policy.RegistryNodeBudgetError, ValueError)

    # Confirm load_device_policy maps RegistryNodeBudgetError to state='unreadable'
    fleet_file = tmp_path / "fleet.yaml"
    fleet_file.write_text("a: [1, 2]\n")
    with patch.object(
        device_policy,
        "_extract_personal_serials",
        side_effect=device_policy.RegistryNodeBudgetError("walk budget exceeded"),
    ):
        policy = load_device_policy(fleet_file)
    assert policy.state == "unreadable"
    assert policy.error == "walk budget exceeded"
