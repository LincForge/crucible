"""Fleet-path classification contracts for ``load_device_policy`` (REVIEW B4a).

Only a clean ``FileNotFoundError`` on the fleet path fails open (``"absent"``); every
entry that exists but is not a readable regular file fails closed (``"unreadable"``).
"""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from pathlib import Path
from typing import get_args

import pytest
import structlog

import linc_crucible.server_context as server_context
from linc_crucible.core import device_policy
from linc_crucible.core.device_policy import (
    DevicePolicy,
    DevicePolicyState,
    _probe_fleet_path,
    load_device_policy,
)


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


_REFUSAL_PREFIX = "device registry "


def _assert_fails_closed(policy: DevicePolicy, source: Path | str) -> None:
    assert policy.state == "unreadable"
    assert policy.loaded is False
    assert policy.personal_serials == frozenset()
    assert policy.error
    assert policy.is_personal("ANYSERIAL") is True
    assert policy.is_personal("") is True
    refusal = policy.refusal("ANYSERIAL")
    assert refusal is not None
    assert refusal.startswith(f"{_REFUSAL_PREFIX}{source} could not be read")
    assert policy.error in refusal


def _load_quietly(path: Path | str | None = None) -> tuple[DevicePolicy, list[dict]]:
    with structlog.testing.capture_logs() as logs:
        policy = load_device_policy(None if path is None else Path(path))
    return policy, logs


def test_dangling_symlink_fails_closed(tmp_path: Path) -> None:
    """[contract] Symlink whose target is missing -> 'unreadable', not 'absent'."""
    link = tmp_path / "device-fleet.yaml"
    link.symlink_to(tmp_path / "gone.yaml")

    policy, logs = _load_quietly(link)

    _assert_fails_closed(policy, link)
    assert not [e for e in logs if e.get("event") == "device_fleet_absent"]
    assert [e for e in logs if e.get("event") == "device_fleet_policy_read_failed"]


def test_symlink_loop_fails_closed(tmp_path: Path) -> None:
    """[contract] Self-referencing symlink (ELOOP) -> 'unreadable', never raises."""
    loop_a = tmp_path / "a.yaml"
    loop_b = tmp_path / "b.yaml"
    loop_a.symlink_to(loop_b)
    loop_b.symlink_to(loop_a)

    policy, _ = _load_quietly(loop_a)

    _assert_fails_closed(policy, loop_a)


def test_directory_at_fleet_path_fails_closed(tmp_path: Path) -> None:
    """[contract] A directory at the fleet path -> 'unreadable' (not a regular file)."""
    as_dir = tmp_path / "device-fleet.yaml"
    as_dir.mkdir()

    policy, _ = _load_quietly(as_dir)

    _assert_fails_closed(policy, as_dir)
    assert "not a regular file" in policy.error


def test_fifo_at_fleet_path_fails_closed(tmp_path: Path) -> None:
    """[contract] Hostile: a FIFO at the fleet path -> 'unreadable', never blocks on read."""
    fifo = tmp_path / "device-fleet.yaml"
    os.mkfifo(fifo)

    policy, _ = _load_quietly(fifo)

    _assert_fails_closed(policy, fifo)


def test_parent_not_a_directory_fails_closed(tmp_path: Path) -> None:
    """[contract] Hostile: a path component is a regular file (ENOTDIR) -> 'unreadable'."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    target = blocker / "device-fleet.yaml"

    policy, _ = _load_quietly(target)

    _assert_fails_closed(policy, target)


def test_embedded_nul_byte_path_fails_closed() -> None:
    """[contract] Hostile: a path with an embedded NUL -> 'unreadable', never raises."""
    target = "fleet\x00.yaml"

    policy, _ = _load_quietly(target)

    _assert_fails_closed(policy, target)


def test_symlink_to_regular_file_loads(tmp_path: Path) -> None:
    """A symlink resolving to a regular fleet file still loads normally."""
    real = tmp_path / "real.yaml"
    real.write_text('devices:\n  - serial: "PERSONALWATCH01"\n    personal: true\n')
    link = tmp_path / "device-fleet.yaml"
    link.symlink_to(real)

    policy, _ = _load_quietly(link)

    assert policy.state == "loaded"
    assert policy.is_personal("PERSONALWATCH01") is True


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permission checks")
def test_untraversable_parent_fails_closed_without_raising(tmp_path: Path) -> None:
    """[contract] EACCES on stat (parent mode 000) -> 'unreadable', no PermissionError."""
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    target = locked_dir / "device-fleet.yaml"
    target.write_text("devices: []\n")
    locked_dir.chmod(0o000)
    try:
        policy, _ = _load_quietly(target)
    finally:
        locked_dir.chmod(0o700)

    _assert_fails_closed(policy, target)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permission checks")
def test_server_context_caches_unreadable_on_untraversable_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] get_device_policy returns and caches 'unreadable' instead of re-raising."""
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    target = locked_dir / "device-fleet.yaml"
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(target))
    monkeypatch.setattr(server_context, "_cached_device_policy", None)
    locked_dir.chmod(0o000)
    try:
        with structlog.testing.capture_logs():
            first = server_context.get_device_policy()
            second = server_context.get_device_policy()
    finally:
        locked_dir.chmod(0o700)

    assert first.state == "unreadable"
    assert second is first


@pytest.mark.parametrize("bad_state", ["", "LOADED", "Absent", "bogus", "loaded "])
def test_unknown_state_is_rejected(bad_state: str) -> None:
    """[contract] DevicePolicy refuses to construct with a state outside the three known."""
    with pytest.raises(ValueError, match="state"):
        DevicePolicy(personal_serials=frozenset(), state=bad_state)  # type: ignore[arg-type]


def test_unknown_state_injected_post_init_fails_closed() -> None:
    """[contract] A state forced past __init__ is treated as unreadable, never as loaded."""
    policy = DevicePolicy(personal_serials=frozenset({"X"}), state="loaded", source="s")
    object.__setattr__(policy, "state", "bogus")

    assert policy.is_personal("BENCH") is True
    assert policy.is_personal("") is True
    refusal = policy.refusal("BENCH")
    assert refusal is not None
    assert refusal.startswith(f"{_REFUSAL_PREFIX}s could not be read")


def test_non_utf8_fleet_file_fails_closed(tmp_path: Path) -> None:
    """[contract] Hostile: binary / non-UTF-8 fleet file -> 'unreadable', never raises."""
    binary = tmp_path / "device-fleet.yaml"
    binary.write_bytes(b"\xff\xfe\x00devices: \x80\x81\n")

    policy, _ = _load_quietly(binary)

    _assert_fails_closed(policy, binary)
    assert "unacceptable character" in policy.error


def test_invalid_utf8_without_bom_fails_closed(tmp_path: Path) -> None:
    """[contract] Hostile: invalid UTF-8 bytes (no BOM) -> 'unreadable', never raises."""
    bad = tmp_path / "device-fleet.yaml"
    bad.write_bytes(b"devices:\n  - serial: P1\n    personal: true\n    note: \x80\x81\n")

    policy, logs = _load_quietly(bad)

    _assert_fails_closed(policy, bad)
    assert "unacceptable character" in policy.error
    assert any(e["event"] == "device_fleet_policy_read_failed" for e in logs)


def test_utf16_bom_fleet_file_loads_personal_serials(tmp_path: Path) -> None:
    """A UTF-16 (BOM) registry is decoded by YAML itself and its personal flags honored."""
    fleet = tmp_path / "device-fleet.yaml"
    fleet.write_bytes(
        "devices:\n  - serial: P1\n    personal: true\n  - serial: B1\n".encode("utf-16")
    )

    policy, _ = _load_quietly(fleet)

    assert policy.state == "loaded"
    assert policy.error == ""
    assert policy.personal_serials == frozenset({"P1"})
    assert policy.is_personal("P1") is True
    assert policy.is_personal("B1") is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permission checks")
def test_ios_signing_team_read_untraversable_parent_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] ios_signing._read_fleet_team returns None on EACCES, never raises."""
    from linc_crucible.core import ios_signing

    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    (locked_dir / "device-fleet.yaml").write_text("ios_signing:\n  team_id: TEAMID0000\n")
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(locked_dir / "device-fleet.yaml"))
    locked_dir.chmod(0o000)
    try:
        team = ios_signing._read_fleet_team()
    finally:
        locked_dir.chmod(0o700)

    assert team is None


@pytest.mark.parametrize("kind", ["dangling", "directory"])
def test_ios_signing_team_read_non_regular_entry_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    """[contract] Hostile: dangling symlink / directory at the fleet path -> None."""
    from linc_crucible.core import ios_signing

    target = tmp_path / "device-fleet.yaml"
    if kind == "dangling":
        target.symlink_to(tmp_path / "gone.yaml")
    else:
        target.mkdir()
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(target))

    assert ios_signing._read_fleet_team() is None


def test_state_vocabulary_is_single_sourced() -> None:
    """The runtime state set matches the ``DevicePolicyState`` Literal exactly."""
    states = device_policy._STATES
    assert states == frozenset(get_args(DevicePolicyState))
    assert states == {"loaded", "absent", "unreadable"}


def test_probe_classifies_each_entry_kind(tmp_path: Path) -> None:
    """[contract] _probe_fleet_path: absent / unreadable(+errno text) / regular file."""
    missing = tmp_path / "missing.yaml"
    regular = tmp_path / "regular.yaml"
    regular.write_text("devices: []\n")
    dangling = tmp_path / "dangling.yaml"
    dangling.symlink_to(tmp_path / "gone.yaml")
    blocker = tmp_path / "blocker"
    blocker.write_text("x")

    assert _probe_fleet_path(missing) == ("absent", f"file not found: {missing}")
    assert _probe_fleet_path(regular) is None
    state, error = _probe_fleet_path(dangling)  # type: ignore[misc]
    assert state == "unreadable"
    assert os.strerror(errno.ENOENT) in error
    state, error = _probe_fleet_path(blocker / "x.yaml")  # type: ignore[misc]
    assert state == "unreadable"
    assert os.strerror(errno.ENOTDIR) in error
    assert _probe_fleet_path(tmp_path) == ("unreadable", f"not a regular file: {tmp_path}")


def test_unreadable_and_absent_carry_source_and_logged_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both non-loaded states record the source path; unreadable logs path + error."""
    monkeypatch.setattr(device_policy, "_absent_warned", False)
    missing = tmp_path / "missing.yaml"
    as_dir = tmp_path / "dir.yaml"
    as_dir.mkdir()

    absent, absent_logs = _load_quietly(missing)
    unreadable, logs = _load_quietly(as_dir)

    assert absent.source == str(missing)
    assert absent_logs[0]["hint"] == device_policy._ABSENT_HINT
    failures = [e for e in logs if e.get("event") == "device_fleet_policy_read_failed"]
    assert failures == [
        {
            "event": "device_fleet_policy_read_failed",
            "log_level": "warning",
            "domain": "device_policy",
            "path": str(as_dir),
            "error": unreadable.error,
        }
    ]
    assert unreadable.error == f"not a regular file: {as_dir}"
