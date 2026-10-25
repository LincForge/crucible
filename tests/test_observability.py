"""Tests for MCP tool call observability — JSONL logger."""

from __future__ import annotations

import errno
import json
import os
import stat
import sys
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture(autouse=True)
def _clean_usage_env(monkeypatch: MonkeyPatch) -> None:
    """Never inherit the host's usage-log mode or run id into these tests."""
    for var in ("CRUCIBLE_USAGE_LOG", "CRUCIBLE_RUN_ID", "FORGE_RUN_ID"):
        monkeypatch.delenv(var, raising=False)


def test_writes_jsonl_entry(tmp_path: Path) -> None:
    """log_mcp_call writes a valid JSONL entry with expected fields."""
    from linc_crucible.core.observability import log_mcp_call

    log_file = tmp_path / "calls.jsonl"
    log_mcp_call(
        server="linc-crucible",
        tool="crucible_pour",
        params={"project": "DawnCup"},
        duration_ms=123.4,
        success=True,
        log_path=log_file,
    )

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["server"] == "linc-crucible"
    assert entry["tool"] == "crucible_pour"
    assert entry["params"] == {"project": "DawnCup"}
    assert entry["duration_ms"] == 123.4
    assert entry["success"] is True
    assert entry["error"] is None
    assert "timestamp" in entry
    assert "session_id" in entry


def test_logs_error(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """log_mcp_call records error string on failure."""
    from linc_crucible.core.observability import log_mcp_call

    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "full")
    log_file = tmp_path / "calls.jsonl"
    log_mcp_call(
        server="linc-crucible",
        tool="crucible_assay",
        params={"project": "ZenBalance"},
        duration_ms=50.0,
        success=False,
        error="Device not found",
        log_path=log_file,
    )

    entry = json.loads(log_file.read_text().strip())
    assert entry["success"] is False
    assert entry["error"] == "Device not found"


def test_appends_multiple_entries(tmp_path: Path) -> None:
    """Multiple calls append to the same file."""
    from linc_crucible.core.observability import log_mcp_call

    log_file = tmp_path / "calls.jsonl"
    for i in range(3):
        log_mcp_call(
            server="linc-crucible",
            tool=f"tool_{i}",
            params={},
            duration_ms=float(i),
            success=True,
            log_path=log_file,
        )

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        assert json.loads(line)["tool"] == f"tool_{i}"


def test_redacts_bytes(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Bytes values in params are redacted with length info."""
    from linc_crucible.core.observability import log_mcp_call

    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "full")
    log_file = tmp_path / "calls.jsonl"
    log_mcp_call(
        server="linc-crucible",
        tool="crucible_baseline",
        params={"image": b"\x89PNG" * 100},
        duration_ms=10.0,
        success=True,
        log_path=log_file,
    )

    entry = json.loads(log_file.read_text().strip())
    assert entry["params"]["image"] == "<bytes:400>"


def test_truncates_long_strings(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Strings longer than 500 chars are truncated."""
    from linc_crucible.core.observability import log_mcp_call

    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "full")
    log_file = tmp_path / "calls.jsonl"
    long_val = "x" * 1000
    log_mcp_call(
        server="linc-crucible",
        tool="crucible_fleet",
        params={"data": long_val},
        duration_ms=5.0,
        success=True,
        log_path=log_file,
    )

    entry = json.loads(log_file.read_text().strip())
    assert len(entry["params"]["data"]) == 503  # 500 + "..."
    assert entry["params"]["data"].endswith("...")


def test_creates_parent_directory(tmp_path: Path) -> None:
    """log_mcp_call creates parent directories if they don't exist."""
    from linc_crucible.core.observability import log_mcp_call

    log_file = tmp_path / "nested" / "deep" / "calls.jsonl"
    log_mcp_call(
        server="linc-crucible",
        tool="crucible_score",
        params={},
        duration_ms=1.0,
        success=True,
        log_path=log_file,
    )

    assert log_file.exists()
    entry = json.loads(log_file.read_text().strip())
    assert entry["tool"] == "crucible_score"


def test_rotation_on_max_size(tmp_path: Path) -> None:
    """Log file is rotated when it exceeds max_size_bytes."""
    from linc_crucible.core.observability import log_mcp_call

    log_file = tmp_path / "calls.jsonl"
    # Write enough to exceed a small limit
    log_file.write_text("x" * 100)

    log_mcp_call(
        server="linc-crucible",
        tool="crucible_pour",
        params={},
        duration_ms=1.0,
        success=True,
        log_path=log_file,
        max_size_bytes=50,
    )

    # Original file should have the new entry
    assert log_file.exists()
    entry = json.loads(log_file.read_text().strip())
    assert entry["tool"] == "crucible_pour"

    # Rotated file should exist
    rotated = list(tmp_path.glob("calls.*.jsonl"))
    assert len(rotated) == 1


def test_session_id_from_env(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """session_id comes from FORGE_RUN_ID env var."""
    from linc_crucible.core.observability import log_mcp_call

    monkeypatch.setenv("FORGE_RUN_ID", "run-abc-123")
    log_file = tmp_path / "calls.jsonl"
    log_mcp_call(
        server="linc-crucible",
        tool="crucible_preflight",
        params={},
        duration_ms=2.0,
        success=True,
        log_path=log_file,
    )

    entry = json.loads(log_file.read_text().strip())
    assert entry["session_id"] == "run-abc-123"


def test_session_id_defaults_to_manual(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """session_id defaults to 'manual' when FORGE_RUN_ID is not set."""
    from linc_crucible.core.observability import log_mcp_call

    monkeypatch.delenv("FORGE_RUN_ID", raising=False)
    log_file = tmp_path / "calls.jsonl"
    log_mcp_call(
        server="linc-crucible",
        tool="crucible_preflight",
        params={},
        duration_ms=2.0,
        success=True,
        log_path=log_file,
    )

    entry = json.loads(log_file.read_text().strip())
    assert entry["session_id"] == "manual"


# --- REVIEW B8: usage-log privacy -------------------------------------------------

_SERIAL = "FAKESERIAL01"
_VIDEO = "/tmp/recordings/session-42/clip.mp4"


def _log(log_file: Path, params: dict, error: str | None = None) -> None:
    from linc_crucible.core.observability import log_mcp_call

    log_mcp_call(
        server="linc-crucible",
        tool="crucible_extract_assets",
        params=params,
        duration_ms=1.0,
        success=error is None,
        error=error,
        log_path=log_file,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "redacted"),
        ("", "redacted"),
        ("redacted", "redacted"),
        ("off", "off"),
        (" OFF ", "off"),
        ("full", "full"),
        ("Full", "full"),
        ("verbose", "redacted"),
        ("0", "redacted"),
    ],
)
def test_usage_log_mode_parses_env(
    monkeypatch: MonkeyPatch, raw: str | None, expected: str
) -> None:
    """[contract] CRUCIBLE_USAGE_LOG → off|full; default and any unknown value → redacted."""
    from linc_crucible.core.observability import usage_log_mode

    if raw is not None:
        monkeypatch.setenv("CRUCIBLE_USAGE_LOG", raw)
    assert usage_log_mode() == expected


def test_default_env_writes_redacted_entry(tmp_path: Path) -> None:
    """No env set → the default mode is redacted: serial and path never reach the file."""
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {"project": "DawnCup", "device": _SERIAL, "video_path": _VIDEO})

    entry = json.loads(log_file.read_text())
    assert entry["params"] == {
        "project": "DawnCup",
        "device": "<redacted len=12>",
        "video_path": f"<redacted len={len(_VIDEO)}>",
    }


def test_off_mode_writes_nothing_and_creates_no_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """CRUCIBLE_USAGE_LOG=off → log_mcp_call returns before touching the filesystem."""
    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "off")
    log_dir = tmp_path / "never"
    _log(log_dir / "calls.jsonl", {"device": _SERIAL})
    assert not log_dir.exists()


def test_off_mode_default_path_creates_no_linc_home(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Off mode with the default path leaves LINC_HOME uncreated."""
    from linc_crucible.core.observability import log_mcp_call

    home = tmp_path / "home"
    monkeypatch.setenv("LINC_HOME", str(home))
    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "off")
    log_mcp_call("linc-crucible", "crucible_pour", {}, 1.0, True)
    assert not home.exists()


def test_redacted_mode_hides_serial_and_video_path(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """[test] device serial and video_path appear only as <redacted len=N>."""
    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "redacted")
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {"device": _SERIAL, "video_path": _VIDEO, "mode": "quick"})

    text = log_file.read_text()
    assert _SERIAL not in text
    assert "recordings" not in text
    entry = json.loads(text)
    assert entry["params"]["device"] == "<redacted len=12>"
    assert entry["params"]["video_path"] == "<redacted len=35>"
    assert entry["params"]["mode"] == "quick"


@pytest.mark.parametrize(
    "key",
    [
        "device",
        "device_serial",
        "serial",
        "udid",
        "project_dir",
        "output_dir",
        "apk_path",
        "config_file",
        "video",
        "api_key",
        "token",
        "secret",
        "password",
        "DEVICE",
        "Video_Path",
    ],
)
def test_redacted_keys(tmp_path: Path, key: str) -> None:
    """Every serial/path/credential key is replaced length-only (case-insensitive)."""
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {key: "sekrit-value"})
    assert json.loads(log_file.read_text())["params"] == {key: "<redacted len=12>"}


def test_redacted_keeps_project_bool_number_mode(tmp_path: Path) -> None:
    """Project names, booleans, numbers and mode strings survive redaction."""
    log_file = tmp_path / "calls.jsonl"
    params = {"project": "ZenBalance", "build_only": True, "limit": 5, "mode": "full", "x": None}
    _log(log_file, params)
    assert json.loads(log_file.read_text())["params"] == params


def test_redacted_devices_list_and_nested(tmp_path: Path) -> None:
    """Strings inside a devices list and nested dicts/lists are redacted recursively."""
    log_file = tmp_path / "calls.jsonl"
    params = {
        "devices": ["emulator-5554", "FAKESERIAL01", None],
        "action": {"serial": "abc", "mode": [{"udid": "0000-1111"}, "tap"], "limit": 2},
        "screen_name": b"\x00" * 7,
        "serial": b"\x01\x02",
        "token": 12345,
        "secret": {"inner": ["x", 9]},
    }
    _log(log_file, params)
    text = log_file.read_text()
    assert "emulator-5554" not in text
    assert "0000-1111" not in text
    assert json.loads(text)["params"] == {
        "devices": ["<redacted len=13>", "<redacted len=12>", None],
        "action": {
            "serial": "<redacted len=3>",
            "mode": [{"udid": "<redacted len=9>"}, "tap"],
            "limit": 2,
        },
        "screen_name": "<bytes:7>",
        "serial": "<redacted len=2>",
        "token": "<redacted len=5>",
        "secret": {"inner": ["<redacted len=1>", "<redacted len=1>"]},
    }


def test_redacted_truncates_long_unredacted_strings(tmp_path: Path) -> None:
    """Allowlisted strings over 500 chars are still truncated (500 + '...')."""
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {"project": "y" * 600, "screen_name": ["z" * 501]})
    params = json.loads(log_file.read_text())["params"]
    assert params["project"] == "y" * 500 + "..."
    assert params["screen_name"] == ["z" * 500 + "..."]


def test_redacted_hostile_long_path_with_newline(tmp_path: Path) -> None:
    """[contract] hostile input: a 400-char path with an embedded newline never leaks."""
    hostile = "/private/" + "a" * 190 + "\nINJECTED\n" + "b" * 191
    assert len(hostile) == 400
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {"project_dir": hostile}, error=f"cannot open {hostile}: denied")

    text = log_file.read_text()
    assert len(text.splitlines()) == 1
    assert "INJECTED" not in text
    assert "aaaa" not in text
    assert json.dumps(hostile)[1:-1] not in text
    entry = json.loads(text)
    assert entry["params"] == {"project_dir": "<redacted len=400>"}
    assert entry["error"] == "cannot open <redacted len=400>: denied"


def test_redacted_error_scrubs_every_redacted_value(tmp_path: Path) -> None:
    """Error strings are scrubbed of each redacted value, longest first."""
    log_file = tmp_path / "calls.jsonl"
    _log(
        log_file,
        {"device": "ABC", "devices": ["ABCDEF"], "project": "DawnCup", "token": ""},
        error="ABCDEF busy; ABC offline for DawnCup",
    )
    entry = json.loads(log_file.read_text())
    assert entry["error"] == "<redacted len=6> busy; <redacted len=3> offline for DawnCup"


def test_full_mode_is_unchanged(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """CRUCIBLE_USAGE_LOG=full keeps today's behaviour: raw serial/path values and error."""
    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "full")
    log_file = tmp_path / "calls.jsonl"
    params = {"device": _SERIAL, "video_path": _VIDEO, "devices": ["a"], "nested": {"k": "v"}}
    _log(log_file, params, error=f"{_SERIAL} offline")
    entry = json.loads(log_file.read_text())
    assert entry["params"] == params
    assert entry["error"] == f"{_SERIAL} offline"


_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@_POSIX_ONLY
def test_log_file_created_0600(tmp_path: Path) -> None:
    """[contract] the log file is owner-only (0600) after the first write."""
    log_file = tmp_path / "calls.jsonl"
    old = os.umask(0o000)
    try:
        _log(log_file, {})
    finally:
        os.umask(old)
    assert _mode(log_file) == 0o600


@_POSIX_ONLY
def test_preexisting_world_readable_log_is_tightened(tmp_path: Path) -> None:
    """[contract] an older 0644 log is tightened to 0600 on the next append."""
    log_file = tmp_path / "calls.jsonl"
    log_file.write_text("")
    log_file.chmod(0o644)
    _log(log_file, {})
    assert _mode(log_file) == 0o600


@_POSIX_ONLY
def test_log_file_0600_after_rotation(tmp_path: Path) -> None:
    """[contract] both the fresh log and the rotated file stay 0600 across a rotation."""
    from linc_crucible.core.observability import log_mcp_call

    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {"note": "x" * 200})
    log_mcp_call("linc-crucible", "crucible_pour", {}, 1.0, True, None, log_file, 50)

    rotated = list(tmp_path.glob("calls.*.jsonl"))
    assert len(rotated) == 1
    assert _mode(log_file) == 0o600
    assert _mode(rotated[0]) == 0o600
    assert json.loads(log_file.read_text())["tool"] == "crucible_pour"


@_POSIX_ONLY
def test_symlinked_log_path_refused(tmp_path: Path) -> None:
    """[contract] a symlink at the log path raises ELOOP and nothing is written through it."""
    target = tmp_path / "victim.txt"
    target.write_text("original\n")
    log_file = tmp_path / "calls.jsonl"
    log_file.symlink_to(target)

    with pytest.raises(OSError) as excinfo:
        _log(log_file, {"device": _SERIAL})
    assert excinfo.value.errno == errno.ELOOP
    assert target.read_text() == "original\n"


def test_crucible_run_id_wins_over_forge_run_id(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """CRUCIBLE_RUN_ID takes precedence over the deprecated FORGE_RUN_ID."""
    monkeypatch.setenv("CRUCIBLE_RUN_ID", "crucible-run-1")
    monkeypatch.setenv("FORGE_RUN_ID", "forge-run-2")
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {})
    assert json.loads(log_file.read_text())["session_id"] == "crucible-run-1"


def test_empty_crucible_run_id_falls_back(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """An empty CRUCIBLE_RUN_ID falls through to FORGE_RUN_ID (deprecated, one release)."""
    monkeypatch.setenv("CRUCIBLE_RUN_ID", "")
    monkeypatch.setenv("FORGE_RUN_ID", "forge-run-2")
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {})
    assert json.loads(log_file.read_text())["session_id"] == "forge-run-2"


def test_redacted_error_scrubs_nested_values_under_redacted_key(tmp_path: Path) -> None:
    """A string nested inside a redacted key's dict is scrubbed from the error too."""
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {"secret": {"inner": ["hunter2"]}}, error="bad hunter2")
    assert json.loads(log_file.read_text())["error"] == "bad <redacted len=7>"


# --- REVIEW B8 follow-ups: run-id validation, derived-path scrub, log-failure isolation ----


@pytest.mark.parametrize(
    "hostile",
    ["run\nid", "../../etc", "a/b", "x" * 200, "x" * 129, "run id", "run;id", "é"],
)
def test_hostile_crucible_run_id_becomes_manual(
    tmp_path: Path, monkeypatch: MonkeyPatch, hostile: str
) -> None:
    """[contract] CRUCIBLE_RUN_ID outside [A-Za-z0-9._-]{1,128} is logged as "manual"."""
    monkeypatch.setenv("CRUCIBLE_RUN_ID", hostile)
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {})
    assert json.loads(log_file.read_text())["session_id"] == "manual"


def test_hostile_forge_run_id_becomes_manual(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """[contract] the deprecated FORGE_RUN_ID fallback is validated the same way."""
    monkeypatch.setenv("FORGE_RUN_ID", "evil\n{}")
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {})
    assert json.loads(log_file.read_text())["session_id"] == "manual"


def test_empty_run_ids_become_manual(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """[contract] empty CRUCIBLE_RUN_ID and FORGE_RUN_ID fall back to "manual"."""
    monkeypatch.setenv("CRUCIBLE_RUN_ID", "")
    monkeypatch.setenv("FORGE_RUN_ID", "")
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {})
    assert json.loads(log_file.read_text())["session_id"] == "manual"


def test_valid_run_id_at_max_length_is_kept(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """A 128-char id of the allowed alphabet is kept verbatim."""
    run_id = ("Ab0._-" * 22)[:128]
    monkeypatch.setenv("CRUCIBLE_RUN_ID", run_id)
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {})
    assert json.loads(log_file.read_text())["session_id"] == run_id


@pytest.mark.parametrize("env_var", ["HOME", "LINC_HOME", "LINC_PROJECTS_DIR"])
def test_redacted_error_scrubs_derived_host_paths(
    tmp_path: Path, monkeypatch: MonkeyPatch, env_var: str
) -> None:
    """[contract] a host path derived from HOME/LINC_HOME/LINC_PROJECTS_DIR inside the
    error text (e.g. a stdlib FileNotFoundError) never reaches the redacted log."""
    prefix = str(tmp_path / f"hostroot-{env_var.lower()}")
    monkeypatch.setenv(env_var, prefix)
    log_file = tmp_path / "out" / "calls.jsonl"
    error = f"[Errno 2] No such file or directory: '{prefix}/proj/build.gradle'"
    _log(log_file, {"project": "demo"}, error=error)

    text = log_file.read_text()
    assert prefix not in text
    assert json.loads(text)["error"].endswith("/proj/build.gradle'")


def test_full_mode_error_keeps_host_paths(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """full mode keeps the error verbatim, host paths included."""
    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "full")
    prefix = str(tmp_path / "hostroot")
    monkeypatch.setenv("LINC_PROJECTS_DIR", prefix)
    log_file = tmp_path / "out" / "calls.jsonl"
    _log(log_file, {}, error=f"missing {prefix}/x")
    assert json.loads(log_file.read_text())["error"] == f"missing {prefix}/x"


@pytest.mark.parametrize("trivial", ["", "/"])
def test_trivial_host_prefix_does_not_mangle_error(
    tmp_path: Path, monkeypatch: MonkeyPatch, trivial: str
) -> None:
    """[contract] an empty or root LINC_PROJECTS_DIR is not treated as a secret prefix."""
    monkeypatch.setenv("LINC_PROJECTS_DIR", trivial)
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {}, error="a/b c")
    assert json.loads(log_file.read_text())["error"] == "a/b c"


@pytest.mark.parametrize("suffix", ["/", ""])
def test_host_prefix_scrub_is_exact(tmp_path: Path, monkeypatch: MonkeyPatch, suffix: str) -> None:
    """[contract] a host root (trailing slash or not) is replaced, keeping the relative tail."""
    from linc_crucible.core.observability import _redacted

    prefix = str(tmp_path / "projects")
    monkeypatch.setenv("LINC_PROJECTS_DIR", prefix + suffix)
    log_file = tmp_path / "out" / "calls.jsonl"
    _log(log_file, {}, error=f"missing {prefix}/x")
    assert json.loads(log_file.read_text())["error"] == f"missing {_redacted(prefix)}/x"


def test_two_char_host_prefix_is_scrubbed(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """A two-character root (e.g. ``/p``) is still a real path prefix and is scrubbed."""
    monkeypatch.setenv("LINC_PROJECTS_DIR", "/p")
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {}, error="missing /p/x")
    assert json.loads(log_file.read_text())["error"] == "missing <redacted len=2>/x"


def test_single_char_host_prefix_is_ignored(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """[contract] a one-character LINC_PROJECTS_DIR (e.g. ``.``) never mangles error text."""
    monkeypatch.setenv("LINC_PROJECTS_DIR", ".")
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {}, error="a.b c")
    assert json.loads(log_file.read_text())["error"] == "a.b c"


def test_scrub_replaces_longest_value_first(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """[contract] a short redacted value inside a host root cannot split the root and leak it."""
    home = tmp_path / "hzz9h"
    monkeypatch.setenv("HOME", str(home))
    log_file = tmp_path / "out" / "calls.jsonl"
    _log(log_file, {"device": "zz9"}, error=f"open {home}/f on zz9")
    text = log_file.read_text()
    assert str(tmp_path) not in text
    assert json.loads(text)["error"].endswith("/f on <redacted len=3>")


def test_string_at_max_length_is_not_truncated(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """A string of exactly 500 chars is logged verbatim (truncation starts above it)."""
    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "full")
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {"note": "n" * 500})
    assert json.loads(log_file.read_text())["params"]["note"] == "n" * 500


def test_redacted_error_keeps_unredacted_param_values(tmp_path: Path) -> None:
    """Only redacted values are scrubbed from the error; a project name stays readable."""
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, {"project": "demo"}, error="project demo failed")
    assert json.loads(log_file.read_text())["error"] == "project demo failed"


def test_no_rotation_at_exactly_max_size(tmp_path: Path) -> None:
    """A log of exactly max_size_bytes is appended to, not rotated (rotation is strictly >)."""
    from linc_crucible.core.observability import log_mcp_call

    log_file = tmp_path / "calls.jsonl"
    log_file.write_text("x" * 49 + "\n")
    log_mcp_call(
        "linc-crucible", "crucible_pour", {}, 1.0, True, log_path=log_file, max_size_bytes=50
    )
    assert list(tmp_path.glob("calls.*.jsonl")) == []
    assert log_file.read_text().startswith("x" * 49 + "\n")


@_POSIX_ONLY
def test_fresh_log_is_created_0600_before_any_chmod(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """[contract] a new log is born 0600 by ``os.open`` itself — no world-readable window
    exists before the follow-up ``fchmod`` (stubbed out here) tightens it."""
    from linc_crucible.core import observability

    monkeypatch.setattr(observability.os, "fchmod", lambda _fd, _mode: None)
    log_file = tmp_path / "calls.jsonl"
    old = os.umask(0o000)
    try:
        _log(log_file, {})
    finally:
        os.umask(old)
    assert _mode(log_file) == 0o600


def test_unset_host_roots_never_scrub_placeholder_text(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """[contract] an unset LINC_HOME/LINC_PROJECTS_DIR contributes no scrub prefix at all."""
    from linc_crucible.core.observability import _host_prefixes

    monkeypatch.delenv("LINC_HOME", raising=False)
    monkeypatch.delenv("LINC_PROJECTS_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _host_prefixes() == [str(tmp_path / "home")]


@pytest.mark.parametrize("root", ["/srv/linX", "/srv/linX/", "/srv/linX//"])
def test_host_root_keeps_trailing_non_slash_chars(
    monkeypatch: MonkeyPatch, tmp_path: Path, root: str
) -> None:
    """[contract] only trailing path separators are dropped from a host root."""
    from linc_crucible.core.observability import _host_prefixes

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LINC_HOME", root)
    monkeypatch.delenv("LINC_PROJECTS_DIR", raising=False)
    assert _host_prefixes() == [str(tmp_path), "/srv/linX"]


def test_redacted_strings_top_level_is_not_redacted() -> None:
    """[contract] only values under a redacted key are collected — never the whole params."""
    from linc_crucible.core.observability import _redacted_strings

    assert _redacted_strings({"project": "demo", "device": "zz9"}) == ["zz9"]
    assert _redacted_strings(["a", {"serial": "s1"}]) == ["s1"]
    assert _redacted_strings("loose") == []


def test_redacted_strings_collects_only_nonempty_string_leaves_under_redacted_key() -> None:
    """[contract] under a redacted key every nested non-empty str is collected; numbers,
    ``None`` and empty strings never become scrub targets (``""`` would mangle all text)."""
    from linc_crucible.core.observability import _redacted_strings

    params = {"devices": ["a1", "", 7, None, ("b2",), {"inner": "c3"}], "mode": "quick"}
    assert _redacted_strings(params) == ["a1", "b2", "c3"]


def test_redacted_error_scrubs_redacted_key_nested_under_plain_key(tmp_path: Path) -> None:
    """[contract] a serial nested under an allowlisted key is redacted in params AND scrubbed
    from the error text — the scrub walk descends into non-redacted containers too."""
    from linc_crucible.core.observability import _redacted_strings

    params = {"action": {"device": "zz9"}, "mode": [{"serial": "s1"}]}
    assert _redacted_strings(params) == ["zz9", "s1"]
    log_file = tmp_path / "calls.jsonl"
    _log(log_file, params, error="zz9 and s1 offline")
    entry = json.loads(log_file.read_text())
    assert entry["params"]["action"] == {"device": "<redacted len=3>"}
    assert entry["error"] == "<redacted len=3> and <redacted len=2> offline"


# --- R4 Low 7 + Info NaN + Info LINC_HOME 0755 -----------------------------------------

_EXPECTED_UNREDACTED_KEYS = frozenset(
    {
        "project",
        "mode",
        "action",
        "platform",
        "device_type",
        "perf",
        "a11y",
        "a11y_tree",
        "history",
        "limit",
        "keep",
        "build_only",
        "skip_build",
        "launch",
        "build_type",
        "allow_personal",
        "allow_empty",
        "since_hours",
        "top_n",
        "max_assets",
        "screen_name",
    }
)


def test_every_registered_tool_param_is_classified_and_honours_allowlist(tmp_path: Path) -> None:
    """[contract] every registered MCP tool parameter is in _UNREDACTED_KEYS or _REDACT_KEYS;
    allowlisted parameters log unredacted, non-allowlisted and unknown parameters are redacted."""
    import inspect

    from linc_crucible.core.observability import _REDACT_KEYS, _UNREDACTED_KEYS
    from linc_crucible.mcp_server import server

    assert _UNREDACTED_KEYS == _EXPECTED_UNREDACTED_KEYS
    assert _UNREDACTED_KEYS.isdisjoint(_REDACT_KEYS)

    tool_params = {
        param
        for tool in server._tool_manager.list_tools()
        for param in inspect.signature(tool.fn).parameters
    }
    unclassified = tool_params - (_UNREDACTED_KEYS | _REDACT_KEYS)
    assert not unclassified, f"unclassified tool parameters: {sorted(unclassified)}"

    log_file = tmp_path / "calls.jsonl"
    sample = {param: f"val-{param}" for param in sorted(tool_params)}
    sample["unknown_future_param"] = "secret-token"
    sample["budget_s"] = 120
    sample["compare_against_baseline"] = True
    _log(log_file, sample)

    logged = json.loads(log_file.read_text())["params"]
    for param in sorted(tool_params):
        if param in _UNREDACTED_KEYS:
            assert logged[param] == f"val-{param}", f"allowlisted {param!r} was redacted"
        else:
            expected_len = len(str(sample[param]))
            assert logged[param] == f"<redacted len={expected_len}>", (
                f"non-allowlisted {param!r} was not redacted: {logged[param]!r}"
            )
    assert logged["unknown_future_param"] == f"<redacted len={len('secret-token')}>"


def test_redacted_mode_scrubs_device_info_and_outside_root_error_paths(tmp_path: Path) -> None:
    """[contract] in redacted mode, a serial inside device_info and an absolute path outside
    every known root inside error (POSIX and Windows drive paths) never appear in the log file."""
    log_file = tmp_path / "calls.jsonl"
    serial = "R58N000TEST"
    posix_secret = "/private/var/x/secret-dir/y.png"
    win_secret = r"C:\Users\alice\secret-dir\y.png"
    _log(
        log_file,
        {"project": "demo", "device_info": f"Pixel {serial}"},
        error=f"cannot open '{posix_secret}' or {win_secret} on Pixel {serial}",
    )

    raw = log_file.read_text()
    assert serial not in raw
    assert posix_secret not in raw
    assert win_secret not in raw
    assert "secret-dir" not in raw
    assert "alice" not in raw
    entry = json.loads(raw)
    assert entry["params"] == {"project": "demo", "device_info": "<redacted len=17>"}
    assert entry["error"] == "cannot open '<path>' or <path> on <redacted len=17>"


def test_non_finite_float_params_produce_strict_json(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """[contract] NaN/Infinity/-Infinity parameters produce valid JSON lines that parse with
    strict json.loads(..., parse_constant=...)."""
    from linc_crucible.core.observability import log_mcp_call

    def _forbid_non_finite(token: str) -> None:
        raise ValueError(f"non-standard JSON constant: {token}")

    log_file = tmp_path / "calls.jsonl"
    log_mcp_call(
        "linc-crucible",
        "crucible_usage_stats",
        {
            "since_hours": float("nan"),
            "action": {"limit": float("inf"), "keep": float("-inf"), "top_n": 1.5},
            "mode": [float("nan"), float("inf"), float("-inf")],
        },
        float("nan"),
        True,
        log_path=log_file,
    )
    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", "full")
    log_mcp_call(
        "linc-crucible",
        "crucible_bvt",
        {"budget_s": float("inf"), "source_score": float("-inf"), "nan_val": float("nan")},
        float("inf"),
        True,
        log_path=log_file,
    )

    lines = log_file.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0], parse_constant=_forbid_non_finite)
    assert first["params"]["since_hours"] == "NaN"
    assert first["params"]["action"] == {"limit": "Infinity", "keep": "-Infinity", "top_n": 1.5}
    assert first["params"]["mode"] == ["NaN", "Infinity", "-Infinity"]
    assert first["duration_ms"] == "NaN"

    second = json.loads(lines[1], parse_constant=_forbid_non_finite)
    assert second["params"] == {
        "budget_s": "Infinity",
        "source_score": "-Infinity",
        "nan_val": "NaN",
    }
    assert second["duration_ms"] == "Infinity"

    log_mcp_call("linc-crucible", "crucible_bvt", {}, 1.24, True, log_path=log_file)
    third = json.loads(log_file.read_text().splitlines()[2], parse_constant=_forbid_non_finite)
    assert third["duration_ms"] == 1.2


def test_write_log_entry_enforces_strict_json_and_default_str(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """_write_log_entry uses default=str and rejects unnormalized NaN."""
    from linc_crucible.core import observability

    log_file = tmp_path / "strict.jsonl"
    observability._write_log_entry(log_file, {"path_obj": Path("/tmp/demo")}, 1024)
    assert json.loads(log_file.read_text()) == {"path_obj": "/tmp/demo"}

    monkeypatch.setattr(observability, "_finite", lambda value: value)
    with pytest.raises(ValueError):
        observability._write_log_entry(log_file, {"bad": float("nan")}, 1024)


@_POSIX_ONLY
def test_fresh_linc_home_created_0700_and_preexisting_0755_preserved(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """[contract] a fresh LINC_HOME is created 0700 when writing mcp-calls.jsonl, while a
    pre-existing 0755 LINC_HOME keeps its 0755 permissions."""
    from linc_crucible.core.observability import log_mcp_call

    fresh_home = tmp_path / "fresh" / "h"
    monkeypatch.setenv("LINC_HOME", str(fresh_home))
    old = os.umask(0o022)
    try:
        log_mcp_call("linc-crucible", "crucible_score", {"project": "demo"}, 1.0, True)
    finally:
        os.umask(old)
    assert _mode(fresh_home) == 0o700
    assert _mode(fresh_home / "mcp-calls.jsonl") == 0o600

    existing_home = tmp_path / "existing"
    existing_home.mkdir(mode=0o755)
    os.chmod(existing_home, 0o755)
    monkeypatch.setenv("LINC_HOME", str(existing_home))
    log_mcp_call("linc-crucible", "crucible_score", {"project": "demo"}, 1.0, True)
    assert _mode(existing_home) == 0o755
    assert _mode(existing_home / "mcp-calls.jsonl") == 0o600
