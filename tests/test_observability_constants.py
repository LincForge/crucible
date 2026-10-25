"""[contract] REVIEW B8: the usage-log policy tables in ``core.observability`` are pinned.

Each module-level policy constant is exercised through the public logger so a change to
the table (a dropped redact key, a looser file mode, a new mode name) fails here.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from linc_crucible.core import observability
from linc_crucible.core.observability import (
    _DEFAULT_MODE,
    _EXPLICIT_MODES,
    _HOST_PREFIX_VARS,
    _LOG_FILE_MODE,
    _MAX_STR_LEN,
    _PATH_KEY_SUFFIXES,
    _REDACT_KEYS,
    _RUN_ID_RE,
    log_mcp_call,
    usage_log_mode,
)


@pytest.fixture(autouse=True)
def _clean_usage_env(monkeypatch: MonkeyPatch) -> None:
    """Never inherit the host's usage-log mode or run id into these tests."""
    for var in ("CRUCIBLE_USAGE_LOG", "CRUCIBLE_RUN_ID", "FORGE_RUN_ID"):
        monkeypatch.delenv(var, raising=False)


def _entry(tmp_path: Path, params: dict, error: str | None = None) -> dict:
    log_file = tmp_path / "calls.jsonl"
    log_mcp_call("linc-crucible", "crucible_pour", params, 1.0, error is None, error, log_file)
    return json.loads(log_file.read_text().splitlines()[-1])


def test_log_file_mode_is_owner_read_write_only() -> None:
    """[contract] the create/fchmod mode is exactly 0600 — no group or other bits."""
    assert _LOG_FILE_MODE == 0o600
    assert _LOG_FILE_MODE == stat.S_IRUSR | stat.S_IWUSR


def test_mode_names() -> None:
    """[contract] redacted is the default; only off and full are explicit opt-outs."""
    assert _DEFAULT_MODE == "redacted"
    assert _EXPLICIT_MODES == ("off", "full")
    assert usage_log_mode() == _DEFAULT_MODE


@pytest.mark.parametrize("mode", _EXPLICIT_MODES)
def test_each_explicit_mode_is_selectable(monkeypatch: MonkeyPatch, mode: str) -> None:
    """[contract] every explicit mode name round-trips through ``CRUCIBLE_USAGE_LOG``."""
    monkeypatch.setenv("CRUCIBLE_USAGE_LOG", mode.upper())
    assert usage_log_mode() == mode


def test_redact_keys_cover_serials_paths_and_credentials() -> None:
    """[contract] the redact and unredacted tables partition registered tool parameters."""
    unredacted = observability._UNREDACTED_KEYS
    assert {
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
    } == unredacted
    assert {
        "api_key",
        "app_path",
        "app_version",
        "budget_s",
        "compare_against_baseline",
        "device",
        "device_info",
        "device_serial",
        "devices",
        "exclude_tags",
        "image_path",
        "include_rotated",
        "launch_intent",
        "module",
        "output_dir",
        "password",
        "project_dir",
        "secret",
        "serial",
        "server_filter",
        "source_score",
        "suite_dir",
        "tags",
        "token",
        "udid",
        "video_path",
    } == _REDACT_KEYS
    assert unredacted.isdisjoint(_REDACT_KEYS)


@pytest.mark.parametrize("key", sorted(_REDACT_KEYS))
def test_every_redact_key_is_redacted(tmp_path: Path, key: str) -> None:
    """[contract] each key in the redact table is logged length-only."""
    assert _entry(tmp_path, {key: "abc"})["params"] == {key: "<redacted len=3>"}


@pytest.mark.parametrize("key", sorted(getattr(observability, "_UNREDACTED_KEYS", ("project",))))
def test_every_unredacted_key_is_kept(tmp_path: Path, key: str) -> None:
    """[contract] each key in the allowlist table is logged unredacted."""
    assert _entry(tmp_path, {key: "abc"})["params"] == {key: "abc"}


@pytest.mark.parametrize("suffix", _PATH_KEY_SUFFIXES)
def test_every_path_key_suffix_is_redacted(tmp_path: Path, suffix: str) -> None:
    """[contract] any key ending in a host-path suffix is logged length-only."""
    assert _PATH_KEY_SUFFIXES == ("_path", "_dir", "file", "video")
    key = f"out{suffix}"
    assert _entry(tmp_path, {key: "/x/y"})["params"] == {key: "<redacted len=4>"}


def test_long_unredacted_string_truncated_at_max_len(tmp_path: Path) -> None:
    """[contract] an unredacted string longer than _MAX_STR_LEN keeps that many chars + '...'."""
    assert _MAX_STR_LEN == 500
    logged = _entry(tmp_path, {"project": "n" * (_MAX_STR_LEN + 1)})["params"]["project"]
    assert logged == "n" * _MAX_STR_LEN + "..."


@pytest.mark.parametrize(
    ("run_id", "ok"),
    [("ci-1.a_B", True), ("x" * 128, True), ("x" * 129, False), ("a/b", False), ("a\nb", False)],
)
def test_run_id_pattern(run_id: str, ok: bool) -> None:
    """[contract] a run id is one short token of ``[A-Za-z0-9._-]``."""
    assert (_RUN_ID_RE.fullmatch(run_id) is not None) is ok


@pytest.mark.parametrize("var", _HOST_PREFIX_VARS)
def test_every_host_prefix_var_is_scrubbed(
    tmp_path: Path, monkeypatch: MonkeyPatch, var: str
) -> None:
    """[contract] each host-root env var in the table is scrubbed from redacted errors."""
    assert _HOST_PREFIX_VARS == ("LINC_HOME", "LINC_PROJECTS_DIR")
    for other in _HOST_PREFIX_VARS:
        monkeypatch.delenv(other, raising=False)
    root = str(tmp_path / "root")
    monkeypatch.setenv(var, root)
    error = _entry(tmp_path, {}, error=f"open {root}/f")["error"]
    assert error == f"open {observability._redacted(root)}/f"
