"""`crucible fuelcheck` hardening (REVIEW B4b round 3).

* stderr is as paste-safe as stdout: the warnings the real probes log (device-policy absent,
  duplicate project manifests) never carry the absolute home dir;
* ``redact_home`` is an identity for a relative or one-character HOME;
* registries are read with a size cap: an oversized file is rejected, never parsed.

There is deliberately no autouse ``capture_logs`` here: the leak lives in what structlog
renders to stderr, so these tests render it.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import structlog
import structlog.testing
from click.testing import CliRunner

from linc_crucible import _redaction
from linc_crucible import fuelcheck as fc
from linc_crucible._bounded_read import RegistryTooLargeError, read_capped
from linc_crucible.cli import cli
from linc_crucible.core import device_policy, forge_integration

ONE_DEVICE = fc.Probe("fleet", "ok", "1 device(s) in fleet registry")


class _LiveStderr:
    """structlog logger writing to whatever ``sys.stderr`` is at call time (CliRunner's)."""

    def msg(self, message: str) -> None:
        sys.stderr.write(message + "\n")

    debug = info = warning = warn = error = critical = exception = fatal = log = msg


@pytest.fixture
def rendered_logs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Production wiring: loggers bound ONCE (as at import), WARNING filter, console renderer.

    Binding eagerly matters: an import-time-bound logger keeps the processor list object and
    the wrapper class it was bound with, so a fix that reassigns structlog's config (instead
    of editing the live processor chain) passes a lazily-bound rig and still leaks for real.
    """
    old = structlog.get_config()
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=False),
        ]
    )
    try:
        for module in (device_policy, forge_integration):
            bound = structlog.wrap_logger(
                _LiveStderr(), wrapper_class=structlog.make_filtering_bound_logger("warning")
            ).bind(domain=module.__name__)
            monkeypatch.setattr(module, "logger", bound)
        monkeypatch.setattr(device_policy, "_absent_warned", False)
        yield
    finally:
        structlog.configure(**old)


def _stranger_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """A fresh $HOME: no registry, two manifests claiming one project name."""
    for name in ("app", "app-copy"):
        (home / "projects" / name).mkdir(parents=True)
        (home / "projects" / name / "linc-manifest.yaml").write_text("identity:\n  name: app\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LINC_HOME", str(home / ".linc"))
    monkeypatch.setenv("LINC_PROJECTS_DIR", str(home / "projects"))
    for var in ("LINC_DEVICE_FLEET_FILE", "FORGE_PROJECTS_FILE", "CRUCIBLE_FLEET_FILE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def absent_env(tmp_path: Path) -> fc.ProbeEnv:
    """Every probe faked absent; nothing touches the host."""
    return fc.ProbeEnv(
        which=lambda _name: None,
        check_install=lambda: None,
        linc_workspace=lambda: tmp_path / "no-workspace",
        list_projects=list,
        fleet_file=lambda: tmp_path / "no-fleet.yaml",
        load_policy=lambda: device_policy.DevicePolicy(frozenset(), state="absent", source="x"),
        claude_json=lambda: None,
        docker_ps=lambda: "",
        nerve_up=lambda: False,
        home=lambda: tmp_path,
    )


_REAL_PROBE_ENV = fc.default_probe_env


def _offline_real_env() -> fc.ProbeEnv:
    """The real probe env with only the network, PATH and docker faked out."""
    return replace(
        _REAL_PROBE_ENV(),
        which=lambda _name: None,
        nerve_up=lambda: False,
        docker_ps=lambda: "",
    )


def test_rig_sees_the_absolute_home_path_outside_fuelcheck(
    rendered_logs: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control: the rig is not vacuous — the raw device-policy warning names the home dir."""
    _stranger_home(monkeypatch, tmp_path)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        device_policy.load_device_policy()
    assert f"{tmp_path}/.linc/device-fleet.yaml" in err.getvalue()


@pytest.mark.parametrize("args", [(), ("--json",)])
def test_fuelcheck_stderr_never_carries_the_home_dir(
    rendered_logs: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: tuple[str, ...]
) -> None:
    """[contract] real load_device_policy + real project discovery, absent $LINC_HOME."""
    _stranger_home(monkeypatch, tmp_path)
    with patch.object(fc, "default_probe_env", _offline_real_env):
        result = CliRunner().invoke(cli, ["fuelcheck", *args])
    assert result.exit_code == 0, result.output
    assert str(tmp_path) not in result.stdout
    assert str(tmp_path) not in result.stderr
    assert "device_fleet_absent" in result.stderr
    assert "~/.linc/device-fleet.yaml" in result.stderr
    assert "forge_manifest_duplicate_name_kept_existing" in result.stderr
    assert "~/projects/app-copy" in result.stderr


def test_run_probes_restores_the_live_processor_chain(absent_env: fc.ProbeEnv) -> None:
    """[contract] the redaction step is removed again, in place, even when a probe raises."""
    chain = structlog.get_config()["processors"]
    before = list(chain)
    fc.run_probes(absent_env)
    assert structlog.get_config()["processors"] is chain
    assert chain == before

    def boom(_name: str) -> str | None:
        raise RuntimeError("probe crashed")

    with pytest.raises(RuntimeError, match="probe crashed"):
        fc.run_probes(replace(absent_env, which=boom))
    assert chain == before


def test_redacted_logs_redacts_events_logged_inside_it(tmp_path: Path) -> None:
    with structlog.testing.capture_logs() as logs, fc.redacted_logs(tmp_path):
        structlog.get_logger().warning("e", path=tmp_path / "x", note=f"at {tmp_path}")
    assert logs == [{"event": "e", "log_level": "warning", "path": "~/x", "note": "at ~"}]


def test_redacted_logs_without_a_home_passes_events_through() -> None:
    with structlog.testing.capture_logs() as logs, fc.redacted_logs(None):
        structlog.get_logger().warning("e", path="/h/u/x")
    assert logs == [{"event": "e", "log_level": "warning", "path": "/h/u/x"}]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/h/u/a", "~/a"),
        (Path("/h/u/b"), "~/b"),
        (["/h/u/c", 1, None], ["~/c", 1, None]),
        (("/h/u/d",), ["~/d"]),
        ({"k": "/h/u/e", "n": 2.5}, {"k": "~/e", "n": 2.5}),
        (42, 42),
        (True, True),
    ],
)
def test_redact_value_walks_strings_paths_and_containers(value: Any, expected: Any) -> None:
    assert _redaction._redact_value(value, Path("/h/u")) == expected


def test_home_redactor_formats_and_redacts_exc_info() -> None:
    """A logged traceback names files under the home dir: it is formatted, then redacted."""
    redact = _redaction._home_redactor(Path("/h/u"))
    try:
        raise OSError("cannot open /h/u/secret.yaml")
    except OSError:
        event = redact(None, "error", {"event": "x", "exc_info": True})
    assert "exc_info" not in event
    assert "OSError: cannot open ~/secret.yaml" in event["traceback"]
    assert "/h/u/" not in event["traceback"]


# --- redact_home with an unusable HOME ---------------------------------------------------


@pytest.mark.parametrize("home", [Path("."), Path("s"), Path(""), Path("rel/home")])
@pytest.mark.parametrize("text", ["e.g. this. /x/y", "s/s ss", "rel/home/x"])
def test_redact_home_with_relative_or_short_home_is_identity(home: Path, text: str) -> None:
    """[contract] HOME=. or a one-character/relative HOME must not mangle the text."""
    assert fc.redact_home(text, home) == text


# --- registry size cap -------------------------------------------------------------------


def _sparse(path: Path, size: int) -> Path:
    with path.open("wb") as fh:
        fh.truncate(size)
    return path


def test_oversized_fleet_registry_fails_without_parsing(tmp_path: Path) -> None:
    """[contract] a fleet registry over the cap is rejected naming size and cap, unparsed."""
    big = _sparse(tmp_path / "fleet.yaml", fc._MAX_REGISTRY_BYTES + 1)
    with patch.object(fc.yaml, "safe_load", side_effect=AssertionError("parsed")):
        probe = fc.probe_fleet(lambda: big)
    assert probe == fc.Probe(
        "fleet",
        "fail",
        f"fleet registry {big} is too large: {fc._MAX_REGISTRY_BYTES + 1} bytes, "
        f"over the {fc._MAX_REGISTRY_BYTES}-byte cap",
    )


def test_fleet_registry_at_the_cap_is_parsed(tmp_path: Path) -> None:
    body = b"fleet:\n  devices:\n    - name: a\n"
    fleet = tmp_path / "fleet.yaml"
    fleet.write_bytes(body + b"#" * (fc._MAX_REGISTRY_BYTES - len(body)))
    assert fc.probe_fleet(lambda: fleet) == fc.Probe(
        "fleet", "ok", "1 device(s) in fleet registry"
    )


def test_oversized_claude_json_warns_without_parsing(tmp_path: Path) -> None:
    """[contract] ~/.claude.json over its cap is a warn (the client owns it), never parsed."""
    big = _sparse(tmp_path / "claude.json", fc._MAX_CLAUDE_JSON_BYTES + 1)
    with patch.object(fc.json, "loads", side_effect=AssertionError("parsed")):
        probe = fc.probe_mcp_config(lambda: big)
    assert probe == fc.Probe(
        "mcp-config",
        "warn",
        f"~/.claude.json is too large: {fc._MAX_CLAUDE_JSON_BYTES + 1} bytes, "
        f"over the {fc._MAX_CLAUDE_JSON_BYTES}-byte cap",
    )


def test_read_capped_reads_at_most_cap_plus_one_bytes(tmp_path: Path) -> None:
    """Growth after stat cannot defeat the cap: the read itself is bounded."""
    path = tmp_path / "f"
    path.write_bytes(b"abcdef")
    assert read_capped(path, 6) == b"abcdef"
    with pytest.raises(RegistryTooLargeError, match="6 bytes, over the 5-byte cap"):
        read_capped(path, 5)


def test_registry_caps_are_pinned() -> None:
    from linc_crucible._bounded_read import _MAX_REGISTRY_BYTES
    from linc_crucible.core import fleet_registry

    assert _MAX_REGISTRY_BYTES == 1024 * 1024
    assert fc._MAX_REGISTRY_BYTES == _MAX_REGISTRY_BYTES
    assert device_policy._MAX_REGISTRY_BYTES == _MAX_REGISTRY_BYTES
    assert fleet_registry._MAX_REGISTRY_BYTES == _MAX_REGISTRY_BYTES
    assert fc._MAX_CLAUDE_JSON_BYTES == 16 * 1024 * 1024


def test_oversized_device_policy_probe_reports_too_large(tmp_path: Path) -> None:
    """[test] fuelcheck's device-policy probe reports 'is too large' for an oversized registry."""
    big = _sparse(tmp_path / "device-fleet.yaml", fc._MAX_REGISTRY_BYTES + 1)
    probe = fc.probe_device_policy(lambda: device_policy.load_device_policy(big))
    assert probe == fc.Probe(
        "device-policy",
        "fail",
        f"unreadable: personal-device registry {big} could not be read "
        f"(is too large: {fc._MAX_REGISTRY_BYTES + 1} bytes, "
        f"over the {fc._MAX_REGISTRY_BYTES}-byte cap) — "
        "every device operation is refused unless allow_personal=True",
    )


def test_utf16_bom_fleet_registry_parses(tmp_path: Path) -> None:
    """PyYAML honours a BOM: a UTF-16 registry is read, only undecodable bytes are ``fail``."""
    fleet = tmp_path / "fleet.yaml"
    fleet.write_bytes("fleet:\n  devices:\n    - name: a\n".encode("utf-16"))
    assert fc.probe_fleet(lambda: fleet) == fc.Probe(
        "fleet", "ok", "1 device(s) in fleet registry"
    )
