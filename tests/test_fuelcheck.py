"""Stranger-safe `crucible fuelcheck` (REVIEW B4b).

Every probe takes injectable callables, so these tests never touch the network, docker or the
real home dir: a host with no LINC workspace, no nerve daemon and no docker exits 0 with only
ok/warn/skip lines; only a broken install or a present-but-unparsable config file is `fail`.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import structlog
import structlog.testing
from click.testing import CliRunner

from linc_crucible import fuelcheck as fc
from linc_crucible.cli import cli
from linc_crucible.config_contract import render_env_table
from linc_crucible.core.device_policy import DevicePolicy

NERVE_HINT = (
    "nerve daemon not running — BVT, baseline capture and pour need it: "
    "pipx install linc-nerve && nerve daemon"
)


@pytest.fixture(autouse=True)
def _capture_structlog() -> Iterator[None]:
    """Record structlog events instead of printing them to an import-time-bound stream.

    Importing ``linc_crucible.cli`` points structlog at the then-current ``sys.stderr``, a
    pytest capture stream that a first in-process ``pytest.main()`` session closes; the
    device-policy warnings real ``load_device_policy`` calls emit here must not reach it.
    """
    with structlog.testing.capture_logs():
        yield


def _absent_policy() -> DevicePolicy:
    return DevicePolicy(frozenset(), state="absent", source="x", error="file not found: x")


def _no_docker() -> str:
    raise AssertionError("docker must not be queried when it is not installed")


def _absent_env(tmp_path: Path) -> fc.ProbeEnv:
    """Everything a stranger's laptop lacks: no tools, workspace, projects, fleet, daemon."""
    return fc.ProbeEnv(
        which=lambda _name: None,
        check_install=lambda: None,
        linc_workspace=lambda: tmp_path / "no-workspace",
        list_projects=list,
        fleet_file=lambda: tmp_path / "no-fleet.yaml",
        load_policy=_absent_policy,
        claude_json=lambda: tmp_path / "no-claude.json",
        docker_ps=_no_docker,
        nerve_up=lambda: False,
        home=lambda: None,
    )


@pytest.fixture
def absent(tmp_path: Path) -> fc.ProbeEnv:
    return _absent_env(tmp_path)


def _invoke(env: fc.ProbeEnv, *args: str) -> Any:
    with patch.object(fc, "default_probe_env", return_value=env):
        return CliRunner().invoke(cli, ["fuelcheck", *args])


def _by_name(probes: list[fc.Probe]) -> dict[str, fc.Probe]:
    return {p.name: p for p in probes}


# --- CLI ---------------------------------------------------------------------------------


def test_cli_all_absent_exits_zero_with_only_ok_warn_skip_lines(absent: fc.ProbeEnv) -> None:
    result = _invoke(absent)
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == len(fc.run_probes(absent))
    for line in lines:
        assert line.split()[0] in {"ok", "warn", "skip"}, line
    assert NERVE_HINT in result.stdout


def test_cli_json_lists_every_probe_with_a_status(absent: fc.ProbeEnv) -> None:
    result = _invoke(absent, "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [p["name"] for p in payload] == [p.name for p in fc.run_probes(absent)]
    for item in payload:
        assert set(item) == {"name", "status", "detail"}
        assert item["status"] in {"ok", "warn", "skip"}


def test_cli_one_failed_probe_exits_one(absent: fc.ProbeEnv) -> None:
    broken = replace(absent, check_install=lambda: "No module named 'fastmcp'")
    result = _invoke(broken)
    assert result.exit_code == 1
    assert "fail  install: broken install: No module named 'fastmcp'" in result.stdout


def test_cli_json_failed_probe_exits_one(absent: fc.ProbeEnv) -> None:
    broken = replace(absent, check_install=lambda: "boom")
    result = _invoke(broken, "--json")
    assert result.exit_code == 1
    assert _by_name([fc.Probe(**p) for p in json.loads(result.stdout)])["install"].status == "fail"


def test_cli_env_table_prints_the_pinned_table() -> None:
    result = CliRunner().invoke(cli, ["fuelcheck", "--env-table"])
    assert result.exit_code == 0
    assert result.stdout == render_env_table() + "\n"


def test_env_table_option_is_hidden() -> None:
    result = CliRunner().invoke(cli, ["fuelcheck", "--help"])
    assert "--env-table" not in result.stdout
    assert "--json" in result.stdout


# --- Probe model -------------------------------------------------------------------------


def test_probe_rejects_unknown_status() -> None:
    """[contract] a probe status outside ok/warn/skip/fail is a programming error."""
    with pytest.raises(ValueError, match="unknown probe status 'bad'"):
        fc.Probe("x", "bad", "d")


def test_probe_to_dict() -> None:
    assert fc.Probe("n", "ok", "d").to_dict() == {"name": "n", "status": "ok", "detail": "d"}


def test_exit_code() -> None:
    assert fc.exit_code([fc.Probe("a", "ok", ""), fc.Probe("b", "warn", "")]) == 0
    assert fc.exit_code([fc.Probe("a", "skip", ""), fc.Probe("b", "fail", "")]) == 1
    assert fc.exit_code([]) == 0


def test_format_line_pads_status_plain_text() -> None:
    assert fc.format_line(fc.Probe("n", "warn", "[ai] extra")) == "warn  n: [ai] extra"
    assert fc.format_line(fc.Probe("n", "ok", "d")) == "ok    n: d"
    assert fc.format_line(fc.Probe("n", "skip", "d")) == "skip  n: d"
    assert fc.format_line(fc.Probe("n", "fail", "d")) == "fail  n: d"


# --- Individual probes -------------------------------------------------------------------


def test_absent_statuses_are_pinned(absent: fc.ProbeEnv) -> None:
    got = {p.name: p.status for p in fc.run_probes(absent)}
    assert got == {
        "install": "ok",
        "tool:adb": "skip",
        "tool:xcrun": "skip",
        "tool:ruff": "skip",
        "tool:pytest": "skip",
        "linc-workspace": "skip",
        "projects": "warn",
        "fleet": "warn",
        "device-policy": "warn",
        "mcp-config": "skip",
        "nerve": "warn",
        "mobsf": "skip",
    }


def test_absent_hints_name_the_env_var_or_command(absent: fc.ProbeEnv) -> None:
    got = _by_name(fc.run_probes(absent))
    assert got["nerve"].detail == NERVE_HINT
    assert "LINC_WORKSPACE" in got["linc-workspace"].detail
    assert (
        "preflight reads store-listing.md and brand-brief.json from <project>/store/"
        " (or LINC_WORKSPACE); missing ones score as fail"
    ) in got["linc-workspace"].detail
    assert "LINC_PROJECTS_DIR" in got["projects"].detail
    assert "crucible fleet discover" in got["fleet"].detail
    assert "CRUCIBLE_FLEET_FILE" in got["fleet"].detail
    assert "LINC_DEVICE_FLEET_FILE" in got["device-policy"].detail
    assert "crucible bellows" in got["mcp-config"].detail
    assert got["mobsf"].detail == (
        "docker not found — only needed for assay mode=full (MobSF security scan)"
    )
    assert got["tool:adb"].detail.startswith("adb not found — ")


def test_present_statuses_are_ok(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    fleet = tmp_path / "fleet.yaml"
    fleet.write_text("fleet:\n  devices:\n    - serial: A\n    - serial: B\n")
    claude = tmp_path / "claude.json"
    claude.write_text(json.dumps({"mcpServers": {"linc-crucible": {"command": "crucible"}}}))
    env = replace(
        _absent_env(tmp_path),
        which=lambda name: f"/bin/{name}",
        linc_workspace=lambda: ws,
        list_projects=lambda: ["p1", "p2", "p3"],
        fleet_file=lambda: fleet,
        load_policy=lambda: DevicePolicy(frozenset({"S"}), state="loaded", source="f"),
        claude_json=lambda: claude,
        docker_ps=lambda: "Up 3 hours",
        nerve_up=lambda: True,
    )
    got = _by_name(fc.run_probes(env))
    assert {p.status for p in got.values()} == {"ok"}
    assert got["projects"].detail == "3 project(s) discovered"
    assert got["fleet"].detail == "2 device(s) in fleet registry"
    assert got["device-policy"].detail == "loaded: 1 personal device(s) protected"
    assert got["mcp-config"].detail == "linc-crucible registered in ~/.claude.json"
    assert got["mobsf"].detail == "MobSF container running: Up 3 hours"
    assert got["nerve"].detail == "nerve daemon reachable"
    assert got["tool:adb"].detail == "/bin/adb"
    assert got["linc-workspace"].detail == str(ws)


def test_install_ok_detail_names_version(absent: fc.ProbeEnv) -> None:
    from linc_crucible import __version__

    assert _by_name(fc.run_probes(absent))["install"].detail == f"linc-crucible {__version__}"


def test_unparsable_fleet_file_fails(tmp_path: Path) -> None:
    bad = tmp_path / "fleet.yaml"
    bad.write_text("fleet: [unclosed\n")
    probe = fc.probe_fleet(lambda: bad)
    assert (probe.name, probe.status) == ("fleet", "fail")
    assert probe.detail.startswith(f"fleet registry {bad} is unparsable: ")


def test_non_utf8_fleet_file_fails(tmp_path: Path) -> None:
    """[contract] undecodable registry bytes are ``fail``, never a crash (a BOM is honoured)."""
    bad = tmp_path / "fleet.yaml"
    bad.write_bytes(b"fleet:\n  devices:\n    - name: \xff\xfe\x80\n")
    probe = fc.probe_fleet(lambda: bad)
    assert (probe.name, probe.status) == ("fleet", "fail")
    assert probe.detail.startswith(f"fleet registry {bad} is unparsable: ")


def test_fleet_file_is_read_as_utf8_bytes(tmp_path: Path) -> None:
    """A non-ASCII device name in a UTF-8 registry counts as a device (bytes in, no locale)."""
    fleet = tmp_path / "fleet.yaml"
    fleet.write_bytes("fleet:\n  devices:\n    - name: Café ✓\n".encode())
    probe = fc.probe_fleet(lambda: fleet)
    assert probe == fc.Probe("fleet", "ok", "1 device(s) in fleet registry")


def test_fleet_file_with_unexpected_shape_counts_zero(tmp_path: Path) -> None:
    odd = tmp_path / "fleet.yaml"
    odd.write_text("- just a list\n")
    probe = fc.probe_fleet(lambda: odd)
    assert probe == fc.Probe("fleet", "warn", fc.FLEET_EMPTY_HINT)


def test_projects_probe_raising_fails() -> None:
    def boom() -> list[Any]:
        raise OSError("disk gone")

    assert fc.probe_projects(boom) == fc.Probe(
        "projects", "fail", "could not read project registry: disk gone"
    )


ABSENT_DETAIL = (
    "absent: no personal devices registered ({path} not found) — "
    "set LINC_DEVICE_FLEET_FILE to protect personal devices"
)
UNREADABLE_DETAIL = (
    "unreadable: personal-device registry {path} could not be read ({error}) — "
    "every device operation is refused unless allow_personal=True"
)


def test_device_policy_loaded_is_ok() -> None:
    policy = DevicePolicy(frozenset({"S1", "S2"}), state="loaded", source="p")
    assert fc.probe_device_policy(lambda: policy) == fc.Probe(
        "device-policy", "ok", "loaded: 2 personal device(s) protected"
    )


def test_device_policy_absent_fails_open_with_a_warn_hint() -> None:
    """Main's DevicePolicy: a missing registry means no personal devices -> warn, not fail."""
    assert fc.probe_device_policy(_absent_policy) == fc.Probe(
        "device-policy", "warn", ABSENT_DETAIL.format(path="x")
    )


def test_device_policy_unreadable_fails_closed() -> None:
    policy = DevicePolicy(frozenset(), state="unreadable", source="p", error="bad yaml")
    assert fc.probe_device_policy(lambda: policy) == fc.Probe(
        "device-policy", "fail", UNREADABLE_DETAIL.format(path="p", error="bad yaml")
    )


def test_device_policy_status_follows_state_not_serials() -> None:
    """[contract] an unreadable policy with leftover serials still fails; ``state`` decides."""
    policy = DevicePolicy(frozenset({"S1"}), state="unreadable", source="p", error="e")
    assert fc.probe_device_policy(lambda: policy).status == "fail"
    absent = DevicePolicy(frozenset({"S1"}), state="absent", source="p", error="e")
    assert fc.probe_device_policy(lambda: absent).status == "warn"


def _real_policy_probe() -> fc.Probe:
    from linc_crucible.core.device_policy import load_device_policy

    return fc.probe_device_policy(load_device_policy)


@pytest.mark.parametrize("override", [None, ""])
def test_real_default_registry_is_linc_home_device_fleet_and_absent_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, override: str | None
) -> None:
    """Unset or empty LINC_DEVICE_FLEET_FILE -> ``$LINC_HOME/device-fleet.yaml`` -> warn."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path / "lh"))
    if override is None:
        monkeypatch.delenv("LINC_DEVICE_FLEET_FILE", raising=False)
    else:
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", override)
    path = tmp_path / "lh" / "device-fleet.yaml"
    assert _real_policy_probe() == fc.Probe(
        "device-policy", "warn", ABSENT_DETAIL.format(path=path)
    )


def test_real_registry_override_loaded_is_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reg = tmp_path / "reg.yaml"
    reg.write_text("devices:\n  - serial: S1\n    personal: true\n")
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(reg))
    assert _real_policy_probe() == fc.Probe(
        "device-policy", "ok", "loaded: 1 personal device(s) protected"
    )


def _directory(tmp_path: Path) -> Path:
    target = tmp_path / "reg.yaml"
    target.mkdir()
    return target


def _dangling_symlink(tmp_path: Path) -> Path:
    target = tmp_path / "reg.yaml"
    target.symlink_to(tmp_path / "gone.yaml")
    return target


def _malformed(tmp_path: Path) -> Path:
    target = tmp_path / "reg.yaml"
    target.write_text("devices: [unclosed\n")
    return target


def _not_a_mapping(tmp_path: Path) -> Path:
    target = tmp_path / "reg.yaml"
    target.write_text("- serial: S1\n")
    return target


def _not_utf8(tmp_path: Path) -> Path:
    target = tmp_path / "reg.yaml"
    target.write_bytes(b"devices:\n  - serial: \xff\xfe\x80\n")
    return target


@pytest.mark.parametrize(
    "make", [_directory, _dangling_symlink, _malformed, _not_a_mapping, _not_utf8]
)
def test_real_unreadable_registry_fails_naming_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make: Any
) -> None:
    """[contract] a registry entry that exists but cannot be used fails closed -> ``fail``."""
    target = make(tmp_path)
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(target))
    probe = _real_policy_probe()
    assert (probe.name, probe.status) == ("device-policy", "fail")
    assert probe.detail.startswith(
        f"unreadable: personal-device registry {target} could not be read ("
    )
    assert probe.detail.endswith(
        ") — every device operation is refused unless allow_personal=True"
    )


def test_real_registry_path_with_nul_byte_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """[contract] a hostile LINC_DEVICE_FLEET_FILE never crashes the probe."""
    monkeypatch.setattr(os, "environ", {**os.environ, "LINC_DEVICE_FLEET_FILE": "a\x00b"})
    probe = _real_policy_probe()
    assert (probe.name, probe.status) == ("device-policy", "fail")
    prefix = "unreadable: personal-device registry a\x00b could not be read ("
    assert probe.detail.startswith(prefix)
    assert "embedded null" in probe.detail


def test_absent_default_registry_under_home_renders_as_tilde(tmp_path: Path) -> None:
    """[contract] the absent line names ``~/.linc/device-fleet.yaml``, not the home dir."""
    home = tmp_path / "home"
    source = str(home / ".linc" / "device-fleet.yaml")
    env = replace(
        _absent_env(tmp_path),
        load_policy=lambda: DevicePolicy(frozenset(), state="absent", source=source),
        home=lambda: home,
    )
    got = _by_name(fc.run_probes(env))["device-policy"]
    assert got == fc.Probe(
        "device-policy", "warn", ABSENT_DETAIL.format(path="~/.linc/device-fleet.yaml")
    )


def test_claude_json_unparsable_warns(tmp_path: Path) -> None:
    bad = tmp_path / "claude.json"
    bad.write_text("{not json")
    probe = fc.probe_mcp_config(lambda: bad)
    assert (probe.name, probe.status) == ("mcp-config", "warn")
    assert probe.detail.startswith("~/.claude.json is not valid JSON: ")


def test_claude_json_not_utf8_warns(tmp_path: Path) -> None:
    """[contract] a non-UTF-8 ``~/.claude.json`` is ``warn``, never a crash."""
    bad = tmp_path / "claude.json"
    bad.write_bytes(b'{"mcpServers": {"\xff\xfe\x80": {}}}')
    probe = fc.probe_mcp_config(lambda: bad)
    assert (probe.name, probe.status) == ("mcp-config", "warn")
    assert probe.detail.startswith("~/.claude.json is not valid JSON: ")


def test_claude_json_without_crucible_warns(tmp_path: Path) -> None:
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {}}}))
    probe = fc.probe_mcp_config(lambda: cfg)
    assert probe == fc.Probe(
        "mcp-config",
        "warn",
        f"linc-crucible not registered in ~/.claude.json — {fc.MCP_REGISTER_HINT} "
        "(the `crucible bellows` stdio server)",
    )


def test_claude_json_non_mapping_warns(tmp_path: Path) -> None:
    cfg = tmp_path / "claude.json"
    cfg.write_text("[1, 2]")
    assert fc.probe_mcp_config(lambda: cfg).status == "warn"


def test_mobsf_not_running_skips() -> None:
    probe = fc.probe_mobsf(lambda _n: "/bin/docker", lambda: "")
    assert probe == fc.Probe("mobsf", "skip", fc.MOBSF_NOT_RUNNING_HINT)


@pytest.mark.parametrize(
    "exc", [OSError("no"), subprocess.TimeoutExpired(["docker"], 5), subprocess.SubprocessError()]
)
def test_mobsf_docker_error_skips(exc: Exception) -> None:
    def boom() -> str:
        raise exc

    probe = fc.probe_mobsf(lambda n: "/bin/docker" if n == "docker" else None, boom)
    assert (probe.name, probe.status) == ("mobsf", "skip")
    assert probe.detail.startswith("could not query docker: ")


def test_default_probe_env_wires_real_callables() -> None:
    env = fc.default_probe_env()
    assert env.which("definitely-not-a-binary-xyz") is None
    assert env.check_install() is None
    assert env.fleet_file().name == "fleet.yaml"
    assert env.claude_json().name == ".claude.json"


def test_default_nerve_up_false_when_unreachable() -> None:
    with patch("linc_crucible.core.nerve_client.NerveClient.health", side_effect=ConnectionError):
        assert fc.default_probe_env().nerve_up() is False


def test_default_nerve_up_true_when_healthy() -> None:
    with patch("linc_crucible.core.nerve_client.NerveClient.health", return_value={"ok": 1}):
        assert fc.default_probe_env().nerve_up() is True


def test_default_docker_ps_runs_docker_without_shell() -> None:
    with patch("subprocess.run") as run:
        run.return_value.stdout = " Up 1 minute \n"
        assert fc.default_probe_env().docker_ps() == "Up 1 minute"
    argv = run.call_args.args[0]
    assert argv[:2] == ["docker", "ps"]
    assert run.call_args.kwargs == {"capture_output": True, "text": True, "timeout": 5}


def test_default_check_install_reports_import_error() -> None:
    with patch("importlib.import_module", side_effect=ImportError("no fastmcp")):
        assert fc.default_probe_env().check_install() == "no fastmcp"


def test_default_paths_honour_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CRUCIBLE_FLEET_FILE", str(tmp_path / "f.yaml"))
    monkeypatch.setenv("LINC_WORKSPACE", str(tmp_path / "ws"))
    env = fc.default_probe_env()
    assert env.fleet_file() == tmp_path / "f.yaml"
    assert env.linc_workspace() == tmp_path / "ws"


def test_default_paths_fall_back_under_linc_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRUCIBLE_FLEET_FILE", raising=False)
    monkeypatch.delenv("LINC_WORKSPACE", raising=False)
    env = fc.default_probe_env()
    home = Path(__import__("os").environ["LINC_HOME"])
    assert env.fleet_file() == home / "crucible" / "fleet.yaml"
    assert env.linc_workspace() == home / "projects" / "LINC"


def test_default_list_projects_uses_forge_reader() -> None:
    with patch("linc_crucible.core.forge_integration.ForgeProjectReader") as reader:
        reader.return_value.list_projects.return_value = ["p"]
        assert fc.default_probe_env().list_projects() == ["p"]


def test_format_rich_line_colours_status_and_escapes_detail() -> None:
    assert fc.format_rich_line(fc.Probe("n", "fail", "[ai]")) == (
        "[bold red]fail  [/bold red]n: \\[ai]"
    )


# --- [contract] hostile environment ------------------------------------------------------


def test_redact_home_substitutes_tilde() -> None:
    """[contract] home paths never reach output that strangers paste into public issues."""
    assert fc.redact_home("/h/u/ws and /h/u", Path("/h/u")) == "~/ws and ~"
    assert fc.redact_home("at /h/u: x", Path("/h/u")) == "at ~: x"


@pytest.mark.parametrize("home", [None, Path("/")])
@pytest.mark.parametrize("text", ["/h/u/ws", "a / b", "XXXX"])
def test_redact_home_without_a_usable_home_is_identity(home: Path | None, text: str) -> None:
    """[contract] HOME unset (or ``/``) leaves the text untouched instead of mangling it."""
    assert fc.redact_home(text, home) == text


def test_redact_home_keeps_trailing_word_characters_of_home() -> None:
    """Only a trailing ``/`` is trimmed from home: ``/h/uX`` must not shrink to ``/h/u``."""
    assert fc.redact_home("/h/uX/ws and /h/uX", Path("/h/uX")) == "~/ws and ~"
    assert fc.redact_home("/h/u/ws", Path("/h/uX")) == "/h/u/ws"


def test_redact_home_does_not_touch_sibling_prefixes() -> None:
    assert fc.redact_home("/h/user2/ws /h/u.bak", Path("/h/u")) == "/h/user2/ws /h/u.bak"


def _home_env(tmp_path: Path) -> fc.ProbeEnv:
    home = tmp_path / "home"
    ws = home / "ws"
    ws.mkdir(parents=True)
    fleet = home / "fleet.yaml"
    fleet.write_text("fleet: [unclosed\n")
    return replace(
        _absent_env(tmp_path),
        which=lambda name: str(home / "bin" / name),
        docker_ps=lambda: "",
        linc_workspace=lambda: ws,
        fleet_file=lambda: fleet,
        load_policy=lambda: DevicePolicy(
            frozenset(), state="unreadable", source=str(home / "reg.yaml"), error="bad"
        ),
        home=lambda: home,
    )


@pytest.mark.parametrize("args", [(), ("--json",)])
def test_cli_output_renders_home_as_tilde(tmp_path: Path, args: tuple[str, ...]) -> None:
    """[contract] neither the text nor the --json output carries the absolute home dir."""
    result = _invoke(_home_env(tmp_path), *args)
    assert str(tmp_path / "home") not in result.stdout
    assert "~/ws" in result.stdout
    assert "~/bin/adb" in result.stdout
    assert "~/fleet.yaml" in result.stdout
    assert "~/reg.yaml" in result.stdout


def test_home_dir_none_when_home_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    """[contract] HOME unset and no passwd entry: no crash, just no home."""
    monkeypatch.delenv("HOME", raising=False)
    with patch.object(Path, "home", side_effect=RuntimeError("no home")):
        assert fc._home_dir() is None
        assert fc._claude_json() is None


def test_mcp_config_skips_when_home_unresolvable() -> None:
    assert fc.probe_mcp_config(lambda: None) == fc.Probe(
        "mcp-config",
        "skip",
        "~/.claude.json not found — to use the MCP server, register `crucible bellows` "
        "(stdio) in your MCP client",
    )


def test_projects_probe_empty_hint_is_exact() -> None:
    assert fc.probe_projects(list) == fc.Probe(
        "projects",
        "warn",
        "no projects found — add a linc-manifest.yaml to a project under LINC_PROJECTS_DIR",
    )


def test_nerve_up_probes_once_without_retry_storms() -> None:
    with patch("linc_crucible.core.nerve_client.NerveClient") as client:
        client.return_value.health.return_value = {"status": "ok"}
        assert fc._nerve_up() is True
    client.assert_called_once_with(max_retries=1)


def test_device_policy_unresolvable_registry_path_warns() -> None:
    """[contract] main's registry default needs Path.home(); HOME unset must not crash."""

    def boom() -> Any:
        raise RuntimeError("Could not determine home directory.")

    probe = fc.probe_device_policy(boom)
    assert probe.status == "warn"
    assert "LINC_DEVICE_FLEET_FILE" in probe.detail
    assert "Could not determine home directory." in probe.detail


def _stubbed_default_env() -> fc.ProbeEnv:
    return replace(
        fc.default_probe_env(),
        which=lambda _n: None,
        nerve_up=lambda: False,
        list_projects=list,
    )


def test_cli_with_home_unset_exits_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """[contract] HOME unset and unresolvable: every probe still answers, exit 0."""
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("LINC_DEVICE_FLEET_FILE", raising=False)
    monkeypatch.setenv("LINC_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CRUCIBLE_FLEET_FILE", str(tmp_path / "fleet.yaml"))
    with patch.object(Path, "home", side_effect=RuntimeError("no home")):
        result = _invoke(_stubbed_default_env(), "--json")
    assert result.exit_code == 0, result.output
    got = {p["name"]: p["status"] for p in json.loads(result.stdout)}
    assert got["mcp-config"] == "skip"
    assert got["device-policy"] == "warn"


def test_cli_in_unwritable_cwd_exits_zero_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, absent: fc.ProbeEnv
) -> None:
    """[contract] fuelcheck is read-only: an unwritable cwd neither fails it nor gets files."""
    cwd = tmp_path / "ro"
    cwd.mkdir()
    cwd.chmod(0o500)
    monkeypatch.chdir(cwd)
    try:
        for args in ((), ("--json",)):
            result = _invoke(absent, *args)
            assert result.exit_code == 0, result.output
        assert os.listdir(cwd) == []
    finally:
        cwd.chmod(0o700)


@pytest.mark.parametrize("garbage", ["<html>not nerve</html>", ["x"], None, 42])
def test_nerve_up_false_when_daemon_returns_garbage(garbage: Any) -> None:
    """[contract] something answering on the nerve port is not a healthy nerve daemon."""
    with patch("linc_crucible.core.nerve_client.NerveClient.health", return_value=garbage):
        assert fc._nerve_up() is False


def test_nerve_up_false_when_daemon_body_is_not_json() -> None:
    """[contract] a non-JSON /health body (json decode error) is down, not a crash."""
    err = json.JSONDecodeError("Expecting value", "garbage", 0)
    with patch("linc_crucible.core.nerve_client.NerveClient.health", side_effect=err):
        assert fc._nerve_up() is False


def test_module_constants_are_pinned() -> None:
    assert fc.STATUSES == ("ok", "warn", "skip", "fail")
    assert set(fc._STATUS_STYLE) == set(fc.STATUSES)
    assert set(fc.TOOL_HINTS) == {"adb", "xcrun", "ruff", "pytest"}
    assert fc.NERVE_DOWN_HINT == NERVE_HINT
    assert fc._MOBSF_PS[:2] == ["docker", "ps"]
