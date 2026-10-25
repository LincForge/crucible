"""Stranger-safe environment probes behind ``crucible fuelcheck``.

Each probe returns a :class:`Probe` ``(name, status, detail)`` with status in
``ok``/``warn``/``skip``/``fail``. Missing optional pieces — a LINC workspace, registered
projects, a fleet registry, ``~/.claude.json``, docker, the nerve daemon — are ``warn`` or
``skip`` with a one-line hint naming the env var or command. Only a broken install or a config
file that IS present but unparsable is ``fail``; ``fuelcheck`` exits 1 only then.

Every side effect goes through a :class:`ProbeEnv` callable so tests inject fakes and never
touch the network, docker or the real home dir. Output is meant to be pasted into public
issues, so the user's home directory is rendered as ``~`` in every detail AND in every log
event the probes emit to stderr. Registry files are read with a size cap: an oversized one is
rejected (naming its size and the cap), never parsed.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.markup import escape

from . import __version__
from ._bounded_read import _MAX_REGISTRY_BYTES, read_capped, read_failure
from ._redaction import redact_home, redacted_logs
from .core.device_policy import DevicePolicy

STATUSES: tuple[str, ...] = ("ok", "warn", "skip", "fail")

_STATUS_STYLE = {"ok": "green", "warn": "yellow", "skip": "dim", "fail": "bold red"}

TOOL_HINTS: dict[str, str] = {
    "adb": "Android platform-tools, used by the nerve daemon for Android devices",
    "xcrun": "Xcode command-line tools, needed for iOS builds",
    "ruff": "used by assay when a project's lint command calls it",
    "pytest": "used by assay when a project's test command calls it",
}

NERVE_DOWN_HINT = (
    "nerve daemon not running — BVT, baseline capture and pour need it: "
    "pipx install linc-nerve && nerve daemon"
)
FLEET_EMPTY_HINT = (
    "no devices in fleet registry — run `crucible fleet discover` (needs the nerve daemon) "
    "or set CRUCIBLE_FLEET_FILE"
)
MOBSF_NOT_RUNNING_HINT = (
    "MobSF container not running — only needed for assay mode=full (MobSF security scan)"
)
MCP_REGISTER_HINT = 'add "linc-crucible": {"command": "crucible", "args": ["bellows"]}'

# Size cap for ~/.claude.json (registry cap _MAX_REGISTRY_BYTES lives in _bounded_read).
# ~/.claude.json belongs to the MCP client and legitimately grows larger.
_MAX_CLAUDE_JSON_BYTES = 16 * 1024 * 1024

_MOBSF_PS = [
    "docker",
    "ps",
    "--filter",
    "ancestor=opensecurity/mobile-security-framework-mobsf",
    "--format",
    "{{.Status}}",
]


@dataclass(frozen=True)
class Probe:
    """One fuelcheck result line."""

    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown probe status {self.status!r}")

    def to_dict(self) -> dict[str, str]:
        """Return the ``--json`` shape: ``{"name", "status", "detail"}``."""
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class ProbeEnv:
    """Injectable side effects for every probe."""

    which: Callable[[str], str | None]
    check_install: Callable[[], str | None]
    linc_workspace: Callable[[], Path]
    list_projects: Callable[[], Sequence[Any]]
    fleet_file: Callable[[], Path]
    load_policy: Callable[[], DevicePolicy]
    claude_json: Callable[[], Path | None]
    docker_ps: Callable[[], str]
    nerve_up: Callable[[], bool]
    home: Callable[[], Path | None]


def probe_install(check_install: Callable[[], str | None]) -> Probe:
    """``fail`` when crucible's own modules cannot be imported."""
    error = check_install()
    if error is not None:
        return Probe("install", "fail", f"broken install: {error}")
    return Probe("install", "ok", f"linc-crucible {__version__}")


def probe_tool(name: str, which: Callable[[str], str | None]) -> Probe:
    """``ok`` with the resolved path, else ``skip`` naming what the tool is for."""
    path = which(name)
    if path is None:
        return Probe(f"tool:{name}", "skip", f"{name} not found — {TOOL_HINTS[name]}")
    return Probe(f"tool:{name}", "ok", path)


def probe_linc_workspace(linc_workspace: Callable[[], Path]) -> Probe:
    """The LINC workspace is optional: absent is ``skip``."""
    path = linc_workspace()
    if path.is_dir():
        return Probe("linc-workspace", "ok", str(path))
    return Probe(
        "linc-workspace",
        "skip",
        f"no LINC workspace at {path} — preflight reads store-listing.md and brand-brief.json"
        " from <project>/store/ (or LINC_WORKSPACE); missing ones score as fail",
    )


def probe_projects(list_projects: Callable[[], Sequence[Any]]) -> Probe:
    """``warn`` when no project is discoverable; ``fail`` only when discovery itself breaks."""
    try:
        projects = list_projects()
    except Exception as exc:
        return Probe("projects", "fail", f"could not read project registry: {exc}")
    if not projects:
        return Probe(
            "projects",
            "warn",
            "no projects found — add a linc-manifest.yaml to a project under LINC_PROJECTS_DIR",
        )
    return Probe("projects", "ok", f"{len(projects)} project(s) discovered")


def _fleet_device_count(data: Any) -> int:
    """Count ``fleet.devices`` entries in a parsed fleet registry; 0 for any other shape."""
    fleet = data.get("fleet") if isinstance(data, dict) else None
    devices = fleet.get("devices") if isinstance(fleet, dict) else None
    return len(devices) if isinstance(devices, list) else 0


def probe_fleet(fleet_file: Callable[[], Path]) -> Probe:
    """``warn`` when the fleet registry is absent or empty; ``fail`` when present but broken.

    Broken means oversized, unreadable or not YAML. The file is parsed from bytes, so PyYAML
    honours a UTF-8/UTF-16 BOM; only undecodable bytes are a decode ``fail``.
    """
    path = fleet_file()
    if not path.is_file():
        return Probe("fleet", "warn", FLEET_EMPTY_HINT)
    try:
        data = yaml.safe_load(read_capped(path, _MAX_REGISTRY_BYTES))
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        return Probe(
            "fleet", "fail", f"fleet registry {path} {read_failure(exc, 'is unparsable')}"
        )
    count = _fleet_device_count(data)
    if count == 0:
        return Probe("fleet", "warn", FLEET_EMPTY_HINT)
    return Probe("fleet", "ok", f"{count} device(s) in fleet registry")


def _policy_probe(policy: DevicePolicy) -> Probe:
    """Map a :class:`DevicePolicy` ``state`` onto one probe line.

    ``loaded`` is ``ok``. ``absent`` (no registry file) fails open — no personal devices are
    registered — so it is a ``warn`` hint. ``unreadable`` fails closed and refuses every device,
    so it is ``fail``.
    """
    if policy.state == "loaded":
        count = len(policy.personal_serials)
        return Probe("device-policy", "ok", f"loaded: {count} personal device(s) protected")
    if policy.state == "absent":
        return Probe(
            "device-policy",
            "warn",
            f"absent: no personal devices registered ({policy.source} not found) — "
            "set LINC_DEVICE_FLEET_FILE to protect personal devices",
        )
    return Probe(
        "device-policy",
        "fail",
        f"unreadable: personal-device registry {policy.source} could not be read "
        f"({policy.error}) — every device operation is refused unless allow_personal=True",
    )


def probe_device_policy(load_policy: Callable[[], DevicePolicy]) -> Probe:
    """Report the personal-device policy state; an unreadable registry is ``fail``."""
    try:
        policy = load_policy()
    except (RuntimeError, OSError) as exc:
        return Probe(
            "device-policy",
            "warn",
            f"cannot locate the personal-device registry ({exc}) — set LINC_DEVICE_FLEET_FILE",
        )
    return _policy_probe(policy)


def _registers_crucible(config: Any) -> bool:
    """True when a parsed ``~/.claude.json`` registers the ``linc-crucible`` MCP server."""
    servers = config.get("mcpServers") if isinstance(config, dict) else None
    return isinstance(servers, dict) and "linc-crucible" in servers


def probe_mcp_config(claude_json: Callable[[], Path | None]) -> Probe:
    """``~/.claude.json`` belongs to the MCP client, so problems there are never ``fail``."""
    path = claude_json()
    if path is None or not path.is_file():
        return Probe(
            "mcp-config",
            "skip",
            "~/.claude.json not found — to use the MCP server, register `crucible bellows` "
            "(stdio) in your MCP client",
        )
    try:
        config = json.loads(read_capped(path, _MAX_CLAUDE_JSON_BYTES))
    except (ValueError, OSError) as exc:
        detail = read_failure(exc, "is not valid JSON")
        return Probe("mcp-config", "warn", f"~/.claude.json {detail}")
    if _registers_crucible(config):
        return Probe("mcp-config", "ok", "linc-crucible registered in ~/.claude.json")
    return Probe(
        "mcp-config",
        "warn",
        f"linc-crucible not registered in ~/.claude.json — {MCP_REGISTER_HINT} "
        "(the `crucible bellows` stdio server)",
    )


def probe_nerve(nerve_up: Callable[[], bool]) -> Probe:
    """The nerve daemon is optional for analysis-only use: down is ``warn``."""
    if nerve_up():
        return Probe("nerve", "ok", "nerve daemon reachable")
    return Probe("nerve", "warn", NERVE_DOWN_HINT)


def probe_mobsf(which: Callable[[str], str | None], docker_ps: Callable[[], str]) -> Probe:
    """MobSF is only for ``assay`` mode ``full``: docker absent or not running is ``skip``."""
    if which("docker") is None:
        return Probe(
            "mobsf",
            "skip",
            "docker not found — only needed for assay mode=full (MobSF security scan)",
        )
    try:
        status = docker_ps()
    except (OSError, subprocess.SubprocessError) as exc:
        return Probe("mobsf", "skip", f"could not query docker: {exc}")
    if status:
        return Probe("mobsf", "ok", f"MobSF container running: {status}")
    return Probe("mobsf", "skip", MOBSF_NOT_RUNNING_HINT)


def run_probes(env: ProbeEnv) -> list[Probe]:
    """Run every probe in display order, rendering the home dir as ``~`` in each detail.

    The probes run inside :func:`redacted_logs`, so the warnings real loaders log to stderr
    (e.g. ``device_fleet_absent path=...``) are home-redacted too.
    """
    home = env.home()
    with redacted_logs(home):
        probes = [
            probe_install(env.check_install),
            *(probe_tool(name, env.which) for name in TOOL_HINTS),
            probe_linc_workspace(env.linc_workspace),
            probe_projects(env.list_projects),
            probe_fleet(env.fleet_file),
            probe_device_policy(env.load_policy),
            probe_mcp_config(env.claude_json),
            probe_nerve(env.nerve_up),
            probe_mobsf(env.which, env.docker_ps),
        ]
    return [Probe(p.name, p.status, redact_home(p.detail, home)) for p in probes]


def exit_code(probes: Sequence[Probe]) -> int:
    """1 when any probe failed, else 0."""
    return 1 if any(p.status == "fail" for p in probes) else 0


def format_line(probe: Probe) -> str:
    """Plain text line: status padded to 6 columns, then ``name: detail``."""
    return f"{probe.status:<6}{probe.name}: {probe.detail}"


def format_rich_line(probe: Probe) -> str:
    """:func:`format_line` with the status coloured and the rest markup-escaped."""
    style = _STATUS_STYLE[probe.status]
    return f"[{style}]{probe.status:<6}[/{style}]{escape(f'{probe.name}: {probe.detail}')}"


# --- Real side effects (wired by default_probe_env) ---------------------------------------


def _check_install() -> str | None:
    try:
        importlib.import_module("linc_crucible.mcp_server")
    except ImportError as exc:
        return str(exc)
    return None


def _list_projects() -> Sequence[Any]:
    from .core.forge_integration import ForgeProjectReader

    return ForgeProjectReader().list_projects()


def _fleet_file() -> Path:
    """Same resolution as ``FleetRegistry``: ``CRUCIBLE_FLEET_FILE``, else the state root."""
    from .core.fleet_registry import _default_fleet_path

    return Path(os.environ.get("CRUCIBLE_FLEET_FILE") or _default_fleet_path())


def _linc_workspace() -> Path:
    """Same resolution as ``PreflightChecker``: ``LINC_WORKSPACE``, else the default."""
    from .core.preflight import _default_linc_workspace

    return Path(os.environ.get("LINC_WORKSPACE") or _default_linc_workspace())


def _docker_ps() -> str:
    result = subprocess.run(_MOBSF_PS, capture_output=True, text=True, timeout=5)
    return result.stdout.strip()


def _nerve_up() -> bool:
    """True only when ``/health`` answers with a JSON object — anything else is not nerve."""
    from .core.nerve_client import NerveClient

    try:
        health = NerveClient(max_retries=1).health()
    except Exception:
        return False
    return isinstance(health, dict)


def _home_dir() -> Path | None:
    """The user's home dir, or ``None`` when HOME is unset and unresolvable."""
    try:
        return Path.home()
    except (RuntimeError, KeyError):
        return None


def _claude_json() -> Path | None:
    home = _home_dir()
    return None if home is None else home / ".claude.json"


def default_probe_env() -> ProbeEnv:
    """Wire every probe to the real host (PATH, home dir, docker, the nerve daemon)."""
    from .core.device_policy import load_device_policy

    return ProbeEnv(
        which=shutil.which,
        check_install=_check_install,
        linc_workspace=_linc_workspace,
        list_projects=_list_projects,
        fleet_file=_fleet_file,
        load_policy=load_device_policy,
        claude_json=_claude_json,
        docker_ps=_docker_ps,
        nerve_up=_nerve_up,
        home=_home_dir,
    )
