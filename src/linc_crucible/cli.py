"""Command-line interface for Linc Crucible."""

import contextlib
import io
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import click
from linc_cli_kit import (
    LocalTransport,
    ToolError,
    UsageError,
    current_mode,
    emit,
    emit_exception,
    install_globals,
    mount_tools,
)

from . import _logging

# CRITICAL: structlog to stderr, BEFORE any linc_crucible module import. server_context.py
# (`logger = structlog.get_logger().bind(domain="build_pipeline")`) and bvt_tools.py bind a
# structlog logger at IMPORT time, and those modules get imported below (server_context
# directly; the *_tools modules lazily via _registry_server(), called at the bottom of this
# file). structlog's default logger factory writes to stdout, which would corrupt the CLI's
# JSON payload on `crucible <cmd> | jq`. Regression test: tests/test_cli_stdout_hygiene.py.
# `configure_cli_structlog()` also layers a WARNING-default level filter (env `LINC_LOG_LEVEL`)
# on top of the kit's stderr redirect — see `_logging.py` and tests/test_cli_log_level.py.
_logging.configure_cli_structlog()

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

import linc_crucible.server_context as server_ctx  # noqa: E402
from linc_crucible import __version__  # noqa: E402
from linc_crucible import fuelcheck as _fuel  # noqa: E402
from linc_crucible.config_contract import render_env_table  # noqa: E402

console = Console()

CRUCIBLE_BANNER = rf"""
    .  *  .  *  .
   . \|/ . \|/ .
  ---[=]---[=]---      [bold red]Linc Crucible[/bold red] v{__version__}
   . /|\ . /|\ .      [dim]Where quality is tested[/dim]
    .  *  .  *  .      [dim]and code is proven[/dim]
   _______________
  |               |
  |   [bold yellow]~ ~ ~ ~ ~[/bold yellow]   |
  |   [bold yellow]~ ~ ~ ~ ~[/bold yellow]   |
  |_______________|
  |_____[=]_______|
"""


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Linc Crucible - Enterprise test, device management, and quality control.

    \b
    Examples:
        crucible fleet discover       # Find connected devices
        crucible fleet list           # Show fleet
        crucible pour --project myapp   # Build + deploy (JSON on stdout)
        crucible assay --project myapp  # Run quality checks
        crucible score myapp            # Show quality score
        crucible bellows              # Start MCP server
        crucible fuelcheck            # Verify environment
    """
    pass


install_globals(cli)


def _human_text(build: Callable[[Console], None]) -> str:
    """Render Rich output to plain text for a `--human` renderer (spec: no ANSI on stdout)."""
    buf = io.StringIO()
    con = Console(file=buf, no_color=True, width=200)
    build(con)
    return buf.getvalue()


# --- Fleet commands ---


@cli.group()
def fleet() -> None:
    """Fleet management — discover, list, add, and remove devices."""
    pass


def _device_dict(d: Any) -> dict[str, Any]:
    """Serialize a `FleetDevice` with explicit keys (keeping empty strings)."""
    return {
        "serial": d.serial,
        "platform": d.platform.value,
        "device_type": d.device_type.value,
        "model": d.model,
        "os_version": d.os_version,
        "nickname": d.nickname,
        "state": d.state,
    }


def _skipped_personal_count(registry: Any) -> int:
    skipped = getattr(registry, "last_discover_skipped_personal", 0)
    if isinstance(skipped, int) and not isinstance(skipped, bool) and skipped > 0:
        return skipped
    return 0


def _build_fleet_discover_payload(
    new_devices: list[Any], all_devices: list[Any], skipped: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "count": len(new_devices),
        "fleet_total": len(all_devices),
        "skipped_personal": skipped,
        "discovered": [_device_dict(d) for d in new_devices],
    }
    if skipped > 0:
        payload["message"] = f"skipped {skipped} personal device(s)"
    elif not new_devices:
        payload["message"] = "No new devices discovered."
    return payload


def _render_fleet_discover(payload: dict[str, Any]) -> str:
    def _build(con: Console) -> None:
        con.print(CRUCIBLE_BANNER)
        con.print("[bold cyan]Fleet Discovery[/bold cyan]\n")
        new_devices = payload.get("discovered") or []
        if new_devices:
            con.print(f"[green]Discovered {len(new_devices)} new device(s):[/green]")
            for d in new_devices:
                con.print(f"  + {d['serial']} — {d['model']} ({d['platform']}/{d['device_type']})")
        else:
            con.print("[yellow]No new devices discovered.[/yellow]")
        skipped = payload.get("skipped_personal", 0)
        if skipped > 0:
            con.print(f"[dim]skipped {skipped} personal device(s)[/dim]")
        total = payload.get("fleet_total", 0)
        if total:
            con.print(f"\n[bold]Fleet total: {total} device(s)[/bold]")

    return _human_text(_build)


@fleet.command()
def discover() -> None:
    """Auto-discover ADB + iOS simulator devices."""
    from .core.fleet_registry import FleetRegistry

    registry = FleetRegistry()
    new_devices = registry.discover()
    skipped = _skipped_personal_count(registry)
    all_devices = registry.list_devices()
    payload = _build_fleet_discover_payload(new_devices, all_devices, skipped)

    ctx = click.get_current_context()
    ctx.exit(
        emit(
            payload,
            mode=current_mode(),
            renderer=_render_fleet_discover,
            tool="fleet_discover",
        )
    )


def _build_fleet_payload(devices: list[Any], platform: str | None) -> dict[str, Any]:
    """Build the `fleet list` JSON payload — explicit keys, empty strings kept.

    Deliberately does NOT reuse `FleetDevice.to_dict()`: that method drops empty fields for
    a compact `fleet.yaml`, which would silently vary this payload's shape device-to-device.
    """
    payload: dict[str, Any] = {
        "count": len(devices),
        "platform": platform,
        "devices": [_device_dict(d) for d in devices],
    }
    if not devices:
        payload["message"] = "No devices registered. Run 'crucible fleet discover' first."
    return payload


def _render_fleet_list(payload: dict[str, Any]) -> str:
    def _build(con: Console) -> None:
        devices = payload.get("devices") or []
        if not devices:
            con.print(
                "[yellow]No devices registered. Run 'crucible fleet discover' first.[/yellow]"
            )
            return

        table = Table(title=f"Crucible Fleet ({len(devices)} devices)")
        table.add_column("Serial", style="bold")
        table.add_column("Platform")
        table.add_column("Type")
        table.add_column("Model")
        table.add_column("OS")
        table.add_column("Nickname")
        table.add_column("State")

        for d in devices:
            state_style = "green" if d["state"] == "online" else "red"
            table.add_row(
                d["serial"],
                d["platform"],
                d["device_type"],
                d["model"],
                d["os_version"],
                d["nickname"] or "—",
                f"[{state_style}]{d['state']}[/{state_style}]",
            )

        con.print(table)

    return _human_text(_build)


@fleet.command("list")
@click.option("--platform", type=click.Choice(["android", "ios"]), default=None)
def fleet_list(platform: str | None) -> None:
    """Show registered devices."""
    from .core.device import Platform
    from .core.fleet_registry import FleetRegistry

    registry = FleetRegistry()
    plat = Platform(platform) if platform else None
    devices = registry.list_devices(platform=plat)
    payload = _build_fleet_payload(devices, platform)

    ctx = click.get_current_context()
    ctx.exit(emit(payload, mode=current_mode(), renderer=_render_fleet_list, tool="fleet_list"))


def _render_fleet_add(payload: dict[str, Any]) -> str:
    dev = payload["device"]
    serial = dev["serial"]
    nickname = dev["nickname"]
    return _human_text(
        lambda con: con.print(
            f"[green]Added:[/green] {serial}" + (f" ({nickname})" if nickname else "")
        )
    )


@fleet.command()
@click.argument("serial")
@click.option("--nickname", default="", help="Friendly name for the device")
@click.option(
    "--type",
    "device_type",
    default="phone",
    type=click.Choice(["phone", "tablet", "watch", "tv", "emulator", "simulator"]),
)
def add(serial: str, nickname: str, device_type: str) -> None:
    """Add a device to the fleet manually."""
    from .core.device import DeviceType, FleetDevice, Platform
    from .core.fleet_registry import FleetRegistry

    registry = FleetRegistry()

    device = FleetDevice(
        serial=serial,
        platform=Platform.ANDROID,  # Default; discover will correct
        device_type=DeviceType(device_type),
        nickname=nickname,
        state="online",
    )

    registry.add(device)
    payload: dict[str, Any] = {"added": True, "device": _device_dict(device)}
    ctx = click.get_current_context()
    ctx.exit(emit(payload, mode=current_mode(), renderer=_render_fleet_add, tool="fleet_add"))


def _render_fleet_remove(payload: dict[str, Any]) -> str:
    return _human_text(lambda con: con.print(f"[yellow]Removed:[/yellow] {payload['serial']}"))


@fleet.command()
@click.argument("serial")
def remove(serial: str) -> None:
    """Remove a device from the fleet."""
    from .core.fleet_registry import FleetRegistry

    registry = FleetRegistry()
    removed = bool(registry.remove(serial))
    payload: dict[str, Any] = {"removed": removed, "serial": serial}
    if not removed:
        payload["error"] = f"Device '{serial}' not found"

    ctx = click.get_current_context()
    ctx.exit(
        emit(payload, mode=current_mode(), renderer=_render_fleet_remove, tool="fleet_remove")
    )


# --- Pour / Assay / Score: kit-adopted CLI commands ---
#
# `pour` and `assay` collide with the MCP tools `crucible_pour`/`crucible_assay` and are
# handled per collision rule (a): the hand-written commands are deleted below in favor of
# the generated ones from `mount_tools()` (bottom of this module); their Rich console
# rendering is preserved as `renderers={...}` entries for `--human`.
#
# `score` collides with `crucible_score` and is handled per rule (b): kept hand-written
# and routed through the group's `linc_transport`, because the JSON contract needs a key
# the tool itself doesn't emit — `.score` (the tool returns `.composite`) — pinned
# alongside the tool's own `.ship_gate` for a machine caller that doesn't want to know the
# scorer's internal field name. See `tests/test_cli_parity.py` (`samples["crucible_score"]`)
# for the conformance suite's override coverage of this command.


def _render_pour(res: dict[str, Any]) -> str:
    def _build(con: Console) -> None:
        con.print(CRUCIBLE_BANNER)
        con.print(f"[bold cyan]Pouring {res.get('project', '')}...[/bold cyan]\n")

        if res.get("build_only"):
            # No `if res.get("error")` split here: the kit's output layer
            # (`linc_cli_kit/output.py`) classifies any error-bearing payload and returns
            # BEFORE a renderer is ever invoked, so `_render_pour` only ever sees
            # error-free results — a failure branch here would be unreachable dead code.
            con.print("[bold green]Build complete (build_only).[/bold green]")
            return

        devices = res.get("devices") or []
        if not devices:
            con.print("[red]No deployment results.[/red]")

        for d in devices:
            icon = "+" if d.get("install_success") else "x"
            module_tag = f" \\[{d['module']}]" if d.get("module") else ""
            con.print(f"  {icon} {d.get('device_serial')} ({d.get('device_model')}){module_tag}")
            if d.get("artifact_path"):
                con.print(f"    Artifact: {d['artifact_path']}")
            if d.get("install_success"):
                con.print("    [green]Installed[/green]")
            if d.get("launch_success"):
                con.print("    [green]Launched[/green]")
            if d.get("error"):
                con.print(f"    [red]{d['error']}[/red]")

        if res.get("success"):
            con.print("\n[bold green]Pour complete.[/bold green]")
        else:
            con.print("\n[bold red]Pour failed.[/bold red]")

    return _human_text(_build)


def _print_assay_perf(con: Console, res: dict[str, Any]) -> None:
    perf = res.get("performance")
    if perf:
        perf_icon = "+" if perf.get("success") else "x"
        parts = []
        if perf.get("apk_size_mb") is not None:
            parts.append(f"APK: {perf['apk_size_mb']:.2f} MB")
        if perf.get("ipa_size_mb") is not None:
            parts.append(f"iOS: {perf['ipa_size_mb']:.2f} MB")
        perf_output = ", ".join(parts) if parts else perf.get("output", "")
        con.print(f"  {perf_icon} Perf: {perf_output}")


def _print_assay_a11y(con: Console, res: dict[str, Any]) -> None:
    a11y = res.get("accessibility")
    if a11y:
        a11y_icon = "+" if a11y.get("success") else "x"
        con.print(
            f"  {a11y_icon} A11y: {a11y.get('score', 0):.0f}/100 "
            f"(critical={a11y.get('critical', 0)}, "
            f"major={a11y.get('major', 0)}, "
            f"minor={a11y.get('minor', 0)})"
        )
        for detail in a11y.get("details", [])[:10]:
            con.print(f"    - {detail}")


def _print_assay_quality_score(con: Console, res: dict[str, Any]) -> None:
    qs = res.get("quality_score")
    if qs:
        gate_icon = "[green]PASS[/green]" if qs.get("ship_gate") else "[red]FAIL[/red]"
        con.print(
            f"\n  [bold]Quality Score: {qs.get('composite', 0):.1f}/100[/bold]  "
            f"Ship Gate: {gate_icon}"
        )
        for dim in qs.get("dimensions", []):
            con.print(
                f"    {dim.get('name')}: {dim.get('score', 0):.0f} x "
                f"{dim.get('weight', 0):.2f} = {dim.get('weighted_score', 0):.1f}"
            )


def _format_pass_rate(tests: dict[str, Any]) -> str:
    rate = tests.get("pass_rate")
    return "unmeasured" if rate is None else f"{rate:.0%}"


def _render_assay(res: dict[str, Any]) -> str:
    def _build(con: Console) -> None:
        con.print(CRUCIBLE_BANNER)
        mode = res.get("mode", "quick")
        con.print(f"[bold cyan]Assaying {res.get('project', '')} ({mode} mode)...[/bold cyan]\n")

        lint = res.get("lint") or {}
        lint_icon = "+" if lint.get("success") else "x"
        con.print(f"  {lint_icon} Lint: {lint.get('issues', 0)} issue(s)")

        tests = res.get("tests") or {}
        test_icon = "+" if tests.get("success") else "x"
        con.print(
            f"  {test_icon} Tests: {tests.get('passed', 0)}/{tests.get('total', 0)} passed "
            f"({_format_pass_rate(tests)})"
        )

        security = res.get("security")
        if security:
            sec_icon = "+" if security.get("success") else "x"
            con.print(f"  {sec_icon} Security: {security.get('score', 0):.0f}/100")

        _print_assay_perf(con, res)
        _print_assay_a11y(con, res)
        _print_assay_quality_score(con, res)

    return _human_text(_build)


def _render_score(res: dict[str, Any]) -> str:
    def _build(con: Console) -> None:
        if "scores" in res:
            scores = res.get("scores") or []
            if not scores:
                con.print(f"[yellow]No scores found for {res.get('project', '')}[/yellow]")
                if res.get("message"):
                    con.print(f"[yellow]{res['message']}[/yellow]")
                return

            table = Table(title=f"Score History: {res.get('project', '')}")
            table.add_column("Timestamp")
            table.add_column("Mode")
            table.add_column("Score", justify="right")
            table.add_column("Ship Gate")

            for s in scores:
                gate = "[green]PASS[/green]" if s.get("ship_gate") else "[red]FAIL[/red]"
                table.add_row(
                    s.get("timestamp", ""), s.get("mode", ""), f"{s.get('composite', 0):.1f}", gate
                )

            con.print(table)
            return

        gate_icon = "[green]PASS[/green]" if res.get("ship_gate") else "[red]FAIL[/red]"
        project = res.get("project", "")
        timestamp = res.get("timestamp", "")
        con.print(f"\n[bold cyan]{project}[/bold cyan] — {timestamp}")
        con.print(f"  Mode: {res.get('mode', '')}")
        con.print(
            f"  [bold]Score: {res.get('composite', 0):.1f}/100[/bold]  Ship Gate: {gate_icon}"
        )

        for dim in res.get("dimensions") or []:
            con.print(
                f"    {dim.get('name')}: {dim.get('score', 0):.0f} x "
                f"{dim.get('weight', 0):.2f} = {dim.get('weighted_score', 0):.1f}"
            )

    return _human_text(_build)


def _echo_human_inventory_message(result: dict[str, Any]) -> None:
    """Print a store-inventory `.message` that `emit()`'s error path would drop.

    `emit()`'s `--human` error path only ever prints the bare `.error` string
    (+ `.hint`, always empty for a plain dict return) — the honest-inventory
    `.message` this tool adds alongside `.error` (see `core/inventory.py`)
    would otherwise be silently dropped for `--human` runs; it survives
    unaided only in the JSON envelope's `error.details`.
    """
    if result.get("error") and result.get("message"):
        click.echo(result["message"], err=True)


_ERROR_CATEGORY_MAP: dict[str, tuple[str, str, int]] = {
    "not_found": ("NOT_FOUND", "not_found", 1),
    "personal_device": ("PERSONAL_DEVICE", "permission", 1),
    "validation": ("VALIDATION_ERROR", "validation", 2),
}

_BVT_REASON_CATEGORIES: dict[str, str] = {
    "unknown-project": "not_found",
    "suite-dir-missing": "not_found",
    "personal_device": "permission",
    "device-reserved": "unavailable",
    "device-not-ready": "unavailable",
    "nerve-daemon-unreachable": "unavailable",
}


def _category_error(result: Any) -> ToolError | None:
    """Build a categorized `ToolError` for a tool's error dict, or `None` if not applicable.

    Maps a dict with a truthy `error` key and `error_category` in `_ERROR_CATEGORY_MAP`
    (`not_found`, `personal_device`, `validation`) to a `ToolError` with the mapped code,
    category (`not_found` / `permission` / `validation`), and exit code (`2` for
    `validation`, `1` otherwise) instead of `linc_cli_kit`'s blanket `internal`.
    """
    if not isinstance(result, dict) or not result.get("error"):
        return None
    mapped = _ERROR_CATEGORY_MAP.get(result.get("error_category"))
    if mapped is None:
        return None
    code, category, exit_code = mapped
    details = {k: v for k, v in result.items() if k not in ("error", "validation_error")}
    return ToolError(
        str(result["error"]),
        code=code,
        category=category,
        details=details,
        exit_code=exit_code,
    )


def _bvt_not_ran_error(result: dict[str, Any]) -> ToolError:
    """Build a `ToolError` for a BVT payload with `ran is False`."""
    reason = str(result.get("reason") or result.get("error_category") or "")
    detail = str(result.get("detail") or result.get("error") or "")
    code = reason.replace("-", "_").upper()
    category = _BVT_REASON_CATEGORIES.get(reason) or "internal"
    return ToolError(detail, code=code, category=category, details=result)


def _cli_result_or_raise(result: dict[str, Any]) -> dict[str, Any]:
    """Raise a CLI `ToolError` on `ran: false`, `status: 'unmeasured'`, or `error_category`."""
    if result.get("ran") is False:
        raise _bvt_not_ran_error(result)
    if result.get("status") == "unmeasured":
        project = result.get("project") or ""
        raise ToolError(
            f"Assay for project '{project}' measured nothing",
            code="UNMEASURED",
            category="internal",
            details=result,
        )
    err = _category_error(result)
    if err is not None:
        raise err
    return result


def _exit_if_categorized(ctx: click.Context, result: Any) -> None:
    """Exit via `emit_exception` when `_category_error` recognizes `result`'s category.

    Split out of `score()` so the branch this adds doesn't grow that command's own
    complexity — `ctx.exit()` raises (see `click.Context.exit`), so control never returns
    here and `score()` never falls through to its own `emit()` call below.
    """
    err = _category_error(result)
    if err is not None:
        ctx.exit(emit_exception(err, mode=current_mode()))


@cli.command()
@click.argument("project")
@click.option("--history", is_flag=True, help="Show score history")
@click.option("--limit", default=10, help="Number of history entries")
def score(project: str, history: bool, limit: int) -> None:
    """Show quality score for a project.

    PROJECT: Forge project name

    JSON output pins `.score` (number) and `.ship_gate` (boolean) alongside the tool's own
    fields. `--human` renders the same table this command has always rendered.
    """
    from linc_cli_kit.registry import registry_of

    ctx = click.get_current_context()
    transport = ctx.find_root().command.linc_transport
    spec = next(s for s in registry_of(_registry_server()) if s.name == "crucible_score")
    payload = {"project": project, "history": history, "limit": limit}
    result = transport.call(spec, payload, timeout=None)

    if isinstance(result, dict) and not result.get("error") and "composite" in result:
        # Pin `.score` as an alias of the scorer's `.composite` field — the JSON contract
        # a caller who has never heard of "composite" can still read (`.ship_gate` is
        # already the tool's own field name, no alias needed). Gated on `composite`
        # actually being present: `--history` payloads (`{project, count, scores}`) have
        # no `composite` field, so without this guard every history call would get a
        # literal `"score": null` injected.
        result["score"] = result["composite"]

    if current_mode() == "human" and isinstance(result, dict):
        _echo_human_inventory_message(result)

    _exit_if_categorized(ctx, result)

    ctx.exit(emit(result, mode=current_mode(), renderer=_render_score, tool="score"))


# --- Bellows command (MCP server) ---


@cli.command()
def bellows() -> None:
    """Start the MCP server — keeps the crucible bubbling.

    \b
    Runs a stdio-based MCP server for Claude Code integration.

    Configure in ~/.claude.json:
        {
          "mcpServers": {
            "linc-crucible": {
              "command": "crucible",
              "args": ["bellows"],
              "type": "stdio"
            }
          }
        }
    """
    import asyncio

    from .mcp_server import server
    from .stdio import stdio_server

    async def _run_stdio() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server._mcp_server.run(
                read_stream,
                write_stream,
                server._mcp_server.create_initialization_options(),
            )

    asyncio.run(_run_stdio())


# --- Fuelcheck command ---


def _emit_probes(probes: list[_fuel.Probe], *, as_json: bool) -> None:
    """Print probes as one JSON list (``--json``) or one Rich-coloured line each."""
    if as_json:
        click.echo(json.dumps([p.to_dict() for p in probes]))
        return
    for probe in probes:
        console.print(_fuel.format_rich_line(probe), soft_wrap=True, highlight=False)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Print the probes as a JSON list.")
@click.option("--env-table", is_flag=True, hidden=True, help="Print the README env table.")
def fuelcheck(as_json: bool, env_table: bool) -> None:
    """Fuel check — probe the environment; exit 1 only on a broken install or config file.

    Missing optional pieces (LINC workspace, projects, fleet registry, ~/.claude.json, docker,
    the nerve daemon) are reported as warn/skip lines with a one-line hint.
    """
    if env_table:
        click.echo(render_env_table())
        return
    probes = _fuel.run_probes(_fuel.default_probe_env())
    _emit_probes(probes, as_json=as_json)
    sys.exit(_fuel.exit_code(probes))


# --- Baseline commands ---


@cli.group()
def baseline() -> None:
    """Baseline management — capture, list, and prune visual baselines."""
    pass


@contextlib.contextmanager
def _store_name_usage_errors() -> Iterator[None]:
    """Turn a refused store name into a usage error: exit 2, validation envelope on stderr.

    The baseline group calls ``BaselineManager`` directly (bypassing the MCP
    layer), so a hostile PROJECT/SCREEN surfaces here as ``InvalidStoreName``.
    A context manager keeps the try/except out of each command's complexity.
    """
    from .core.store_names import InvalidStoreName

    try:
        yield
    except InvalidStoreName as exc:
        err = UsageError(str(exc), details={"field": exc.field})
        click.get_current_context().exit(emit_exception(err, mode=current_mode()))


def _build_baseline_capture_payload(entry: Any) -> dict[str, Any]:
    return {
        **entry.to_dict(),
        "dimensions": f"{entry.width}x{entry.height}",
        "message": f"Baseline captured: {entry.project}/{entry.screen_name}",
    }


def _render_baseline_capture(payload: dict[str, Any]) -> str:
    def _build(con: Console) -> None:
        con.print(
            f"[green]Baseline captured:[/green] {payload['project']}/{payload['screen_name']}"
        )
        con.print(f"  Timestamp: {payload['capture_time']}")
        con.print(f"  Dimensions: {payload['dimensions']}")
        con.print(f"  Path: {payload['screenshot_path']}")

    return _human_text(_build)


@baseline.command()
@click.argument("project")
@click.argument("screen_name")
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--device", default="", help="Device info string")
@click.option("--app-version", default="", help="App version")
def capture(
    project: str, screen_name: str, image_path: str, device: str, app_version: str
) -> None:
    """Capture a baseline screenshot for a project screen."""
    from .core.baseline import BaselineManager

    mgr = BaselineManager()
    with _store_name_usage_errors():
        entry = mgr.capture(
            project=project,
            screen_name=screen_name,
            image_path=Path(image_path),
            device_info=device,
            app_version=app_version,
        )
    payload = _build_baseline_capture_payload(entry)
    ctx = click.get_current_context()
    ctx.exit(
        emit(
            payload,
            mode=current_mode(),
            renderer=_render_baseline_capture,
            tool="baseline_capture",
        )
    )


def _build_baseline_payload(project: str, screen: str, entries: list[Any]) -> dict[str, Any]:
    """Build the `baseline list` JSON payload: each entry's `to_dict()` plus `dimensions`."""
    payload: dict[str, Any] = {
        "project": project,
        "screen": screen,
        "count": len(entries),
        "baselines": [{**e.to_dict(), "dimensions": f"{e.width}x{e.height}"} for e in entries],
    }
    if not entries:
        payload["message"] = "No baselines found."
    return payload


def _render_baseline_list(payload: dict[str, Any]) -> str:
    def _build(con: Console) -> None:
        entries = payload.get("baselines") or []
        if not entries:
            con.print("[yellow]No baselines found.[/yellow]")
            return

        project = payload.get("project", "")
        screen = payload.get("screen", "")
        table = Table(title=f"Baselines for {project}" + (f"/{screen}" if screen else ""))
        table.add_column("Screen", style="bold")
        table.add_column("Timestamp")
        table.add_column("Dimensions")
        table.add_column("Device")
        table.add_column("Version")

        for e in entries:
            table.add_row(
                e["screen_name"],
                e["capture_time"],
                e["dimensions"],
                e["device_info"] or "—",
                e["app_version"] or "—",
            )

        con.print(table)

    return _human_text(_build)


@baseline.command("list")
@click.argument("project")
@click.option("--screen", default="", help="Filter by screen name")
def baseline_list(project: str, screen: str) -> None:
    """List stored baselines for a project."""
    from .core.baseline import BaselineManager

    mgr = BaselineManager()
    with _store_name_usage_errors():
        entries = mgr.list_baselines(project, screen_name=screen)
    payload = _build_baseline_payload(project, screen, entries)

    ctx = click.get_current_context()
    ctx.exit(
        emit(payload, mode=current_mode(), renderer=_render_baseline_list, tool="baseline_list")
    )


def _build_baseline_prune_payload(
    project: str, screen: str, keep: int, count: int
) -> dict[str, Any]:
    return {
        "project": project,
        "screen": screen,
        "keep": keep,
        "pruned": count,
        "message": (
            f"Pruned {count} baseline(s) (kept newest {keep})" if count else "Nothing to prune."
        ),
    }


def _render_baseline_prune(payload: dict[str, Any]) -> str:
    def _build(con: Console) -> None:
        count = payload.get("pruned", 0)
        keep = payload.get("keep", 5)
        if count:
            con.print(f"[yellow]Pruned {count} baseline(s)[/yellow] (kept newest {keep})")
        else:
            con.print("[green]Nothing to prune.[/green]")

    return _human_text(_build)


@baseline.command()
@click.argument("project")
@click.option("--screen", default="", help="Prune specific screen only")
@click.option("--keep", default=5, help="Number of baselines to keep per screen")
def prune(project: str, screen: str, keep: int) -> None:
    """Prune old baselines beyond retention limit."""
    from .core.baseline import BaselineManager

    mgr = BaselineManager()
    with _store_name_usage_errors():
        count = mgr.prune(project, screen_name=screen, keep=keep)
    payload = _build_baseline_prune_payload(project, screen, keep, count)
    ctx = click.get_current_context()
    ctx.exit(
        emit(
            payload,
            mode=current_mode(),
            renderer=_render_baseline_prune,
            tool="baseline_prune",
        )
    )


# --- Logo command ---


@cli.command()
def logo() -> None:
    """Display the Crucible banner."""
    console.print(CRUCIBLE_BANNER)


# --- linc-cli-kit mount (MCP-to-CLI parity) ---


def _registry_server() -> Any:
    """Return the FastMCP server with every tool module imported (registration is at import).

    Deliberately does NOT import ``linc_crucible.mcp_server`` — that module wires the stdio
    MCP entrypoint (``bellows``); a CLI process only needs the tool registry, not the stdio
    transport. The six modules below are exactly the ones that decorate a function with
    ``@server.tool()`` (grep: ``deploy_tools``, ``assay_tools`` — assay + score + preflight,
    ``baseline_tools``, ``asset_tools``, ``bvt_tools``, ``stats_tools``).
    """
    import importlib

    for module_path in (
        "linc_crucible.deploy_tools",
        "linc_crucible.assay_tools",
        "linc_crucible.baseline_tools",
        "linc_crucible.asset_tools",
        "linc_crucible.bvt_tools",
        "linc_crucible.stats_tools",
    ):
        importlib.import_module(module_path)
    return server_ctx.server


# Two tools pre-serialize their own return value to a JSON *string* rather than returning a
# dict (`crucible_extract_assets`, `crucible_bvt` — see their docstrings: "Returns: JSON
# string: ..."), unlike every other tool in this registry. Left as-is, the kit's `emit()`
# would json.dumps() that string AGAIN (double-encoded stdout: a quoted, escaped JSON blob)
# and `classify()` would never see their `{"error": ...}` payloads (a bare `str` never
# matches `isinstance(result, dict)`, so a known failure would report exit 0). Parsing the
# string back to a dict at this CLI-layer boundary — never touching asset_tools.py/
# bvt_tools.py themselves, which are out of scope — restores both single-encoding and error
# classification. Mirrors linc-nerve's `_local_fn` pattern (cli/main.py).
_JSON_STRING_TOOLS = frozenset({"crucible_extract_assets", "crucible_bvt"})


def _decode_tool_result(spec_name: str, raw: Any) -> dict[str, Any]:
    import json

    data = json.loads(raw) if spec_name in _JSON_STRING_TOOLS else raw
    if spec_name == "crucible_score":
        return data
    return _cli_result_or_raise(data)


def _local_fn(spec: Any) -> Any:
    import inspect

    fn = spec.fn
    if inspect.iscoroutinefunction(fn):

        async def _async_wrapped(**payload: Any) -> Any:
            return _decode_tool_result(spec.name, await fn(**payload))

        return _async_wrapped

    def _sync_wrapped(**payload: Any) -> Any:
        return _decode_tool_result(spec.name, fn(**payload))

    return _sync_wrapped


mount_tools(
    cli,
    _registry_server(),
    prefix="crucible_",
    transport=LocalTransport(fn_for=_local_fn, before=server_ctx._ensure_initialized),
    skip={
        "crucible_baseline": (
            "the hand-written `baseline` group (capture/list/prune, defined above) already "
            "covers this tool's screenshot-mode surface. `crucible_baseline` also supports "
            "an a11y mode and latest/promote actions the group doesn't expose yet — left "
            "MCP-only pending a Phase 2 CLI surface, not silently dropped."
        )
    },
    renderers={"pour": _render_pour, "assay": _render_assay},
)


def main() -> None:
    """Main entry point.

    Structlog is already configured at module-import time above (WARNING-default,
    `LINC_LOG_LEVEL`-driven) via `_logging.configure_cli_structlog()` — calling
    `_logging.configure()` here would re-point structlog at stdlib logging with an INFO
    default, contradicting that default and re-enabling the `[debug]`/`[info]` stderr leak
    this module's import-time call exists to prevent.
    """
    cli()


if __name__ == "__main__":
    main()
