"""Deployer — build, find artifact, install, and launch on fleet devices.

Orchestrates the ``crucible pour`` command: build the project, locate
the artifact (APK or .app), install on target device(s), optionally launch.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from linc_crucible.core.subprocess_utils import (
    CommandParseError,
    ShellMetacharError,
    resolve_java_home,
    safe_subprocess_run,
    trim_output,
)
from linc_crucible.core.tool_args import check_build_type

from . import ios_signing
from .app_id import (  # noqa: F401 — re-exported: pre-C4 import path compatibility
    _infer_bundle_id,
    _infer_package,
    _parse_application_id,
    _parse_application_id_from_file,
    _parse_bundle_id,
)
from .device import Platform, is_simulator_udid
from .device_policy import DevicePolicy
from .forge_integration import ForgeProject, ForgeProjectReader, ModuleConfig
from .nerve_client import NerveClient, find_reservation

logger = structlog.get_logger().bind(domain="build_pipeline")

_DEFAULT_BUILD_TIMEOUT = 600
_BUILD_TIMEOUT_ENV = "LINC_CRUCIBLE_BUILD_TIMEOUT"


def _build_timeout_seconds() -> int:
    """Build timeout in seconds — ``LINC_CRUCIBLE_BUILD_TIMEOUT`` override, default 600.

    Invalid or non-positive values fall back to the default (with a warning) so a
    typo'd env var can never disable or zero the timeout.
    """
    raw = os.environ.get(_BUILD_TIMEOUT_ENV)
    if not raw:
        return _DEFAULT_BUILD_TIMEOUT
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning(
            "invalid_build_timeout_env",
            env_var=_BUILD_TIMEOUT_ENV,
            value=raw,
            fallback=_DEFAULT_BUILD_TIMEOUT,
        )
        return _DEFAULT_BUILD_TIMEOUT
    return value


@dataclass
class DeployResult:
    """Result of deploying to a single device."""

    device_serial: str
    device_model: str = ""
    platform: str = ""
    build_success: bool = True
    artifact_path: str = ""
    install_success: bool = False
    launch_success: bool = False
    error: str = ""
    module: str = ""
    provenance: dict = field(default_factory=dict)
    artifact_older_than_head: bool | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d = {
            "device_serial": self.device_serial,
            "device_model": self.device_model,
            "platform": self.platform,
            "build_success": self.build_success,
            "artifact_path": self.artifact_path,
            "install_success": self.install_success,
            "launch_success": self.launch_success,
            "error": self.error,
        }
        if self.module:
            d["module"] = self.module
        d.update(_provenance_fields(self.provenance, self.artifact_older_than_head))
        return d


_NO_BUILD_COMMAND_MSG = (
    "No build command configured; pass skip_build=True to install the existing artifact"
)

_BUILD_ONLY_NO_COMMAND_MSG = (
    "No build command configured; build_only has nothing to build"
    " (set build_command in the project manifest)"
)

NERVE_UNREACHABLE_MSG = "Nerve unreachable at http://localhost:8111 — start it with `nerve daemon`"


def _build_only_output(output: str) -> str:
    """Swap the deploy-oriented no-build-command hint for one that fits build_only.

    ``skip_build=True`` is meaningless advice for a build-only pour, which
    exists precisely to run the build.
    """
    return output.replace(_NO_BUILD_COMMAND_MSG, _BUILD_ONLY_NO_COMMAND_MSG)


def _find_module(project: ForgeProject, name: str) -> ModuleConfig | None:
    """Return the project's module called *name*, or None."""
    return next((m for m in project.modules if m.name == name), None)


def _module_not_found(project: ForgeProject, module: str) -> str:
    """Return the error for a module the project does not declare."""
    return f"Module '{module}' not found in project '{project.name}'"


def _pour_succeeded(result: PourResult) -> bool:
    """Return whether a pour succeeded.

    A build-only pour succeeds when its build does; a deploying pour succeeds
    when at least one device install did.
    """
    if result.build_only:
        return result.build_success
    return any(r.install_success for r in result.results)


def _device_section(result: PourResult) -> dict:
    """Return the target part of a pour payload.

    Build-only pours never touch devices, so they report ``build_only: True``
    instead of a ``devices`` list.
    """
    if result.build_only:
        return {"build_only": True}
    return {"devices": [r.to_dict() for r in result.results]}


def _build_only_platforms(project: ForgeProject, platform: str) -> set[Platform]:
    """Return the platforms a build-only pour should build.

    An explicit ``android``/``ios`` platform wins; otherwise build every platform
    the project has a build command for, defaulting to Android.
    """
    norm = platform.lower()
    if norm in ("android", "ios"):
        return {Platform(norm)}
    needed: set[Platform] = set()
    if project.build_command:
        needed.add(Platform.ANDROID)
    if project.ios_build_command:
        needed.add(Platform.IOS)
    return needed or {Platform.ANDROID}


@dataclass
class PourResult:
    """Result of a complete pour operation across devices."""

    project: str
    build_output: str = ""
    results: list[DeployResult] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    timed_out: bool = False
    build_only: bool = False
    build_success: bool = True
    provenance: dict = field(default_factory=dict)
    artifact_older_than_head: bool | None = None
    excluded_personal: list[str] = field(default_factory=list)
    refusal: str = ""

    @property
    def success(self) -> bool:
        """True if at least one device deployed successfully, or build succeeded."""
        return _pour_succeeded(self)

    @property
    def error(self) -> str:
        """Summary error when no devices were deployed."""
        if self.results:
            errors = [r.error for r in self.results if r.error]
            return "; ".join(errors) if errors else ""
        if self.build_output:
            # Truncate build output to last meaningful lines
            lines = self.build_output.strip().splitlines()
            return "\n".join(lines[-10:]) if len(lines) > 10 else self.build_output.strip()
        return ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d: dict = {
            "project": self.project,
            "success": self.success,
            **_device_section(self),
        }
        if not self.success:
            err = self.error
            if err:
                d["error"] = err
            if self.build_output and not self.results:
                d["build_error"] = True
            if self.timed_out:
                # Machine-distinguish "build hit the wall clock" from a real
                # compile failure (compiler output present, no timeout).
                d["reason"] = "build-timeout"
        if self.artifacts:
            d["artifacts"] = self.artifacts
        d.update(_provenance_fields(self.provenance, self.artifact_older_than_head))
        d.update(_policy_fields(self))
        return d


PERSONAL_DEVICE_CATEGORY = "personal_device"


def personal_device_error(refusal: str) -> dict:
    """Uniform tool-level refusal payload for a personal (or unverifiable) device."""
    return {"success": False, "error": refusal, "error_category": PERSONAL_DEVICE_CATEGORY}


def _policy_fields(result: PourResult) -> dict:
    """Personal-device fields for ``PourResult.to_dict`` (empty when the policy was silent)."""
    fields: dict = {}
    if result.excluded_personal:
        fields["excluded_personal"] = list(result.excluded_personal)
    if result.refusal:
        fields.update(personal_device_error(result.refusal))
    return fields


_DEVICE_ID_KEYS = ("serial", "id", "hardware_id")


def _device_is_personal(device: dict, policy: DevicePolicy) -> bool:
    """True when a discovered device's serial, id, or hardware_id is registered personal."""
    ids = [device.get(key) for key in _DEVICE_ID_KEYS]
    return any(policy.is_personal(str(i)) for i in ids if i)


def _device_matches(device: dict, target: str) -> bool:
    """True when ``target`` matches a device's serial, id, or hardware_id (casefolded)."""
    cleaned = str(target).strip().casefold()
    if not cleaned:
        return False
    return any(
        str(device.get(k)).strip().casefold() == cleaned for k in _DEVICE_ID_KEYS if device.get(k)
    )


def _apply_device_policy(
    devices: list[dict],
    explicit: str,
    policy: DevicePolicy | None,
    allow_personal: bool,
) -> tuple[list[dict], list[str], str | None]:
    """Gate candidate deploy targets through the personal-device policy.

    Returns ``(allowed targets, excluded personal serials, refusal text)``:

    * no policy, or ``allow_personal`` -> everything passes untouched;
    * an explicit serial (or a registry that could not be read — fail
      closed) -> nothing passes and the refusal text says why;
    * fan-out -> personal devices are dropped and reported as excluded.
    """
    if policy is None or allow_personal:
        return list(devices), [], None
    if explicit or not policy.loaded:
        refusal = policy.refusal(explicit)
        if not refusal:
            for d in devices:
                if _device_matches(d, explicit) and _device_is_personal(d, policy):
                    refusal = (
                        f"device {explicit} is registered personal: true in {policy.source}; "
                        "pass allow_personal=True to override"
                    )
                    break
        return ([], [], refusal) if refusal else (list(devices), [], None)
    allowed = [d for d in devices if not _device_is_personal(d, policy)]
    excluded = [str(d.get("serial", "")) for d in devices if _device_is_personal(d, policy)]
    return allowed, excluded, None


def discover_for_policy(nerve: Any) -> list[dict]:
    """Return discovered devices for policy checks, or ``[]`` on ``ConnectionError``."""
    with contextlib.suppress(ConnectionError):
        discovered = nerve.discover()
        if isinstance(discovered, list):
            return discovered
    return []


def _discover_or_raise(nerve: Any) -> list[dict]:
    """Discover devices via nerve, probing health when empty so a down daemon raises."""
    devices = nerve.discover()
    if not devices:
        nerve.health()
    return devices


def _run_pour_catching_nerve_unreachable(fn: Any, project_name: str, **kwargs: Any) -> PourResult:
    """Run a pour function and convert nerve ConnectionError into a structured PourResult."""
    try:
        return fn(project_name, **kwargs)
    except ConnectionError:
        return PourResult(project=project_name, build_output=NERVE_UNREACHABLE_MSG)


def resolve_policy_target(
    devices: list[dict],
    explicit: str,
    policy: DevicePolicy | None,
    allow_personal: bool,
) -> tuple[dict | None, str | None]:
    """Resolve one target device through the personal-device policy."""
    allowed, excluded, refusal = _apply_device_policy(devices, explicit, policy, allow_personal)
    if refusal:
        return None, refusal
    if explicit:
        return {"serial": explicit, "id": explicit}, None
    if allowed:
        return allowed[0], None
    if excluded:
        return (
            None,
            f"No non-personal device available (excluded personal: {', '.join(excluded)}); "
            "pass allow_personal=True to override",
        )
    return None, None


def _excluded_personal_note(excluded: list[str], source: str) -> str:
    return (
        f"Excluded personal device(s) {', '.join(excluded)} (registered personal: true "
        f"in {source}); pass allow_personal=True to target them."
    )


class _PolicyContext(threading.local):
    """Per-thread personal-device policy for the pour in flight.

    The Deployer is shared by concurrent MCP calls (one worker thread each), so
    the policy must never live on the instance — one caller's allow_personal
    would leak into another caller's fan-out.
    """

    def __init__(self) -> None:
        self.policy: DevicePolicy | None = None
        self.allow_personal = False
        self.excluded: list[str] = []

    def begin(self, policy: DevicePolicy | None, allow_personal: bool) -> None:
        self.policy, self.allow_personal, self.excluded = policy, allow_personal, []

    def end(self) -> list[str]:
        excluded = self.excluded
        self.begin(None, False)
        return excluded

    def filter(self, explicit: str, devices: list[dict]) -> list[dict]:
        allowed, excluded, _ = _apply_device_policy(
            devices, explicit, self.policy, self.allow_personal
        )
        self.excluded.extend(s for s in excluded if s not in self.excluded)
        return allowed


_GIT_LOG_TIMEOUT_S = 10

_STALE_ARTIFACT_WARNING = (
    "Warning: installed artifact predates HEAD commit ({path}, artifact_older_than_head=True)"
)


def _artifact_mtime(path: str) -> float | None:
    """Return the artifact's mtime (epoch seconds), or None if it cannot be read."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _head_commit_iso(project_dir: str) -> str | None:
    """Return HEAD's committer date (ISO 8601) for *project_dir*, or None.

    None covers every "can't tell" case: no directory, git missing, a
    non-repo, a timeout, or a path the subprocess helper refuses.
    """
    if not (project_dir and os.path.isdir(project_dir)):
        return None
    cmd = f"git -C {shlex.quote(project_dir)} log -1 --format=%cI"
    try:
        proc = safe_subprocess_run(cmd, cwd=project_dir, timeout=_GIT_LOG_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    out = (proc.stdout or "").strip()
    return out if proc.returncode == 0 and out else None


def _parse_commit_time(raw: str | None) -> float | None:
    """Parse a git ``%cI`` timestamp to epoch seconds (naive treated as UTC)."""
    if raw is None:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _is_older(artifact_ts: float | None, head_ts: float | None) -> bool | None:
    """True when the artifact strictly predates HEAD; None when either is unknown."""
    if artifact_ts is None or head_ts is None:
        return None
    return artifact_ts < head_ts


def _artifact_provenance(path: str, project_dir: str) -> dict:
    """Return provenance metadata for an artifact relative to git HEAD.

    Args:
        path: Path to the artifact on disk.
        project_dir: Project root directory containing the git repository.

    Returns:
        dict with ``artifact_mtime`` (ISO 8601), ``artifact_age_s`` (float),
        ``head_committed_at`` (ISO 8601) and ``artifact_older_than_head``
        (bool). Any field that cannot be determined is None; a missing
        artifact yields all-None without invoking git.
    """
    mtime = _artifact_mtime(path)
    if mtime is None:
        return {
            "artifact_mtime": None,
            "artifact_age_s": None,
            "head_committed_at": None,
            "artifact_older_than_head": None,
        }
    head_raw = _head_commit_iso(project_dir)
    head_ts = _parse_commit_time(head_raw)
    return {
        "artifact_mtime": datetime.fromtimestamp(mtime, tz=UTC).isoformat(),
        "artifact_age_s": float(time.time() - mtime),
        "head_committed_at": head_raw if head_ts is not None else None,
        "artifact_older_than_head": _is_older(mtime, head_ts),
    }


def _provenance_fields(provenance: dict, older_than_head: bool | None) -> dict:
    """Return the optional provenance keys a result serialises (empty when unset)."""
    fields: dict = {}
    if provenance:
        fields["provenance"] = provenance
    if older_than_head is not None:
        fields["artifact_older_than_head"] = older_than_head
    return fields


def _attach_provenance(deploy_result: DeployResult, project_dir: str) -> None:
    """Record the installed artifact's provenance on *deploy_result* (no-op without one)."""
    if not deploy_result.artifact_path:
        return
    prov = _artifact_provenance(deploy_result.artifact_path, project_dir)
    deploy_result.provenance = prov
    deploy_result.artifact_older_than_head = prov["artifact_older_than_head"]


def _stale_artifact_warning(artifact_path: str) -> str:
    """Return the build_output warning line for an artifact older than HEAD."""
    return _STALE_ARTIFACT_WARNING.format(path=artifact_path)


def _append_lines(text: str, lines: list[str]) -> str:
    """Join *text* and *lines* with newlines, skipping empty parts."""
    return "\n".join(part for part in (text, *lines) if part)


def _aggregate_staleness(flags: Iterable[bool | None]) -> bool | None:
    """True if any flag is True, else False if any is False, else None."""
    seen = set(flags)
    if True in seen:
        return True
    if False in seen:
        return False
    return None


def artifacts_older_than_head(paths: Iterable[str], project_dir: str) -> bool | None:
    """Aggregate ``artifact_older_than_head`` over several artifacts.

    Args:
        paths: Artifact paths on disk.
        project_dir: Project root directory containing the git repository.

    Returns:
        True if any artifact predates HEAD, False if all known ones do not,
        None when staleness could not be determined for any of them.
    """
    return _aggregate_staleness(
        _artifact_provenance(p, project_dir)["artifact_older_than_head"] for p in paths
    )


def _finalize_pour_provenance(result: PourResult) -> None:
    """Summarise per-device provenance onto *result* and warn about stale installs."""
    warnings = [
        _stale_artifact_warning(r.artifact_path)
        for r in result.results
        if r.artifact_older_than_head
    ]
    result.build_output = _append_lines(result.build_output, warnings)
    result.artifact_older_than_head = _aggregate_staleness(
        r.artifact_older_than_head for r in result.results
    )
    result.provenance = next((r.provenance for r in result.results if r.provenance), {})


class Deployer:
    """Orchestrates build -> find artifact -> install -> launch."""

    # Class-level and thread-local: per-call state, safe across concurrent pours.
    _policy_ctx = _PolicyContext()

    def __init__(
        self,
        forge_reader: ForgeProjectReader | None = None,
        nerve_client: NerveClient | None = None,
    ) -> None:
        self._forge = forge_reader or ForgeProjectReader()
        self._nerve = nerve_client or NerveClient()
        self._build_timed_out = False

    def pour(
        self,
        project_name: str,
        device: str = "",
        platform: str = "",
        launch: bool = False,
        skip_build: bool = False,
        module: str = "",
        device_type: str = "",
        launch_intent: str = "",
        build_type: str = "debug",
        policy: DevicePolicy | None = None,
        allow_personal: bool = False,
        build_only: bool = False,
    ) -> PourResult:
        """Deploy a project to target device(s), or only build it.

        Args:
            project_name: Forge project name.
            device: Specific device serial, or empty for all matching.
            platform: Filter by platform ("android" or "ios").
            launch: Whether to launch the app after install.
            skip_build: Skip the build step, use existing artifact.
            module: Deploy only this module (for multi-target projects).
            device_type: Filter devices by type ("phone", "watch", etc.).
            build_type: Gradle build type for module-scoped Android builds
                ("debug" or "release"); only consulted when the build command
                is auto-derived — an explicit build_command always wins.
            policy: Personal-device policy. ``None`` disables the gate; the MCP
                tool always passes the server's loaded policy.
            allow_personal: Explicit opt-in to target ``personal: true`` devices.
            build_only: Only build — never resolve devices, install, or launch.
                Dispatched before multi-target/module routing, so it holds for
                every project shape; ``skip_build`` is ignored. It also bypasses
                the personal-device gate, which guards device access only: a
                build-only pour cannot reach a device, so ``device`` is ignored
                and nothing is refused.

        Returns:
            PourResult with per-device results, or a build-only result with
            ``build_only`` set and no devices. A refused explicit personal
            serial (or an unreadable registry) carries ``refusal``; fan-out
            skips personal devices and lists them in ``excluded_personal``.
        """
        # Reset per-pour so an earlier timed-out build never leaks into this
        # pour's payload; _run_build sets the flag on TimeoutExpired.
        self._build_timed_out = False
        if build_only:
            # No device is ever touched on this path, so the policy gate is moot.
            result = self._pour_build_only(
                project_name, module=module, platform=platform, build_type=build_type
            )
        else:
            result = self._policy_gated_pour(
                project_name,
                policy,
                allow_personal,
                device=device,
                platform=platform,
                launch=launch,
                skip_build=skip_build,
                module=module,
                device_type=device_type,
                launch_intent=launch_intent,
                build_type=build_type,
            )
        result.timed_out = self._build_timed_out
        return result

    def _policy_gated_pour(
        self,
        project_name: str,
        policy: DevicePolicy | None,
        allow_personal: bool,
        **kwargs: Any,
    ) -> PourResult:
        """Refuse before any build when the policy forbids the pour, else run it
        with the policy active so every target resolution skips personal devices."""
        devices = (
            discover_for_policy(self._nerve)
            if (policy is not None and not allow_personal and policy.loaded and kwargs["device"])
            else []
        )
        _, _, refusal = _apply_device_policy(devices, kwargs["device"], policy, allow_personal)
        if refusal:
            logger.warning("pour_refused_personal_device", device=kwargs["device"])
            return PourResult(project=project_name, refusal=refusal)
        self._policy_ctx.begin(policy, allow_personal)
        try:
            result = _run_pour_catching_nerve_unreachable(self._pour, project_name, **kwargs)
        finally:
            excluded = self._policy_ctx.end()
        if excluded:
            result.excluded_personal = excluded
            result.build_output = _append_lines(
                result.build_output, [_excluded_personal_note(excluded, policy.source)]
            )
        return result

    def _pour(
        self,
        project_name: str,
        *,
        device: str,
        platform: str,
        launch: bool,
        skip_build: bool,
        module: str,
        device_type: str,
        launch_intent: str,
        build_type: str = "debug",
    ) -> PourResult:
        """pour() body — see pour() for the contract; split so the public entry
        can bracket it with timed-out flag reset/copy on every return path."""
        result = PourResult(project=project_name)

        # Resolve project
        project = self._forge.get(project_name)
        if not project:
            result.build_output = f"Project '{project_name}' not found in forge registry"
            return result

        # Multi-target routing
        if project.is_multi_target and not module:
            return self._pour_multi_target(
                project,
                device=device,
                platform=platform,
                launch=launch,
                skip_build=skip_build,
                device_type=device_type,
                launch_intent=launch_intent,
            )

        # A module was explicitly requested — route it through the
        # single-module path even if the project has zero declared modules
        # (e.g. a manifest without deployment.modules, or a project that
        # predates module support). That path already turns "module not in
        # project.modules" into an explicit error; silently falling through
        # to the single-target pour below would ignore --module and install
        # whatever the single-target artifact glob happens to match.
        if module:
            return self._pour_single_module(
                project,
                module=module,
                device=device,
                platform=platform,
                launch=launch,
                skip_build=skip_build,
                device_type=device_type,
                launch_intent=launch_intent,
                build_type=build_type,
            )

        # Single-target path

        # Step 1: Resolve target devices first (so we know which platforms to build for)
        targets = self._resolve_targets(device, platform, device_type)
        if not targets:
            hint = (
                "\nNo matching devices found."
                " Ensure devices are connected and linc-nerve is running."
            )
            result.build_output += hint
            return result

        # Step 2: Build for needed platforms (unless skipped)
        if not skip_build:
            needed_platforms = {
                Platform(t.get("platform", "android"))
                for t in targets
                if t.get("platform") in ("android", "ios")
            }
            build_ok, build_output = self._build_for_platforms(
                project,
                needed_platforms,
                targets=targets,
            )
            result.build_output = build_output
            result.build_success = build_ok
            if not build_ok:
                return result

        # Collect artifact paths
        artifacts: dict[str, str] = {}
        apk_path = self._forge.find_apk(project.name)
        if apk_path:
            artifacts["android"] = apk_path
        ios_app = self._forge.find_ios_app(project.name)
        if ios_app:
            artifacts["ios"] = ios_app
        result.artifacts = artifacts

        # Step 3: Deploy to each target
        for dev in targets:
            deploy_result = self._deploy_to_device(project, dev, launch, launch_intent)
            result.results.append(deploy_result)
        _finalize_pour_provenance(result)

        return result

    def _pour_build_only(
        self,
        project_name: str,
        *,
        module: str,
        platform: str,
        build_type: str,
    ) -> PourResult:
        """Build a project without resolving targets or touching devices.

        Mirrors the build a deploying pour of the same shape would run —
        module-scoped, all-modules, or per-platform — then reports artifacts.
        """
        result = PourResult(project=project_name, build_only=True, build_success=False)
        project = self._forge.get(project_name)
        if not project:
            result.build_output = f"Project '{project_name}' not found in forge registry"
            return result

        build_ok, build_output = self._build_only_build(project, module, platform, build_type)
        result.build_success = build_ok
        result.build_output = _build_only_output(build_output)
        if build_ok:
            self._find_project_artifacts(project, result)
        return result

    def _build_only_build(
        self, project: ForgeProject, module: str, platform: str, build_type: str
    ) -> tuple[bool, str]:
        """Run the build matching the pour's routing, minus any device-derived choice."""
        if module:
            return self._build_only_module(project, module, platform, build_type)
        if project.is_multi_target:
            return self._build_modules(project)
        return self._build_for_platforms(
            project, _build_only_platforms(project, platform), targets=None
        )

    def _build_only_module(
        self, project: ForgeProject, module: str, platform: str, build_type: str
    ) -> tuple[bool, str]:
        """Build one named module, failing cleanly when the project lacks it."""
        module_config = _find_module(project, module)
        if module_config is None:
            return (False, _module_not_found(project, module))
        return self._build_module(project, module_config, platform, build_type)

    def _build_module(
        self, project: ForgeProject, module_config: ModuleConfig, platform: str, build_type: str
    ) -> tuple[bool, str]:
        """Build a single module: its own command, a scoped Gradle task, or the project build."""
        build_cmd = module_config.build_command or self._module_scoped_android_build_command(
            module_config, platform, build_type
        )
        if build_cmd:
            return self._run_build(build_cmd, project.path)
        return self._build(project)

    def _build(self, project: ForgeProject) -> tuple:
        """Build the project using Gradle. Returns (success, output)."""
        build_cmd = project.build_command
        if not build_cmd:
            return (False, project.config_error_for("build_command", _NO_BUILD_COMMAND_MSG))
        return self._run_build(build_cmd, project.path)

    def _build_for_platforms(
        self,
        project: ForgeProject,
        platforms: set[Platform],
        targets: list[dict] | None = None,
    ) -> tuple[bool, str]:
        """Build for the specified target platforms.

        Args:
            project: Forge project configuration.
            platforms: Set of platforms to build for.
            targets: Optional list of target device dicts. Used to extract the
                simulator UDID for xcodebuild ``-destination`` flag.

        Routes to Gradle for Android and xcodebuild for iOS.
        """
        outputs: list[str] = []

        if Platform.ANDROID in platforms:
            ok, output = self._build(project)
            outputs.append(output)
            if not ok:
                return (False, "\n".join(outputs))

        if Platform.IOS in platforms:
            ok, output = self._build_ios_platform(project, targets)
            outputs.append(output)
            if not ok:
                return (False, "\n".join(outputs))

        return (True, "\n".join(outputs))

    @staticmethod
    def _device_is_simulator(device: dict) -> bool:
        """True when *device* is virtual (simulator/emulator), independent of form factor.

        ``device_type`` alone conflates two axes: form factor (phone/watch/
        tablet) and virtual-vs-physical. That collapse is harmless for Android
        (emulator serials get ``device_type="emulator"``) and for iPhone-family
        simulators (``device_type`` is ``"emulator"``/``"simulator"``) — but a
        watch SIMULATOR still reports ``device_type="watch"`` (nerve matches
        watch simulators by name, same as physical Apple Watches — see
        ios_discovery.py), so it's indistinguishable from a physical Apple
        Watch by ``device_type`` alone.

        Falls back to the serial/UDID shape heuristic (``is_simulator_udid``)
        for iOS devices whose ``device_type`` doesn't already signal
        virtuality: simulator UDIDs are UUID-format (dashes), physical
        UDIDs aren't.
        """
        device_type = device.get("device_type", "")
        if device_type in ("emulator", "simulator"):
            return True
        if device.get("platform") == "ios":
            return is_simulator_udid(device.get("serial", ""))
        return False

    def _build_ios_platform(
        self,
        project: ForgeProject,
        targets: list[dict] | None = None,
    ) -> tuple[bool, str]:
        """Build the iOS side of a project for the targeted device.

        Resolves the target device (simulator vs physical), gates physical
        builds on a signing team, and runs the resolved xcodebuild command.
        """
        device_id = ""
        device_type_str = ""
        is_sim = True
        if targets:
            ios_targets = [t for t in targets if t.get("platform") == "ios"]
            if ios_targets:
                device_id = ios_targets[0].get("serial", "")
                device_type_str = ios_targets[0].get("device_type", "")
                is_sim = self._device_is_simulator(ios_targets[0])

        signing_team = ""
        if not is_sim:
            team = ios_signing.resolve_team_id()
            if not team:
                msg = (
                    "Physical iOS build requires a signing team. Set "
                    "ios_signing.team_id in the device fleet file "
                    "(LINC_DEVICE_FLEET_FILE, default $LINC_HOME/device-fleet.yaml) "
                    "or export CRUCIBLE_IOS_TEAM_ID. If a signed build later "
                    "fails on signing, verify the Apple Developer account is "
                    "active in Xcode → Settings → Accounts."
                )
                logger.error("ios_signing_team_missing", project=project.name)
                return (False, msg)
            signing_team = team

        ios_cmd = self._resolve_ios_build_command(
            project,
            device_id=device_id,
            device_type=device_type_str,
            is_simulator=is_sim,
            signing_team=signing_team,
        )
        if ios_cmd:
            return self._run_build(ios_cmd, project.path)

        msg = "No iOS build command configured and no iosApp/ detected, skipping iOS build"
        logger.info("ios_build_skipped", project=project.name, reason=msg)
        return (True, msg)

    def _find_watch_scheme_via_xcodebuild(self, root: Path, is_workspace: bool, name: str) -> str:
        import json
        import subprocess

        cmd = ["xcodebuild", "-list", "-json"]
        if is_workspace:
            cmd.extend(["-workspace", f"{name}.xcworkspace"])
        else:
            cmd.extend(["-project", f"{name}.xcodeproj"])

        try:
            res = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=15)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                schemes = data.get("project", {}).get("schemes", [])
                if not schemes:
                    schemes = data.get("workspace", {}).get("schemes", [])

                for s in schemes:
                    if "watch" in s.lower() and self._verify_scheme_is_watchos(
                        root, is_workspace, name, s
                    ):
                        return s
        except Exception:
            pass
        return ""

    def _verify_scheme_is_watchos(
        self, root: Path, is_workspace: bool, name: str, scheme: str
    ) -> bool:
        """Verify the scheme's SDKROOT is a watchOS SDK.

        Older Xcode emits the short name (``SDKROOT = watchos``); modern
        Xcode emits the full SDK path (``.../WatchOS26.4.sdk``) — match
        either, case-insensitively, on the SDKROOT line only.
        """
        import subprocess

        cmd = ["xcodebuild", "-showBuildSettings", "-scheme", scheme]
        if is_workspace:
            cmd.extend(["-workspace", f"{name}.xcworkspace"])
        else:
            cmd.extend(["-project", f"{name}.xcodeproj"])
        try:
            res = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=15)
            return any(
                "SDKROOT" in line and "watchos" in line.lower() for line in res.stdout.splitlines()
            )
        except Exception:
            return False

    def _resolve_platform_names(self, is_watch: bool, is_simulator: bool) -> tuple[str, str]:
        """Return (platform_name, sdk_name) for the given device type."""
        if is_watch:
            platform_name = "watchOS Simulator" if is_simulator else "watchOS"
            sdk_name = "watchsimulator" if is_simulator else "watchos"
        else:
            platform_name = "iOS Simulator" if is_simulator else "iOS"
            sdk_name = "iphonesimulator" if is_simulator else "iphoneos"
        return platform_name, sdk_name

    def _resolve_watch_scheme(
        self,
        default_name: str,
        is_watch: bool,
        declared: str,
        search_dir: Path,
        is_workspace: bool,
    ) -> str:
        """Pick the scheme for an iOS/watchOS build.

        Non-watch builds use the project/workspace name. Watch builds
        prefer a manifest-declared scheme (no probe), then fall back to
        xcodebuild discovery, then the default name.
        """
        if not is_watch:
            return default_name
        if declared:
            return declared
        return (
            self._find_watch_scheme_via_xcodebuild(search_dir, is_workspace, default_name)
            or default_name
        )

    def _scan_dir_for_xcode_build(
        self,
        search_dir: Path,
        prefix: str,
        sdk_or_dest: str,
        is_watch: bool,
        declared_watch_scheme: str = "",
    ) -> str:
        """Scan a directory for xcworkspace/xcodeproj and return build command.

        Prefers xcworkspace (CocoaPods) over xcodeproj.  When *is_watch*
        is True, resolves a Watch scheme via ``_resolve_watch_scheme`` —
        a manifest-declared ``declared_watch_scheme`` wins and skips the
        xcodebuild discovery probes; otherwise falls back to discovery.

        Returns an empty string if no Xcode project is found.
        """
        source = prefix.rstrip("/") if prefix else "root"

        # Prefer xcworkspace (CocoaPods) over xcodeproj
        workspaces = list(search_dir.glob("*.xcworkspace"))
        if workspaces:
            ws = workspaces[0]
            name = ws.name.removesuffix(".xcworkspace")
            scheme = self._resolve_watch_scheme(
                name, is_watch, declared_watch_scheme, search_dir, is_workspace=True
            )
            ws_path = f"{prefix}{ws.name}" if prefix else ws.name
            cmd = (
                f"xcodebuild -workspace {shlex.quote(ws_path)} -scheme {shlex.quote(scheme)}"
                f" {sdk_or_dest} -configuration Debug -derivedDataPath ./build/derived build"
            )
            logger.info("ios_build_auto_detected", command=cmd, source=source)
            return cmd

        xcodeprojs = list(search_dir.glob("*.xcodeproj"))
        if xcodeprojs:
            proj = xcodeprojs[0]
            name = proj.name.removesuffix(".xcodeproj")
            scheme = self._resolve_watch_scheme(
                name, is_watch, declared_watch_scheme, search_dir, is_workspace=False
            )
            proj_path = f"{prefix}{proj.name}" if prefix else proj.name
            cmd = (
                f"xcodebuild -project {shlex.quote(proj_path)} -scheme {shlex.quote(scheme)}"
                f" {sdk_or_dest} -configuration Debug -derivedDataPath ./build/derived build"
            )
            logger.info("ios_build_auto_detected", command=cmd, source=source)
            return cmd

        return ""

    @staticmethod
    def _inject_signing_flags(cmd: str, team_id: str) -> str:
        """Append device code-signing flags, respecting flags already present.

        Explicit ios_build_command values take precedence: if it already
        carries CODE_SIGN_IDENTITY or DEVELOPMENT_TEAM, those are kept —
        when signing fails on such a project, check the explicit command
        for conflicting values.
        Assumes CODE_SIGN_STYLE=Automatic in the target xcodeproj;
        -allowProvisioningUpdates is silently ignored on manual signing.
        """
        if "-allowProvisioningUpdates" not in cmd:
            cmd += " -allowProvisioningUpdates"
        if "DEVELOPMENT_TEAM" not in cmd:
            cmd += f" DEVELOPMENT_TEAM={team_id}"
        if "CODE_SIGN_IDENTITY" not in cmd:
            cmd += f' CODE_SIGN_IDENTITY="{ios_signing.CODE_SIGN_IDENTITY}"'
        return cmd

    def _resolve_ios_build_command(
        self,
        project: ForgeProject,
        device_id: str = "",
        device_type: str = "",
        is_simulator: bool = True,
        signing_team: str = "",
    ) -> str:
        """Resolve the iOS build command — explicit config or auto-detect from iosApp/.

        Args:
            project: Forge project configuration.
            device_id: Optional device UDID. When provided, emits
                ``-destination`` instead of ``-sdk``.
            device_type: Target device type (e.g. ``"watch"``). When
                ``"watch"``, uses watchOS platform and scans for
                a Watch-suffixed scheme.
            is_simulator: Whether the target device is a simulator.
                Controls platform name (``iOS Simulator`` vs ``iOS``)
                and SDK (``iphonesimulator`` vs ``iphoneos``).
            signing_team: Apple development team for physical device
                builds. When set and ``is_simulator`` is False, signing
                flags are appended to the resolved command.

        Returns an empty string if no iOS build can be determined.
        """
        is_watch = device_type == "watch"
        platform_name, sdk_name = self._resolve_platform_names(is_watch, is_simulator)

        def _signed(cmd: str) -> str:
            if not is_simulator and signing_team:
                return self._inject_signing_flags(cmd, signing_team)
            return cmd

        if project.ios_build_command:
            cmd = project.ios_build_command
            # Append -destination if device_id provided and not already present
            if device_id and "-destination" not in cmd:
                cmd += f" -destination {shlex.quote(f'platform={platform_name},id={device_id}')}"
            if "-derivedDataPath" not in cmd:
                cmd += " -derivedDataPath ./build/derived"
            return _signed(cmd)

        sdk_or_dest = (
            f"-destination {shlex.quote(f'platform={platform_name},id={device_id}')}"
            if device_id
            else f"-sdk {sdk_name}"
        )

        # Auto-detect: check for iosApp/ directory with xcodeproj or xcworkspace
        ios_dir = Path(project.path) / "iosApp"
        if ios_dir.is_dir():
            cmd = self._scan_dir_for_xcode_build(
                ios_dir, "iosApp/", sdk_or_dest, is_watch, project.ios_watch_scheme
            )
            if cmd:
                return _signed(cmd)

        # Fallback: root-level xcworkspace or xcodeproj
        root = Path(project.path)
        cmd = self._scan_dir_for_xcode_build(
            root, "", sdk_or_dest, is_watch, project.ios_watch_scheme
        )
        if cmd:
            return _signed(cmd)

        return ""

    @staticmethod
    def _gate_command(command: str) -> str:
        """Prefix the build with linc-build-gate when the build-slot mutex is enabled
        (LINC_BUILD_GATE=1, set on the mini). Inert no-op everywhere else. Extracted
        from _run_build so the gate branch doesn't inflate that function's CRAP — see
        feedback_crap_ratchet_extract_dont_inline."""
        if os.environ.get("LINC_BUILD_GATE") == "1":
            return f"linc-build-gate --desc crucible -- {command}"
        return command

    def _build_env_with_java_home(
        self, env: Mapping[str, str] | None, *, command: str, cwd: str
    ) -> dict[str, str]:
        """Merge ``env`` over ``os.environ`` and inject a resolved JAVA_HOME when
        it's absent (launchd strips the interactive JAVA_HOME/PATH).

        Precedence: caller-supplied ``env`` > pre-existing JAVA_HOME > resolved.
        Extracted from ``_run_build`` to keep that method's complexity flat."""
        build_env: dict[str, str] = dict(os.environ)
        if env:
            build_env.update(env)
        if "JAVA_HOME" not in build_env:
            java_home = resolve_java_home()
            if java_home:
                build_env["JAVA_HOME"] = java_home
            else:
                logger.warning("java_home_resolution_failed", command=command, cwd=cwd)
        return build_env

    def _run_build(
        self, command: str, cwd: str, *, env: Mapping[str, str] | None = None
    ) -> tuple[bool, str]:
        """Execute a build command, serialized via linc-build-gate when enabled.

        Injects JAVA_HOME (resolved via ``resolve_java_home()``) into the build
        subprocess env when it's absent from the merged env — launchd strips the
        interactive-shell PATH/JAVA_HOME, so a keg-only Homebrew JDK that works
        fine in a terminal is invisible to a launchd-spawned build.
        Precedence: caller-supplied ``env`` > pre-existing JAVA_HOME > resolved.
        """
        command = self._gate_command(command)
        timeout = _build_timeout_seconds()
        build_env = self._build_env_with_java_home(env, command=command, cwd=cwd)

        try:
            proc = safe_subprocess_run(command, cwd=cwd, timeout=timeout, env=build_env)
            output = proc.stdout + proc.stderr
            return (proc.returncode == 0, trim_output(output))
        except subprocess.TimeoutExpired:
            self._build_timed_out = True
            return (False, f"Build timed out after {timeout}s (set {_BUILD_TIMEOUT_ENV} to raise)")
        except (FileNotFoundError, ShellMetacharError, CommandParseError) as exc:
            return (False, str(exc))

    def _resolve_targets(
        self,
        device: str,
        platform: str,
        device_type: str = "",
    ) -> list[dict]:
        """Resolve target devices for deployment via nerve discovery."""
        devices = _discover_or_raise(self._nerve)
        if not devices:
            return []

        if device:
            return self._policy_ctx.filter(
                device, [d for d in devices if _device_matches(d, device)]
            )

        result = devices
        if platform:
            result = [d for d in result if d.get("platform") == platform.lower()]
        if device_type:
            result = [d for d in result if d.get("device_type") == device_type.lower()]
        # Blocklist: a device is deployable only if it explicitly reports
        # state=="online". Absent/unknown state -> offline, never online
        # (mirrors nerve ios.py). A permissive default here would silently
        # disable this filter when state is dropped over HTTP.
        result = [d for d in result if d.get("state") == "online"]
        return self._policy_ctx.filter("", result)

    def _deploy_to_device(
        self,
        project: ForgeProject,
        device: dict,
        launch: bool,
        launch_intent: str = "",
    ) -> DeployResult:
        """Deploy to a single device (represented as a plain dict)."""
        serial = device.get("serial", "")
        model = device.get("model", "")
        platform_str = device.get("platform", "")
        result = DeployResult(
            device_serial=serial,
            device_model=model,
            platform=platform_str,
        )

        if platform_str == "android":
            self._deploy_android(project, device, launch, result, launch_intent)
        elif platform_str == "ios":
            self._deploy_ios(project, device, launch, result)
        else:
            result.error = f"Unsupported platform: {platform_str}"

        _attach_provenance(result, project.path)

        return result

    def _deploy_android(
        self,
        project: ForgeProject,
        device: dict,
        launch: bool,
        result: DeployResult,
        launch_intent: str = "",
    ) -> None:
        """Deploy APK to an Android device."""
        serial = device.get("serial", "")
        # Find APK
        apk_path = self._forge.find_apk(project.name)
        if not apk_path:
            result.error = f"No APK found for {project.name}"
            return

        result.artifact_path = apk_path
        self._install_android_apk(apk_path, serial, result)

        if result.install_success and launch:
            self._launch_android(project, serial, result, launch_intent)

    def _reservation_error(self, serial: str) -> str | None:
        """Refuse a device another nerve session actively holds.

        Checked once here so every install path (single-target, multi-target
        module, physical-iOS) is covered without duplicating the guard at
        each call site. ``nerve_pool`` degrades gracefully: a ConnectionError
        (daemon down / unreachable) means the ledger couldn't be read, so the
        deploy proceeds as before rather than blocking on missing info — the
        same graceful-degradation contract as ``crucible_bvt``'s guard.
        """
        try:
            pool_result = self._nerve.pool()
        except ConnectionError:
            return None
        if not pool_result:
            return None
        reservation = find_reservation(pool_result.get("reservations", []), serial)
        if not reservation:
            return None
        return (
            f"Refused: device {serial} reserved by session "
            f"{reservation.get('session_id')} (purpose: {reservation.get('purpose')}) "
            "— release it or wait (nerve_pool)"
        )

    def _install_android_apk(
        self,
        apk_path: str,
        serial: str,
        result: DeployResult,
    ) -> None:
        """Install an APK on an Android device via nerve."""
        blocked = self._reservation_error(serial)
        if blocked:
            result.error = blocked
            return
        try:
            self._nerve.install(serial, apk_path)
            result.install_success = True
        except Exception as exc:
            result.error = f"Install failed: {exc}"

    def _launch_android(
        self,
        project: ForgeProject,
        serial: str,
        result: DeployResult,
        launch_intent: str = "",
    ) -> None:
        """Launch an Android app after install via nerve."""
        try:
            package = _infer_package(project, apk_path=result.artifact_path)
            if package:
                if "://" in launch_intent:
                    self._nerve.launch(serial, package, uri=launch_intent)
                else:
                    self._nerve.launch(serial, package)
                result.launch_success = True
            else:
                result.error = (
                    "Could not infer package name for launch; "
                    "set identity.package in linc-manifest.yaml"
                )
        except Exception as exc:
            result.error = f"Launch failed: {exc}"

    def _deploy_ios(
        self,
        project: ForgeProject,
        device: dict,
        launch: bool,
        result: DeployResult,
    ) -> None:
        """Deploy .app to an iOS simulator."""
        device_type_str = device.get("device_type", "")
        is_sim = self._device_is_simulator(device)
        app_path = self._forge.find_ios_app(
            project.name,
            device_type=device_type_str,
            is_simulator=is_sim,
        )
        if not app_path:
            result.error = f"No iOS .app found for {project.name}"
            return

        self._deploy_ios_artifact(app_path, device, project, launch, result)

    def _deploy_ios_artifact(
        self,
        app_path: str,
        device: dict,
        project: ForgeProject,
        launch: bool,
        result: DeployResult,
    ) -> None:
        """Install and optionally launch an iOS .app on a device via nerve.

        Nerve handles simulator boot and platform routing internally,
        so we don't need separate physical/simulator paths here.
        """
        serial = device.get("serial", "")
        result.artifact_path = app_path

        blocked = self._reservation_error(serial)
        if blocked:
            result.error = blocked
            return

        try:
            self._nerve.install(serial, app_path)
            result.install_success = True
        except Exception as exc:
            result.error = f"Install failed: {exc}"
            return

        if launch:
            bundle_id = _infer_bundle_id(project)
            if bundle_id:
                try:
                    self._nerve.launch(serial, bundle_id)
                    result.launch_success = True
                except Exception as exc:
                    result.error = f"Launch failed: {exc}"
            else:
                result.error = (
                    "Could not infer bundle ID for launch; "
                    "set identity.bundle_id in linc-manifest.yaml"
                )

    # --- Multi-target deployment ---

    def _find_project_artifacts(self, project: ForgeProject, result: PourResult) -> None:
        """Find project-level artifacts and add them to result."""
        artifacts: dict[str, str] = {}
        apk_path = self._forge.find_apk(project.name)
        if apk_path:
            artifacts["android"] = apk_path
        ios_app = self._forge.find_ios_app(project.name)
        if ios_app:
            artifacts["ios"] = ios_app
        result.artifacts = artifacts

    def _deploy_module_to_target(
        self,
        project: ForgeProject,
        module: ModuleConfig,
        dev: dict,
        has_apk: bool,
        has_ios_app: bool,
        module_apks: dict[str, str],
        module_ios_apps: dict[str, str],
        launch: bool,
        launch_intent: str,
        result: PourResult,
    ) -> None:
        """Deploy a specific module to a specific target device."""
        dev_platform = dev.get("platform", "")
        if dev_platform == "ios" and not has_ios_app:
            return
        if dev_platform != "ios" and not has_apk:
            return

        deploy_result = DeployResult(
            device_serial=dev.get("serial", ""),
            device_model=dev.get("model", ""),
            platform=dev_platform,
            module=module.name,
        )

        if dev_platform == "ios":
            app_path = module_ios_apps.get(module.name)
            if app_path:
                self._deploy_ios_artifact(
                    app_path,
                    dev,
                    project,
                    launch,
                    deploy_result,
                )
            else:
                deploy_result.error = f"No iOS .app found for module '{module.name}'"
        else:
            apk_path = module_apks.get(module.name)
            if not apk_path:
                deploy_result.error = f"No APK found for module '{module.name}'"
            else:
                deploy_result.artifact_path = apk_path
                self._install_android_apk(apk_path, dev.get("serial", ""), deploy_result)
                if deploy_result.install_success and launch:
                    self._launch_android(
                        project,
                        dev.get("serial", ""),
                        deploy_result,
                        launch_intent,
                    )
        _attach_provenance(deploy_result, project.path)
        result.results.append(deploy_result)

    def _pour_multi_target(
        self,
        project: ForgeProject,
        *,
        device: str,
        platform: str,
        launch: bool,
        skip_build: bool,
        device_type: str,
        launch_intent: str = "",
    ) -> PourResult:
        """Deploy all modules to their matching device types."""
        result = PourResult(project=project.name)

        # Build
        if not skip_build:
            build_ok, build_output = self._build_modules(project)
            result.build_output = build_output
            result.build_success = build_ok
            if not build_ok:
                return result

        # Collect project-level artifacts
        self._find_project_artifacts(project, result)

        # Physical iOS devices: module build_commands are Gradle/Android-only,
        # so device builds (with code signing) route through the platform-aware
        # iOS build and device-SDK artifact discovery instead of the
        # simulator-only module glob.
        handled_ios = self._deploy_physical_ios_targets(
            project,
            device=device,
            platform=platform,
            device_type=device_type,
            skip_build=skip_build,
            launch=launch,
            result=result,
        )
        if handled_ios is None:  # iOS device build failed — stop the pour
            return result

        self._deploy_modules_to_targets(
            project,
            device=device,
            platform=platform,
            device_type=device_type,
            handled_ios=handled_ios,
            launch=launch,
            launch_intent=launch_intent,
            result=result,
        )

        _finalize_pour_provenance(result)

        if not result.results:
            result.build_output += (
                "\nNo matching devices found for any module."
                " Ensure devices are connected and linc-nerve is running."
            )

        return result

    def _deploy_modules_to_targets(
        self,
        project: ForgeProject,
        *,
        device: str,
        platform: str,
        device_type: str,
        handled_ios: set[str],
        launch: bool,
        launch_intent: str,
        result: PourResult,
    ) -> None:
        """Route each module's artifacts to its matching target devices.

        Serials in *handled_ios* were already deployed via the
        physical-iOS path and are skipped here.
        """
        module_apks = self._forge.find_module_apks(project.name)
        module_ios_apps = self._forge.find_module_ios_apps(project.name)

        # applicationIds already installed per device serial — so a second
        # module sharing one (the Wear phone+watch pairing pattern) can't
        # overwrite the first on a single device.
        installed_app_ids: dict[str, set[str]] = {}

        for module in project.modules:
            has_apk = module.name in module_apks
            has_ios_app = module.name in module_ios_apps
            if not has_apk and not has_ios_app:
                continue  # No artifacts for this module — skip

            targets = self._resolve_targets_for_module(
                module,
                device=device,
                platform=platform,
                device_type=device_type,
                artifact_platforms=self._module_artifact_platforms(has_apk, has_ios_app),
            )
            self._deploy_module_targets(
                project,
                module,
                targets,
                handled_ios=handled_ios,
                installed_app_ids=installed_app_ids,
                module_apks=module_apks,
                module_ios_apps=module_ios_apps,
                has_apk=has_apk,
                has_ios_app=has_ios_app,
                launch=launch,
                launch_intent=launch_intent,
                result=result,
            )

    @staticmethod
    def _module_artifact_platforms(has_apk: bool, has_ios_app: bool) -> set[Platform]:
        """Platforms this module has built artifacts for."""
        platforms: set[Platform] = set()
        if has_apk:
            platforms.add(Platform.ANDROID)
        if has_ios_app:
            platforms.add(Platform.IOS)
        return platforms

    def _deploy_module_targets(
        self,
        project: ForgeProject,
        module: ModuleConfig,
        targets: list[dict],
        *,
        handled_ios: set[str],
        installed_app_ids: dict[str, set[str]],
        module_apks: dict[str, str],
        module_ios_apps: dict[str, str],
        has_apk: bool,
        has_ios_app: bool,
        launch: bool,
        launch_intent: str,
        result: PourResult,
    ) -> None:
        """Deploy one module to each of its resolved target devices.

        Skips serials already handled by the physical-iOS path, and refuses
        an install whose applicationId already landed on the device.
        """
        for dev in targets:
            if dev.get("serial", "") in handled_ios:
                continue  # already deployed via the physical-iOS path
            blocked = self._claim_app_id_or_block(project, module, dev, installed_app_ids)
            if blocked is not None:
                result.results.append(blocked)
                continue
            self._deploy_module_to_target(
                project=project,
                module=module,
                dev=dev,
                has_apk=has_apk,
                has_ios_app=has_ios_app,
                module_apks=module_apks,
                module_ios_apps=module_ios_apps,
                launch=launch,
                launch_intent=launch_intent,
                result=result,
            )

    def _claim_app_id_or_block(
        self,
        project: ForgeProject,
        module: ModuleConfig,
        dev: dict,
        installed_app_ids: dict[str, set[str]],
    ) -> DeployResult | None:
        """Reserve the module's applicationId on the device, or block it.

        Returns ``None`` (and records the id as claimed) when the install may
        proceed. Returns a failed ``DeployResult`` when another module already
        installed the same applicationId on this device — installing this one
        would overwrite it. Only Android modules carry a collidable
        applicationId; iOS and unparseable modules never block.
        """
        if dev.get("platform", "") == "ios":
            return None
        app_id = self._module_application_id(project, module)
        if not app_id:
            return None
        claimed = installed_app_ids.setdefault(dev.get("serial", ""), set())
        if app_id in claimed:
            return DeployResult(
                device_serial=dev.get("serial", ""),
                device_model=dev.get("model", ""),
                platform=dev.get("platform", ""),
                module=module.name,
                error=(
                    f"Refused: applicationId '{app_id}' already installed on "
                    f"{dev.get('serial', '')} by another module — installing "
                    f"'{module.name}' would overwrite it (shared-applicationId "
                    f"collision)."
                ),
            )
        claimed.add(app_id)
        return None

    def _module_application_id(self, project: ForgeProject, module: ModuleConfig) -> str | None:
        """Parse the applicationId from a module's build.gradle.kts.

        Used to detect two modules that share one id (the Wear phone+watch
        pairing pattern) before they overwrite each other on one device.
        """
        if not project.path:
            return None
        gradle = Path(project.path) / module.name / "build.gradle.kts"
        return _parse_application_id_from_file(gradle)

    def _deploy_physical_ios_targets(
        self,
        project: ForgeProject,
        *,
        device: str,
        platform: str,
        device_type: str,
        skip_build: bool,
        launch: bool,
        result: PourResult,
    ) -> set[str] | None:
        """Build + deploy to physical iOS targets in a multi-target pour.

        Builds per device (phone and watch need different SDKs) via the
        platform-aware iOS build, then installs the device-SDK artifact.

        Returns the set of handled device serials, or None when an iOS
        device build failed (caller should stop the pour).
        """
        if platform not in ("", "ios"):
            return set()
        # _resolve_targets ignores the platform arg when an explicit device
        # serial is given, so guard platform here — otherwise an explicit
        # Android serial leaks into the physical-iOS path instead of routing
        # through normal module deployment.
        targets = [
            t
            for t in self._resolve_targets(device, "ios", device_type)
            if t.get("platform") == "ios" and not self._device_is_simulator(t)
        ]
        if not targets:
            return set()

        handled: set[str] = set()
        for dev in targets:
            if not skip_build:
                ok, output = self._build_for_platforms(project, {Platform.IOS}, targets=[dev])
                sep = "\n" if result.build_output else ""
                result.build_output += sep + output
                if not ok:
                    return None

            dtype = "watch" if dev.get("device_type") == "watch" else "phone"
            app_path = self._forge.find_ios_app(
                project.name, device_type=dtype, is_simulator=False
            )
            deploy_result = DeployResult(
                device_serial=dev.get("serial", ""),
                device_model=dev.get("model", ""),
                platform="ios",
            )
            if app_path:
                self._deploy_ios_artifact(app_path, dev, project, launch, deploy_result)
            else:
                sdk_dir = "Debug-watchos" if dtype == "watch" else "Debug-iphoneos"
                deploy_result.error = (
                    f"No physical-device iOS .app found for {project.name} "
                    f"(expected {sdk_dir} build output)"
                )
            result.results.append(deploy_result)
            handled.add(dev.get("serial", ""))
        return handled

    @staticmethod
    def _module_scoped_android_build_command(
        module: ModuleConfig, platform: str, build_type: str = "debug"
    ) -> str:
        """Module-scoped Gradle assemble command for single-platform Android deploys.

        A single-module Android pour must not run the project-wide
        ``build_command`` (a broad ``./gradlew build``): in composite-build KMP
        projects that compiles Apple metadata too, so an Apple-side error
        blocks an unrelated Android install — and every pour over-builds.
        Only the explicit ``platform="android"`` narrowing opts
        in; an empty or non-Android platform returns ``""`` so the caller
        falls back to the project-level build. ``build_type`` picks the
        assemble variant (debug → assembleDebug, release → assembleRelease).
        """
        variant = check_build_type(build_type, "build_type")
        if platform.lower() != "android":
            return ""
        return f"./gradlew :{module.name}:assemble{variant[:1].upper()}{variant[1:].lower()}"

    def _pour_single_module(
        self,
        project: ForgeProject,
        *,
        module: str,
        device: str,
        platform: str,
        launch: bool,
        skip_build: bool,
        device_type: str,
        launch_intent: str = "",
        build_type: str = "debug",
    ) -> PourResult:
        """Deploy a specific module to matching devices."""
        result = PourResult(project=project.name)

        module_config = _find_module(project, module)
        if not module_config:
            result.build_output = _module_not_found(project, module)
            return result

        # Build
        if not skip_build:
            build_ok, build_output = self._build_module(
                project, module_config, platform, build_type
            )
            result.build_output = build_output
            result.build_success = build_ok
            if not build_ok:
                return result

        # Find artifacts for this module
        module_apks = self._forge.find_module_apks(project.name)
        module_ios_apps = self._forge.find_module_ios_apps(project.name)

        # Collect artifact paths
        artifacts: dict[str, str] = {}
        apk_path = self._forge.find_apk(project.name)
        if apk_path:
            artifacts["android"] = apk_path
        ios_app = self._forge.find_ios_app(project.name)
        if ios_app:
            artifacts["ios"] = ios_app
        result.artifacts = artifacts

        # Build the set of platforms this module has artifacts for
        artifact_platforms: set[Platform] = set()
        if module_config.name in module_apks:
            artifact_platforms.add(Platform.ANDROID)
        if module_config.name in module_ios_apps:
            artifact_platforms.add(Platform.IOS)

        # Resolve targets
        targets = self._resolve_targets_for_module(
            module_config,
            device=device,
            platform=platform,
            device_type=device_type,
            artifact_platforms=artifact_platforms,
        )
        if not targets:
            result.build_output += (
                "\nNo matching devices found for module."
                " Ensure devices are connected and linc-nerve is running."
            )
            return result

        for dev in targets:
            deploy_result = self._deploy_module_to_device(
                dev, module_config, project, module_apks, module_ios_apps, launch, launch_intent
            )
            result.results.append(deploy_result)
        _finalize_pour_provenance(result)

        return result

    def _deploy_module_to_device(
        self,
        dev: dict,
        module_config: ModuleConfig,
        project: ForgeProject,
        module_apks: dict,
        module_ios_apps: dict,
        launch: bool,
        launch_intent: str,
    ) -> DeployResult:
        """Deploy a single module to a single device.

        Handles platform branching (iOS vs Android), artifact lookup,
        install, and optional launch.
        """
        dev_platform = dev.get("platform", "")
        deploy_result = DeployResult(
            device_serial=dev.get("serial", ""),
            device_model=dev.get("model", ""),
            platform=dev_platform,
            module=module_config.name,
        )
        if dev_platform == "ios":
            app_path = module_ios_apps.get(module_config.name)
            if app_path:
                self._deploy_ios_artifact(
                    app_path,
                    dev,
                    project,
                    launch,
                    deploy_result,
                )
            else:
                deploy_result.error = f"No iOS .app found for module '{module_config.name}'"
        else:
            apk_path = module_apks.get(module_config.name)
            if not apk_path:
                deploy_result.error = f"No APK found for module '{module_config.name}'"
            else:
                deploy_result.artifact_path = apk_path
                self._install_android_apk(apk_path, dev.get("serial", ""), deploy_result)
                if deploy_result.install_success and launch:
                    self._launch_android(
                        project, dev.get("serial", ""), deploy_result, launch_intent
                    )
        _attach_provenance(deploy_result, project.path)
        return deploy_result

    def _resolve_targets_for_module(
        self,
        module: ModuleConfig,
        *,
        device: str,
        platform: str,
        device_type: str,
        artifact_platforms: set[Platform] | None = None,
    ) -> list[dict]:
        """Resolve target devices for a specific module.

        An explicit device serial is honored (user override), but is still
        filtered by the module's ``device_types`` form factor — otherwise a
        single phone serial gets BOTH composeApp and watchApp, and the watch
        module (sharing an applicationId) overwrites the phone app.
        Otherwise, filter online devices by device_types.
        When ``artifact_platforms`` is provided, devices whose platform
        matches an available artifact are also included even if their
        device_type is not in the module's ``device_types`` list.  This
        handles iOS simulators (device_type ``emulator``) matching iOS
        modules that declare ``device_types=["phone"]``.
        """
        if device:
            # User override — respect explicit device selection.
            targets = self._resolve_targets(device, platform)
        else:
            # Get all online devices
            targets = self._resolve_targets("", platform, device_type)

        # Filter by module's device_types, but also allow iOS platform matches.
        # iOS simulators have device_type "emulator" which won't match a module
        # declaring device_types=["phone"], so we fall back to platform matching
        # for iOS devices when the module has iOS artifacts.
        if module.device_types:
            targets = [
                d
                for d in targets
                if d.get("device_type", "") in module.device_types
                or (
                    artifact_platforms
                    and d.get("platform") == "ios"
                    and Platform.IOS in artifact_platforms
                )
            ]

        return targets

    def _build_modules(self, project: ForgeProject) -> tuple[bool, str]:
        """Build modules — per-module commands or single project build."""
        has_module_commands = any(m.build_command or m.ios_build_command for m in project.modules)

        if has_module_commands:
            outputs = []
            for module in project.modules:
                cmd = module.ios_build_command or module.build_command or project.build_command
                if not cmd:
                    continue
                ok, output = self._run_build(cmd, project.path)
                outputs.append(f"[{module.name}] {output}")
                if not ok:
                    return (False, "\n".join(outputs))
            return (True, "\n".join(outputs))

        # Fall back to single project-level build
        return self._build(project)
