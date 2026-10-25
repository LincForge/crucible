"""Read-only integration with linc-forge's project registry.

Discovers projects from ``linc-manifest.yaml`` files in the projects
directory, with fallback to ``~/.forge/projects.yaml``.  Merges each
project's ``.forge.yaml`` independently — no Python import of ``link_forge``.
"""

from __future__ import annotations

import glob
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger().bind(domain="build_pipeline")


def _default_forge_registry() -> Path:
    """Return the default forge project registry path, honouring ``LINC_HOME``.

    Returns:
        ``$LINC_HOME/.forge/projects.yaml`` when ``LINC_HOME`` is set,
        otherwise ``~/.forge/projects.yaml``.  Re-evaluated on every call so
        that ``os.environ`` monkeypatching in tests takes effect.
    """
    home = os.getenv("LINC_HOME")
    if home:
        return Path(home) / ".forge" / "projects.yaml"
    return Path.home() / ".forge" / "projects.yaml"


@dataclass
class ModuleConfig:
    """Per-module configuration for multi-target projects (e.g. phone + watch)."""

    name: str
    device_types: list[str] = field(default_factory=list)
    build_command: str = ""
    apk_pattern: str = ""
    app_pattern: str = ""  # iOS .app glob pattern
    ios_build_command: str = ""  # xcodebuild command for iOS modules


def _parse_module_configs(raw_modules: object) -> list[ModuleConfig]:
    """Parse a list of module dicts (from .forge.yaml or a manifest's
    ``deployment.modules``) into ``ModuleConfig`` objects, skipping any
    entry that isn't a dict or lacks a ``name``.
    """
    configs: list[ModuleConfig] = []
    if not isinstance(raw_modules, list):
        return configs
    for entry in raw_modules:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        configs.append(
            ModuleConfig(
                name=entry["name"],
                device_types=entry.get("device_types", []),
                build_command=entry.get("build_command", ""),
                apk_pattern=entry.get("apk_pattern", ""),
                app_pattern=entry.get("app_pattern", ""),
                ios_build_command=entry.get("ios_build_command", ""),
            )
        )
    return configs


def _manifest_deployment_modules(raw: dict) -> list[ModuleConfig]:
    """Parse ``deployment.modules`` out of a manifest dict.

    Tolerates a missing or malformed ``deployment`` block (returns no
    modules) so a manifest without a deployment section still discovers
    fine — mirrors ``_parse_module_configs``'s tolerance for malformed
    entries. Extracted from ``_discover_from_manifests`` so the isinstance
    branch lives in its own fully-covered unit instead of inline in the
    discovery loop.
    """
    deployment = raw.get("deployment", {})
    if not isinstance(deployment, dict):
        return []
    return _parse_module_configs(deployment.get("modules", []))


def _apply_manifest_modules(project: ForgeProject, manifest_modules: list[ModuleConfig]) -> None:
    """Overwrite ``project.modules`` with manifest-declared modules, if any.

    A manifest's ``deployment.modules`` wins over anything ``.forge.yaml``
    declares — same manifest-wins-on-path precedence used elsewhere in
    discovery. An empty ``manifest_modules`` list means the manifest didn't
    declare modules, so whatever ``.forge.yaml`` already set is left alone.
    """
    if manifest_modules:
        project.modules = manifest_modules


def _is_git_worktree_of(candidate_path: str, existing_path: str) -> bool:
    """Best-effort check whether ``candidate_path`` is a ``git worktree`` of the
    repo rooted at ``existing_path``.

    Only consulted when the cheap exact-name/shortest-path heuristic in
    ``_should_keep_existing`` is tied — this shells out to git, so it's the
    last resort, not the first check. Any failure (git absent, not a repo,
    timeout) is treated as "not a worktree" so environments without git never
    hard-fail discovery.
    """
    try:
        result = subprocess.run(
            ["git", "-C", existing_path, "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    if result.returncode != 0:
        return False

    try:
        candidate_resolved = Path(candidate_path).resolve()
    except OSError:
        return False

    for line in result.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        wt_path = line[len("worktree ") :].strip()
        try:
            if Path(wt_path).resolve() == candidate_resolved:
                return True
        except OSError:
            continue
    return False


def _should_keep_existing(existing_path: str, candidate_path: str, name: str) -> bool:
    """Decide whether ``existing_path`` should win over ``candidate_path`` when
    two manifests both discover project ``name`` (worktree/duplicate resolution).

    Cheap heuristics first, git worktree check only when still ambiguous:
      1. A basename that exactly matches ``name`` wins over one that doesn't.
      2. If both (or neither) match exactly, the shorter path wins — worktree
         copies conventionally live under a longer, suffixed directory name
         (e.g. ``MyApp-v033`` vs. ``MyApp``).
      3. If still tied, ``existing_path`` wins if ``candidate_path`` is a git
         worktree member of the repo at ``existing_path``.
    """
    existing_exact = Path(existing_path).name == name
    candidate_exact = Path(candidate_path).name == name
    if existing_exact != candidate_exact:
        return existing_exact

    if len(existing_path) != len(candidate_path):
        return len(existing_path) < len(candidate_path)

    return _is_git_worktree_of(candidate_path, existing_path)


@dataclass
class ForgeProject:
    """A project from the forge registry with merged .forge.yaml config."""

    name: str = ""
    path: str = ""
    package: str = ""
    bundle_id: str = ""
    github_project: str = ""
    test_command: str = ""
    build_command: str = ""
    ios_build_command: str = ""
    ios_watch_scheme: str = ""  # declared watchOS scheme; skips xcodebuild discovery probes
    lint_command: str = ""
    framework: str = ""
    target_branch_default: str = ""
    linked_projects: list[str] = field(default_factory=list)
    modules: list[ModuleConfig] = field(default_factory=list)
    config_errors: list[str] = field(default_factory=list)

    def config_error_for(self, field_name: str, default: str = "") -> str:
        """Return the recorded ``.forge.yaml`` error for ``field_name``, or ``default``."""
        prefix = f"{field_name}:"
        return next((err for err in self.config_errors if err.startswith(prefix)), default)

    @property
    def is_multi_target(self) -> bool:
        """True if this project has multiple deployment modules."""
        return len(self.modules) > 0


def _read_command_field(forge: dict[str, object], project: ForgeProject, field_name: str) -> str:
    """Extract a string command field from ``.forge.yaml``, logging and dropping non-strings."""
    raw = forge.get(field_name)
    if raw is None:
        return ""
    if not isinstance(raw, str):
        type_name = type(raw).__name__
        logger.warning(
            "forge_yaml_command_not_string",
            field=field_name,
            type=type_name,
        )
        project.config_errors.append(
            f"{field_name}: .forge.yaml value must be a string (got {type_name})"
        )
        return ""
    return raw


_EMPTY_IDENTITY: dict[str, object] = {}


def _read_identity_str(identity: dict[str, object], field_name: str) -> str:
    """Extract a string field from manifest ``identity``, logging and dropping non-strings."""
    raw = identity.get(field_name)
    if raw is None:
        return ""
    if not isinstance(raw, str):
        logger.warning(
            "forge_manifest_identity_not_string",
            field=field_name,
            type=type(raw).__name__,
        )
        return ""
    return raw


def _apply_manifest_identity(project: ForgeProject, identity: dict[str, object]) -> None:
    """Apply ``identity.package`` and ``identity.bundle_id`` from ``linc-manifest.yaml``."""
    project.package = _read_identity_str(identity, "package")
    project.bundle_id = _read_identity_str(identity, "bundle_id")


class ForgeProjectReader:
    """Read-only access to linc-forge's project registry.

    Discovers projects from ``linc-manifest.yaml`` files in the projects
    directory, with fallback to ``~/.forge/projects.yaml``.  Merges each
    project's ``.forge.yaml`` for a unified view. Does NOT import ``link_forge``.
    """

    def __init__(self, registry_path: Path | None = None) -> None:
        env_path = os.environ.get("FORGE_PROJECTS_FILE")
        if registry_path:
            self._path = registry_path
        elif env_path:
            self._path = Path(env_path)
        else:
            self._path = _default_forge_registry()

        # Auto-discover from manifests unless an explicit registry_path argument was passed
        self._auto_discover = registry_path is None

        # Guards concurrent re-scans triggered by get() lookup misses or stale
        # paths when the MCP server invokes tools off-thread.
        self._reload_lock = threading.Lock()

        self._projects: dict[str, ForgeProject] = {}
        self._load()

    def _load(self) -> None:
        """Parse the forge registry, discover manifests, and merge .forge.yaml files."""
        self._projects = {}

        # 1. Load from projects.yaml (legacy fallback)
        if self._path.is_file():
            try:
                with open(self._path) as f:
                    data = yaml.safe_load(f)
            except (yaml.YAMLError, OSError) as exc:
                logger.warning("forge_registry_read_failed", path=str(self._path), error=str(exc))
                data = None

            if isinstance(data, dict):
                for entry in data.get("projects", []):
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name", "")
                    if not name:
                        continue

                    project = ForgeProject(
                        name=name,
                        path=entry.get("path", ""),
                        github_project=entry.get("github_project", ""),
                    )
                    self._merge_forge_yaml(project)
                    self._projects[name] = project
        else:
            logger.debug("forge_registry_not_found", path=str(self._path))

        # 2. Discover from manifests (preferred — manifests win over projects.yaml)
        if self._auto_discover:
            self._discover_from_manifests()

    def _discover_from_manifests(self) -> None:
        """Scan the projects directory for linc-manifest.yaml files."""
        projects_root = Path(os.environ.get("LINC_PROJECTS_DIR", str(Path.home() / "projects")))
        if not projects_root.is_dir():
            return

        # Names already claimed by a manifest seen earlier in *this* scan —
        # only these collisions go through the worktree/duplicate heuristic.
        # A name whose only prior registration came from legacy projects.yaml
        # keeps the existing "manifest always wins" behavior (its first
        # manifest sighting is not a worktree collision).
        manifest_seen: set[str] = set()

        for manifest_path in sorted(projects_root.glob("*/linc-manifest.yaml")):
            try:
                with open(manifest_path) as f:
                    raw = yaml.safe_load(f)
                if not isinstance(raw, dict):
                    continue
                identity = raw.get("identity", {})
                if not isinstance(identity, dict):
                    continue
                name = identity.get("name", "")
                if not name:
                    continue
            except (yaml.YAMLError, OSError):
                continue

            project_dir = str(manifest_path.parent)

            # deployment.modules is the manifest's own module declaration —
            # wins over anything .forge.yaml declares, same as manifest path
            # winning over the legacy projects.yaml path below.
            manifest_modules = _manifest_deployment_modules(raw)

            if name in self._projects:
                self._update_existing_project(
                    name,
                    project_dir,
                    manifest_modules,
                    manifest_seen,
                    identity,
                )
            else:
                project = ForgeProject(name=name, path=project_dir)
                _apply_manifest_identity(project, identity)
                self._merge_forge_yaml(project)
                _apply_manifest_modules(project, manifest_modules)
                self._projects[name] = project

            manifest_seen.add(name)

    def _update_existing_project(
        self,
        name: str,
        project_dir: str,
        manifest_modules: list[ModuleConfig],
        manifest_seen: set[str],
        identity: dict[str, object] = _EMPTY_IDENTITY,
    ) -> None:
        """Update an already-registered project's path/modules for a manifest sighting.

        Extracted out of ``_discover_from_manifests`` (same complexity-isolation
        pattern as ``_manifest_deployment_modules``) so the
        worktree/duplicate-collision decision lives in its own fully-covered
        unit instead of inline in the discovery loop.

        A name already claimed by a manifest earlier in *this* scan
        (``name in manifest_seen``) is a worktree/duplicate collision, resolved
        deterministically via ``_should_keep_existing`` instead of last-wins. An
        existing-wins verdict leaves the registration untouched entirely (no
        path or module update) so the shadowed manifest's modules never leak
        in. Otherwise (first sighting this scan, or the candidate won the
        collision) the manifest's path and modules win, same as the
        always-win behavior for a name whose only prior registration came
        from the legacy projects.yaml.
        """
        if name in manifest_seen:
            existing_path = self._projects[name].path
            if _should_keep_existing(existing_path, project_dir, name):
                logger.warning(
                    "forge_manifest_duplicate_name_kept_existing",
                    name=name,
                    kept_path=existing_path,
                    shadowed_path=project_dir,
                )
                return
            logger.warning(
                "forge_manifest_duplicate_name_resolved",
                name=name,
                kept_path=project_dir,
                shadowed_path=existing_path,
            )

        self._projects[name].path = project_dir
        _apply_manifest_identity(self._projects[name], identity)
        _apply_manifest_modules(self._projects[name], manifest_modules)

    def _merge_forge_yaml(self, project: ForgeProject) -> None:
        """Merge per-project .forge.yaml fields."""
        if not project.path:
            return

        forge_path = Path(project.path) / ".forge.yaml"
        if not forge_path.is_file():
            return

        try:
            with open(forge_path) as f:
                forge = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            return

        if not isinstance(forge, dict):
            return

        # Unwrap nested forge: key (matches linc-forge convention)
        if "forge" in forge and isinstance(forge["forge"], dict):
            forge = forge["forge"]

        project.test_command = _read_command_field(forge, project, "test_command")
        project.build_command = _read_command_field(forge, project, "build_command")
        project.ios_build_command = _read_command_field(forge, project, "ios_build_command")
        project.ios_watch_scheme = _read_command_field(forge, project, "ios_watch_scheme")
        project.lint_command = _read_command_field(forge, project, "lint_command")
        project.framework = forge.get("framework", "")
        project.target_branch_default = forge.get("target_branch_default", "")
        project.linked_projects = forge.get("linked_projects", [])

        # Parse modules for multi-target deployment
        project.modules.extend(_parse_module_configs(forge.get("modules", [])))

    def list_projects(self) -> list[ForgeProject]:
        """Return all registered forge projects."""
        return list(self._projects.values())

    def get(self, name: str) -> ForgeProject | None:
        """Look up a project by name.

        Self-heals in two cases (bounded to at most one re-scan per call —
        no loops): a lookup miss (name not yet registered, e.g. a manifest
        was added after this reader was constructed), or a cached path that
        no longer exists on disk (e.g. a worktree was deleted). Both trigger
        one ``_load()`` re-scan before giving up. Re-scans are skipped when
        auto-discovery is off (an explicit ``registry_path`` was given) since
        there is nothing new to find.
        See ``_get_with_self_heal`` for the implementation.
        """
        return self._get_with_self_heal(name)

    @staticmethod
    def _is_usable(project: ForgeProject | None) -> bool:
        """True if ``project`` is registered and its cached path still exists on disk."""
        return project is not None and bool(project.path) and Path(project.path).is_dir()

    def _get_with_self_heal(self, name: str) -> ForgeProject | None:
        """Implementation of ``get`` — extracted (same complexity-isolation
        pattern as ``_update_existing_project``) so the public one-line
        dispatcher's own complexity stays flat while this unit carries the
        added self-heal branching.
        """
        project = self._projects.get(name)
        if self._is_usable(project):
            return project

        if not self._auto_discover:
            return project

        with self._reload_lock:
            # Re-check inside the lock — another thread may have already
            # completed the re-scan we're about to do.
            project = self._projects.get(name)
            if self._is_usable(project):
                return project
            self._load()

        return self._projects.get(name)

    def find_module_apks(self, name: str) -> dict[str, str]:
        """Find APKs for each module in a multi-target project.

        Returns a dict of module_name → APK path. Modules without a
        matching APK are omitted.
        """
        project = self._projects.get(name)
        if not project or not project.path or not project.modules:
            return {}

        result: dict[str, str] = {}
        for module in project.modules:
            pattern = module.apk_pattern or f"{module.name}/build/outputs/apk/debug/*.apk"
            full_pattern = os.path.join(project.path, pattern)
            matches = sorted(glob.glob(full_pattern), key=os.path.getmtime, reverse=True)
            if matches:
                result[module.name] = matches[0]

        return result

    def find_apk(self, name: str) -> str | None:
        """Find the most recent debug APK for a project.

        Searches common Gradle output paths for debug APKs.
        """
        project = self._projects.get(name)
        if not project or not project.path:
            return None

        # Common APK output locations
        patterns = [
            "app/build/outputs/apk/debug/*.apk",
            "composeApp/build/outputs/apk/debug/*.apk",
            "androidApp/build/outputs/apk/debug/*.apk",
            "*/build/outputs/apk/debug/*.apk",
        ]

        for pattern in patterns:
            full_pattern = os.path.join(project.path, pattern)
            matches = sorted(glob.glob(full_pattern), key=os.path.getmtime, reverse=True)
            if matches:
                return matches[0]

        return None

    @staticmethod
    def _resolve_watch_nested(app_path: str, device_type: str) -> str:
        """Return nested Watch App.app if this is a watch target, else the original path."""
        if device_type == "watch":
            nested = os.path.join(app_path, "Watch App.app")
            if os.path.isdir(nested):
                return nested
        return app_path

    def _scan_system_derived_data(
        self,
        project_name: str,
        sdk_dir: str,
        device_type: str,
    ) -> str | None:
        """Scan ~/Library/Developer/Xcode/DerivedData for matching .app bundles."""
        import glob

        derived_data = Path.home() / "Library" / "Developer" / "Xcode" / "DerivedData"
        if not derived_data.is_dir():
            return None

        project_prefix = (
            project_name.lower().replace("-", "").replace("_", "") if project_name else ""
        )
        alt_prefixes = ["iosapp", "watchapp"]

        all_matches = []
        for dd_dir in derived_data.iterdir():
            if not dd_dir.is_dir():
                continue
            dirname = dd_dir.name.lower()
            is_match = (project_prefix and dirname.startswith(project_prefix)) or any(
                dirname.startswith(p) for p in alt_prefixes
            )
            if is_match:
                pattern = str(dd_dir / "Build" / "Products" / sdk_dir / "*.app")
                all_matches.extend(glob.glob(pattern))

        if all_matches:
            match = sorted(all_matches, key=os.path.getmtime, reverse=True)[0]
            return self._resolve_watch_nested(match, device_type)
        return None

    def find_ios_app(
        self, name: str, device_type: str = "", is_simulator: bool = True
    ) -> str | None:
        """Find the most recent iOS .app for a project."""
        import glob

        project = self._projects.get(name)
        if not project or not project.path:
            return None

        if is_simulator:
            sdk_dir = "Debug-watchsimulator" if device_type == "watch" else "Debug-iphonesimulator"
        else:
            sdk_dir = "Debug-watchos" if device_type == "watch" else "Debug-iphoneos"

        # 1. Deterministic derived data path
        patterns = [
            f"build/derived/Build/Products/{sdk_dir}/*.app",
            f"iosApp/build/derived/Build/Products/{sdk_dir}/*.app",
            f"build/Build/Products/{sdk_dir}/*.app",
            f"iosApp/build/Build/Products/{sdk_dir}/*.app",
        ]

        for pattern in patterns:
            full_pattern = os.path.join(project.path, pattern)
            matches = sorted(glob.glob(full_pattern), key=os.path.getmtime, reverse=True)
            if matches:
                return self._resolve_watch_nested(matches[0], device_type)

        # 2. System DerivedData
        return self._scan_system_derived_data(
            getattr(project, "name", ""),
            sdk_dir,
            device_type,
        )

    def find_module_ios_apps(self, name: str) -> dict[str, str]:
        """Find iOS .app bundles for each module in a multi-target project.

        Returns a dict of module_name → .app path. Modules without a
        matching .app are omitted.
        """
        project = self._projects.get(name)
        if not project or not project.path or not project.modules:
            return {}

        result: dict[str, str] = {}
        for module in project.modules:
            if module.app_pattern:
                pattern = module.app_pattern
            else:
                sdk_dir = (
                    "Debug-watchsimulator"
                    if "watch" in module.device_types
                    else "Debug-iphonesimulator"
                )
                pattern = f"{module.name}/build/Build/Products/{sdk_dir}/*.app"
            full_pattern = os.path.join(project.path, pattern)
            matches = sorted(glob.glob(full_pattern), key=os.path.getmtime, reverse=True)
            if matches:
                result[module.name] = matches[0]

        return result
