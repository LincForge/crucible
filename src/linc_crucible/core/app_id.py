"""App identity inference — Android applicationId and iOS bundle ID.

Extracted from ``deployer.py`` (C4, 2026-07-02): pure functions that
resolve which package/bundle a project deploys as, from APK badging,
``linc-manifest.yaml`` ``identity.package`` / ``identity.bundle_id``,
Gradle source, or iOS project files. ``deployer`` re-exports these names,
so existing import paths keep working.
"""

from __future__ import annotations

import re
from pathlib import Path

from .forge_integration import ForgeProject

_APPLICATION_ID_RE = re.compile(r'applicationId\s*=\s*"([^"]+)"')


def _parse_application_id_from_file(gradle_file: Path) -> str | None:
    """Parse a single ``applicationId`` from one build.gradle.kts file."""
    if not gradle_file.is_file():
        return None
    try:
        text = gradle_file.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _APPLICATION_ID_RE.search(text)
    return match.group(1) if match else None


def _parse_application_id(project_path: str) -> str | None:
    """Parse ``applicationId`` from Gradle build files.

    Checks well-known module dirs first, then scans all immediate
    subdirectories for a ``build.gradle.kts`` containing ``applicationId``.
    """
    root = Path(project_path)
    well_known = ["app", "composeApp", "androidApp", "demo-app"]
    candidates = [root / name / "build.gradle.kts" for name in well_known]

    # Also scan any other immediate subdirectory with a build file
    try:
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name not in well_known:
                gradle = child / "build.gradle.kts"
                if gradle.is_file():
                    candidates.append(gradle)
    except OSError:
        pass

    for gradle_file in candidates:
        parsed = _parse_application_id_from_file(gradle_file)
        if parsed:
            return parsed

    return None


def _parse_bundle_id_from_yml(yml_path: Path) -> str | None:
    """Parse bundle ID from a project.yml file."""
    if not yml_path.is_file():
        return None
    try:
        text = yml_path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    match = re.search(r"PRODUCT_BUNDLE_IDENTIFIER:\s*(\S+)", text)
    if match:
        return match.group(1)
    match = re.search(r"bundleIdPrefix:\s*(\S+)", text)
    if match:
        return match.group(1)
    return None


def _parse_bundle_id_from_pbxproj(pbxproj_path: Path) -> str | None:
    """Parse bundle ID from a project.pbxproj file."""
    if not pbxproj_path.is_file():
        return None
    try:
        text = pbxproj_path.read_text(encoding="utf-8")
    except OSError:
        return None
    # Take the first non-parameterized bundle ID
    for m in re.finditer(
        r'PRODUCT_BUNDLE_IDENTIFIER\s*=\s*"?([^";$]+)"?\s*;',
        text,
    ):
        val = m.group(1).strip()
        if "$" not in val:
            return val
    return None


def _parse_bundle_id(project_path: str) -> str | None:
    """Parse iOS bundle identifier from project config files.

    Searches ``iosApp/project.yml`` for ``bundleIdPrefix`` and
    ``iosApp/*.xcodeproj/project.pbxproj`` for ``PRODUCT_BUNDLE_IDENTIFIER``.
    """
    root = Path(project_path)

    # Try project.yml first (XcodeGen-based KMP projects)
    parsed = _parse_bundle_id_from_yml(root / "iosApp" / "project.yml")
    if parsed:
        return parsed

    # Try pbxproj (Xcode project) — iosApp/ first, then root-level fallback
    search_dirs = [root / "iosApp", root]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue

        # Check project.yml at this level (root-level fallback)
        if search_dir == root:
            parsed = _parse_bundle_id_from_yml(search_dir / "project.yml")
            if parsed:
                return parsed

        for xcodeproj in search_dir.glob("*.xcodeproj"):
            parsed = _parse_bundle_id_from_pbxproj(xcodeproj / "project.pbxproj")
            if parsed:
                return parsed

    return None


def _infer_package(project: ForgeProject, apk_path: str | None = None) -> str:
    """Infer the effective Android package for the debug build.

    Resolution order: badged APK (``apk_path``) → manifest ``identity.package``
    → Gradle source-parse (``applicationId`` + the debug ``applicationIdSuffix``)
    → ``""`` when none resolves.
    """
    from .package_resolver import resolve_effective_package

    resolved = resolve_effective_package(
        project.path,
        apk_path=apk_path,
        manifest_package=project.package,
    )
    return resolved or ""


def _infer_bundle_id(project: ForgeProject) -> str:
    """Infer iOS bundle ID from manifest ``identity.bundle_id`` or project config files.

    Prefers ``project.bundle_id`` (from ``linc-manifest.yaml`` ``identity.bundle_id``),
    then parses ``bundleIdPrefix`` or ``PRODUCT_BUNDLE_IDENTIFIER`` from iOS
    project files, returning ``""`` when neither resolves.
    """
    if project.bundle_id:
        return project.bundle_id
    if project.path:
        parsed = _parse_bundle_id(project.path)
        if parsed:
            return parsed
    return ""
