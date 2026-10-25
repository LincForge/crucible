"""Resolve the effective Android package for a build variant.

Two tiers: (1) badge an installed/built APK via aapt (ground truth, handles
flavor + buildType suffix + manifest placeholders); (2) source-parse Gradle for
``applicationId`` + the buildType ``applicationIdSuffix``. Tier 1 degrades to
Tier 2 on any aapt miss/error. See
docs/superpowers/specs/2026-06-25-fleet-dev-suffix-package-design.md.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger().bind(domain="package_resolver")

_APP_ID_RE = re.compile(r'applicationId\s*=\s*"([^"]+)"')
_SUFFIX_RE = re.compile(r'applicationIdSuffix\s*=\s*"([^"]+)"')
_BADGING_NAME_RE = re.compile(r"package:\s*name='([^']+)'")
_WELL_KNOWN = ("app", "composeApp", "androidApp", "demo-app")


def parse_base_application_id(gradle_text: str) -> str | None:
    """Return the base ``applicationId`` from one Gradle file's text."""
    m = _APP_ID_RE.search(gradle_text)
    return m.group(1) if m else None


def _extract_block(text: str, open_brace_idx: int) -> str:
    """Return the substring inside the balanced ``{...}`` starting at open_brace_idx."""
    depth = 0
    for i in range(open_brace_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx + 1 : i]
    return text[open_brace_idx + 1 :]


def parse_build_type_suffix(gradle_text: str, build_type: str) -> str | None:
    """Return the ``applicationIdSuffix`` declared in the build_type block, or None.

    Recognizes ``getByName("debug")``, ``named("debug")``, and bare ``debug {``.
    """
    bt = re.escape(build_type)
    decl = re.compile(rf'(?:getByName\(\s*"{bt}"\s*\)|named\(\s*"{bt}"\s*\)|\b{bt}\b)\s*\{{')
    m = decl.search(gradle_text)
    if not m:
        return None
    block = _extract_block(gradle_text, m.end() - 1)
    sm = _SUFFIX_RE.search(block)
    return sm.group(1) if sm else None


def _find_gradle_with_app_id(project_path: str) -> Path | None:
    root = Path(project_path)
    candidates = [root / name / "build.gradle.kts" for name in _WELL_KNOWN]
    try:
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name not in _WELL_KNOWN:
                candidates.append(child / "build.gradle.kts")
    except OSError:
        pass
    for gradle in candidates:
        if gradle.is_file():
            try:
                if _APP_ID_RE.search(gradle.read_text(encoding="utf-8")):
                    return gradle
            except OSError:
                continue
    return None


def source_parse_package(project_path: str, build_type: str = "debug") -> str | None:
    """Tier 2: base applicationId + buildType suffix, parsed from Gradle source."""
    gradle = _find_gradle_with_app_id(project_path)
    if gradle is None:
        return None
    text = gradle.read_text(encoding="utf-8")
    base = parse_base_application_id(text)
    if not base:
        return None
    suffix = parse_build_type_suffix(text, build_type)
    return f"{base}{suffix}" if suffix else base


def find_aapt() -> str | None:
    """Locate aapt2/aapt: PATH first, then $ANDROID_HOME/build-tools/*/."""
    for tool in ("aapt2", "aapt"):
        found = shutil.which(tool)
        if found:
            return found
    home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if home:
        hits = sorted(
            glob.glob(os.path.join(home, "build-tools", "*", "aapt2"))
            + glob.glob(os.path.join(home, "build-tools", "*", "aapt"))
        )
        if hits:
            return hits[-1]
    return None


def badge_package(apk_path: str) -> str | None:
    """Tier 1: parse the package name from ``aapt dump badging <apk>``. None on any miss."""
    aapt = find_aapt()
    if not aapt:
        return None
    try:
        cp = subprocess.run(
            [aapt, "dump", "badging", apk_path],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("aapt_badging_failed", error=str(exc))
        return None
    if cp.returncode != 0:
        logger.debug("aapt_badging_nonzero", code=cp.returncode, stderr=cp.stderr[:200])
        return None
    m = _BADGING_NAME_RE.search(cp.stdout)
    return m.group(1) if m else None


def resolve_effective_package(
    project_path: str,
    build_type: str = "debug",
    apk_path: str | None = None,
    manifest_package: str = "",
) -> str | None:
    """Resolve the effective package: APK badging → manifest identity.package → Gradle source."""
    if apk_path:
        badged = badge_package(apk_path)
        if badged:
            return badged
    if manifest_package:
        return manifest_package
    if project_path:
        return source_parse_package(project_path, build_type)
    return None
