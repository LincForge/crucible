"""Preflight checker — app store readiness analysis.

Inspects a project's code quality, store listing, brand brief,
visual assets, and build artifacts to produce a readiness report.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from .deployer import artifacts_older_than_head
from .forge_integration import ForgeProject, ForgeProjectReader
from .scorer import QualityScore, QualityScorer

logger = structlog.get_logger().bind(domain="build_pipeline")


def _default_linc_workspace() -> Path:
    """Return the default LINC workspace path, honouring ``LINC_HOME``.

    Returns:
        ``$LINC_HOME/projects/LINC`` when ``LINC_HOME`` is set,
        otherwise ``~/projects/LINC``.  Re-evaluated on every call so
        that ``os.environ`` monkeypatching in tests takes effect.
    """
    home = os.getenv("LINC_HOME")
    if home:
        return Path(home) / "projects" / "LINC"
    return Path.home() / "projects" / "LINC"


READINESS_THRESHOLD = 80
SCORE_MAX_AGE_DAYS = 7

PREFLIGHT_WEIGHTS = {
    "code_quality": 0.30,
    "store_listing": 0.20,
    "brand_brief": 0.15,
    "visual_assets": 0.20,
    "build_artifacts": 0.15,
}

# Required fields in brand-brief.json (matches schema required list)
BRAND_BRIEF_REQUIRED = [
    "app_name",
    "tagline",
    "emotion",
    "primary_color",
    "accent_color",
    "background_color",
    "icon_concept",
    "style",
    "platforms",
]

# Recommended fields that improve readiness
BRAND_BRIEF_RECOMMENDED = [
    "bundle_id",
    "target_audience",
    "marketing_themes",
    "screenshot_captions",
]

# Required sections in store-listing.md (detected by heading patterns)
STORE_LISTING_REQUIRED = [
    "short description",
    "full description",
    "keywords",
    "privacy policy",
]

STORE_LISTING_RECOMMENDED = [
    "screenshots",
    "feature graphic",
    "app identity",
]


@dataclass
class CheckItem:
    """A single item within a preflight check."""

    name: str
    found: bool
    detail: str = ""


@dataclass
class PreflightCheck:
    """Result of a single preflight check category."""

    name: str
    status: str = "skip"  # "pass", "warn", "fail", "skip"
    score: float = 0.0  # 0-100
    weight: float = 0.0
    detail: str = ""
    items: list[CheckItem] = field(default_factory=list)


@dataclass
class PreflightReport:
    """Complete preflight readiness report for a project."""

    project: str
    timestamp: str = ""
    checks: list[PreflightCheck] = field(default_factory=list)
    readiness_score: float = 0.0
    ready: bool = False
    summary: str = ""
    # Names of checks that veto `ready` regardless of composite readiness_score
    # (today only "code_quality", when its status is not "pass" — i.e. the
    # assay's own ship gate did not pass). Empty when nothing vetoes.
    blocking: list[str] = field(default_factory=list)
    not_found: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "project": self.project,
            "timestamp": self.timestamp,
            "readiness_score": round(self.readiness_score, 1),
            "ready": self.ready,
            "blocking": list(self.blocking),
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "score": round(c.score, 1),
                    "weight": c.weight,
                    "detail": c.detail,
                    "items": [
                        {"name": i.name, "found": i.found, "detail": i.detail} for i in c.items
                    ],
                }
                for c in self.checks
            ],
        }


def _check_artifact(name: str, path: str | None, *, is_dir: bool = False) -> CheckItem:
    """Evaluate whether a build artifact exists on disk."""
    if not path:
        return CheckItem(name=name, found=False, detail="not found")
    found = os.path.isdir(path) if is_dir else os.path.isfile(path)
    return CheckItem(name=name, found=found, detail=path)


def _artifact_staleness_detail(items: list[CheckItem], project_dir: str) -> str:
    """Return the ``artifact_older_than_head`` suffix for found artifacts ("" if none)."""
    paths = [item.detail for item in items if item.found]
    if not paths:
        return ""
    return f" (artifact_older_than_head={artifacts_older_than_head(paths, project_dir)})"


def _parse_score_timestamp(raw_timestamp: str | None) -> datetime | None:
    """Parse an ISO timestamp from a quality score, treating naive as UTC."""
    if not isinstance(raw_timestamp, str):
        return None
    raw = raw_timestamp.strip()
    if not raw:
        return None
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        return None


def _check_score_staleness(latest: QualityScore) -> str | None:
    """Validate that a quality score is current-schema and not stale.

    Returns:
        Failure detail string if stale or unparseable, else None.
    """
    if latest.stale:
        return "Quality score was computed under an older scoring schema — re-run crucible_assay"

    parsed_dt = _parse_score_timestamp(latest.timestamp)
    if parsed_dt is None:
        return "Quality score has unparseable timestamp — re-run crucible_assay"

    age = datetime.now(UTC) - parsed_dt
    if age > timedelta(days=SCORE_MAX_AGE_DAYS):
        return (
            f"Quality score is {age.days} days old"
            f" (max {SCORE_MAX_AGE_DAYS}) — re-run crucible_assay"
        )
    return None


def _compute_blocking(checks: list[PreflightCheck]) -> list[str]:
    """Return the check names that veto `ready` regardless of composite score.

    Today only "code_quality" vetoes, when its ship-gate status is not "pass" —
    a "warn"/"fail" ship-gate status blocks `ready` even when the composite
    clears READINESS_THRESHOLD on the strength of the other (metadata-presence)
    checks. Extracted out of `PreflightChecker.preflight` to
    keep that method's own complexity from absorbing the veto's branching.
    """
    code_quality_check = next((c for c in checks if c.name == "code_quality"), None)
    if code_quality_check is not None and code_quality_check.status != "pass":
        return ["code_quality"]
    return []


def _build_summary(
    readiness_score: float, checks: list[PreflightCheck], blocking: list[str]
) -> str:
    """Build the human-readable one-line summary for a preflight report."""
    passed = sum(1 for c in checks if c.status == "pass")
    warned = sum(1 for c in checks if c.status == "warn")
    failed = sum(1 for c in checks if c.status == "fail")
    summary = f"{readiness_score}% ready — {passed} pass, {warned} warn, {failed} fail"
    if blocking:
        blocked_by = ", ".join(blocking)
        summary += f" — blocked by {blocked_by} (ship gate not passed)"
    return summary


class PreflightChecker:
    """Inspects a project for app store readiness across multiple dimensions."""

    def __init__(
        self,
        forge_reader: ForgeProjectReader | None = None,
        scorer: QualityScorer | None = None,
        linc_workspace: Path | None = None,
    ) -> None:
        self._forge = forge_reader or ForgeProjectReader()
        self._scorer = scorer or QualityScorer()

        env_path = os.environ.get("LINC_WORKSPACE")
        if linc_workspace:
            self._linc_workspace = linc_workspace
        elif env_path:
            self._linc_workspace = Path(env_path)
        else:
            self._linc_workspace = _default_linc_workspace()

    def preflight(self, project_name: str) -> PreflightReport:
        """Run a complete preflight readiness check.

        Args:
            project_name: Forge project name (e.g. "myapp").

        Returns:
            PreflightReport with per-category checks and composite readiness score.
        """
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        report = PreflightReport(project=project_name, timestamp=now)

        project = self._forge.get(project_name)
        if not project:
            report.not_found = True
            report.summary = f"Project '{project_name}' not found in forge registry"
            return report

        docs_dir = self._resolve_docs_dir(project_name, project.path)

        report.checks = [
            self._check_code_quality(project_name),
            self._check_store_listing(docs_dir),
            self._check_brand_brief(docs_dir),
            self._check_visual_assets(project, docs_dir),
            self._check_build_artifacts(project),
        ]

        # Compute composite readiness score
        composite = sum(c.score * c.weight for c in report.checks)
        # Gate on the displayed (rounded) readiness_score so ready never contradicts
        # the shown score — mirroring scorer.py:493-502. A raw composite in
        # [79.95, 80.0) displays as 80.0 and must not fail the gate.
        report.readiness_score = round(composite, 1)

        # code_quality is a per-check veto, not just a weighted contributor —
        # see `_compute_blocking`.
        report.blocking = _compute_blocking(report.checks)
        report.ready = report.readiness_score >= READINESS_THRESHOLD and not report.blocking
        report.summary = _build_summary(report.readiness_score, report.checks, report.blocking)

        return report

    def _resolve_docs_dir(self, project_name: str, project_path: str = "") -> Path:
        """Resolve the store-metadata directory for a project."""
        return _probe_docs_dir(self._linc_workspace, project_name, project_path=project_path)

    def _check_code_quality(self, project_name: str) -> PreflightCheck:
        """Check code quality from the latest crucible score."""
        check = PreflightCheck(
            name="code_quality",
            weight=PREFLIGHT_WEIGHTS["code_quality"],
        )

        latest = self._scorer.load_latest(project_name)
        if not latest:
            check.status = "fail"
            check.score = 0.0
            check.detail = "No quality score found — run crucible_assay first"
            return check

        staleness_detail = _check_score_staleness(latest)
        if staleness_detail:
            check.status = "fail"
            check.score = 0.0
            check.detail = staleness_detail
            return check

        check.score = min(100.0, latest.composite)

        for dim in latest.dimensions:
            check.items.append(
                CheckItem(
                    name=dim.name,
                    found=dim.score >= 50,
                    detail=f"score={dim.score:.0f}, weight={dim.weight}",
                )
            )

        check.items.append(
            CheckItem(
                name="ship_gate",
                found=latest.ship_gate,
                detail=f"composite={latest.composite:.1f} (gate=85)",
            )
        )

        if latest.ship_gate:
            check.status = "pass"
            check.detail = f"Quality score {latest.composite:.1f} — ship gate passed"
        elif latest.composite >= 70:
            check.status = "warn"
            check.detail = f"Quality score {latest.composite:.1f} — below ship gate (85)"
        else:
            check.status = "fail"
            check.detail = f"Quality score {latest.composite:.1f} — needs improvement"

        return check

    def _check_store_listing(self, docs_dir: Path) -> PreflightCheck:
        """Check store listing completeness."""
        check = PreflightCheck(
            name="store_listing",
            weight=PREFLIGHT_WEIGHTS["store_listing"],
        )

        listing_path = docs_dir / "store-listing.md"
        if not listing_path.is_file():
            check.status = "fail"
            check.score = 0.0
            check.detail = "store-listing.md not found"
            check.items.append(CheckItem(name="store-listing.md", found=False))
            return check

        check.items.append(CheckItem(name="store-listing.md", found=True))

        try:
            content = listing_path.read_text().lower()
        except OSError:
            check.status = "fail"
            check.score = 0.0
            check.detail = "Could not read store-listing.md"
            return check

        # Check required sections
        required_found = 0
        for section in STORE_LISTING_REQUIRED:
            found = section in content
            check.items.append(CheckItem(name=f"section:{section}", found=found))
            if found:
                required_found += 1

        # Check recommended sections
        recommended_found = 0
        for section in STORE_LISTING_RECOMMENDED:
            found = section in content
            check.items.append(
                CheckItem(
                    name=f"section:{section}",
                    found=found,
                    detail="recommended",
                )
            )
            if found:
                recommended_found += 1

        # Check for placeholder markers
        has_placeholders = "[placeholder" in content or "todo" in content or "tbd" in content
        if has_placeholders:
            check.items.append(
                CheckItem(
                    name="no_placeholders",
                    found=False,
                    detail="Contains placeholder/TODO text",
                )
            )

        # Score: required sections are 70%, recommended are 20%, no placeholders is 10%
        req_total = len(STORE_LISTING_REQUIRED)
        rec_total = len(STORE_LISTING_RECOMMENDED)
        score = (
            (required_found / req_total * 70)
            + (recommended_found / rec_total * 20)
            + (0 if has_placeholders else 10)
        )
        check.score = min(100.0, score)

        if required_found == req_total and not has_placeholders:
            check.status = "pass"
            check.detail = f"All {req_total} required sections present"
        elif required_found == req_total:
            check.status = "warn"
            check.detail = "Required sections present but has placeholders"
        else:
            missing = req_total - required_found
            check.status = "fail" if missing > 1 else "warn"
            check.detail = f"{missing} required section(s) missing"

        return check

    def _check_brand_brief(self, docs_dir: Path) -> PreflightCheck:
        """Check brand brief completeness."""
        check = PreflightCheck(
            name="brand_brief",
            weight=PREFLIGHT_WEIGHTS["brand_brief"],
        )

        brief_path = docs_dir / "brand-brief.json"
        if not brief_path.is_file():
            check.status = "fail"
            check.score = 0.0
            check.detail = "brand-brief.json not found"
            check.items.append(CheckItem(name="brand-brief.json", found=False))
            return check

        check.items.append(CheckItem(name="brand-brief.json", found=True))

        try:
            with open(brief_path) as f:
                brief = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            check.status = "fail"
            check.score = 0.0
            check.detail = f"Could not parse brand-brief.json: {exc}"
            return check

        if not isinstance(brief, dict):
            check.status = "fail"
            check.score = 0.0
            check.detail = "brand-brief.json is not a JSON object"
            return check

        # Check required fields
        required_found = 0
        for field_name in BRAND_BRIEF_REQUIRED:
            value = brief.get(field_name)
            found = value is not None and value != "" and value != []
            check.items.append(CheckItem(name=f"field:{field_name}", found=found))
            if found:
                required_found += 1

        # Check recommended fields
        recommended_found = 0
        for field_name in BRAND_BRIEF_RECOMMENDED:
            value = brief.get(field_name)
            found = value is not None and value != "" and value != []
            check.items.append(
                CheckItem(
                    name=f"field:{field_name}",
                    found=found,
                    detail="recommended",
                )
            )
            if found:
                recommended_found += 1

        # Score: required 75%, recommended 25%
        req_total = len(BRAND_BRIEF_REQUIRED)
        rec_total = len(BRAND_BRIEF_RECOMMENDED)
        score = (required_found / req_total * 75) + (recommended_found / rec_total * 25)
        check.score = min(100.0, score)

        if required_found == req_total:
            check.status = "pass"
            check.detail = f"All {req_total} required fields present"
        else:
            missing = req_total - required_found
            check.status = "fail" if missing > 2 else "warn"
            check.detail = f"{missing} required field(s) missing"

        return check

    def _has_android_icon(self, project_path: Path) -> bool:
        return any(
            (project_path / f"app/src/androidMain/res/mipmap-{d}").is_dir()
            or (project_path / f"app/src/main/res/mipmap-{d}").is_dir()
            for d in ["hdpi", "xhdpi", "xxhdpi", "xxxhdpi"]
        )

    def _has_ios_icon(self, project_path: Path) -> bool:
        return (project_path / "iosApp" / "iosApp" / "Assets.xcassets").is_dir() or (
            project_path / "iosApp" / "Assets.xcassets"
        ).is_dir()

    def _has_marketing_images(self, generated_dir: Path) -> bool:
        return any(list(generated_dir.glob(ext)) for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"))

    def _check_visual_assets(self, project: ForgeProject, docs_dir: Path) -> PreflightCheck:
        """Check visual asset inventory (icons, screenshots)."""
        check = PreflightCheck(
            name="visual_assets",
            weight=PREFLIGHT_WEIGHTS["visual_assets"],
        )

        project_path = Path(project.path) if project.path else None
        if not project_path or not project_path.is_dir():
            check.status = "skip"
            check.detail = "Project path not accessible"
            return check

        found_count = 0
        total_count = 0

        # Android adaptive icon
        total_count += 1
        android_icon = self._has_android_icon(project_path)
        check.items.append(
            CheckItem(name="android_icon", found=android_icon, detail="mipmap directories")
        )
        if android_icon:
            found_count += 1

        # iOS icon
        total_count += 1
        ios_icon = self._has_ios_icon(project_path)
        check.items.append(CheckItem(name="ios_icon", found=ios_icon, detail="Assets.xcassets"))
        if ios_icon:
            found_count += 1

        # Generated assets in LINC docs
        total_count += 1
        assets_dir = docs_dir / "assets"
        has_assets_dir = assets_dir.is_dir()
        check.items.append(
            CheckItem(name="linc_assets_dir", found=has_assets_dir, detail=str(assets_dir))
        )
        if has_assets_dir:
            found_count += 1

        # Generated assets subdirectory
        total_count += 1
        generated_dir = assets_dir / "generated"
        has_generated = generated_dir.is_dir()
        check.items.append(
            CheckItem(name="generated_assets", found=has_generated, detail="assets/generated/")
        )
        if has_generated:
            found_count += 1

        # Feature graphic or marketing images
        total_count += 1
        has_marketing = False
        if has_generated:
            has_marketing = self._has_marketing_images(generated_dir)
        check.items.append(
            CheckItem(
                name="marketing_images", found=has_marketing, detail="images in assets/generated/"
            )
        )
        if has_marketing:
            found_count += 1

        # Score
        check.score = (found_count / total_count * 100) if total_count > 0 else 0.0

        if found_count == total_count:
            check.status = "pass"
            check.detail = f"All {total_count} asset checks passed"
        elif found_count >= total_count * 0.6:
            check.status = "warn"
            check.detail = f"{found_count}/{total_count} asset checks passed"
        else:
            check.status = "fail"
            check.detail = f"{found_count}/{total_count} asset checks passed"

        return check

    def _check_build_artifacts(self, project: ForgeProject) -> PreflightCheck:
        """Check for existing build artifacts."""
        check = PreflightCheck(
            name="build_artifacts",
            weight=PREFLIGHT_WEIGHTS["build_artifacts"],
        )

        found_count = 0
        total_count = 0

        # Android APK
        total_count += 1
        item = _check_artifact("android_apk", self._forge.find_apk(project.name))
        check.items.append(item)
        if item.found:
            found_count += 1

        # iOS app
        total_count += 1
        item = _check_artifact("ios_app", self._forge.find_ios_app(project.name), is_dir=True)
        check.items.append(item)
        if item.found:
            found_count += 1

        # Multi-target modules
        if project.is_multi_target:
            module_apks = self._forge.find_module_apks(project.name)
            for module in project.modules:
                total_count += 1
                item = _check_artifact(f"module:{module.name}", module_apks.get(module.name))
                check.items.append(item)
                if item.found:
                    found_count += 1

        # Score
        check.score = (found_count / total_count * 100) if total_count > 0 else 0.0

        if found_count == total_count:
            check.status = "pass"
            check.detail = f"All {total_count} build artifacts found"
        elif found_count > 0:
            check.status = "warn"
            check.detail = f"{found_count}/{total_count} build artifacts found"
        else:
            check.status = "fail"
            check.detail = "No build artifacts found — run a build first"

        check.detail += _artifact_staleness_detail(check.items, project.path)

        return check


def _probe_docs_dir(workspace: Path, project_name: str, project_path: str = "") -> Path:
    if project_path:
        local_store = Path(project_path) / "store"
        if local_store.is_dir():
            return local_store

    lower_name = project_name.lower()
    default_path = workspace / "projects" / "active" / lower_name
    if default_path.is_dir():
        return default_path

    # Dynamic directory probe
    active_dir = workspace / "projects" / "active"
    if active_dir.is_dir():
        norm_target = _normalize(project_name)
        for child in active_dir.iterdir():
            if child.is_dir() and _normalize(child.name) == norm_target:
                return child

    return default_path


def _normalize(name: str) -> str:
    n = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if n.endswith("app") and len(n) > 3:
        n = n[:-3]
    return n
