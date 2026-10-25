"""Assayer — quality analysis orchestrator.

Runs lint, tests, and optional MobSF security scans to produce
an ``AssayReport`` with quality scores and ship gate status.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import structlog

from linc_crucible.core.subprocess_utils import (
    CommandParseError,
    ShellMetacharError,
    safe_subprocess_run,
    trim_output,
)

from .forge_integration import ForgeProject, ForgeProjectReader
from .nerve_client import UiTreeUnavailableError
from .scorer import QualityScore, QualityScorer

logger = structlog.get_logger().bind(domain="build_pipeline")

MOBSF_DEFAULT_URL = "http://localhost:8000"


@dataclass
class LintResult:
    """Result of a lint run."""

    issues: int = 0
    output: str = ""
    success: bool = True
    # False when no lint command is configured (nothing ran). Drives the
    # scorer to exclude lint rather than score an unrun check as a perfect 100.
    configured: bool = True
    # True once a lint command was actually invoked (subprocess launched),
    # regardless of outcome — set on both the clean-run and the crash/refusal
    # paths, but NOT on the "no lint command configured" guard. Distinct from
    # `configured`: `configured=False` also covers "nothing to check here",
    # which must NOT force the ship gate closed.
    # `attempted_unmeasured` is the signal that does that, and it needs both.
    attempted: bool = False

    @property
    def score_input(self) -> int | None:
        """Lint issue count for scoring, or None when nothing was measured.

        None when: no lint command was configured at all (`configured` is
        False), OR the run exited non-zero and nothing parseable came back
        (`success` is False and `issues` is 0 — an exit-nonzero linter with
        no recognisable issue lines is unparsed, not clean). A linter that
        exits non-zero WITH parsed issues still scores those issues.
        """
        if not self.configured or (not self.success and self.issues == 0):
            return None
        return self.issues

    @property
    def attempted_unmeasured(self) -> bool:
        """True when a lint run was attempted but produced no usable score.

        Distinct from "not configured": a project with no lint command at all
        is legitimately excluded from scoring (never attempted, renormalise
        around it). A command that WAS invoked and then crashed, was refused,
        or exited without parseable output is `attempted=True` with
        `score_input is None` — that state must be gate-VISIBLE, not silently
        indistinguishable from "there was nothing to check here".
        """
        return self.attempted and self.score_input is None


@dataclass
class RunResult:
    """Result of a test run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    output: str = ""
    success: bool = True
    # False when no test command is configured (nothing ran).
    configured: bool = True
    # See LintResult.attempted — same semantics, mirrored for the test path.
    attempted: bool = False

    @property
    def pass_rate(self) -> float:
        """Test pass rate as a fraction (0.0 - 1.0)."""
        if self.total == 0:
            return 1.0
        return self.passed / self.total

    @property
    def score_input(self) -> float | None:
        """Test pass rate for scoring, or None when nothing was measured.

        None when: no test command was configured at all (`configured` is
        False), OR the run produced zero parseable results AND did not exit
        cleanly (`total == 0` and `success` is False — a build break, a
        collection error, or unparseable output). A genuinely empty-but-green
        suite (`total == 0`, `success` is True) is a SEPARATE, deliberate
        case: it still scores the vacuous `pass_rate` of 1.0 rather than
        being lumped in with a crash.
        """
        if not self.configured or (self.total == 0 and not self.success):
            return None
        return self.pass_rate

    @property
    def attempted_unmeasured(self) -> bool:
        """True when a test run was attempted but produced no usable score.

        See LintResult.attempted_unmeasured — same semantics.
        """
        return self.attempted and self.score_input is None

    def serialized_pass_rate(self) -> float | None:
        """Rounded pass rate for JSON output, or None when nothing was measured."""
        if self.score_input is None:
            return None
        return round(self.pass_rate, 3)


@dataclass
class SecurityResult:
    """Result of a MobSF security scan."""

    score: float = 0.0
    findings: list[dict[str, Any]] = field(default_factory=list)
    output: str = ""
    success: bool = True


@dataclass
class PerformanceResult:
    """Result of performance measurements."""

    apk_size_mb: float | None = None
    ipa_size_mb: float | None = None
    output: str = ""
    success: bool = True

    @property
    def measured(self) -> bool:
        """True when artifact size measurement succeeded with at least one size."""
        return self.success and any((self.apk_size_mb is not None, self.ipa_size_mb is not None))


@dataclass
class AccessibilityResult:
    """Result of accessibility analysis."""

    critical: int = 0
    major: int = 0
    minor: int = 0
    score: float = 100.0
    details: list[str] = field(default_factory=list)
    output: str = ""
    success: bool = True


@dataclass
class A11yTreeResult:
    """Result of a live a11y tree BVT check."""

    structural_score: float | None = None
    rules_score: float | None = 100.0
    diff_report: Any | None = None
    violations: list[Any] | None = None
    success: bool = True
    error: str = ""
    output: str = ""
    configured: bool = True
    attempted: bool = False
    error_category: str = ""

    @property
    def attempted_unmeasured(self) -> bool:
        """True when an a11y-tree run was attempted but failed/raised."""
        return self.attempted and not self.success

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        d: dict[str, Any] = {
            "structural_score": self.structural_score,
            "rules_score": self.rules_score,
            "success": self.success,
            "error": self.error,
            "output": self.output,
            "configured": self.configured,
            "attempted_unmeasured": self.attempted_unmeasured,
        }
        if self.error_category:
            d["error_category"] = self.error_category
        return d


def _has_measured_dimension(report: AssayReport) -> bool:
    """Return True when at least one assay dimension produced a score."""
    return any(
        (
            report.lint.score_input is not None,
            report.tests.score_input is not None,
            bool(report.security and report.security.success),
            bool(report.performance and report.performance.measured),
            bool(report.accessibility and report.accessibility.success),
            bool(report.a11y_tree and report.a11y_tree.success),
            bool(report.quality_score and report.quality_score.dimensions),
        )
    )


def _has_attempted_unmeasured(report: AssayReport) -> bool:
    """Return True when any attempted dimension on the report produced no score."""
    return any(
        (
            report.lint.attempted_unmeasured,
            report.tests.attempted_unmeasured,
            bool(report.security and not report.security.success),
            bool(report.performance and not report.performance.measured),
            bool(report.accessibility and not report.accessibility.success),
            bool(report.a11y_tree and report.a11y_tree.attempted_unmeasured),
            bool(report.quality_score and report.quality_score.attempted_unmeasured),
        )
    )


def _assay_status(report: AssayReport, measured: bool) -> str:
    """Classify an assay report as 'measured', 'partial', or 'unmeasured'."""
    if not measured:
        return "unmeasured"
    if _has_attempted_unmeasured(report):
        return "partial"
    return "measured"


@dataclass
class AssayReport:
    """Complete assay report for a project."""

    project: str
    mode: str  # "quick" or "full"
    lint: LintResult = field(default_factory=LintResult)
    tests: RunResult = field(default_factory=RunResult)
    security: SecurityResult | None = None
    performance: PerformanceResult | None = None
    accessibility: AccessibilityResult | None = None
    a11y_tree: A11yTreeResult | None = None
    quality_score: QualityScore | None = None
    not_found: bool = False

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        measured = _has_measured_dimension(self)
        d: dict = {
            "project": self.project,
            "mode": self.mode,
            "measured": measured,
            "status": _assay_status(self, measured),
            "lint": {
                "issues": self.lint.issues,
                "success": self.lint.success,
                # The agent-visible payload must be able to tell a measured
                # "0 issues" apart from a crashed run that never measured anything.
                "configured": self.lint.configured,
                "attempted_unmeasured": self.lint.attempted_unmeasured,
                "output": self.lint.output,
            },
            "tests": {
                "total": self.tests.total,
                "passed": self.tests.passed,
                "failed": self.tests.failed,
                "skipped": self.tests.skipped,
                "pass_rate": self.tests.serialized_pass_rate(),
                "success": self.tests.success,
                "configured": self.tests.configured,
                "attempted_unmeasured": self.tests.attempted_unmeasured,
                "output": self.tests.output,
            },
        }
        if self.security:
            d["security"] = {
                "score": self.security.score,
                "success": self.security.success,
            }
        if self.performance:
            perf_d: dict = {"success": self.performance.success}
            if self.performance.apk_size_mb is not None:
                perf_d["apk_size_mb"] = round(self.performance.apk_size_mb, 2)
            if self.performance.ipa_size_mb is not None:
                perf_d["ipa_size_mb"] = round(self.performance.ipa_size_mb, 2)
            d["performance"] = perf_d
        if self.accessibility:
            d["accessibility"] = {
                "critical": self.accessibility.critical,
                "major": self.accessibility.major,
                "minor": self.accessibility.minor,
                "score": round(self.accessibility.score, 1),
                "success": self.accessibility.success,
                "details": self.accessibility.details,
            }
        if self.a11y_tree:
            d["a11y_tree"] = self.a11y_tree.to_dict()
        if self.quality_score:
            d["quality_score"] = self.quality_score.to_dict()
        return d


class Assayer:
    """Orchestrates quality analysis: lint -> tests -> optional security scan."""

    def __init__(
        self,
        forge_reader: ForgeProjectReader | None = None,
        scorer: QualityScorer | None = None,
        mobsf_url: str = MOBSF_DEFAULT_URL,
        mobsf_api_key: str = "",
        nerve_client: Any | None = None,
        tree_capture: Any | None = None,
        device_policy: Any | None = None,
    ) -> None:
        self._forge = forge_reader or ForgeProjectReader()
        self._scorer = scorer or QualityScorer()
        self._mobsf_url = mobsf_url
        self._mobsf_api_key = mobsf_api_key
        self._nerve_client = nerve_client
        self._tree_capture = tree_capture
        self._device_policy = device_policy

    def _resolve_device_policy(self) -> Any | None:
        if self._device_policy is not None:
            return self._device_policy() if callable(self._device_policy) else self._device_policy
        from linc_crucible import server_context as ctx

        return ctx._cached_device_policy

    def assay(
        self,
        project_name: str,
        mode: str = "quick",
        perf: bool = False,
        a11y: bool = False,
        a11y_tree: bool = False,
        device: str = "",
        allow_personal: bool = False,
    ) -> AssayReport:
        """Run a complete quality assay on a project.

        Args:
            project_name: Forge project name.
            mode: "quick" (lint + tests) or "full" (+ MobSF security).
            perf: If True, measure artifact sizes (APK/IPA).
            a11y: If True, run static accessibility checks.
            a11y_tree: If True, run live a11y tree BVT via nerve.
            device: Device ID for a11y_tree check. Auto-discovers if empty.
            allow_personal: If True, permit a11y_tree checks on personal devices.

        Returns:
            AssayReport with results and quality score.
        """
        project = self._forge.get(project_name)
        if not project:
            report = AssayReport(project=project_name, mode=mode, not_found=True)
            report.tests.output = f"Project '{project_name}' not found in forge registry"
            report.tests.success = False
            return report

        report = AssayReport(project=project_name, mode=mode)

        # Step 1: Lint
        report.lint = self._run_lint(project)

        # Step 2: Tests
        report.tests = self._run_tests(project)

        # Steps 3-5: Optional dimensions (security, perf, a11y)
        optional_scores, unmeasured_optional = self._run_optional_dims(
            project, mode, perf, a11y, report
        )

        # Step 6: A11y tree BVT (live device check via nerve)
        a11y_structural_score_val = None
        a11y_rules_score_val = None
        if a11y_tree:
            report.a11y_tree = self._run_a11y_tree(
                project_name, device, allow_personal=allow_personal
            )
            if report.a11y_tree.success:
                a11y_structural_score_val = report.a11y_tree.structural_score
                a11y_rules_score_val = report.a11y_tree.rules_score

        # Resolve commit_sha if project.path is a git repo
        commit_sha = _resolve_commit_sha(project.path)

        # A dimension that was ATTEMPTED (configured or requested) but produced
        # no usable score must be gate-VISIBLE — the scorer needs to know this
        # is different from "no command was configured", or it silently
        # renormalises weight onto whatever sibling dimension survived and ships
        # anyway.
        attempted_unmeasured_dims: list[str] = []
        if report.lint.attempted_unmeasured:
            attempted_unmeasured_dims.append("lint")
        if report.tests.attempted_unmeasured:
            attempted_unmeasured_dims.append("tests")
        if report.a11y_tree and report.a11y_tree.attempted_unmeasured:
            attempted_unmeasured_dims.append("a11y_tree")
        attempted_unmeasured_dims.extend(unmeasured_optional)

        # Step 7: Compute quality score. Pass None for any check that did not
        # actually run (no command configured) so the scorer excludes it instead
        # of treating an unrun check as a perfect pass (false-green guard).
        quality_score = self._scorer.compute(
            project=project_name,
            test_pass_rate=report.tests.score_input,
            lint_issues=report.lint.score_input,
            security_score=optional_scores.get("security"),
            perf_score=optional_scores.get("perf"),
            a11y_score=optional_scores.get("a11y"),
            a11y_structural_score=a11y_structural_score_val,
            a11y_rules_score=a11y_rules_score_val,
            commit_sha=commit_sha,
            attempted_unmeasured=attempted_unmeasured_dims,
        )
        self._scorer.save(quality_score)
        report.quality_score = quality_score

        return report

    def _run_optional_dims(
        self,
        project: ForgeProject,
        mode: str,
        perf: bool,
        a11y: bool,
        report: AssayReport,
    ) -> tuple[dict[str, float], list[str]]:
        """Run requested optional dimensions (security, perf, a11y).

        Returns:
            Tuple of (scores dict for QualityScorer, unmeasured dimension names).
        """
        scores: dict[str, float] = {}
        unmeasured: list[str] = []

        if mode == "full":
            report.security = self._run_security(project)
            if report.security.success:
                scores["security"] = report.security.score
            else:
                unmeasured.append("security")

        if perf:
            report.performance = self._measure_artifact_size(project)
            if report.performance.success and report.performance.apk_size_mb is not None:
                scores["perf"] = report.performance.apk_size_mb
            elif report.performance.success and report.performance.ipa_size_mb is not None:
                scores["perf"] = report.performance.ipa_size_mb
            else:
                unmeasured.append("perf")

        if a11y:
            report.accessibility = self._run_accessibility(project)
            if report.accessibility.success:
                scores["a11y"] = report.accessibility.score
            else:
                unmeasured.append("a11y")

        return scores, unmeasured

    def _run_lint(self, project: ForgeProject) -> LintResult:
        """Run lint check on a project."""
        lint_cmd = project.lint_command
        if not lint_cmd:
            return _missing_lint_result(project)

        try:
            result = safe_subprocess_run(lint_cmd, cwd=project.path, timeout=120)
            output = result.stdout + result.stderr
            issues = _count_lint_issues(output, project.framework)

            return LintResult(
                issues=issues,
                output=trim_output(output),
                success=result.returncode == 0,
                attempted=True,
            )
        except (
            subprocess.TimeoutExpired,
            OSError,
            ShellMetacharError,
            CommandParseError,
            ValueError,
        ) as exc:
            # A refused/crashed run is UNMEASURED, not a clean pass — mark
            # configured=False so score_input excludes it instead of the
            # scorer reading "0 issues" as a perfect lint dimension.
            # `attempted=True` is what lets attempted_unmeasured tell this
            # apart from "no lint command configured" — a command WAS
            # invoked here, it just never produced a usable result.
            # OSError also covers PermissionError (binary not executable) and
            # NotADirectoryError (a .forge.yaml path pointing at a file);
            # CommandParseError/ValueError covers shlex.split() on an unbalanced quote.
            # FileNotFoundError is already an OSError subclass.
            return LintResult(output=str(exc), success=False, configured=False, attempted=True)

    def _run_tests(self, project: ForgeProject) -> RunResult:
        """Run tests on a project."""
        test_cmd = project.test_command
        if not test_cmd:
            return _missing_test_result(project)

        try:
            result = safe_subprocess_run(test_cmd, cwd=project.path, timeout=300)
            return _build_test_run_result(result, project.framework)
        except (
            subprocess.TimeoutExpired,
            OSError,
            ShellMetacharError,
            CommandParseError,
            ValueError,
        ) as exc:
            # A refused/crashed run is UNMEASURED, not a clean pass — mark
            # configured=False so score_input excludes it instead of the
            # scorer reading the vacuous total==0 pass_rate as a perfect 1.0.
            # `attempted=True` — see the matching comment in _run_lint above.
            return RunResult(output=str(exc), success=False, configured=False, attempted=True)

    def _measure_artifact_size(self, project: ForgeProject) -> PerformanceResult:
        """Measure APK and/or IPA artifact sizes."""
        apk_size = None
        ipa_size = None

        apk_path = self._forge.find_apk(project.name)
        if apk_path and os.path.isfile(apk_path):
            apk_size = os.path.getsize(apk_path) / (1024 * 1024)

        ios_path = self._forge.find_ios_app(project.name)
        if ios_path and os.path.isdir(ios_path):
            # .app is a directory — sum all file sizes
            total = 0
            for dirpath, _dirnames, filenames in os.walk(ios_path):
                for f in filenames:
                    total += os.path.getsize(os.path.join(dirpath, f))
            ipa_size = total / (1024 * 1024)
        elif ios_path and os.path.isfile(ios_path):
            ipa_size = os.path.getsize(ios_path) / (1024 * 1024)

        if apk_size is None and ipa_size is None:
            return PerformanceResult(
                output="No APK or iOS app found",
                success=False,
            )

        parts = []
        if apk_size is not None:
            parts.append(f"APK: {apk_size:.2f} MB")
        if ipa_size is not None:
            parts.append(f"iOS: {ipa_size:.2f} MB")

        return PerformanceResult(
            apk_size_mb=apk_size,
            ipa_size_mb=ipa_size,
            output=", ".join(parts),
            success=True,
        )

    def _run_security(self, project: ForgeProject) -> SecurityResult:
        """Run MobSF security scan on a project's APK."""
        if not self._mobsf_api_key:
            return SecurityResult(output="MobSF API key not configured", success=False)

        # Find APK
        apk_path = self._forge.find_apk(project.name)
        if not apk_path:
            return SecurityResult(output=f"No APK found for {project.name}", success=False)

        try:
            # MobSF is only ever local here (no env var configures the URL; the
            # constructor defaults to loopback), so a remote MobSF behind a proxy
            # is not supported and trust_env=False prevents HTTP(S)_PROXY /
            # ALL_PROXY from routing loopback traffic or the Authorization header
            # off-host.
            with httpx.Client(trust_env=False, timeout=30) as client:
                # Upload
                with open(apk_path, "rb") as f:
                    upload_resp = client.post(
                        f"{self._mobsf_url}/api/v1/upload",
                        files={"file": f},
                        headers={"Authorization": self._mobsf_api_key},
                        timeout=60,
                    )
                upload_resp.raise_for_status()
                file_hash = upload_resp.json().get("hash", "")

                # Scan
                scan_resp = client.post(
                    f"{self._mobsf_url}/api/v1/scan",
                    data={"hash": file_hash},
                    headers={"Authorization": self._mobsf_api_key},
                    timeout=120,
                )
                scan_resp.raise_for_status()

                # Get report
                report_resp = client.post(
                    f"{self._mobsf_url}/api/v1/report_json",
                    data={"hash": file_hash},
                    headers={"Authorization": self._mobsf_api_key},
                    timeout=30,
                )
            report_resp.raise_for_status()
            report_data = report_resp.json()

            return SecurityResult(
                score=float(report_data.get("security_score", 0)),
                success=True,
            )

        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return SecurityResult(output=str(exc), success=False)

    def _run_accessibility(self, project: ForgeProject) -> AccessibilityResult:
        """Run static accessibility analysis on project source.

        Checks:
        1. Missing contentDescription in Android XML layouts (ImageView/ImageButton).
        2. Missing contentDescription in Compose Image()/Icon() calls.
        3. Touch target sizes < 48dp in XML layouts.
        """
        from .scorer import a11y_issues_to_score

        critical, major, minor = 0, 0, 0
        details: list[str] = []
        project_path = Path(project.path)

        if not project_path.is_dir():
            return AccessibilityResult(
                output=f"Project path not found: {project.path}",
                success=False,
            )

        try:
            self._check_xml_content_descriptions(project_path, details)
            self._check_compose_content_descriptions(project_path, details)
            self._check_touch_target_sizes(project_path, details)
            major = len(details)

        except OSError as exc:
            return AccessibilityResult(output=str(exc), success=False)

        score = a11y_issues_to_score(critical=critical, major=major, minor=minor)

        return AccessibilityResult(
            critical=critical,
            major=major,
            minor=minor,
            score=score,
            details=details,
            success=True,
        )

    def _check_xml_content_descriptions(self, project_path: Path, details: list[str]) -> None:
        """Check 1: Missing contentDescription in Android XML layouts."""
        for xml_file in project_path.rglob("*.xml"):
            if "layout" not in xml_file.relative_to(project_path).parts:
                continue
            try:
                content = xml_file.read_text(errors="replace")
            except OSError:
                continue
            img_pattern = re.compile(
                r"<(ImageView|ImageButton)\b(?:(?!/>|</)[^>])*?(/?>)",
                re.DOTALL,
            )
            for match in img_pattern.finditer(content):
                element = match.group(0)
                if "contentDescription" not in element:
                    rel = xml_file.relative_to(project_path)
                    details.append(f"Missing contentDescription: {match.group(1)} in {rel}")

    def _check_compose_content_descriptions(self, project_path: Path, details: list[str]) -> None:
        """Check 2: Missing contentDescription in Compose Image()/Icon() calls."""
        for kt_file in project_path.rglob("*.kt"):
            try:
                content = kt_file.read_text(errors="replace")
            except OSError:
                continue
            compose_pattern = re.compile(r"\b(Image|Icon)\s*\(")
            for match in compose_pattern.finditer(content):
                start = match.start()
                snippet = content[start : start + 500]
                if "contentDescription" not in snippet:
                    rel = kt_file.relative_to(project_path)
                    details.append(f"Missing contentDescription: {match.group(1)}() in {rel}")

    def _check_touch_target_sizes(self, project_path: Path, details: list[str]) -> None:
        """Check 3: Touch target sizes < 48dp in XML layouts."""
        for xml_file in project_path.rglob("*.xml"):
            if "layout" not in xml_file.relative_to(project_path).parts:
                continue
            try:
                content = xml_file.read_text(errors="replace")
            except OSError:
                continue
            size_pattern = re.compile(r'(?:layout_width|layout_height)\s*=\s*"(\d+)dp"')
            if "android:clickable" not in content and "android:onClick" not in content:
                continue
            for match in size_pattern.finditer(content):
                dp_val = int(match.group(1))
                if dp_val < 48:
                    rel = xml_file.relative_to(project_path)
                    details.append(f"Touch target too small ({dp_val}dp < 48dp) in {rel}")

    def _capture_a11y_elements(self, device: str) -> list[dict] | str:
        """Capture raw UI elements via nerve_client.elements_strict.

        Returns a list of raw element dictionaries, or a reason string when the
        UI tree cannot be read (UiTreeUnavailableError) or is empty. An empty
        tree is treated as an attempted-unmeasured measurement failure
        (empty UI tree), not perfect accessibility.
        """
        try:
            elements = self._nerve_client.elements_strict(device)
        except UiTreeUnavailableError as exc:
            return f"ui tree unavailable: {exc}"

        if not elements:
            return "ui tree empty: 0 elements — measurement failure, not perfect accessibility"

        return elements

    def _resolve_a11y_policy(
        self, device: str, allow_personal: bool
    ) -> tuple[list[dict], A11yTreeResult | None]:
        """Resolve the target device through DevicePolicy before reading the UI tree."""
        from .deployer import (
            PERSONAL_DEVICE_CATEGORY,
            discover_for_policy,
            resolve_policy_target,
        )

        discover_response = (
            discover_for_policy(self._nerve_client) if device else self._nerve_client.discover()
        )
        target, refusal = resolve_policy_target(
            discover_response, device, self._resolve_device_policy(), allow_personal
        )
        if refusal:
            return (
                discover_response,
                A11yTreeResult(
                    success=False,
                    error=refusal,
                    output=refusal,
                    error_category=PERSONAL_DEVICE_CATEGORY,
                ),
            )
        if not device:
            discover_response = [target] if target is not None else []
        return discover_response, None

    def _run_a11y_tree(
        self, project_name: str, device: str, allow_personal: bool = False
    ) -> A11yTreeResult:
        """Run live a11y tree BVT via nerve.

        Fetches the current UI element tree with elements_strict, diffs it against
        the stored baseline, and checks WCAG rules. Treats an unreadable or empty
        tree as attempted-unmeasured (ship gate closed), never as a11y_rules = 100.
        Returns an A11yTreeResult with structural and rules scores.
        """
        if self._nerve_client is None:
            return A11yTreeResult(
                success=False,
                error="nerve client not configured",
                output="nerve client not configured",
                rules_score=None,
                configured=False,
                attempted=False,
            )

        try:
            from linc_device_kit import Platform

            from .nerve_convert import convert_nerve_elements
            from .rule_checker import WCAGRuleChecker
            from .tree_differ import TreeDiffer

            # Auto-discover device if not specified
            active_device = device
            discover_response, refusal_result = self._resolve_a11y_policy(device, allow_personal)
            if refusal_result is not None:
                return refusal_result
            if not active_device:
                if not discover_response:
                    return A11yTreeResult(
                        success=False,
                        error="no devices available",
                        output="no devices available",
                        rules_score=None,
                        configured=True,
                        attempted=True,
                    )
                active_device = discover_response[0]["id"]

            # Determine platform from discover response
            platform = Platform.ANDROID
            for dev in discover_response:
                if dev.get("id") == active_device:
                    plat_str = dev.get("platform", "android").lower()
                    if plat_str == "ios":
                        platform = Platform.IOS
                    break

            # Fetch elements and convert to ElementNode list
            raw_elements = self._capture_a11y_elements(active_device)
            if isinstance(raw_elements, str):
                return A11yTreeResult(
                    success=False,
                    error=raw_elements,
                    output=raw_elements,
                    rules_score=None,
                    violations=None,
                    configured=True,
                    attempted=True,
                )
            current_elements = convert_nerve_elements(raw_elements)

            # Structural diff against baseline
            structural_score = None
            diff_report = None
            if self._tree_capture is not None:
                baseline_entry = self._tree_capture.get_latest(project_name, "default")
                if baseline_entry is not None:
                    baseline_elements = self._tree_capture.load_elements(baseline_entry)
                    diff_report = TreeDiffer.diff(baseline_elements, current_elements)
                    structural_score = float(diff_report.score)

            # WCAG rules check
            violations = WCAGRuleChecker.check(current_elements, platform)
            rules_score = float(WCAGRuleChecker.violations_to_score(violations))

            return A11yTreeResult(
                structural_score=structural_score,
                rules_score=rules_score,
                diff_report=diff_report,
                violations=violations,
                success=True,
                error="",
                output="",
                configured=True,
                attempted=True,
            )

        except Exception as exc:
            return A11yTreeResult(
                success=False,
                error=str(exc),
                output=str(exc),
                rules_score=None,
                violations=None,
                configured=True,
                attempted=True,
            )


def _count_lint_issues(output: str, framework: str) -> int:
    """Count lint issues from output based on framework.

    Heuristic: count lines that look like issue reports.
    """
    if not output:
        return 0

    # Ruff-style: "path.py:line:col: CODE message"
    ruff_pattern = re.compile(r"^\S+\.py:\d+:\d+: [A-Z]\d+", re.MULTILINE)
    ruff_matches = ruff_pattern.findall(output)
    if ruff_matches:
        return len(ruff_matches)

    # Gradle/ktlint style
    if "gradle" in framework:
        warning_count = output.count("w: ") + output.count("e: ")
        if warning_count > 0:
            return warning_count

    # Generic: count lines with "warning" or "error"
    count = 0
    for line in output.splitlines():
        lower = line.lower()
        if "warning:" in lower or "error:" in lower:
            count += 1
    return count


def _exit_contradicts_summary(returncode: int, total: int, failed: int) -> bool:
    """Return True when test process exited non-zero but zero failures were parsed."""
    return returncode != 0 and failed == 0


def _build_test_run_result(result: Any, framework: str) -> RunResult:
    """Build RunResult from completed subprocess test run and framework."""
    output = result.stdout + result.stderr
    total, passed, failed, skipped = _parse_test_results(output, framework)

    if _exit_contradicts_summary(result.returncode, total, failed):
        prefix = f"UNMEASURED: exit {result.returncode} with no parsed failures"
        unmeasured_output = f"{prefix}\n{output}" if output else prefix
        return RunResult(
            total=0,
            passed=0,
            failed=0,
            skipped=0,
            output=trim_output(unmeasured_output),
            success=False,
            attempted=True,
        )

    return RunResult(
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        output=trim_output(output),
        success=result.returncode == 0,
        attempted=True,
    )


def _pytest_counts(output: str) -> tuple[int, int, int, int]:
    """Parse pytest summary tokens into (total, passed, failed, skipped).

    error/errors are counted into failed; xfailed/xpassed are counted into total only.
    """
    matches = re.findall(
        r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)(?!\s+during\b)", output
    )
    if not matches:
        return (0, 0, 0, 0)

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    for count_str, outcome in matches:
        count = int(count_str)
        total += count
        if outcome == "passed":
            passed += count
        elif outcome in ("failed", "error", "errors"):
            failed += count
        elif outcome == "skipped":
            skipped += count
        # xfailed and xpassed are counted into total only
    return (total, passed, failed, skipped)


def _parse_test_results(output: str, framework: str) -> tuple:
    """Parse test results from output.

    Returns (total, passed, failed, skipped).
    """
    if not output:
        return (0, 0, 0, 0)

    # Gradle style: "Tests run: X, Failures: Y, Errors: Z, Skipped: W"
    gradle_match = re.search(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)(?:,\s*Skipped:\s*(\d+))?",
        output,
    )
    if gradle_match:
        total = int(gradle_match.group(1))
        failures = int(gradle_match.group(2))
        errors = int(gradle_match.group(3))
        skipped = int(gradle_match.group(4) or 0)
        failed = failures + errors
        passed = total - failed - skipped
        return (total, max(0, passed), failed, skipped)

    # Gradle Kotlin test: "X tests completed, Y failed"
    kt_match = re.search(r"(\d+) tests? completed(?:.*?(\d+) failed)?", output)
    if kt_match:
        total = int(kt_match.group(1))
        failed = int(kt_match.group(2) or 0)
        passed = total - failed
        return (total, passed, failed, 0)

    # pytest style: token scan
    counts = _pytest_counts(output)
    if counts[0] > 0:
        return counts

    return (0, 0, 0, 0)


def _resolve_commit_sha(project_path: str | Path | None) -> str | None:
    """Resolve commit SHA if the project path is a git repository."""
    if not project_path:
        return None
    path = Path(project_path)
    if not path.exists():
        return None
    try:
        git_res = safe_subprocess_run(
            "git rev-parse HEAD",
            cwd=str(path),
            timeout=10,
        )
        if git_res.returncode == 0:
            return git_res.stdout.strip()
    except Exception:
        pass
    return None


def _missing_lint_result(project: ForgeProject) -> LintResult:
    """Return the unconfigured or malformed-config result when ``lint_command`` is empty."""
    cfg_err = project.config_error_for("lint_command")
    out = cfg_err or "No lint command configured"
    res = LintResult(output=out, success=False, configured=False, attempted=bool(cfg_err))
    return res


def _missing_test_result(project: ForgeProject) -> RunResult:
    """Return the unconfigured or malformed-config result when ``test_command`` is empty."""
    cfg_err = project.config_error_for("test_command")
    out = cfg_err or "No test command configured"
    res = RunResult(output=out, success=False, configured=False, attempted=bool(cfg_err))
    return res
