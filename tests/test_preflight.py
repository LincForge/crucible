"""Tests for preflight readiness checker."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from linc_crucible.core.forge_integration import ForgeProject, ForgeProjectReader, ModuleConfig
from linc_crucible.core.preflight import (
    PREFLIGHT_WEIGHTS,
    READINESS_THRESHOLD,
    CheckItem,
    PreflightCheck,
    PreflightChecker,
    PreflightReport,
    _artifact_staleness_detail,
)
from linc_crucible.core.scorer import (
    SHIP_GATE_THRESHOLD,
    QualityScore,
    QualityScorer,
    ScoreDimension,
)
from tests.test_subprocess_utils import bridge_safe_run_to_subprocess_run


@pytest.fixture(autouse=True)
def _safe_run_via_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module fakes `git log` with @patch("subprocess.run"); safe_subprocess_run now
    spawns via Popen (linc-crucible-64q.5), so bridge its spawn back to that seam."""
    bridge_safe_run_to_subprocess_run(monkeypatch)


@pytest.fixture
def linc_workspace(tmp_path: Path) -> Path:
    """Create a minimal LINC workspace structure."""
    workspace = tmp_path / "LINC"
    workspace.mkdir()
    (workspace / "projects" / "active" / "testproject").mkdir(parents=True)
    return workspace


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    """Create a minimal KMP project directory."""
    proj = tmp_path / "TestProject"
    proj.mkdir()
    # Android mipmap dirs
    for density in ["hdpi", "xhdpi", "xxhdpi", "xxxhdpi"]:
        (proj / "app" / "src" / "androidMain" / "res" / f"mipmap-{density}").mkdir(parents=True)
    # iOS assets
    (proj / "iosApp" / "iosApp" / "Assets.xcassets").mkdir(parents=True)
    return proj


@pytest.fixture
def mock_forge(project_path: Path) -> MagicMock:
    forge = MagicMock(spec=ForgeProjectReader)
    forge.get.return_value = ForgeProject(
        name="TestProject",
        path=str(project_path),
        framework="kmp",
    )
    forge.find_apk.return_value = None
    forge.find_ios_app.return_value = None
    forge.find_module_apks.return_value = {}
    return forge


@pytest.fixture
def scorer(tmp_path: Path) -> QualityScorer:
    return QualityScorer(scores_dir=tmp_path / "scores")


def _write_store_listing(docs_dir: Path, *, placeholders: bool = False) -> None:
    """Write a store-listing.md with all required sections."""
    content = """# TestProject - Store Listing

## App Identity
- **App name:** TestProject

## Short Description (80 chars max)
A great test app for testing.

## Full Description (4000 chars max)
This is a full description of the test project.

## Keywords / Tags
test, app, great

## Privacy Policy
URL: https://example.com/privacy

## Screenshots
1. Main screen
2. Settings

## Feature Graphic (1024x500)
Dark background with app icon centered.
"""
    if placeholders:
        content += "\n[placeholder — to be completed]\n"
    (docs_dir / "store-listing.md").write_text(content)


def _write_brand_brief(docs_dir: Path, *, minimal: bool = False) -> None:
    """Write a brand-brief.json with required fields."""
    brief: dict = {
        "app_name": "TestProject",
        "tagline": "Test all the things",
        "emotion": "precision",
        "primary_color": "#1E40AF",
        "accent_color": "#60A5FA",
        "background_color": "#F8FAFF",
        "icon_concept": "A test icon",
        "style": "minimal",
        "platforms": ["android", "ios"],
    }
    if not minimal:
        brief.update(
            {
                "bundle_id": "com.test.project",
                "target_audience": "Testers",
                "marketing_themes": ["Theme 1"],
                "screenshot_captions": [{"screen": "main", "headline": "Main Screen"}],
            }
        )
    (docs_dir / "brand-brief.json").write_text(json.dumps(brief, indent=2))


def _setup_full_metadata(linc_workspace: Path, project_path: Path, mock_forge: MagicMock) -> None:
    """Write store-listing/brand-brief/assets and stub build artifacts so every
    non-code_quality check passes — the all-pass fixture shape shared by the
    readiness-veto tests below (issue #145 ACs 1-3)."""
    docs_dir = linc_workspace / "projects" / "active" / "testproject"
    _write_store_listing(docs_dir)
    _write_brand_brief(docs_dir)

    generated = docs_dir / "assets" / "generated"
    generated.mkdir(parents=True)
    (generated / "marketing.png").write_bytes(b"PNG")

    apk_file = project_path / "test.apk"
    apk_file.write_bytes(b"APK")
    ios_app_dir = project_path / "test.app"
    ios_app_dir.mkdir()
    mock_forge.find_apk.return_value = str(apk_file)
    mock_forge.find_ios_app.return_value = str(ios_app_dir)


class TestPreflightChecker:
    def test_project_not_found(self, scorer: QualityScorer, linc_workspace: Path) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = None

        checker = PreflightChecker(
            forge_reader=forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("NonExistent")

        assert report.project == "NonExistent"
        assert "not found" in report.summary
        assert report.readiness_score == 0.0
        assert report.ready is False

    def test_full_readiness(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        project_path: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        _write_store_listing(docs_dir)
        _write_brand_brief(docs_dir)

        # Create assets directory with generated images
        generated = docs_dir / "assets" / "generated"
        generated.mkdir(parents=True)
        (generated / "marketing.png").write_bytes(b"PNG")

        # Save a quality score
        score = scorer.compute("TestProject", test_pass_rate=1.0, lint_issues=0)
        scorer.save(score)

        # Mock build artifacts
        apk_file = project_path / "test.apk"
        apk_file.write_bytes(b"APK")
        ios_app_dir = project_path / "test.app"
        ios_app_dir.mkdir()
        mock_forge.find_apk.return_value = str(apk_file)
        mock_forge.find_ios_app.return_value = str(ios_app_dir)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        assert report.ready is True
        assert report.readiness_score >= READINESS_THRESHOLD
        assert len(report.checks) == 5

        # All checks should pass
        for check in report.checks:
            assert check.status == "pass", f"{check.name} was {check.status}: {check.detail}"

    def test_no_quality_score(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        code_check = next(c for c in report.checks if c.name == "code_quality")
        assert code_check.status == "fail"
        assert code_check.score == 0.0

    def test_code_quality_old_score_fails(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        score = QualityScore(
            project="TestProject",
            timestamp="2020-01-01T00:00:00Z",
            mode="quick",
            composite=99.0,
            ship_gate=True,
        )
        scorer.save(score)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        code_check = next(c for c in report.checks if c.name == "code_quality")
        assert code_check.status == "fail"
        assert code_check.score == 0.0
        assert "old" in code_check.detail.lower()

    def test_code_quality_stale_schema_fails(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        # Write score JSON without schema_version (loaded as stale=True)
        project_dir = scorer._scores_dir / "TestProject"
        project_dir.mkdir(parents=True, exist_ok=True)
        legacy = {
            "project": "TestProject",
            "timestamp": "2020-01-01T00:00:00Z",
            "mode": "quick",
            "composite": 99.0,
            "ship_gate": True,
            "commit_sha": None,
            "dimensions": [],
        }
        (project_dir / "2020-01-01T00-00-00Z.json").write_text(json.dumps(legacy))

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        code_check = next(c for c in report.checks if c.name == "code_quality")
        assert code_check.status == "fail"
        assert code_check.score == 0.0
        assert "schema" in code_check.detail.lower()

    @pytest.mark.parametrize("invalid_ts", ["", "not-a-date"])
    def test_code_quality_unparseable_or_empty_timestamp_fails(
        self,
        mock_forge: MagicMock,
        linc_workspace: Path,
        tmp_path: Path,
        invalid_ts: str,
    ) -> None:
        scores_dir = tmp_path / f"scores_{invalid_ts or 'empty'}"
        custom_scorer = QualityScorer(scores_dir=scores_dir)
        score = QualityScore(
            project="TestProject",
            timestamp=invalid_ts,
            mode="quick",
            composite=99.0,
            ship_gate=True,
        )
        custom_scorer.save(score)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=custom_scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        code_check = next(c for c in report.checks if c.name == "code_quality")
        assert code_check.status == "fail"
        assert code_check.score == 0.0

    def test_code_quality_fresh_score_window_6d_and_8d(
        self,
        mock_forge: MagicMock,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        now = datetime.now(UTC)
        # 6 days old passes with score equal to its composite
        ts_6d = (now - timedelta(days=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        scores_dir_6d = tmp_path / "scores_6d"
        scorer_6d = QualityScorer(scores_dir=scores_dir_6d)
        score_6d = QualityScore(
            project="TestProject",
            timestamp=ts_6d,
            mode="quick",
            composite=95.0,
            ship_gate=True,
        )
        scorer_6d.save(score_6d)

        checker_6d = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer_6d, linc_workspace=linc_workspace
        )
        report_6d = checker_6d.preflight("TestProject")
        code_check_6d = next(c for c in report_6d.checks if c.name == "code_quality")
        assert code_check_6d.status == "pass"
        assert code_check_6d.score == 95.0

        # 8 days old fails (hardcoded literal 7-day limit in test)
        ts_8d = (now - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        scores_dir_8d = tmp_path / "scores_8d"
        scorer_8d = QualityScorer(scores_dir=scores_dir_8d)
        score_8d = QualityScore(
            project="TestProject",
            timestamp=ts_8d,
            mode="quick",
            composite=95.0,
            ship_gate=True,
        )
        scorer_8d.save(score_8d)

        checker_8d = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer_8d, linc_workspace=linc_workspace
        )
        report_8d = checker_8d.preflight("TestProject")
        code_check_8d = next(c for c in report_8d.checks if c.name == "code_quality")
        assert code_check_8d.status == "fail"
        assert code_check_8d.score == 0.0

    def test_code_quality_naive_timestamp(
        self,
        mock_forge: MagicMock,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        # Test timestamp without timezone offset (naive, treated as UTC)
        now_utc = datetime.now(UTC)
        naive_ts = now_utc.strftime("%Y-%m-%dT%H:%M:%S")
        scores_dir = tmp_path / "scores_naive"
        scorer = QualityScorer(scores_dir=scores_dir)
        score = QualityScore(
            project="TestProject",
            timestamp=naive_ts,
            mode="quick",
            composite=92.0,
            ship_gate=True,
        )
        scorer.save(score)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")
        code_check = next(c for c in report.checks if c.name == "code_quality")
        assert code_check.status == "pass"
        assert code_check.score == 92.0

    def test_code_quality_warn_and_fail_composite(
        self,
        mock_forge: MagicMock,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        now_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Composite >= 70 but not ship_gate -> warn
        scores_dir_warn = tmp_path / "scores_warn"
        scorer_warn = QualityScorer(scores_dir=scores_dir_warn)
        score_warn = QualityScore(
            project="TestProject",
            timestamp=now_ts,
            mode="quick",
            composite=75.0,
            ship_gate=False,
            dimensions=[
                ScoreDimension(
                    name="tests",
                    raw_value=0.4,
                    score=40.0,
                    weight=0.5,
                    weighted_score=20.0,
                ),
            ],
        )
        scorer_warn.save(score_warn)

        checker_warn = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer_warn, linc_workspace=linc_workspace
        )
        report_warn = checker_warn.preflight("TestProject")
        check_warn = next(c for c in report_warn.checks if c.name == "code_quality")
        assert check_warn.status == "warn"
        assert "below ship gate" in check_warn.detail

        # Composite < 70 -> fail with needs improvement
        scores_dir_fail = tmp_path / "scores_fail"
        scorer_fail = QualityScorer(scores_dir=scores_dir_fail)
        score_fail = QualityScore(
            project="TestProject",
            timestamp=now_ts,
            mode="quick",
            composite=50.0,
            ship_gate=False,
        )
        scorer_fail.save(score_fail)

        checker_fail = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer_fail, linc_workspace=linc_workspace
        )
        report_fail = checker_fail.preflight("TestProject")
        check_fail = next(c for c in report_fail.checks if c.name == "code_quality")
        assert check_fail.status == "fail"
        assert "needs improvement" in check_fail.detail

    def test_low_code_quality_blocks_ready_despite_high_composite(
        self,
        mock_forge: MagicMock,
        linc_workspace: Path,
        project_path: Path,
        tmp_path: Path,
    ) -> None:
        """[#145 AC1] composite 34.0, ship_gate False, every other check
        passing (readiness_score >= 80, the all-pass fixture shape) still
        yields `ready is False`. Pre-fix this was True — the audit observed
        readiness 80.2, ready=True with a code_quality score of 34.0."""
        _setup_full_metadata(linc_workspace, project_path, mock_forge)

        scorer_ = QualityScorer(scores_dir=tmp_path / "scores_ac1")
        now_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        scorer_.save(
            QualityScore(
                project="TestProject",
                timestamp=now_ts,
                mode="quick",
                composite=34.0,
                ship_gate=False,
            )
        )

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer_, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        check = next(c for c in report.checks if c.name == "code_quality")
        assert check.status == "fail"
        assert report.readiness_score >= READINESS_THRESHOLD
        assert report.ready is False
        assert report.blocking == ["code_quality"]

    def test_warn_code_quality_blocks_ready(
        self,
        mock_forge: MagicMock,
        linc_workspace: Path,
        project_path: Path,
        tmp_path: Path,
    ) -> None:
        """[#145 AC2] composite 80.0 (warn band: 70 <= composite < 85),
        ship_gate False, every other check passing still yields
        `ready is False`. Pre-fix this was True."""
        _setup_full_metadata(linc_workspace, project_path, mock_forge)

        scorer_ = QualityScorer(scores_dir=tmp_path / "scores_ac2")
        now_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        scorer_.save(
            QualityScore(
                project="TestProject",
                timestamp=now_ts,
                mode="quick",
                composite=80.0,
                ship_gate=False,
            )
        )

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer_, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        check = next(c for c in report.checks if c.name == "code_quality")
        assert check.status == "warn"
        assert report.ready is False
        assert report.blocking == ["code_quality"]

    def test_blocking_empty_in_healthy_all_pass_case(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        project_path: Path,
    ) -> None:
        """[#145 AC3 / AC4 healthy case] the existing all-pass fixture
        (ship_gate True, every check passing) still yields `ready is True`
        with an empty `blocking` list — the veto must never fire on a
        genuinely healthy project."""
        _setup_full_metadata(linc_workspace, project_path, mock_forge)
        score = scorer.compute("TestProject", test_pass_rate=1.0, lint_issues=0)
        scorer.save(score)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        assert report.ready is True
        assert report.blocking == []

    def test_low_readiness_still_blocks_even_with_passing_code_quality(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        project_path: Path,
    ) -> None:
        """[#145 AC4 mandatory negative] a ship_gate-True (code_quality
        passing) project whose overall composite readiness_score is below
        READINESS_THRESHOLD still yields `ready is False` — the new
        code_quality veto is additive, it does not relax the existing
        composite-threshold gate. No metadata is written here, so every
        other check fails/scores low and composite stays under 80."""
        score = scorer.compute("TestProject", test_pass_rate=1.0, lint_issues=0)
        scorer.save(score)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        check = next(c for c in report.checks if c.name == "code_quality")
        assert check.status == "pass"
        assert report.readiness_score < READINESS_THRESHOLD
        assert report.ready is False
        assert report.blocking == []

    def test_ready_agrees_with_rounded_readiness_score_at_threshold(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        """[kiy.5 acceptance 1] checks whose weighted composite is 79.96
        (choose weights/scores that produce it) -> the report says readiness_score 80.0
        AND ready True (fails pre-fix: 80.0 / False). Mirrors scorer #62."""
        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        # Weights: code_quality 0.30, store_listing 0.20, brand_brief 0.15,
        # visual_assets 0.20, build_artifacts 0.15.
        # 100*0.30 + 100*0.20 + 100*0.15 + 74.8*0.20 + 0*0.15 = 30 + 20 + 15 + 14.96 + 0 = 79.96
        checker._check_code_quality = MagicMock(
            return_value=PreflightCheck(
                name="code_quality", status="pass", score=100.0, weight=0.30
            )
        )
        checker._check_store_listing = MagicMock(
            return_value=PreflightCheck(
                name="store_listing", status="pass", score=100.0, weight=0.20
            )
        )
        checker._check_brand_brief = MagicMock(
            return_value=PreflightCheck(
                name="brand_brief", status="pass", score=100.0, weight=0.15
            )
        )
        checker._check_visual_assets = MagicMock(
            return_value=PreflightCheck(
                name="visual_assets", status="pass", score=74.8, weight=0.20
            )
        )
        checker._check_build_artifacts = MagicMock(
            return_value=PreflightCheck(
                name="build_artifacts", status="pass", score=0.0, weight=0.15
            )
        )

        report = checker.preflight("TestProject")

        assert report.readiness_score == 80.0
        assert report.ready is True
        assert report.blocking == []

    def test_ready_false_when_rounded_readiness_score_below_threshold(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        """[kiy.5 acceptance 2] composite 79.94 -> 79.9 / False (pin the other side)."""
        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        # 100*0.30 + 100*0.20 + 100*0.15 + 74.7*0.20 + 0*0.15 = 30 + 20 + 15 + 14.94 + 0 = 79.94
        checker._check_code_quality = MagicMock(
            return_value=PreflightCheck(
                name="code_quality", status="pass", score=100.0, weight=0.30
            )
        )
        checker._check_store_listing = MagicMock(
            return_value=PreflightCheck(
                name="store_listing", status="pass", score=100.0, weight=0.20
            )
        )
        checker._check_brand_brief = MagicMock(
            return_value=PreflightCheck(
                name="brand_brief", status="pass", score=100.0, weight=0.15
            )
        )
        checker._check_visual_assets = MagicMock(
            return_value=PreflightCheck(
                name="visual_assets", status="pass", score=74.7, weight=0.20
            )
        )
        checker._check_build_artifacts = MagicMock(
            return_value=PreflightCheck(
                name="build_artifacts", status="pass", score=0.0, weight=0.15
            )
        )

        report = checker.preflight("TestProject")

        assert report.readiness_score == 79.9
        assert report.ready is False
        assert report.blocking == []

    def test_blocking_still_vetoes_ready_at_100(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        """[kiy.5 acceptance 3] blocking still vetoes ready at 100.0 (pin)."""
        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        # All checks score 100.0, but code_quality has failed ship gate (status="fail")
        checker._check_code_quality = MagicMock(
            return_value=PreflightCheck(
                name="code_quality", status="fail", score=100.0, weight=0.30
            )
        )
        checker._check_store_listing = MagicMock(
            return_value=PreflightCheck(
                name="store_listing", status="pass", score=100.0, weight=0.20
            )
        )
        checker._check_brand_brief = MagicMock(
            return_value=PreflightCheck(
                name="brand_brief", status="pass", score=100.0, weight=0.15
            )
        )
        checker._check_visual_assets = MagicMock(
            return_value=PreflightCheck(
                name="visual_assets", status="pass", score=100.0, weight=0.20
            )
        )
        checker._check_build_artifacts = MagicMock(
            return_value=PreflightCheck(
                name="build_artifacts", status="pass", score=100.0, weight=0.15
            )
        )

        report = checker.preflight("TestProject")

        assert report.readiness_score == 100.0
        assert report.ready is False
        assert report.blocking == ["code_quality"]

    def test_missing_store_listing(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        listing_check = next(c for c in report.checks if c.name == "store_listing")
        assert listing_check.status == "fail"
        assert listing_check.score == 0.0

    def test_store_listing_with_placeholders(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        _write_store_listing(docs_dir, placeholders=True)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        listing_check = next(c for c in report.checks if c.name == "store_listing")
        assert listing_check.status == "warn"
        # Should have score < 100 due to placeholders
        assert listing_check.score < 100.0

    def test_missing_brand_brief(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        brief_check = next(c for c in report.checks if c.name == "brand_brief")
        assert brief_check.status == "fail"
        assert brief_check.score == 0.0

    def test_brand_brief_minimal(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        _write_brand_brief(docs_dir, minimal=True)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        brief_check = next(c for c in report.checks if c.name == "brand_brief")
        # Required fields present but recommended missing
        assert brief_check.status == "pass"
        assert brief_check.score == 75.0  # 75% from required, 0% from recommended

    def test_brand_brief_complete(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        _write_brand_brief(docs_dir, minimal=False)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        brief_check = next(c for c in report.checks if c.name == "brand_brief")
        assert brief_check.status == "pass"
        assert brief_check.score == 100.0

    def test_no_build_artifacts(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        assert build_check.status == "fail"
        assert build_check.score == 0.0

    def test_partial_build_artifacts(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        apk_file = tmp_path / "test.apk"
        apk_file.write_bytes(b"APK")
        mock_forge.find_apk.return_value = str(apk_file)
        # iOS still missing

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        assert build_check.status == "warn"
        assert build_check.score == 50.0

    def test_multi_target_build_artifacts(
        self,
        scorer: QualityScorer,
        linc_workspace: Path,
        project_path: Path,
    ) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="TestProject",
            path=str(project_path),
            framework="kmp",
            modules=[
                ModuleConfig(name="app", device_types=["phone"]),
                ModuleConfig(name="watchApp", device_types=["watch"]),
            ],
        )
        app_apk = project_path / "app.apk"
        app_apk.write_bytes(b"APK")
        forge.find_apk.return_value = str(app_apk)
        forge.find_ios_app.return_value = None
        forge.find_module_apks.return_value = {"app": str(app_apk)}

        checker = PreflightChecker(
            forge_reader=forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        # 4 total: apk, ios, module:app, module:watchApp -> 2 found
        assert build_check.score == 50.0

    def test_nonexistent_apk_not_found(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """With find_apk returning a missing path under tmp_path, android_apk has found=False."""
        missing_apk = tmp_path / "never_created.apk"
        mock_forge.find_apk.return_value = str(missing_apk)
        mock_forge.find_ios_app.return_value = None

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        apk_item = next(i for i in build_check.items if i.name == "android_apk")
        assert apk_item.found is False
        assert apk_item.detail == str(missing_apk)

    def test_single_target_missing_apk_and_no_ios_fails(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """Single-target project with missing apk and no ios yields fail and score 0.0."""
        missing_apk = tmp_path / "missing.apk"
        mock_forge.find_apk.return_value = str(missing_apk)
        mock_forge.find_ios_app.return_value = None

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        assert build_check.status == "fail"
        assert build_check.score == 0.0

    def test_nonexistent_module_apk_not_found(
        self,
        scorer: QualityScorer,
        linc_workspace: Path,
        project_path: Path,
        tmp_path: Path,
    ) -> None:
        """Multi-target project whose find_module_apks returns missing path yields found=False."""
        missing_watch_apk = tmp_path / "missing_watch.apk"
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="TestProject",
            path=str(project_path),
            framework="kmp",
            modules=[
                ModuleConfig(name="app", device_types=["phone"]),
                ModuleConfig(name="watchApp", device_types=["watch"]),
            ],
        )
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = None
        forge.find_module_apks.return_value = {"watchApp": str(missing_watch_apk)}

        checker = PreflightChecker(
            forge_reader=forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        watch_item = next(i for i in build_check.items if i.name == "module:watchApp")
        assert watch_item.found is False
        assert watch_item.detail == str(missing_watch_apk)

    def test_real_files_pass_and_missing_ios_not_found(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """Real APK file and iOS .app dir count as found (pass/100.0); missing .app does not."""
        real_apk = tmp_path / "real.apk"
        real_apk.write_bytes(b"APK")
        real_app = tmp_path / "real.app"
        real_app.mkdir()

        # Both present and real -> pass, 100.0
        mock_forge.find_apk.return_value = str(real_apk)
        mock_forge.find_ios_app.return_value = str(real_app)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")
        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        assert build_check.status == "pass"
        assert build_check.score == 100.0

        # Now test with missing .app path
        missing_app = tmp_path / "missing.app"
        mock_forge.find_ios_app.return_value = str(missing_app)
        report2 = checker.preflight("TestProject")
        build_check2 = next(c for c in report2.checks if c.name == "build_artifacts")
        ios_item = next(i for i in build_check2.items if i.name == "ios_app")
        assert ios_item.found is False
        assert ios_item.detail == str(missing_app)

    def test_file_shaped_ios_app_not_found(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """A regular file named .app (not a directory bundle) must not count as found."""
        fake_app = tmp_path / "fake.app"
        fake_app.write_text("not a directory")
        mock_forge.find_apk.return_value = None
        mock_forge.find_ios_app.return_value = str(fake_app)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        ios_item = next(i for i in build_check.items if i.name == "ios_app")
        assert ios_item.found is False
        assert ios_item.detail == str(fake_app)

    @patch("subprocess.run")
    def test_preflight_artifact_check_detail_mentions_staleness_flag(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        apk_file = tmp_path / "test.apk"
        apk_file.write_bytes(b"APK")
        mock_forge.find_apk.return_value = str(apk_file)

        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="2026-10-01T12:00:00+00:00\n",
            stderr="",
        )
        older_mtime = 1790850000.0
        os.utime(apk_file, (older_mtime, older_mtime))

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        assert build_check.detail == "1/2 build artifacts found (artifact_older_than_head=True)"

    @patch("subprocess.run")
    def test_preflight_artifact_check_detail_fresh_artifact(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        apk_file = tmp_path / "test.apk"
        apk_file.write_bytes(b"APK")
        mock_forge.find_apk.return_value = str(apk_file)
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="2026-10-01T12:00:00+00:00\n", stderr=""
        )
        head_ts = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC).timestamp()
        os.utime(apk_file, (head_ts, head_ts))

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        assert build_check.detail == "1/2 build artifacts found (artifact_older_than_head=False)"

    def test_preflight_artifact_check_detail_no_artifacts_has_no_flag(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        build_check = next(c for c in report.checks if c.name == "build_artifacts")
        assert build_check.detail == "No build artifacts found — run a build first"

    def test_visual_assets_none(
        self,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        # Project path exists but has no icon dirs
        forge = MagicMock(spec=ForgeProjectReader)
        bare_path = linc_workspace / "bare_project"
        bare_path.mkdir()
        forge.get.return_value = ForgeProject(
            name="TestProject",
            path=str(bare_path),
        )
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = None
        forge.find_module_apks.return_value = {}

        checker = PreflightChecker(
            forge_reader=forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        assets_check = next(c for c in report.checks if c.name == "visual_assets")
        assert assets_check.status == "fail"
        assert assets_check.score == 0.0

    def test_store_listing_read_error(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        from unittest.mock import patch

        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = OSError("Access denied")
            docs_dir = linc_workspace / "projects" / "active" / "testproject"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / "store-listing.md").touch()

            checker = PreflightChecker(
                forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
            )
            report = checker.preflight("TestProject")

            listing_check = next(c for c in report.checks if c.name == "store_listing")
            assert listing_check.status == "fail"
            assert "Could not read" in listing_check.detail

    def test_store_listing_missing_one_required(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        docs_dir.mkdir(parents=True, exist_ok=True)
        content = "short description\nfull description\nkeywords\n"  # missing privacy policy
        (docs_dir / "store-listing.md").write_text(content)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        listing_check = next(c for c in report.checks if c.name == "store_listing")
        assert listing_check.status == "warn"

    def test_store_listing_missing_multiple_required(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        docs_dir.mkdir(parents=True, exist_ok=True)
        content = "short description\n"  # missing 3
        (docs_dir / "store-listing.md").write_text(content)

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        listing_check = next(c for c in report.checks if c.name == "store_listing")
        assert listing_check.status == "fail"

    def test_brand_brief_json_error(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "brand-brief.json").write_text("invalid json")

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        brief_check = next(c for c in report.checks if c.name == "brand_brief")
        assert brief_check.status == "fail"
        assert "Could not parse" in brief_check.detail

    def test_brand_brief_not_dict(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "brand-brief.json").write_text('["a list", "not a dict"]')

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        brief_check = next(c for c in report.checks if c.name == "brand_brief")
        assert brief_check.status == "fail"
        assert "not a JSON object" in brief_check.detail

    def test_brand_brief_missing_multiple_required(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        docs_dir = linc_workspace / "projects" / "active" / "testproject"
        docs_dir.mkdir(parents=True, exist_ok=True)
        brief = {
            "app_name": "TestProject",
            "tagline": "Test all the things",
            "emotion": "precision",
            "primary_color": "#1E40AF",
            "accent_color": "#60A5FA",
            "background_color": "#F8FAFF",
        }
        (docs_dir / "brand-brief.json").write_text(json.dumps(brief))

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        brief_check = next(c for c in report.checks if c.name == "brand_brief")
        assert brief_check.status == "fail"

    def test_visual_assets_path_not_dir(
        self,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(name="TestProject", path="/does/not/exist")

        checker = PreflightChecker(
            forge_reader=forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("TestProject")

        assets_check = next(c for c in report.checks if c.name == "visual_assets")
        assert assets_check.status == "skip"

    def test_visual_assets_alternate_paths(
        self,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        proj = tmp_path / "AltProject"
        proj.mkdir()
        (proj / "app" / "src" / "main" / "res" / "mipmap-hdpi").mkdir(parents=True)
        (proj / "iosApp" / "Assets.xcassets").mkdir(parents=True)

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(name="AltProject", path=str(proj))

        checker = PreflightChecker(
            forge_reader=forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("AltProject")

        assets_check = next(c for c in report.checks if c.name == "visual_assets")
        assert assets_check.status == "fail"

    def test_visual_assets_warn_threshold(
        self,
        scorer: QualityScorer,
        linc_workspace: Path,
        tmp_path: Path,
    ) -> None:
        proj = tmp_path / "WarnProject"
        proj.mkdir()
        (proj / "app" / "src" / "main" / "res" / "mipmap-hdpi").mkdir(parents=True)
        (proj / "iosApp" / "Assets.xcassets").mkdir(parents=True)

        docs_dir = linc_workspace / "projects" / "active" / "warnproject"
        assets_dir = docs_dir / "assets"
        assets_dir.mkdir(parents=True)

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(name="WarnProject", path=str(proj))

        checker = PreflightChecker(
            forge_reader=forge, scorer=scorer, linc_workspace=linc_workspace
        )
        report = checker.preflight("WarnProject")

        assets_check = next(c for c in report.checks if c.name == "visual_assets")
        assert assets_check.status == "warn"

    def test_resolve_docs_dir_custom_mapping(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        linc_workspace: Path,
    ) -> None:
        from linc_crucible.core.preflight import _probe_docs_dir

        assert "slug_map" not in _probe_docs_dir.__code__.co_varnames

        active_dir = linc_workspace / "projects" / "active"

        # 1. widgetapp -> widget, widget-lab -> widgetlab via normalized lookup
        for folder in ("widget", "widgetlab"):
            target_dir = active_dir / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            _write_store_listing(target_dir)
            _write_brand_brief(target_dir)

        mock_forge.get.side_effect = lambda name: (
            ForgeProject(
                name=name,
                path="/tmp/dummy",
            )
            if name in ("widgetapp", "widget-lab", "SomeAbsentProject")
            else None
        )

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=linc_workspace
        )

        for proj_name in ("widgetapp", "widget-lab"):
            report_ok = checker.preflight(proj_name)
            listing_check = next(c for c in report_ok.checks if c.name == "store_listing")
            assert listing_check.status == "pass"
            assert listing_check.score == 100.0

            brief_check = next(c for c in report_ok.checks if c.name == "brand_brief")
            assert brief_check.status == "pass"
            assert brief_check.score == 100.0

        # Test genuinely absent project still reports files missing
        report_absent = checker.preflight("SomeAbsentProject")
        listing_check_absent = next(c for c in report_absent.checks if c.name == "store_listing")
        assert listing_check_absent.status == "fail"
        assert listing_check_absent.score == 0.0

        brief_check_absent = next(c for c in report_absent.checks if c.name == "brand_brief")
        assert brief_check_absent.status == "fail"
        assert brief_check_absent.score == 0.0

    def test_preflight_finds_project_local_store_dir_first(
        self,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        tmp_path: Path,
    ) -> None:
        """[test] preflight finds store-listing.md / brand-brief.json in <project>/store/ first."""
        proj_dir = tmp_path / "myapp"
        local_store = proj_dir / "store"
        local_store.mkdir(parents=True)
        _write_store_listing(local_store)
        _write_brand_brief(local_store)

        absent_ws = tmp_path / "no-such-workspace"
        mock_forge.get.return_value = ForgeProject(name="myapp", path=str(proj_dir))

        checker = PreflightChecker(
            forge_reader=mock_forge, scorer=scorer, linc_workspace=absent_ws
        )
        report = checker.preflight("myapp")

        listing_check = next(c for c in report.checks if c.name == "store_listing")
        assert listing_check.status == "pass"
        assert listing_check.score == 100.0

        brief_check = next(c for c in report.checks if c.name == "brand_brief")
        assert brief_check.status == "pass"
        assert brief_check.score == 100.0

        # When both <project>/store/ and <workspace>/projects/active/myapp exist,
        # <project>/store/ takes precedence.
        ws_dir = absent_ws / "projects" / "active" / "myapp"
        ws_dir.mkdir(parents=True)
        (ws_dir / "store-listing.md").write_text("# incomplete\n")
        (ws_dir / "brand-brief.json").write_text("{}\n")

        report_precedence = checker.preflight("myapp")
        listing_prec = next(c for c in report_precedence.checks if c.name == "store_listing")
        brief_prec = next(c for c in report_precedence.checks if c.name == "brand_brief")
        assert listing_prec.status == "pass"
        assert brief_prec.status == "pass"


class TestPreflightReport:
    def test_to_dict(self) -> None:
        report = PreflightReport(
            project="Test",
            timestamp="2026-02-27T00:00:00Z",
            readiness_score=85.5,
            ready=True,
            summary="85.5% ready — 5 pass, 0 warn, 0 fail",
        )
        d = report.to_dict()
        assert d["project"] == "Test"
        assert d["readiness_score"] == 85.5
        assert d["ready"] is True
        assert "checks" in d
        assert d["blocking"] == []

    def test_to_dict_with_checks(self) -> None:
        from linc_crucible.core.preflight import CheckItem, PreflightCheck

        report = PreflightReport(
            project="Test",
            timestamp="2026-02-27T00:00:00Z",
            checks=[
                PreflightCheck(
                    name="code_quality",
                    status="pass",
                    score=100.0,
                    weight=0.30,
                    detail="Quality score 100.0",
                    items=[
                        CheckItem(name="tests", found=True, detail="score=100"),
                    ],
                ),
            ],
        )
        d = report.to_dict()
        assert len(d["checks"]) == 1
        assert d["checks"][0]["name"] == "code_quality"
        assert d["checks"][0]["items"][0]["name"] == "tests"
        assert d["checks"][0]["items"][0]["found"] is True


class TestPreflightWeights:
    def test_weights_sum_to_one(self) -> None:
        total = sum(PREFLIGHT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_readiness_threshold(self) -> None:
        assert READINESS_THRESHOLD == 80

    def test_ship_gate_threshold_unchanged(self) -> None:
        """[#145 AC4] the fix must not retune either threshold."""
        assert SHIP_GATE_THRESHOLD == 85


class TestMCPToolRegistration:
    """Test the crucible_preflight tool via the MCP server module."""

    def test_preflight_no_project(self) -> None:
        from unittest.mock import patch

        with patch("linc_crucible.server_context._ensure_initialized"):
            from linc_crucible.assay_tools import crucible_preflight

            result = crucible_preflight(project="")
            assert "error" in result

    def test_preflight_success(self) -> None:
        from unittest.mock import patch

        with (
            patch("linc_crucible.server_context._ensure_initialized"),
            patch("linc_crucible.server_context._preflight_checker") as mock_checker,
        ):
            from linc_crucible.assay_tools import crucible_preflight

            mock_checker.preflight.return_value = PreflightReport(
                project="TestProject",
                timestamp="2026-02-27T00:00:00Z",
                readiness_score=92.0,
                ready=True,
                summary="92.0% ready",
            )

            result = crucible_preflight(project="TestProject")
            assert result["project"] == "TestProject"
            assert result["readiness_score"] == 92.0
            assert result["ready"] is True


class TestArtifactStalenessDetail:
    @patch("linc_crucible.core.preflight.artifacts_older_than_head")
    def test_no_found_items_is_empty(self, mock_older: MagicMock) -> None:
        items = [CheckItem(name="android_apk", found=False, detail="not found")]
        assert _artifact_staleness_detail(items, "/proj") == ""
        mock_older.assert_not_called()

    @patch("linc_crucible.core.preflight.artifacts_older_than_head")
    def test_only_found_paths_are_checked(self, mock_older: MagicMock) -> None:
        mock_older.return_value = True
        items = [
            CheckItem(name="android_apk", found=True, detail="/proj/a.apk"),
            CheckItem(name="ios_app", found=False, detail="/proj/b.app"),
            CheckItem(name="module:watch", found=True, detail="/proj/w.apk"),
        ]
        assert _artifact_staleness_detail(items, "/proj") == " (artifact_older_than_head=True)"
        mock_older.assert_called_once_with(["/proj/a.apk", "/proj/w.apk"], "/proj")

    @patch("linc_crucible.core.preflight.artifacts_older_than_head")
    def test_unknown_staleness_is_rendered(self, mock_older: MagicMock) -> None:
        mock_older.return_value = None
        items = [CheckItem(name="android_apk", found=True, detail="/proj/a.apk")]
        assert _artifact_staleness_detail(items, "/proj") == " (artifact_older_than_head=None)"
