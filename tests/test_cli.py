"""Tests for CLI commands.

`pour`/`assay`/`score` are wired through linc-cli-kit (linc-crucible#150). `pour` and
`assay` are now GENERATED commands (collision rule (a) — the hand-written commands were
deleted; `crucible_pour`/`crucible_assay` route through `ctx._deployer`/`ctx._assayer`
singletons, not a freshly-constructed class per call), so these tests mock the
`server_context` singleton attributes rather than patching `Deployer`/`Assayer` classes
directly. `score` is a hand-written override (rule (b)) kept for its `.score`/`.ship_gate`
JSON pin; its tests mock `ctx._scorer` the same way.

Default output is JSON; `--human` renders the Rich text this CLI has always rendered.
`fleet`/`baseline`/`fuelcheck`/`logo` are untouched hand-written commands and keep their
original class-patching style.
"""

from __future__ import annotations

import json
from unittest.mock import ANY, MagicMock, patch

import pytest
from click.testing import CliRunner

import linc_crucible.server_context as server_ctx
from linc_crucible import __version__
from linc_crucible.cli import CRUCIBLE_BANNER, cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_ctx(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Freeze server_context's lazy init and swap in mocks for every singleton it manages.

    `_ensure_initialized()` runs on every kit-mounted command via LocalTransport's `before`
    hook; forcing `_initialized = True` first makes it a no-op so the mocks below survive
    the whole CLI round trip untouched (`monkeypatch` restores everything after the test).
    """
    monkeypatch.setattr(server_ctx, "_initialized", True)
    mocks: dict[str, MagicMock] = {}
    for attr in (
        "_forge_reader",
        "_deployer",
        "_assayer",
        "_scorer",
        "_preflight_checker",
        "_baseline_manager",
        "_nerve_client",
        "_tree_capture",
        "_bvt_nerve_adapter",
    ):
        m = MagicMock()
        monkeypatch.setattr(server_ctx, attr, m)
        mocks[attr] = m
    return mocks


class TestCLIBasic:
    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "fleet" in result.stdout
        assert "pour" in result.stdout
        assert "assay" in result.stdout
        assert "score" in result.stdout
        assert "bellows" in result.stdout
        assert "fuelcheck" in result.stdout

    def test_logo(self, runner: CliRunner) -> None:
        assert f"v{__version__}" in CRUCIBLE_BANNER
        result = runner.invoke(cli, ["logo"])
        assert result.exit_code == 0
        assert "Crucible" in result.stdout
        assert f"v{__version__}" in result.stdout


class TestFleetCLI:
    def test_fleet_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["fleet", "--help"])
        assert result.exit_code == 0
        assert "discover" in result.stdout
        assert "list" in result.stdout
        assert "add" in result.stdout
        assert "remove" in result.stdout

    def test_fleet_discover(self, runner: CliRunner) -> None:
        with patch("linc_crucible.core.fleet_registry.FleetRegistry") as mock_cls:
            mock_registry = MagicMock()
            mock_registry.discover.return_value = []
            mock_registry.list_devices.return_value = []
            mock_cls.return_value = mock_registry

            result = runner.invoke(cli, ["fleet", "discover"])

        assert result.exit_code == 0

    @patch("linc_crucible.core.fleet_registry.FleetRegistry")
    def test_fleet_list_empty(self, mock_cls: MagicMock, runner: CliRunner) -> None:
        mock_registry = MagicMock()
        mock_registry.list_devices.return_value = []
        mock_cls.return_value = mock_registry

        result = runner.invoke(cli, ["fleet", "list"])
        assert result.exit_code == 0

    def test_fleet_add(self, runner: CliRunner) -> None:
        with patch("linc_crucible.core.fleet_registry.FleetRegistry") as mock_cls:
            mock_registry = MagicMock()
            mock_cls.return_value = mock_registry

            result = runner.invoke(cli, ["fleet", "add", "ABC123", "--nickname", "pixel"])

        assert result.exit_code == 0

    def test_fleet_remove_not_found(self, runner: CliRunner) -> None:
        with patch("linc_crucible.core.fleet_registry.FleetRegistry") as mock_cls:
            mock_registry = MagicMock()
            mock_registry.remove.return_value = False
            mock_cls.return_value = mock_registry

            result = runner.invoke(cli, ["fleet", "remove", "NONEXISTENT"])

        assert result.exit_code == 1


class TestPourCLI:
    def test_pour_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["pour", "--help"])
        assert result.exit_code == 0
        assert "--module" in result.stdout
        assert "--device-type" in result.stdout

    def test_pour_json_default(self, mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
        from linc_crucible.core.deployer import DeployResult, PourResult

        mock_ctx["_deployer"].pour.return_value = PourResult(
            project="TestProject",
            results=[DeployResult(device_serial="D1", device_model="Pixel", install_success=True)],
        )

        result = runner.invoke(cli, ["pour", "--project", "TestProject"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["project"] == "TestProject"
        assert payload["success"] is True
        assert payload["devices"][0]["device_serial"] == "D1"

    def test_pour_human_renders_rich_text(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.deployer import DeployResult, PourResult

        mock_ctx["_deployer"].pour.return_value = PourResult(
            project="TestProject",
            results=[
                DeployResult(
                    device_serial="WATCH1",
                    device_model="Watch",
                    install_success=True,
                    module="watchApp",
                )
            ],
        )

        result = runner.invoke(cli, ["--human", "pour", "--project", "TestProject"])
        assert result.exit_code == 0
        assert "watchApp" in result.stdout
        assert "Installed" in result.stdout
        assert "Pour complete" in result.stdout

    def test_pour_no_project_fails(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.deployer import PourResult

        mock_ctx["_deployer"].pour.return_value = PourResult(
            project="NonExistent", results=[], build_output="Project not found in registry"
        )

        result = runner.invoke(cli, ["pour", "--project", "NonExistent"])
        assert result.exit_code == 1
        assert result.stdout == ""
        assert "error" in json.loads(result.stderr)

    def test_pour_with_module_flag(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.deployer import DeployResult, PourResult

        mock_ctx["_deployer"].pour.return_value = PourResult(
            project="TestProject",
            results=[
                DeployResult(
                    device_serial="WATCH1",
                    device_model="Watch",
                    install_success=True,
                    module="watchApp",
                ),
            ],
        )

        result = runner.invoke(
            cli, ["pour", "--project", "TestProject", "--module", "watchApp", "--skip-build"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["devices"][0]["module"] == "watchApp"
        mock_ctx["_deployer"].pour.assert_called_once_with(
            project_name="TestProject",
            device="",
            platform="",
            launch=False,
            skip_build=True,
            module="watchApp",
            device_type="",
            launch_intent="",
            build_type="debug",
            policy=ANY,
            allow_personal=False,
        )

    def test_pour_with_device_type_flag(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.deployer import DeployResult, PourResult

        mock_ctx["_deployer"].pour.return_value = PourResult(
            project="Simple",
            results=[
                DeployResult(device_serial="PHONE1", device_model="Phone", install_success=True)
            ],
        )

        result = runner.invoke(
            cli,
            [
                "pour",
                "--project",
                "Simple",
                "--device-type",
                "phone",
                "--skip-build",
                "--allow-personal",
            ],
        )
        assert result.exit_code == 0
        mock_ctx["_deployer"].pour.assert_called_once_with(
            project_name="Simple",
            device="",
            platform="",
            launch=False,
            skip_build=True,
            module="",
            device_type="phone",
            launch_intent="",
            build_type="debug",
            policy=ANY,
            allow_personal=True,
        )


class TestAssayCLI:
    def test_assay_unmeasured_exits_1_with_unmeasured_envelope(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.assayer import AssayReport, LintResult, RunResult

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="UnmeasuredProj",
            mode="quick",
            lint=LintResult(issues=0, success=False, configured=False),
            tests=RunResult(total=0, passed=0, success=False, configured=False),
        )

        result = runner.invoke(cli, ["assay", "--project", "UnmeasuredProj"])
        assert result.exit_code == 1
        assert result.stdout == ""
        envelope = json.loads(result.stderr)
        assert envelope["error"]["code"] == "UNMEASURED"
        assert envelope["error"]["category"] == "internal"
        assert (
            envelope["error"]["message"] == "Assay for project 'UnmeasuredProj' measured nothing"
        )
        details = envelope["error"]["details"]
        assert details["project"] == "UnmeasuredProj"
        assert details["measured"] is False
        assert details["status"] == "unmeasured"
        assert details["tests"]["pass_rate"] is None

    def test_assay_partial_and_measured_runs_keep_exit_0(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.assayer import AssayReport, LintResult, RunResult
        from linc_crucible.core.scorer import QualityScore

        # Partial run: lint measured, tests attempted & crashed (pass_rate is None)
        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="PartialProj",
            mode="quick",
            lint=LintResult(issues=0, success=True, configured=True, attempted=True),
            tests=RunResult(total=0, passed=0, success=False, configured=True, attempted=True),
            quality_score=QualityScore(
                project="PartialProj",
                timestamp="T",
                mode="quick",
                composite=100.0,
                ship_gate=False,
                attempted_unmeasured=["tests"],
            ),
        )

        result = runner.invoke(cli, ["assay", "--project", "PartialProj"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["measured"] is True
        assert payload["status"] == "partial"
        assert payload["tests"]["pass_rate"] is None
        assert payload["quality_score"]["ship_gate"] is False

        # Human mode on partial run with pass_rate=None also exits 0 cleanly
        human_res = runner.invoke(cli, ["--human", "assay", "--project", "PartialProj"])
        assert human_res.exit_code == 0, human_res.output
        assert "Tests: 0/0 passed (unmeasured)" in human_res.stdout

        # Measured lint-only run: tests not configured, not attempted -> status "measured"
        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="LintOnlyProj",
            mode="quick",
            lint=LintResult(issues=0, success=True, configured=True, attempted=True),
            tests=RunResult(total=0, passed=0, success=False, configured=False, attempted=False),
        )

        measured_res = runner.invoke(cli, ["assay", "--project", "LintOnlyProj"])
        assert measured_res.exit_code == 0, measured_res.output
        measured_payload = json.loads(measured_res.stdout)
        assert measured_payload["measured"] is True
        assert measured_payload["status"] == "measured"
        assert measured_payload["tests"]["pass_rate"] is None


class TestScoreCLI:
    def test_score_no_scores_is_an_error(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        mock_ctx["_scorer"].load_latest.return_value = None

        result = runner.invoke(cli, ["score", "TestProject"])
        assert result.exit_code == 1
        assert result.stdout == ""
        envelope = json.loads(result.stderr)
        assert "No scores" in envelope["error"]["message"]

    def test_score_no_scores_human(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        mock_ctx["_scorer"].load_latest.return_value = None

        result = runner.invoke(cli, ["--human", "score", "TestProject"])
        assert result.exit_code == 1
        assert "No scores" in result.stderr

    def test_score_history_empty(self, mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
        mock_ctx["_scorer"].load_history.return_value = []

        result = runner.invoke(cli, ["score", "TestProject", "--history"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["count"] == 0
        # `--history` payloads have no `composite` field to alias — the `.score` pin must
        # not inject a literal `null` onto a shape that never had the key (crucible#151
        # repair round 0, json-contract-score-alias-fires-on-history-payloads).
        assert "score" not in payload

    def test_score_pins_score_and_ship_gate_keys(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.scorer import QualityScore

        mock_ctx["_scorer"].load_latest.return_value = QualityScore(
            project="P", timestamp="T", mode="full", composite=87.5, ship_gate=True
        )

        result = runner.invoke(cli, ["score", "P"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["score"] == 87.5
        assert payload["ship_gate"] is True
        # `.composite` — the scorer's own field name — is still present; `.score` is an
        # additive alias, not a rename, so nothing that already reads `.composite` breaks.
        assert payload["composite"] == 87.5


class TestPourCLIMultiTarget:
    def test_pour_help_shows_module_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["pour", "--help"])
        assert result.exit_code == 0
        assert "--module" in result.stdout
        assert "--device-type" in result.stdout


class TestAssayCLIOutput:
    """Characterization tests for assay --human output — covers the preserved renderer."""

    def test_assay_basic_output(self, mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
        from linc_crucible.core.assayer import AssayReport, LintResult, RunResult

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=2, success=False),
            tests=RunResult(total=10, passed=8, failed=2, success=False),
        )

        result = runner.invoke(cli, ["--human", "assay", "--project", "TestProject"])
        assert result.exit_code == 0
        assert "Lint: 2 issue(s)" in result.stdout
        assert "Tests: 8/10 passed" in result.stdout

    def test_assay_with_perf(self, mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
        from linc_crucible.core.assayer import (
            AssayReport,
            LintResult,
            PerformanceResult,
            RunResult,
        )

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=5, passed=5, success=True),
            performance=PerformanceResult(apk_size_mb=12.34, ipa_size_mb=45.67, success=True),
        )

        result = runner.invoke(cli, ["--human", "assay", "--project", "TestProject", "--perf"])
        assert result.exit_code == 0
        assert "APK: 12.34 MB" in result.stdout
        assert "iOS: 45.67 MB" in result.stdout
        assert "Perf:" in result.stdout

    def test_assay_with_a11y(self, mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
        from linc_crucible.core.assayer import (
            AccessibilityResult,
            AssayReport,
            LintResult,
            RunResult,
        )

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=5, passed=5, success=True),
            accessibility=AccessibilityResult(
                critical=1,
                major=3,
                minor=5,
                score=72.0,
                success=True,
                details=["Missing contentDescription on ImageView"],
            ),
        )

        result = runner.invoke(cli, ["--human", "assay", "--project", "TestProject", "--a11y"])
        assert result.exit_code == 0
        assert "A11y:" in result.stdout
        assert "72/100" in result.stdout
        assert "critical=1" in result.stdout
        assert "major=3" in result.stdout
        assert "minor=5" in result.stdout
        assert "Missing contentDescription" in result.stdout

    def test_assay_with_quality_score(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.assayer import AssayReport, LintResult, RunResult
        from linc_crucible.core.scorer import QualityScore, ScoreDimension

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=10, passed=10, success=True),
            quality_score=QualityScore(
                project="TestProject",
                timestamp="20260423T120000Z",
                mode="quick",
                composite=95.0,
                ship_gate=True,
                dimensions=[
                    ScoreDimension(
                        name="tests",
                        raw_value=1.0,
                        score=100.0,
                        weight=0.55,
                        weighted_score=55.0,
                    ),
                    ScoreDimension(
                        name="lint",
                        raw_value=0.0,
                        score=100.0,
                        weight=0.45,
                        weighted_score=45.0,
                    ),
                ],
            ),
        )

        result = runner.invoke(cli, ["--human", "assay", "--project", "TestProject"])
        assert result.exit_code == 0
        assert "Quality Score: 95.0/100" in result.stdout
        assert "PASS" in result.stdout
        assert "tests:" in result.stdout or "tests" in result.stdout
        assert "lint:" in result.stdout or "lint" in result.stdout

    def test_assay_with_perf_failure_no_sizes(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        """Covers the perf failure icon and the parts-empty output fallback branch
        (linc-crucible#151 CRAP ratchet — _print_assay_perf)."""
        from linc_crucible.core.assayer import (
            AssayReport,
            LintResult,
            PerformanceResult,
            RunResult,
        )

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=5, passed=5, success=True),
            performance=PerformanceResult(output="build tool not found", success=False),
        )

        result = runner.invoke(cli, ["--human", "assay", "--project", "TestProject", "--perf"])
        assert result.exit_code == 0
        # `output` isn't carried by AssayReport.to_dict() when no size was measured, so the
        # rendered line falls back to the empty-parts branch — assert the failure icon and
        # that no size string leaked through, not the (unreachable-via-JSON) output text.
        assert "x Perf:" in result.stdout
        assert "MB" not in result.stdout.split("Perf:")[1].split("\n")[0]

    def test_assay_with_a11y_failure_no_details(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        """Covers the a11y failure icon branch (linc-crucible#151 CRAP ratchet —
        _print_assay_a11y)."""
        from linc_crucible.core.assayer import (
            AccessibilityResult,
            AssayReport,
            LintResult,
            RunResult,
        )

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=5, passed=5, success=True),
            accessibility=AccessibilityResult(score=0.0, success=False, details=[]),
        )

        result = runner.invoke(cli, ["--human", "assay", "--project", "TestProject", "--a11y"])
        assert result.exit_code == 0
        assert "x A11y: 0/100" in result.stdout

    def test_assay_with_quality_score_fail_gate(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        """Covers the ship-gate FAIL icon branch (linc-crucible#151 CRAP ratchet —
        _print_assay_quality_score)."""
        from linc_crucible.core.assayer import AssayReport, LintResult, RunResult
        from linc_crucible.core.scorer import QualityScore

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=10, passed=4, success=False),
            quality_score=QualityScore(
                project="TestProject",
                timestamp="20260423T120000Z",
                mode="quick",
                composite=40.0,
                ship_gate=False,
                dimensions=[],
            ),
        )

        result = runner.invoke(cli, ["--human", "assay", "--project", "TestProject"])
        assert result.exit_code == 0
        assert "Quality Score: 40.0/100" in result.stdout
        assert "FAIL" in result.stdout

    def test_assay_json_default_carries_quality_score(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.assayer import AssayReport, LintResult, RunResult
        from linc_crucible.core.scorer import QualityScore

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=10, passed=10, success=True),
            quality_score=QualityScore(
                project="TestProject", timestamp="T", mode="quick", composite=95.0, ship_gate=True
            ),
        )

        result = runner.invoke(cli, ["assay", "--project", "TestProject"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["quality_score"]["composite"] == 95.0
        assert payload["quality_score"]["ship_gate"] is True


class TestBaselineListCLI:
    """Characterization tests for baseline list CLI — untouched hand-written group."""

    @patch("linc_crucible.core.baseline.BaselineManager")
    def test_baseline_list_no_entries(self, mock_mgr_cls: MagicMock, runner: CliRunner) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_baselines.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cli, ["baseline", "list", "TestProject"])
        assert result.exit_code == 0
        assert "No baselines found" in result.stdout

    @patch("linc_crucible.core.baseline.BaselineManager")
    def test_baseline_list_with_entries(self, mock_mgr_cls: MagicMock, runner: CliRunner) -> None:
        from linc_crucible.core.baseline import BaselineEntry

        mock_mgr = MagicMock()
        mock_mgr.list_baselines.return_value = [
            BaselineEntry(
                project="TestProject",
                screen_name="home",
                capture_time="20260423T120000Z",
                screenshot_path="/tmp/home.png",
                device_info="Pixel 7",
                app_version="1.0.0",
                width=1080,
                height=2400,
            ),
            BaselineEntry(
                project="TestProject",
                screen_name="settings",
                capture_time="20260423T120100Z",
                screenshot_path="/tmp/settings.png",
                device_info="Pixel 7",
                app_version="1.0.0",
                width=1080,
                height=2400,
            ),
        ]
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cli, ["baseline", "list", "TestProject"])
        assert result.exit_code == 0
        assert "home" in result.stdout
        assert "settings" in result.stdout
        assert "1080x2400" in result.stdout
        assert "Pixel 7" in result.stdout
        assert "1.0.0" in result.stdout

    @patch("linc_crucible.core.baseline.BaselineManager")
    def test_baseline_list_with_screen_filter(
        self, mock_mgr_cls: MagicMock, runner: CliRunner
    ) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_baselines.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cli, ["baseline", "list", "TestProject", "--screen", "home"])
        assert result.exit_code == 0
        mock_mgr.list_baselines.assert_called_once_with("TestProject", screen_name="home")


class TestFuelcheckCLI:
    def test_fuelcheck_runs(self, runner: CliRunner) -> None:
        """Smoke: probes are injected (see tests/test_fuelcheck.py), one line per probe."""
        from linc_crucible import fuelcheck as fc

        env = MagicMock(spec=fc.ProbeEnv)
        with (
            patch.object(fc, "default_probe_env", return_value=env),
            patch.object(fc, "run_probes", return_value=[fc.Probe("install", "ok", "v")]),
        ):
            result = runner.invoke(cli, ["fuelcheck"])

        assert result.exit_code == 0
        assert result.stdout == "ok    install: v\n"


class TestBaselineCLINameValidation:
    """R4 High 1 + Medium 6: the hand-written baseline group bypasses the MCP layer, so a
    hostile project/screen must still exit 2 with a validation envelope on stderr."""

    @staticmethod
    def _png(path):
        import struct

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 4, 4) + b"\x08\x02\x00\x00\x00"
        path.write_bytes(sig + ihdr + b"\x00" * 16)
        return path

    @staticmethod
    def _listing(root):
        import os

        return [(d, sorted(ds), sorted(fs)) for d, ds, fs in os.walk(root)]

    @staticmethod
    def _assert_validation_envelope(result) -> None:
        from linc_cli_kit.errors import parse_error_envelope

        assert result.exit_code == 2, result.output
        assert result.stdout == ""
        envelope = parse_error_envelope(result.stderr)
        assert envelope is not None
        assert envelope["error"]["category"] == "validation"
        assert envelope["error"]["code"] == "USAGE_ERROR"
        assert envelope["error"]["message"].startswith(
            ("invalid project:", "invalid screen_name:")
        )

    def test_prune_absolute_project_exits_2_and_deletes_nothing(
        self, runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CRUCIBLE_BASELINES_DIR", str(tmp_path / "store"))
        victim = tmp_path / "victim"
        for ts in ("20260101T000000Z", "20260102T000000Z"):
            (victim / "home" / ts).mkdir(parents=True)
            (victim / "home" / ts / "metadata.json").write_text("{}")
        before = self._listing(victim)

        result = runner.invoke(cli, ["baseline", "prune", str(victim), "--keep", "0"])

        self._assert_validation_envelope(result)
        assert self._listing(victim) == before

    def test_capture_dotdot_project_exits_2_and_writes_nothing(
        self, runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = tmp_path / "store"
        monkeypatch.setenv("CRUCIBLE_BASELINES_DIR", str(store))
        png = self._png(tmp_path / "shot.png")
        before = self._listing(tmp_path)

        result = runner.invoke(cli, ["baseline", "capture", "../x", "s", str(png)])

        self._assert_validation_envelope(result)
        assert self._listing(tmp_path) == before

    def test_list_hostile_screen_exits_2(
        self, runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CRUCIBLE_BASELINES_DIR", str(tmp_path / "store"))

        result = runner.invoke(cli, ["baseline", "list", "proj", "--screen", "../../etc"])

        self._assert_validation_envelope(result)

    def test_valid_capture_still_exits_0(
        self, runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = tmp_path / "store"
        monkeypatch.setenv("CRUCIBLE_BASELINES_DIR", str(store))
        png = self._png(tmp_path / "shot.png")

        result = runner.invoke(cli, ["baseline", "capture", "proj", "home", str(png)])

        assert result.exit_code == 0, result.output
        assert (store / "proj" / "home").is_dir()
