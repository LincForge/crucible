"""Additional CLI tests to improve CRAP scores.

See tests/test_cli.py's module docstring for why `pour`/`assay`/`score` are exercised via
`server_context` singleton mocks (`mock_ctx`) rather than class patches, and why generated
commands (`pour`, `assay`) take `--project` as an option, not a positional argument.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import linc_crucible.server_context as server_ctx
from linc_crucible.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_ctx(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
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


def test_fleet_list_with_devices(runner: CliRunner) -> None:
    from linc_crucible.core.device import DeviceType, FleetDevice, Platform

    with patch("linc_crucible.core.fleet_registry.FleetRegistry") as mock_cls:
        mock_registry = MagicMock()
        mock_registry.list_devices.return_value = [
            FleetDevice(
                serial="ON",
                platform=Platform.ANDROID,
                device_type=DeviceType.PHONE,
                state="online",
                nickname="myphone",
            ),
            FleetDevice(
                serial="OFF",
                platform=Platform.IOS,
                device_type=DeviceType.SIMULATOR,
                state="offline",
            ),
        ]
        mock_cls.return_value = mock_registry

        result = runner.invoke(cli, ["fleet", "list", "--platform", "android"])
        assert result.exit_code == 0
        assert "myphone" in result.stdout
        assert "online" in result.stdout
        assert "offline" in result.stdout


def test_assay_failing_components(mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
    from linc_crucible.core.assayer import (
        AccessibilityResult,
        AssayReport,
        LintResult,
        PerformanceResult,
        RunResult,
        SecurityResult,
    )
    from linc_crucible.core.scorer import QualityScore, ScoreDimension

    mock_ctx["_assayer"].assay.return_value = AssayReport(
        project="FailProj",
        mode="full",
        lint=LintResult(issues=5, success=False),
        tests=RunResult(total=10, passed=2, failed=8, success=False),
        security=SecurityResult(score=40.0, success=False),
        performance=PerformanceResult(apk_size_mb=99.9, success=False),
        accessibility=AccessibilityResult(
            critical=5, major=0, minor=0, score=20.0, success=False, details=["Bad"]
        ),
        quality_score=QualityScore(
            project="FailProj",
            timestamp="Now",
            mode="full",
            composite=30.0,
            ship_gate=False,
            dimensions=[
                ScoreDimension(
                    name="test", raw_value=0.0, score=0.0, weight=1.0, weighted_score=0.0
                )
            ],
        ),
    )

    result = runner.invoke(cli, ["--human", "assay", "--project", "FailProj"])
    assert result.exit_code == 0
    assert "Security: 40/100" in result.stdout
    assert "Perf: APK: 99.90 MB" in result.stdout
    assert "A11y: 20/100" in result.stdout
    assert "FAIL" in result.stdout


def test_score_latest_present(mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
    from linc_crucible.core.scorer import QualityScore, ScoreDimension

    mock_ctx["_scorer"].load_latest.return_value = QualityScore(
        project="P",
        timestamp="T",
        mode="full",
        composite=99.0,
        ship_gate=True,
        dimensions=[
            ScoreDimension(name="d", raw_value=1.0, score=100.0, weight=1.0, weighted_score=100.0)
        ],
    )

    result = runner.invoke(cli, ["--human", "score", "P"])
    assert result.exit_code == 0
    assert "99.0/100" in result.stdout


def test_score_history_present(mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
    from linc_crucible.core.scorer import QualityScore

    mock_ctx["_scorer"].load_history.return_value = [
        QualityScore(
            project="P",
            timestamp="T1",
            mode="quick",
            composite=50.0,
            ship_gate=False,
            dimensions=[],
        ),
        QualityScore(
            project="P",
            timestamp="T2",
            mode="full",
            composite=90.0,
            ship_gate=True,
            dimensions=[],
        ),
    ]

    result = runner.invoke(cli, ["--human", "score", "P", "--history"])
    assert result.exit_code == 0
    assert "50.0" in result.stdout
    assert "90.0" in result.stdout


def test_pour_no_results(mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
    """No devices deployed -> the kit's output contract fails the call: empty stdout, an
    envelope with the deployer's own error message. There is no more free-standing "No
    deployment results" string — that was the hand-written command's own text; the
    generated command reports whatever `PourResult.error` says.
    """
    from linc_crucible.core.deployer import PourResult

    mock_ctx["_deployer"].pour.return_value = PourResult(
        project="P", results=[], build_output="No artifact produced"
    )

    result = runner.invoke(cli, ["--human", "pour", "--project", "P"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "No artifact produced" in result.stderr


def test_pour_failed_results(mock_ctx: dict[str, MagicMock], runner: CliRunner) -> None:
    from linc_crucible.core.deployer import DeployResult, PourResult

    mock_ctx["_deployer"].pour.return_value = PourResult(
        project="P",
        build_output="Build error",
        results=[
            DeployResult(
                device_serial="D1",
                device_model="M1",
                error="OOM",
                artifact_path="/tmp/a.apk",
                install_success=False,
                launch_success=False,
            )
        ],
    )

    result = runner.invoke(cli, ["--human", "pour", "--project", "P"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "OOM" in result.stderr


def test_pour_human_no_devices_renders_failed(
    mock_ctx: dict[str, MagicMock], runner: CliRunner
) -> None:
    """Empty `results` AND empty `build_output` -> `PourResult.error` is falsy (nothing to
    report), so `to_dict()` carries no top-level `error` key and the kit's `classify()`
    treats this as success (exit 0) — `_render_pour` actually runs. Pins the "no deployment
    results" and "Pour failed" lines the previous rewrite (test_pour_no_results, above)
    stopped exercising once it switched to an error-bearing payload (crucible#151 repair
    round 0, coverage-regression-in-the-one-renderer-this-pr-promises-to-preserve).
    """
    from linc_crucible.core.deployer import PourResult

    mock_ctx["_deployer"].pour.return_value = PourResult(project="P", results=[], build_output="")

    result = runner.invoke(cli, ["--human", "pour", "--project", "P"])
    assert result.exit_code == 0
    assert "No deployment results." in result.stdout
    assert "Pour failed." in result.stdout


def test_pour_human_partial_success_renders_per_device_lines(
    mock_ctx: dict[str, MagicMock], runner: CliRunner
) -> None:
    """One device errors (with an artifact path and a launch) while ANOTHER installs
    successfully -> `PourResult.success` is True (any() over devices), so `to_dict()`
    omits the top-level `error` key even though a device-level error string is present.
    The renderer actually runs and must print the artifact path, "Launched", and the
    per-device error line for the failing device (crucible#151 repair round 0,
    coverage-regression-in-the-one-renderer-this-pr-promises-to-preserve).
    """
    from linc_crucible.core.deployer import DeployResult, PourResult

    mock_ctx["_deployer"].pour.return_value = PourResult(
        project="P",
        results=[
            DeployResult(
                device_serial="D1",
                device_model="M1",
                error="OOM",
                artifact_path="/tmp/a.apk",
                install_success=False,
                launch_success=True,
            ),
            DeployResult(
                device_serial="D2",
                device_model="M2",
                install_success=True,
            ),
        ],
    )

    result = runner.invoke(cli, ["--human", "pour", "--project", "P"])
    assert result.exit_code == 0
    assert "Artifact: /tmp/a.apk" in result.stdout
    assert "Launched" in result.stdout
    assert "OOM" in result.stdout
    assert "Pour complete." in result.stdout
