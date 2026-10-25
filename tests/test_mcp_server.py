"""Tests for MCP server tools."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import structlog

from linc_crucible.core.assayer import AssayReport, LintResult, RunResult
from linc_crucible.core.deployer import DeployResult, PourResult
from linc_crucible.core.device_policy import DevicePolicy
from linc_crucible.core.scorer import QualityScore, ScoreDimension


@pytest.fixture(autouse=True)
def _capture_structlog() -> Iterator[None]:
    """Record structlog events instead of printing them to an import-time-bound stream.

    Importing ``linc_crucible.cli`` points structlog at the then-current ``sys.stderr``,
    a pytest capture stream. mutmut re-runs ``pytest.main()`` in one process; the first
    session closes that stream, so any later log call raised "I/O operation on closed
    file". Mirrors ``captured_logs`` in ``tests/test_crucible_bvt_mcp.py``.
    """
    with structlog.testing.capture_logs():
        yield


_PERSONAL_SERIAL = "PERSONALWATCH01"
_BENCH_POLICY = DevicePolicy(personal_serials=frozenset(), loaded=True, source="test-fleet.yaml")
_PERSONAL_POLICY = DevicePolicy(
    personal_serials=frozenset({_PERSONAL_SERIAL}), loaded=True, source="test-fleet.yaml"
)


@pytest.fixture(autouse=True)
def _bench_device_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded policy with no personal devices, so these tests never depend on
    the host's real device-fleet.yaml (absent in CI -> the policy fails closed)."""
    import linc_crucible.server_context as ctx

    monkeypatch.setattr(ctx, "_cached_device_policy", _BENCH_POLICY)


def _use_policy(monkeypatch: pytest.MonkeyPatch, policy: DevicePolicy) -> None:
    import linc_crucible.server_context as ctx

    monkeypatch.setattr(ctx, "_cached_device_policy", policy)


class TestCruciblePour:
    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_pour_success(self, mock_deployer: MagicMock, mock_init: MagicMock) -> None:
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(
            project="TestProject",
            results=[
                DeployResult(
                    device_serial="ABC",
                    install_success=True,
                    launch_success=True,
                ),
            ],
        )

        result = await crucible_pour(project="TestProject", launch=True)
        assert result["success"] is True
        assert len(result["devices"]) == 1

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_pour_no_project(self, mock_init: MagicMock) -> None:
        from linc_crucible.deploy_tools import crucible_pour

        result = await crucible_pour(project="")
        assert "error" in result


class TestCrucibleAssay:
    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._assayer")
    async def test_assay_success(self, mock_assayer: MagicMock, mock_init: MagicMock) -> None:
        from linc_crucible.assay_tools import crucible_assay

        mock_assayer.assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=10, passed=10, success=True),
            quality_score=QualityScore(
                project="TestProject",
                timestamp="2026-02-20T00:00:00Z",
                mode="quick",
                composite=100.0,
                ship_gate=True,
            ),
        )

        result = await crucible_assay(project="TestProject")
        assert result["project"] == "TestProject"
        assert result["tests"]["pass_rate"] == 1.0

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_assay_no_project(self, mock_init: MagicMock) -> None:
        from linc_crucible.assay_tools import crucible_assay

        result = await crucible_assay(project="")
        assert "error" in result

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_assay_invalid_mode(self, mock_init: MagicMock) -> None:
        from linc_crucible.assay_tools import crucible_assay

        result = await crucible_assay(project="Test", mode="invalid")
        assert "error" in result


class TestCrucibleScore:
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._scorer")
    def test_score_latest(self, mock_scorer: MagicMock, mock_init: MagicMock) -> None:
        from linc_crucible.assay_tools import crucible_score

        mock_scorer.load_latest.return_value = QualityScore(
            project="Test",
            timestamp="2026-02-20T00:00:00Z",
            mode="quick",
            composite=95.0,
            ship_gate=True,
            dimensions=[
                ScoreDimension(
                    name="tests", raw_value=1.0, score=100.0, weight=0.55, weighted_score=55.0
                ),
                ScoreDimension(
                    name="lint", raw_value=2.0, score=96.0, weight=0.45, weighted_score=43.2
                ),
            ],
        )

        result = crucible_score(project="Test")
        assert result["composite"] == 95.0
        assert result["ship_gate"] is True

    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._scorer")
    def test_score_not_found(self, mock_scorer: MagicMock, mock_init: MagicMock) -> None:
        from linc_crucible.assay_tools import crucible_score

        mock_scorer.load_latest.return_value = None

        result = crucible_score(project="Test")
        assert "error" in result

    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._scorer")
    def test_score_history(self, mock_scorer: MagicMock, mock_init: MagicMock) -> None:
        from linc_crucible.assay_tools import crucible_score

        mock_scorer.load_history.return_value = [
            QualityScore(
                project="Test",
                timestamp="2026-02-20",
                mode="quick",
                composite=90.0,
                ship_gate=True,
            ),
        ]

        result = crucible_score(project="Test", history=True)
        assert result["count"] == 1
        assert len(result["scores"]) == 1

    @patch("linc_crucible.server_context._ensure_initialized")
    def test_score_no_project(self, mock_init: MagicMock) -> None:
        from linc_crucible.assay_tools import crucible_score

        result = crucible_score(project="")
        assert "error" in result


class TestCruciblePourBuildOnly:
    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_build_only_strips_devices_and_adds_flag(
        self, mock_deployer: MagicMock, mock_init: MagicMock
    ) -> None:
        """build_only=True strips devices from result and adds build_only=True."""
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(
            project="TestProject",
            build_only=True,
            build_success=True,
        )

        result = await crucible_pour(project="TestProject", build_only=True)
        call_kwargs = mock_deployer.pour.call_args.kwargs
        assert call_kwargs["build_only"] is True
        assert "devices" not in result
        assert result["build_only"] is True

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_build_only_normalises_a_device_shaped_result(
        self, mock_deployer: MagicMock, mock_init: MagicMock
    ) -> None:
        """Even a devices-shaped deployer result comes back as a build_only payload."""
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(
            project="TestProject",
            results=[DeployResult(device_serial="ABC", install_success=True)],
        )

        result = await crucible_pour(project="TestProject", build_only=True)

        assert "devices" not in result
        assert result["build_only"] is True

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_normal_pour_keeps_devices_and_no_build_only_flag(
        self, mock_deployer: MagicMock, mock_init: MagicMock
    ) -> None:
        """A deploying pour's payload is untouched: devices present, no build_only key."""
        from linc_crucible.deploy_tools import crucible_pour

        dr = DeployResult(device_serial="ABC", install_success=True)
        mock_deployer.pour.return_value = PourResult(project="TestProject", results=[dr])

        result = await crucible_pour(project="TestProject")

        assert result == {"project": "TestProject", "success": True, "devices": [dr.to_dict()]}
        assert "build_only" not in mock_deployer.pour.call_args.kwargs

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_build_only_result_is_exact_deployer_payload(
        self, mock_deployer: MagicMock, mock_init: MagicMock
    ) -> None:
        """build_only returns the deployer's build-only payload verbatim."""
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(
            project="TestProject",
            build_only=True,
            build_success=True,
            artifacts={"android": "/tmp/app.apk"},
        )

        result = await crucible_pour(project="TestProject", build_only=True)

        assert result == {
            "project": "TestProject",
            "success": True,
            "build_only": True,
            "artifacts": {"android": "/tmp/app.apk"},
        }
        mock_deployer.pour.assert_called_once_with(
            project_name="TestProject",
            device="",
            platform="",
            launch=False,
            skip_build=False,
            module="",
            device_type="",
            launch_intent="",
            build_type="debug",
            policy=_BENCH_POLICY,
            allow_personal=False,
            build_only=True,
        )

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_build_only_passes_launch_false_and_empty_intent(
        self, mock_deployer: MagicMock, mock_init: MagicMock
    ) -> None:
        """build_only=True passes launch=False and launch_intent='' to deployer."""
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(project="TestProject", build_only=True)

        await crucible_pour(
            project="TestProject",
            build_only=True,
            launch=True,
            launch_intent=".MainActivity",
        )
        call_kwargs = mock_deployer.pour.call_args.kwargs
        assert call_kwargs["launch"] is False
        assert call_kwargs["launch_intent"] == ""
        assert call_kwargs["build_only"] is True

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_build_only_false_leaves_result_intact(
        self, mock_deployer: MagicMock, mock_init: MagicMock
    ) -> None:
        """build_only=False (default) does not strip devices or add build_only key."""
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(
            project="TestProject",
            results=[
                DeployResult(device_serial="ABC", install_success=True),
            ],
        )

        result = await crucible_pour(project="TestProject", build_only=False)
        assert "devices" in result
        assert "build_only" not in result


class TestCruciblePourMultiTarget:
    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_pour_with_module_param(
        self, mock_deployer: MagicMock, mock_init: MagicMock
    ) -> None:
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(
            project="SniperPulseApp",
            results=[
                DeployResult(device_serial="WATCH1", install_success=True, module="watchApp"),
            ],
        )

        result = await crucible_pour(project="SniperPulseApp", module="watchApp", skip_build=True)
        assert result["success"] is True
        assert result["devices"][0]["module"] == "watchApp"
        mock_deployer.pour.assert_called_once_with(
            project_name="SniperPulseApp",
            device="",
            platform="",
            launch=False,
            skip_build=True,
            module="watchApp",
            device_type="",
            launch_intent="",
            build_type="debug",
            policy=_BENCH_POLICY,
            allow_personal=False,
        )

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_pour_with_device_type_param(
        self,
        mock_deployer: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(
            project="SniperPulseApp",
            results=[
                DeployResult(device_serial="WATCH1", install_success=True, module="watchApp"),
            ],
        )

        result = await crucible_pour(
            project="SniperPulseApp", device_type="watch", skip_build=True
        )
        assert result["success"] is True
        mock_deployer.pour.assert_called_once_with(
            project_name="SniperPulseApp",
            device="",
            platform="",
            launch=False,
            skip_build=True,
            module="",
            device_type="watch",
            launch_intent="",
            build_type="debug",
            policy=_BENCH_POLICY,
            allow_personal=False,
        )

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_pour_result_includes_module(
        self,
        mock_deployer: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value = PourResult(
            project="Multi",
            results=[
                DeployResult(device_serial="A", install_success=True, module="app"),
                DeployResult(device_serial="B", install_success=True, module="watchApp"),
            ],
        )

        result = await crucible_pour(project="Multi", skip_build=True)
        assert result["devices"][0]["module"] == "app"
        assert result["devices"][1]["module"] == "watchApp"


class TestCrucibleAssayA11yTree:
    """Tests for a11y_tree and device params on crucible_assay."""

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._assayer")
    async def test_assay_passes_a11y_tree_param(
        self, mock_assayer: MagicMock, mock_init: MagicMock
    ) -> None:
        """a11y_tree=True and device are forwarded to assayer.assay()."""
        from linc_crucible.assay_tools import crucible_assay

        mock_assayer.assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=5, passed=5, success=True),
        )

        await crucible_assay(project="TestProject", a11y_tree=True, device="ABC123")
        mock_assayer.assay.assert_called_once_with(
            "TestProject", mode="quick", perf=False, a11y=False, a11y_tree=True, device="ABC123"
        )

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._assayer")
    async def test_assay_default_a11y_tree_false(
        self, mock_assayer: MagicMock, mock_init: MagicMock
    ) -> None:
        """a11y_tree defaults to False, device defaults to empty string."""
        from linc_crucible.assay_tools import crucible_assay

        mock_assayer.assay.return_value = AssayReport(
            project="TestProject",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=5, passed=5, success=True),
        )

        await crucible_assay(project="TestProject")
        mock_assayer.assay.assert_called_once_with(
            "TestProject", mode="quick", perf=False, a11y=False, a11y_tree=False, device=""
        )


class TestCrucibleBaselineA11yMode:
    """Tests for mode=a11y on crucible_baseline."""

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._tree_capture")
    @patch("linc_crucible.server_context._nerve_client")
    async def test_baseline_a11y_capture(
        self,
        mock_nerve: MagicMock,
        mock_tree_capture: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """mode=a11y with action=capture calls nerve.elements + tree_capture.store."""
        from linc_crucible.baseline_tools import crucible_baseline
        from linc_crucible.core.tree_capture import A11yBaselineEntry

        mock_nerve.discover.return_value = [{"serial": "DEV1"}]
        mock_nerve.elements.return_value = [
            {
                "resource_id": "btn1",
                "class_name": "android.widget.Button",
                "text": "OK",
                "content_desc": "OK Button",
                "bounds": "[0,0][100,50]",
                "clickable": True,
                "focusable": True,
                "visible": True,
            }
        ]
        mock_tree_capture.store.return_value = A11yBaselineEntry(
            project="TestProject",
            screen_name="home",
            capture_time="20260413T120000Z",
            tree_path="/tmp/a11y_tree.json",
            device_info="",
            app_version="",
            element_count=1,
        )

        result = await crucible_baseline(
            action="capture",
            project="TestProject",
            screen_name="home",
            mode="a11y",
            device="DEV1",
        )
        assert result["action"] == "capture"
        assert result["mode"] == "a11y"
        assert result["entry"]["screen_name"] == "home"

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._tree_capture")
    @patch("linc_crucible.server_context._nerve_client")
    async def test_baseline_a11y_capture_no_screen_name(
        self,
        mock_nerve: MagicMock,
        mock_tree_capture: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """mode=a11y capture without screen_name returns error."""
        from linc_crucible.baseline_tools import crucible_baseline

        result = await crucible_baseline(
            action="capture",
            project="TestProject",
            mode="a11y",
        )
        assert "error" in result

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._tree_capture")
    async def test_baseline_a11y_latest(
        self,
        mock_tree_capture: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """mode=a11y with action=latest returns latest entry."""
        from linc_crucible.baseline_tools import crucible_baseline
        from linc_crucible.core.tree_capture import A11yBaselineEntry

        mock_tree_capture.get_latest.return_value = A11yBaselineEntry(
            project="TestProject",
            screen_name="home",
            capture_time="20260413T120000Z",
            tree_path="/tmp/a11y_tree.json",
            device_info="Pixel 8",
            app_version="1.0",
            element_count=3,
        )

        result = await crucible_baseline(
            action="latest",
            project="TestProject",
            screen_name="home",
            mode="a11y",
        )
        assert result["action"] == "latest"
        assert result["mode"] == "a11y"
        assert result["entry"]["element_count"] == 3

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._tree_capture")
    async def test_baseline_a11y_latest_none(
        self,
        mock_tree_capture: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """mode=a11y latest with no baselines returns entry=None with message."""
        from linc_crucible.baseline_tools import crucible_baseline

        mock_tree_capture.get_latest.return_value = None

        result = await crucible_baseline(
            action="latest",
            project="TestProject",
            screen_name="home",
            mode="a11y",
        )
        assert result["action"] == "latest"
        assert result["mode"] == "a11y"
        assert result["entry"] is None
        assert "message" in result

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._tree_capture")
    async def test_baseline_a11y_list(
        self,
        mock_tree_capture: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """mode=a11y list returns all entries."""
        from linc_crucible.baseline_tools import crucible_baseline
        from linc_crucible.core.tree_capture import A11yBaselineEntry

        mock_tree_capture.list_entries.return_value = [
            A11yBaselineEntry(
                project="TestProject",
                screen_name="home",
                capture_time="20260413T120000Z",
                tree_path="/tmp/a11y_tree.json",
                device_info="",
                app_version="",
                element_count=2,
            )
        ]

        result = await crucible_baseline(
            action="list",
            project="TestProject",
            mode="a11y",
        )
        assert result["action"] == "list"
        assert result["mode"] == "a11y"
        assert result["count"] == 1

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._tree_capture")
    async def test_baseline_a11y_prune(
        self,
        mock_tree_capture: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """mode=a11y prune calls tree_capture.prune and returns count."""
        from linc_crucible.baseline_tools import crucible_baseline

        mock_tree_capture.prune.return_value = 3

        result = await crucible_baseline(
            action="prune",
            project="TestProject",
            mode="a11y",
            keep=2,
        )
        assert result["action"] == "prune"
        assert result["mode"] == "a11y"
        assert result["pruned"] == 3
        assert result["keep"] == 2

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._tree_capture")
    async def test_baseline_a11y_unknown_action(
        self,
        mock_tree_capture: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """mode=a11y with unknown action returns error."""
        from linc_crucible.baseline_tools import crucible_baseline

        result = await crucible_baseline(
            action="explode",
            project="TestProject",
            mode="a11y",
        )
        assert "error" in result

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._baseline_manager")
    async def test_baseline_screenshot_mode_unchanged(
        self,
        mock_baseline_manager: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """mode=screenshot (default) still routes to original baseline_manager logic."""
        from linc_crucible.baseline_tools import crucible_baseline

        mock_baseline_manager.list_baselines.return_value = []

        result = await crucible_baseline(
            action="list",
            project="TestProject",
            mode="screenshot",
        )
        assert result["action"] == "list"
        mock_baseline_manager.list_baselines.assert_called_once()


class TestEnsureInitializedA11y:
    """Tests that _ensure_initialized creates _nerve_client and _tree_capture."""

    def test_ensure_initialized_creates_nerve_and_tree_capture(self) -> None:
        """After _ensure_initialized, _nerve_client and _tree_capture are set."""
        import linc_crucible.server_context as ctx

        # Reset singletons
        ctx._initialized = False
        ctx._nerve_client = None
        ctx._tree_capture = None

        with (
            patch("linc_crucible.core.forge_integration.ForgeProjectReader"),
            patch("linc_crucible.core.scorer.QualityScorer"),
            patch("linc_crucible.core.deployer.Deployer"),
            patch("linc_crucible.core.assayer.Assayer"),
            patch("linc_crucible.core.preflight.PreflightChecker"),
            patch("linc_crucible.core.baseline.BaselineManager"),
            patch("linc_crucible.core.nerve_client.NerveClient"),
            patch("linc_crucible.core.tree_capture.TreeCapture"),
        ):
            ctx._ensure_initialized()
            # After init, _nerve_client and _tree_capture module-level vars are set
            assert ctx._nerve_client is not None
            assert ctx._tree_capture is not None

    def test_ensure_initialized_assayer_receives_nerve_and_tree(self) -> None:
        """Assayer is initialized with nerve_client and tree_capture."""
        import linc_crucible.server_context as ctx

        # Reset singletons
        ctx._initialized = False
        ctx._nerve_client = None
        ctx._tree_capture = None

        captured_kwargs: dict = {}

        def fake_assayer(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return MagicMock()

        with (
            patch("linc_crucible.core.forge_integration.ForgeProjectReader"),
            patch("linc_crucible.core.scorer.QualityScorer"),
            patch("linc_crucible.core.deployer.Deployer"),
            patch("linc_crucible.core.assayer.Assayer", side_effect=fake_assayer),
            patch("linc_crucible.core.preflight.PreflightChecker"),
            patch("linc_crucible.core.baseline.BaselineManager"),
            patch("linc_crucible.core.nerve_client.NerveClient"),
            patch("linc_crucible.core.tree_capture.TreeCapture"),
        ):
            ctx._ensure_initialized()
            assert "nerve_client" in captured_kwargs
            assert "tree_capture" in captured_kwargs
            assert captured_kwargs["nerve_client"] is not None
            assert captured_kwargs["tree_capture"] is not None


class TestCrucibleBaselinePromote:
    """Tests for action=promote on crucible_baseline (screenshot mode)."""

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._baseline_manager")
    async def test_promote_delegates_with_provenance(
        self,
        mock_bm: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from linc_crucible.baseline_tools import crucible_baseline
        from linc_crucible.core.baseline import BaselineEntry

        mock_bm.promote.return_value = BaselineEntry(
            project="MyApp",
            screen_name="home",
            capture_time="20260612T000000Z",
            screenshot_path="/tmp/x.png",
            device_info="",
            app_version="",
            width=10,
            height=10,
            source="promoted",
            source_score=0.95,
            promoted_at="20260612T010000Z",
        )

        result = await crucible_baseline(
            action="promote",
            project="MyApp",
            screen_name="home",
            image_path="/tmp/candidate.png",
            source_score=0.95,
        )
        assert result["action"] == "promote"
        assert result["entry"]["source"] == "promoted"
        mock_bm.promote.assert_called_once()
        assert mock_bm.promote.call_args.kwargs["source_score"] == 0.95

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._baseline_manager")
    async def test_promote_requires_image_path(
        self,
        mock_bm: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from linc_crucible.baseline_tools import crucible_baseline

        result = await crucible_baseline(action="promote", project="MyApp", screen_name="home")
        assert "error" in result
        mock_bm.promote.assert_not_called()


class TestCrucibleBaselineA11yEmptyGuard:
    """AC3: zero-element a11y baseline rejected fail-loud, with allow_empty opt-out."""

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.core.nerve_convert.convert_nerve_elements")
    @patch("linc_crucible.server_context._tree_capture")
    @patch("linc_crucible.server_context._nerve_client")
    async def test_zero_elements_rejected_loud_by_default(
        self,
        mock_nerve: MagicMock,
        mock_tree_capture: MagicMock,
        mock_convert: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from linc_crucible.baseline_tools import crucible_baseline

        mock_nerve.elements.return_value = [{"raw": "present-but-converts-to-nothing"}]
        mock_convert.return_value = []  # zero usable elements after conversion

        result = await crucible_baseline(
            action="capture",
            project="MyApp",
            screen_name="home",
            mode="a11y",
            device="DEV1",
        )
        assert "error" in result
        assert "empty a11y baseline" in result["error"]
        mock_tree_capture.store.assert_not_called()

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.core.nerve_convert.convert_nerve_elements")
    @patch("linc_crucible.server_context._tree_capture")
    @patch("linc_crucible.server_context._nerve_client")
    async def test_allow_empty_skips_without_storing(
        self,
        mock_nerve: MagicMock,
        mock_tree_capture: MagicMock,
        mock_convert: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from linc_crucible.baseline_tools import crucible_baseline

        mock_nerve.elements.return_value = [{"raw": "x"}]
        mock_convert.return_value = []

        result = await crucible_baseline(
            action="capture",
            project="MyApp",
            screen_name="home",
            mode="a11y",
            device="DEV1",
            allow_empty=True,
        )
        assert result["skipped"] is True
        assert result["stored"] is False
        assert "opt-out" in result["reason"]
        mock_tree_capture.store.assert_not_called()


class TestCrucibleUsageStats:
    def test_reads_default_log_and_computes_dormancy(self, tmp_path: Any) -> None:
        import json
        import os
        from pathlib import Path

        import linc_crucible.mcp_server  # noqa: F401 — ensure all tools registered on server
        from linc_crucible.stats_tools import crucible_usage_stats

        # autouse _isolate_linc_env points LINC_HOME at a tmp dir; write the log
        # at the default path the tool will read.
        linc_home = Path(os.environ["LINC_HOME"])
        linc_home.mkdir(parents=True, exist_ok=True)
        log = linc_home / "mcp-calls.jsonl"
        with log.open("w") as f:
            for tool, ok in [("crucible_pour", True), ("crucible_pour", False)]:
                f.write(
                    json.dumps(
                        {
                            "timestamp": "2026-06-27T00:00:00Z",
                            "server": "linc-crucible",
                            "tool": tool,
                            "params": {},
                            "duration_ms": 10.0,
                            "success": ok,
                            "error": None,
                        }
                    )
                    + "\n"
                )

        result = crucible_usage_stats(server_filter="linc-crucible")
        assert result["total_calls"] == 2
        by_tool = {t["tool"]: t for t in result["per_tool"]}
        assert by_tool["crucible_pour"]["calls"] == 2
        assert by_tool["crucible_pour"]["success_pct"] == 50.0
        # Dormancy computed against the real registry — other crucible tools unused.
        assert "crucible_assay" in result["dormant"]
        assert result["registered_count"] >= 8


class TestUnknownProjectNotFound:
    @pytest.mark.asyncio
    async def test_assay_unknown_project(self) -> None:
        from linc_crucible.assay_tools import crucible_assay

        result = await crucible_assay(project="NoSuchProject")
        assert result.get("error_category") == "not_found"
        assert result.get("error") == "Project 'NoSuchProject' not found in forge registry"
        assert "lint" not in result
        assert "tests" not in result

    def test_preflight_unknown_project(self) -> None:
        from linc_crucible.assay_tools import crucible_preflight

        result = crucible_preflight(project="NoSuchProject")
        assert result.get("error_category") == "not_found"
        assert result.get("error") == "Project 'NoSuchProject' not found in forge registry"
        assert "checks" not in result

    def test_cli_assay_unknown_project_exits_nonzero(self) -> None:
        from click.testing import CliRunner
        from linc_cli_kit import parse_error_envelope

        from linc_crucible.cli import cli

        runner = CliRunner()
        res = runner.invoke(cli, ["assay", "--project", "NoSuchProject"])
        assert res.exit_code != 0
        envelope = parse_error_envelope(res.output)
        assert envelope is not None
        assert "error" in envelope
        error_info = envelope["error"]
        assert "NoSuchProject" in error_info.get("message", "")
        assert error_info.get("details", {}).get("error_category") == "not_found"


# ── Personal-device policy (epic linc-crucible-6jf, bead 6jf.2) ───────


class TestCruciblePourPersonalDevices:
    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_pour_forwards_policy_and_allow_personal(
        self,
        mock_deployer: MagicMock,
        mock_init: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from linc_crucible.deploy_tools import crucible_pour

        _use_policy(monkeypatch, _PERSONAL_POLICY)
        mock_deployer.pour.return_value = PourResult(project="TestProject")

        await crucible_pour(project="TestProject", device=_PERSONAL_SERIAL, allow_personal=True)

        kwargs = mock_deployer.pour.call_args.kwargs
        assert kwargs["policy"] is _PERSONAL_POLICY
        assert kwargs["allow_personal"] is True

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_pour_explicit_personal_returns_personal_device_error(
        self,
        mock_init: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import linc_crucible.server_context as ctx
        from linc_crucible.core.deployer import Deployer
        from linc_crucible.core.forge_integration import ForgeProject, ForgeProjectReader
        from linc_crucible.core.nerve_client import NerveClient
        from linc_crucible.deploy_tools import crucible_pour

        _use_policy(monkeypatch, _PERSONAL_POLICY)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(name="TestProject", path="/tmp/TestProject")
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [
            {"serial": _PERSONAL_SERIAL, "platform": "android", "state": "online"}
        ]
        monkeypatch.setattr(ctx, "_deployer", Deployer(forge_reader=forge, nerve_client=nerve))

        result = await crucible_pour(project="TestProject", device=_PERSONAL_SERIAL)

        assert result["success"] is False
        assert result["error_category"] == "personal_device"
        assert _PERSONAL_SERIAL in result["error"]
        nerve.install.assert_not_called()

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_build_only_forwards_policy_and_allow_personal(
        self,
        mock_deployer: MagicMock,
        mock_init: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """build_only pours carry the policy kwargs like any other pour."""
        from linc_crucible.deploy_tools import crucible_pour

        _use_policy(monkeypatch, _PERSONAL_POLICY)
        mock_deployer.pour.return_value = PourResult(
            project="TestProject", build_only=True, build_success=True
        )

        await crucible_pour(project="TestProject", build_only=True, allow_personal=True)

        kwargs = mock_deployer.pour.call_args.kwargs
        assert kwargs["policy"] is _PERSONAL_POLICY
        assert kwargs["allow_personal"] is True
        assert kwargs["build_only"] is True

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_build_only_with_personal_serial_builds_and_never_installs(
        self,
        mock_init: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A build-only pour targets no device, so a personal serial is not refused —
        and nothing reaches nerve."""
        import linc_crucible.server_context as ctx
        from linc_crucible.core.deployer import Deployer
        from linc_crucible.core.forge_integration import ForgeProject, ForgeProjectReader
        from linc_crucible.core.nerve_client import NerveClient
        from linc_crucible.deploy_tools import crucible_pour

        _use_policy(monkeypatch, _PERSONAL_POLICY)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="TestProject", path="/tmp/TestProject", build_command="./gradlew assembleDebug"
        )
        forge.find_apk.return_value = "/tmp/TestProject/app-debug.apk"
        forge.find_ios_app.return_value = None
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [
            {"serial": _PERSONAL_SERIAL, "platform": "android", "state": "online"}
        ]
        deployer = Deployer(forge_reader=forge, nerve_client=nerve)
        monkeypatch.setattr(ctx, "_deployer", deployer)

        with patch.object(deployer, "_run_build", return_value=(True, "BUILD OK")):
            result = await crucible_pour(
                project="TestProject", device=_PERSONAL_SERIAL, build_only=True
            )

        assert result == {
            "project": "TestProject",
            "success": True,
            "build_only": True,
            "artifacts": {"android": "/tmp/TestProject/app-debug.apk"},
        }
        nerve.discover.assert_not_called()
        nerve.install.assert_not_called()
        nerve.launch.assert_not_called()

    def test_pour_docstring_documents_allow_personal(self) -> None:
        from linc_crucible.deploy_tools import crucible_pour

        doc = crucible_pour.__doc__ or ""
        assert "allow_personal:" in doc
        assert "Personal devices" in doc


class TestA11yCapturePersonalDevices:
    def _nerve(self, devices: list[dict]) -> MagicMock:
        from linc_crucible.core.nerve_client import NerveClient

        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = devices
        nerve.elements.return_value = [
            {"resource_id": "btn", "class_name": "android.widget.Button", "text": "OK"}
        ]
        return nerve

    def _capture(self, device: str = "", allow_personal: bool = False) -> dict[str, Any]:
        from linc_crucible.baseline_tools import _a11y_capture

        tree_capture = MagicMock()
        tree_capture.store.return_value.to_dict.return_value = {"screen_name": "home"}
        return _a11y_capture(
            tree_capture,
            "TestProject",
            "home",
            device,
            "",
            "",
            False,
            allow_personal=allow_personal,
        )

    def test_auto_pick_skips_personal_first_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import linc_crucible.server_context as ctx

        _use_policy(monkeypatch, _PERSONAL_POLICY)
        nerve = self._nerve([{"serial": _PERSONAL_SERIAL}, {"serial": "BENCH1"}])
        monkeypatch.setattr(ctx, "_nerve_client", nerve)

        result = self._capture()

        assert result["action"] == "capture"
        nerve.elements.assert_called_once_with("BENCH1")

    def test_auto_pick_with_only_personal_devices_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import linc_crucible.server_context as ctx

        _use_policy(monkeypatch, _PERSONAL_POLICY)
        nerve = self._nerve([{"serial": _PERSONAL_SERIAL}])
        monkeypatch.setattr(ctx, "_nerve_client", nerve)

        result = self._capture()

        assert result["error_category"] == "personal_device"
        assert _PERSONAL_SERIAL in result["error"]
        assert result["excluded_personal"] == [_PERSONAL_SERIAL]
        nerve.elements.assert_not_called()

    def test_explicit_personal_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import linc_crucible.server_context as ctx

        _use_policy(monkeypatch, _PERSONAL_POLICY)
        nerve = self._nerve([{"serial": _PERSONAL_SERIAL}])
        monkeypatch.setattr(ctx, "_nerve_client", nerve)

        result = self._capture(device=_PERSONAL_SERIAL)

        assert result["success"] is False
        assert result["error_category"] == "personal_device"
        assert "allow_personal=True" in result["error"]
        nerve.elements.assert_not_called()

    def test_explicit_personal_with_allow_personal_captures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import linc_crucible.server_context as ctx

        _use_policy(monkeypatch, _PERSONAL_POLICY)
        nerve = self._nerve([{"serial": _PERSONAL_SERIAL}])
        monkeypatch.setattr(ctx, "_nerve_client", nerve)

        result = self._capture(device=_PERSONAL_SERIAL, allow_personal=True)

        assert result["action"] == "capture"
        nerve.elements.assert_called_once_with(_PERSONAL_SERIAL)

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._tree_capture")
    @patch("linc_crucible.server_context._nerve_client")
    async def test_baseline_tool_forwards_allow_personal(
        self,
        mock_nerve: MagicMock,
        mock_tree_capture: MagicMock,
        mock_init: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from linc_crucible.baseline_tools import crucible_baseline

        _use_policy(monkeypatch, _PERSONAL_POLICY)

        refused = await crucible_baseline(
            action="capture",
            project="TestProject",
            screen_name="home",
            mode="a11y",
            device=_PERSONAL_SERIAL,
        )
        assert refused["error_category"] == "personal_device"
        mock_nerve.elements.assert_not_called()

        mock_nerve.elements.return_value = []
        allowed = await crucible_baseline(
            action="capture",
            project="TestProject",
            screen_name="home",
            mode="a11y",
            device=_PERSONAL_SERIAL,
            allow_personal=True,
        )
        assert "error_category" not in allowed
        mock_nerve.elements.assert_called_once_with(_PERSONAL_SERIAL)

    def test_baseline_docstring_documents_allow_personal(self) -> None:
        from linc_crucible.baseline_tools import crucible_baseline

        doc = crucible_baseline.__doc__ or ""
        assert "allow_personal:" in doc
        assert "Personal devices" in doc
