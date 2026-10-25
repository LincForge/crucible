"""Regression: `fleet list` / `baseline list` honor the kit's `--json`/`--human` output
contract, and `score`'s no-scores case classifies as `not_found`, not `internal`.

Before this bead, `fleet list` and `baseline list` were hand-written `console.print` Rich
commands that ignored `--json`/the kit's JSON-by-default mode entirely — `json.loads()` on
their stdout raised `JSONDecodeError`. `score`'s no-scores case unconditionally used the
kit's generic `internal`/`TOOL_ERROR` classification even though the underlying condition is
squarely "no such record" — this pins the more specific `not_found`/`NOT_FOUND` category.
"""

from __future__ import annotations

import json
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
        "_cached_device_policy",
    ):
        m = MagicMock()
        monkeypatch.setattr(server_ctx, attr, m)
        mocks[attr] = m
    return mocks


def _two_devices() -> list:
    from linc_crucible.core.device import DeviceType, FleetDevice, Platform

    return [
        FleetDevice(
            serial="ON",
            platform=Platform.ANDROID,
            device_type=DeviceType.PHONE,
            model="Pixel 7",
            os_version="14",
            nickname="myphone",
            state="online",
        ),
        FleetDevice(
            serial="OFF",
            platform=Platform.IOS,
            device_type=DeviceType.SIMULATOR,
            model="iPhone 15",
            os_version="17",
            state="offline",
        ),
    ]


class TestFleetListJSON:
    @patch("linc_crucible.core.fleet_registry.FleetRegistry")
    def test_json_default_and_explicit_flag_agree(
        self, mock_cls: MagicMock, runner: CliRunner
    ) -> None:
        mock_registry = MagicMock()
        mock_registry.list_devices.return_value = _two_devices()
        mock_cls.return_value = mock_registry

        for args in (["--json", "fleet", "list"], ["fleet", "list"]):
            result = runner.invoke(cli, args)
            assert result.exit_code == 0, result.output
            payload = json.loads(result.stdout)
            assert payload["count"] == 2
            serials = {d["serial"] for d in payload["devices"]}
            assert serials == {"ON", "OFF"}
            for d in payload["devices"]:
                for key in (
                    "serial",
                    "platform",
                    "device_type",
                    "model",
                    "os_version",
                    "nickname",
                    "state",
                ):
                    assert key in d

    @patch("linc_crucible.core.fleet_registry.FleetRegistry")
    def test_json_empty_registry(self, mock_cls: MagicMock, runner: CliRunner) -> None:
        mock_registry = MagicMock()
        mock_registry.list_devices.return_value = []
        mock_cls.return_value = mock_registry

        result = runner.invoke(cli, ["--json", "fleet", "list"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload == {
            "count": 0,
            "platform": None,
            "devices": [],
            "message": "No devices registered. Run 'crucible fleet discover' first.",
        }

    @patch("linc_crucible.core.fleet_registry.FleetRegistry")
    def test_human_renders_table_not_json(self, mock_cls: MagicMock, runner: CliRunner) -> None:
        mock_registry = MagicMock()
        mock_registry.list_devices.return_value = _two_devices()
        mock_cls.return_value = mock_registry

        result = runner.invoke(cli, ["--human", "fleet", "list"])
        assert result.exit_code == 0
        assert "Crucible Fleet" in result.stdout
        assert "ON" in result.stdout
        assert "OFF" in result.stdout
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout)


class TestBaselineListJSON:
    def _two_entries(self) -> list:
        from linc_crucible.core.baseline import BaselineEntry

        return [
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

    @patch("linc_crucible.core.baseline.BaselineManager")
    def test_json_with_entries(self, mock_mgr_cls: MagicMock, runner: CliRunner) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_baselines.return_value = self._two_entries()
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cli, ["--json", "baseline", "list", "TestProject"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["count"] == 2
        assert payload["project"] == "TestProject"
        assert payload["baselines"][0]["dimensions"] == "1080x2400"

    @patch("linc_crucible.core.baseline.BaselineManager")
    def test_json_empty(self, mock_mgr_cls: MagicMock, runner: CliRunner) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_baselines.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cli, ["--json", "baseline", "list", "TestProject"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["count"] == 0
        assert payload["message"] == "No baselines found."

    @patch("linc_crucible.core.baseline.BaselineManager")
    def test_human_renders_table(self, mock_mgr_cls: MagicMock, runner: CliRunner) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_baselines.return_value = self._two_entries()
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cli, ["--human", "baseline", "list", "TestProject"])
        assert result.exit_code == 0
        assert "Baselines for TestProject" in result.stdout
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout)


class TestScoreNotFoundCategory:
    def test_no_scores_classifies_not_found(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        mock_ctx["_scorer"].load_latest.return_value = None

        result = runner.invoke(cli, ["score", "TestProject"])
        assert result.exit_code == 1
        assert result.stdout == ""
        envelope = json.loads(result.stderr)
        assert envelope["error"]["category"] == "not_found"
        assert envelope["error"]["code"] == "NOT_FOUND"
        assert "No scores found" in envelope["error"]["message"]

    @patch("linc_crucible.server_context._ensure_initialized")
    def test_crucible_score_tool_carries_error_category(
        self, mock_init: MagicMock, tmp_path
    ) -> None:
        from linc_crucible.assay_tools import crucible_score
        from linc_crucible.core.scorer import QualityScorer

        scorer = QualityScorer(scores_dir=tmp_path / "never-created")
        with patch("linc_crucible.server_context._scorer", scorer):
            result = crucible_score(project="X")

        assert result["error_category"] == "not_found"
        assert result["project"] == "X"
        assert "error" in result


class TestCLIErrorCategoryMapping:
    def test_assay_unknown_project_classifies_not_found(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.assayer import AssayReport

        mock_ctx["_assayer"].assay.return_value = AssayReport(
            project="NoSuchProject", mode="quick", not_found=True
        )

        result = runner.invoke(cli, ["assay", "--project", "NoSuchProject"])
        assert result.exit_code == 1
        assert result.stdout == ""
        envelope = json.loads(result.stderr)
        assert envelope["error"]["category"] == "not_found"
        assert envelope["error"]["code"] == "NOT_FOUND"
        assert "NoSuchProject" in envelope["error"]["message"]
        assert envelope["error"]["details"]["error_category"] == "not_found"

    def test_preflight_unknown_project_classifies_not_found(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.preflight import PreflightReport

        mock_ctx["_preflight_checker"].check.return_value = PreflightReport(
            project="NoSuchProject", not_found=True
        )

        result = runner.invoke(cli, ["preflight", "--project", "NoSuchProject"])
        assert result.exit_code == 1
        assert result.stdout == ""
        envelope = json.loads(result.stderr)
        assert envelope["error"]["category"] == "not_found"
        assert envelope["error"]["code"] == "NOT_FOUND"

    def test_pour_personal_device_refusal_classifies_permission(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner
    ) -> None:
        from linc_crucible.core.deployer import PourResult

        mock_ctx["_deployer"].pour.return_value = PourResult(
            project="TestProject",
            refusal="Refusing to deploy to personal device PERSONAL1",
        )

        result = runner.invoke(cli, ["pour", "--project", "TestProject", "--device", "PERSONAL1"])
        assert result.exit_code == 1
        assert result.stdout == ""
        envelope = json.loads(result.stderr)
        assert envelope["error"]["category"] == "permission"
        assert envelope["error"]["code"] == "PERSONAL_DEVICE"

    @pytest.mark.parametrize(
        "argv",
        [
            ["assay", "--project", "../bad"],
            ["pour", "--project", "TestProject", "--build-type", "bad-type!"],
            ["score", "../bad"],
            ["bvt", "--project", "../bad", "--device", "DEV1"],
        ],
    )
    def test_validation_error_from_tool_input_validation_exits_2(
        self, mock_ctx: dict[str, MagicMock], runner: CliRunner, argv: list[str]
    ) -> None:
        result = runner.invoke(cli, argv)
        assert result.exit_code == 2, result.output
        assert result.stdout == ""
        envelope = json.loads(result.stderr)
        assert envelope["error"]["category"] == "validation"
        assert envelope["error"]["code"] == "VALIDATION_ERROR"
        assert "validation_error" not in envelope["error"]["details"]
        assert envelope["error"]["details"]["error_category"] == "validation"

    def test_category_error_helper_handles_non_matching_inputs(self) -> None:
        from linc_crucible.cli import _category_error

        assert _category_error(None) is None
        assert _category_error({}) is None
        assert _category_error({"error": "plain failure"}) is None
        assert _category_error({"error": "plain failure", "error_category": "unknown"}) is None
        err = _category_error(
            {"error": "bad input", "error_category": "validation", "field": "limit"}
        )
        assert err is not None
        assert err.code == "VALIDATION_ERROR"
        assert err.category == "validation"
        assert err.exit_code == 2
        assert err.details == {"error_category": "validation", "field": "limit"}

    def test_cli_helpers_exact_contracts(self) -> None:
        from types import SimpleNamespace

        from linc_cli_kit.errors import ToolError

        from linc_crucible.cli import (
            _bvt_not_ran_error,
            _cli_result_or_raise,
            _decode_tool_result,
            _format_pass_rate,
            _local_fn,
        )

        assert _format_pass_rate({"pass_rate": 0.75}) == "75%"
        assert _format_pass_rate({"pass_rate": None}) == "unmeasured"
        assert _format_pass_rate({}) == "unmeasured"

        bvt_err1 = _bvt_not_ran_error(
            {"ran": False, "reason": "unknown-project", "detail": "missing"}
        )
        assert bvt_err1.code == "UNKNOWN_PROJECT"
        assert bvt_err1.category == "not_found"
        assert str(bvt_err1) == "missing"

        bvt_err2 = _bvt_not_ran_error(
            {"ran": False, "error_category": "personal_device", "error": "refused"}
        )
        assert bvt_err2.code == "PERSONAL_DEVICE"
        assert bvt_err2.category == "permission"
        assert str(bvt_err2) == "refused"

        bvt_err3 = _bvt_not_ran_error({"ran": False})
        assert bvt_err3.code == ""
        assert bvt_err3.category == "internal"
        assert str(bvt_err3) == ""

        with pytest.raises(ToolError) as exc_demo:
            _cli_result_or_raise({"status": "unmeasured", "project": "Demo"})
        assert exc_demo.value.code == "UNMEASURED"
        assert exc_demo.value.category == "internal"
        assert str(exc_demo.value) == "Assay for project 'Demo' measured nothing"

        with pytest.raises(ToolError) as exc_empty:
            _cli_result_or_raise({"status": "unmeasured"})
        assert exc_empty.value.code == "UNMEASURED"
        assert exc_empty.value.category == "internal"
        assert str(exc_empty.value) == "Assay for project '' measured nothing"

        score_raw = {"status": "unmeasured", "error": "no scores", "error_category": "not_found"}
        assert _decode_tool_result("crucible_score", score_raw) is score_raw

        sync_spec = SimpleNamespace(
            name="crucible_bvt",
            fn=lambda **kw: '{"ran": true, "passed": 1}',
        )
        assert _local_fn(sync_spec)() == {"ran": True, "passed": 1}


class TestFleetAndBaselineSubcommandsJSONHonesty:
    """[test] baseline capture/prune and fleet discover/add/remove emit JSON by default
    and Rich only with --human; exit codes remain unchanged."""

    @patch("linc_crucible.core.fleet_registry.FleetRegistry")
    def test_fleet_discover_json_by_default_and_rich_with_human(
        self, mock_cls: MagicMock, runner: CliRunner
    ) -> None:
        mock_reg = MagicMock()
        mock_reg.discover.return_value = [_two_devices()[0]]
        mock_reg.last_discover_skipped_personal = 0
        mock_reg.list_devices.return_value = _two_devices()
        mock_cls.return_value = mock_reg

        res_json = runner.invoke(cli, ["fleet", "discover"])
        assert res_json.exit_code == 0, res_json.output
        assert res_json.stderr == ""
        data = json.loads(res_json.stdout)
        assert data["count"] == 1
        assert data["fleet_total"] == 2
        assert data["skipped_personal"] == 0
        assert data["discovered"][0]["serial"] == "ON"

        res_human = runner.invoke(cli, ["--human", "fleet", "discover"])
        assert res_human.exit_code == 0
        assert "Fleet Discovery" in res_human.stdout
        assert "Discovered 1 new device(s):" in res_human.stdout
        assert "+ ON — Pixel 7 (android/phone)" in res_human.stdout
        assert "Fleet total: 2 device(s)" in res_human.stdout
        assert "skipped" not in res_human.stdout
        with pytest.raises(json.JSONDecodeError):
            json.loads(res_human.stdout)

        # Empty discovery + skipped personal = 1 vs 0
        mock_reg.discover.return_value = []
        mock_reg.last_discover_skipped_personal = 1
        mock_reg.list_devices.return_value = []
        res_skipped_json = runner.invoke(cli, ["fleet", "discover"])
        assert res_skipped_json.exit_code == 0
        skipped_data = json.loads(res_skipped_json.stdout)
        assert skipped_data == {
            "count": 0,
            "fleet_total": 0,
            "skipped_personal": 1,
            "discovered": [],
            "message": "skipped 1 personal device(s)",
        }
        res_skipped_human = runner.invoke(cli, ["--human", "fleet", "discover"])
        assert res_skipped_human.exit_code == 0
        assert "No new devices discovered." in res_skipped_human.stdout
        assert "skipped 1 personal device(s)" in res_skipped_human.stdout
        assert "Fleet total:" not in res_skipped_human.stdout

        mock_reg.last_discover_skipped_personal = 0
        res_empty_json = runner.invoke(cli, ["fleet", "discover"])
        assert json.loads(res_empty_json.stdout)["message"] == "No new devices discovered."

    def test_skipped_personal_count_edge_cases(self) -> None:
        from types import SimpleNamespace

        from linc_crucible.cli import _skipped_personal_count

        assert _skipped_personal_count(SimpleNamespace()) == 0
        assert _skipped_personal_count(SimpleNamespace(last_discover_skipped_personal=0)) == 0
        assert _skipped_personal_count(SimpleNamespace(last_discover_skipped_personal=-1)) == 0
        assert _skipped_personal_count(SimpleNamespace(last_discover_skipped_personal=True)) == 0
        assert _skipped_personal_count(SimpleNamespace(last_discover_skipped_personal="2")) == 0
        assert _skipped_personal_count(SimpleNamespace(last_discover_skipped_personal=1)) == 1
        assert _skipped_personal_count(SimpleNamespace(last_discover_skipped_personal=3)) == 3

    @patch("linc_crucible.core.fleet_registry.FleetRegistry")
    def test_fleet_add_json_by_default_and_rich_with_human(
        self, mock_cls: MagicMock, runner: CliRunner
    ) -> None:
        mock_cls.return_value = MagicMock()

        res_json = runner.invoke(cli, ["fleet", "add", "ABC123", "--nickname", "pixel"])
        assert res_json.exit_code == 0, res_json.output
        assert res_json.stderr == ""
        data = json.loads(res_json.stdout)
        assert data == {
            "added": True,
            "device": {
                "serial": "ABC123",
                "platform": "android",
                "device_type": "phone",
                "model": "",
                "os_version": "",
                "nickname": "pixel",
                "state": "online",
            },
        }

        res_human = runner.invoke(
            cli, ["--human", "fleet", "add", "ABC123", "--nickname", "pixel"]
        )
        assert res_human.exit_code == 0
        assert res_human.stdout.strip() == "Added: ABC123 (pixel)"
        with pytest.raises(json.JSONDecodeError):
            json.loads(res_human.stdout)

        res_no_nick = runner.invoke(cli, ["--human", "fleet", "add", "ABC123"])
        assert res_no_nick.exit_code == 0
        assert res_no_nick.stdout.strip() == "Added: ABC123"

    @patch("linc_crucible.core.fleet_registry.FleetRegistry")
    def test_fleet_remove_json_by_default_rich_with_human_and_exit_codes_pinned(
        self, mock_cls: MagicMock, runner: CliRunner
    ) -> None:
        mock_reg = MagicMock()
        mock_reg.remove.return_value = True
        mock_cls.return_value = mock_reg

        res_json = runner.invoke(cli, ["fleet", "remove", "ABC123"])
        assert res_json.exit_code == 0, res_json.output
        assert res_json.stderr == ""
        data = json.loads(res_json.stdout)
        assert data == {"removed": True, "serial": "ABC123"}

        res_human = runner.invoke(cli, ["--human", "fleet", "remove", "ABC123"])
        assert res_human.exit_code == 0
        assert res_human.stdout.strip() == "Removed: ABC123"
        with pytest.raises(json.JSONDecodeError):
            json.loads(res_human.stdout)

        mock_reg.remove.return_value = False
        res_missing = runner.invoke(cli, ["fleet", "remove", "NONEXISTENT"])
        assert res_missing.exit_code == 1
        assert res_missing.stdout == ""
        err_env = json.loads(res_missing.stderr)
        assert err_env["error"]["message"] == "Device 'NONEXISTENT' not found"
        assert err_env["error"]["details"] == {"removed": False, "serial": "NONEXISTENT"}

    def test_baseline_capture_json_by_default_and_rich_with_human(
        self, runner: CliRunner, tmp_path
    ) -> None:
        from PIL import Image

        png = tmp_path / "screen.png"
        Image.new("RGB", (32, 48), color=(10, 20, 30)).save(png)
        entry = TestBaselineListJSON()._two_entries()[0]

        with patch("linc_crucible.core.baseline.BaselineManager") as mock_cls:
            mock_cls.return_value.capture.return_value = entry
            res_json = runner.invoke(
                cli,
                [
                    "baseline",
                    "capture",
                    "TestProject",
                    "home",
                    str(png),
                    "--device",
                    "Pixel 7",
                    "--app-version",
                    "1.0.0",
                ],
            )
            assert res_json.exit_code == 0, res_json.output
            assert res_json.stderr == ""
            data = json.loads(res_json.stdout)
            assert data == {
                **entry.to_dict(),
                "dimensions": "1080x2400",
                "message": "Baseline captured: TestProject/home",
            }
            mock_cls.return_value.capture.assert_called_once_with(
                project="TestProject",
                screen_name="home",
                image_path=png,
                device_info="Pixel 7",
                app_version="1.0.0",
            )

            res_human = runner.invoke(
                cli, ["--human", "baseline", "capture", "TestProject", "home", str(png)]
            )
            assert res_human.exit_code == 0
            assert "Baseline captured: TestProject/home" in res_human.stdout
            assert "Timestamp: 20260423T120000Z" in res_human.stdout
            assert "Dimensions: 1080x2400" in res_human.stdout
            assert "Path: /tmp/home.png" in res_human.stdout
            with pytest.raises(json.JSONDecodeError):
                json.loads(res_human.stdout)

    def test_baseline_prune_json_by_default_and_rich_with_human(self, runner: CliRunner) -> None:
        with patch("linc_crucible.core.baseline.BaselineManager") as mock_cls:
            mock_cls.return_value.prune.return_value = 3
            res_json = runner.invoke(
                cli, ["baseline", "prune", "TestProject", "--screen", "home", "--keep", "2"]
            )
            assert res_json.exit_code == 0, res_json.output
            assert res_json.stderr == ""
            data = json.loads(res_json.stdout)
            assert data == {
                "project": "TestProject",
                "screen": "home",
                "keep": 2,
                "pruned": 3,
                "message": "Pruned 3 baseline(s) (kept newest 2)",
            }
            mock_cls.return_value.prune.assert_called_once_with(
                "TestProject", screen_name="home", keep=2
            )

            res_human = runner.invoke(
                cli,
                ["--human", "baseline", "prune", "TestProject", "--keep", "2"],
            )
            assert res_human.exit_code == 0
            assert res_human.stdout.strip() == "Pruned 3 baseline(s) (kept newest 2)"
            with pytest.raises(json.JSONDecodeError):
                json.loads(res_human.stdout)

            mock_cls.return_value.prune.return_value = 0
            res_zero = runner.invoke(cli, ["baseline", "prune", "TestProject"])
            assert res_zero.exit_code == 0
            assert json.loads(res_zero.stdout) == {
                "project": "TestProject",
                "screen": "",
                "keep": 5,
                "pruned": 0,
                "message": "Nothing to prune.",
            }
            res_zero_human = runner.invoke(cli, ["--human", "baseline", "prune", "TestProject"])
            assert res_zero_human.exit_code == 0
            assert res_zero_human.stdout.strip() == "Nothing to prune."
