"""Tests for FleetRegistry._load — characterization tests for CRAP-critical branches."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from linc_crucible.core.fleet_registry import FleetRegistry


@pytest.fixture
def mock_nerve() -> MagicMock:
    """Provide a mock NerveClient so no HTTP calls happen."""
    return MagicMock()


class TestFleetRegistryLoad:
    def test_load_missing_file(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """Non-existent fleet file results in empty devices."""
        registry = FleetRegistry(fleet_path=tmp_path / "nonexistent.yaml", nerve_client=mock_nerve)
        assert registry.list_devices() == []

    def test_load_invalid_yaml(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """Malformed YAML results in empty devices."""
        fleet_file = tmp_path / "fleet.yaml"
        fleet_file.write_text("invalid: yaml: {{broken")
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        assert registry.list_devices() == []

    def test_load_non_dict(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """YAML that parses to a list (not dict) results in empty devices."""
        fleet_file = tmp_path / "fleet.yaml"
        fleet_file.write_text("- item1\n- item2\n")
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        assert registry.list_devices() == []

    def test_load_missing_fleet_key(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """YAML dict without 'fleet' key results in empty devices."""
        fleet_file = tmp_path / "fleet.yaml"
        fleet_file.write_text(yaml.safe_dump({"other_key": "value"}))
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        assert registry.list_devices() == []

    def test_load_valid_devices(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """Proper YAML with devices populates the registry."""
        fleet_file = tmp_path / "fleet.yaml"
        fleet_file.write_text(
            yaml.safe_dump(
                {
                    "fleet": {
                        "devices": [
                            {
                                "serial": "ABC123",
                                "platform": "android",
                                "device_type": "phone",
                                "model": "Pixel 7",
                                "nickname": "pixel",
                            },
                            {
                                "serial": "DEF456",
                                "platform": "android",
                                "device_type": "watch",
                                "model": "Pixel Watch 2",
                            },
                        ]
                    }
                }
            )
        )
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        devices = registry.list_devices()
        assert len(devices) == 2

        pixel = registry.get("ABC123")
        assert pixel is not None
        assert pixel.model == "Pixel 7"
        assert pixel.nickname == "pixel"

        watch = registry.get("DEF456")
        assert watch is not None
        assert watch.model == "Pixel Watch 2"

    def test_load_invalid_device_entry(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """Invalid device entries are skipped; valid ones are loaded."""
        fleet_file = tmp_path / "fleet.yaml"
        fleet_file.write_text(
            yaml.safe_dump(
                {
                    "fleet": {
                        "devices": [
                            {
                                "serial": "GOOD1",
                                "platform": "android",
                                "model": "Valid Device",
                            },
                            "not a dict at all",
                            {
                                "serial": "GOOD2",
                                "platform": "ios",
                                "model": "iPhone 15",
                            },
                        ]
                    }
                }
            )
        )
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        devices = registry.list_devices()
        assert len(devices) == 2
        assert registry.get("GOOD1") is not None
        assert registry.get("GOOD2") is not None

    def test_load_fleet_devices_not_list(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """If fleet.devices is not a list, results in empty devices."""
        fleet_file = tmp_path / "fleet.yaml"
        fleet_file.write_text(yaml.safe_dump({"fleet": {"devices": "not_a_list"}}))
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        assert registry.list_devices() == []

    def test_load_fleet_not_dict(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """If fleet value is not a dict, results in empty devices."""
        fleet_file = tmp_path / "fleet.yaml"
        fleet_file.write_text(yaml.safe_dump({"fleet": "just_a_string"}))
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        assert registry.list_devices() == []

    def test_load_oversized_fleet_file_ignored_with_one_warning(
        self, tmp_path: Path, mock_nerve: MagicMock
    ) -> None:
        """[test] FleetRegistry ignores an oversized fleet.yaml with one warning."""
        import structlog.testing

        fleet_file = tmp_path / "fleet.yaml"
        header = b"fleet:\n  devices:\n    - serial: TOO_BIG\n"
        fleet_file.write_bytes(header + b"#" * (1024 * 1024 + 1 - len(header)))

        with structlog.testing.capture_logs() as logs:
            registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)

        assert registry.list_devices() == []
        warnings = [e for e in logs if e.get("event") == "fleet_file_read_failed"]
        assert len(warnings) == 1
        assert warnings[0]["path"] == str(fleet_file)
        assert "over the 1048576-byte cap" in warnings[0]["error"]


class TestFleetRegistryDiscover:
    def test_discover_unknown_device_type(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """Assert discover() handles unknown device_type without raising
        and registers valid devices.
        """
        mock_nerve.discover.return_value = [
            {
                "serial": "GOOD1",
                "platform": "android",
                "device_type": "phone",
                "model": "Valid Phone",
            },
            {
                "serial": "BAD1",
                "platform": "android",
                "device_type": "unknown",
                "model": "Unknown Type Device",
            },
        ]
        registry = FleetRegistry(fleet_path=tmp_path / "fleet.yaml", nerve_client=mock_nerve)
        new_devices = registry.discover()
        assert len(new_devices) == 2
        assert registry.get("GOOD1") is not None
        assert registry.get("GOOD1").device_type.value == "phone"
        assert registry.get("BAD1") is not None
        assert registry.get("BAD1").device_type.value == "phone"

    def test_discover_unknown_platform(self, tmp_path: Path, mock_nerve: MagicMock) -> None:
        """Assert discover() handles unknown platform without raising
        and registers valid devices.
        """
        mock_nerve.discover.return_value = [
            {
                "serial": "GOOD2",
                "platform": "android",
                "device_type": "phone",
                "model": "Valid Phone",
            },
            {
                "serial": "BAD2",
                "platform": "unknown_platform",
                "device_type": "phone",
                "model": "Unknown Platform Device",
            },
        ]
        registry = FleetRegistry(fleet_path=tmp_path / "fleet.yaml", nerve_client=mock_nerve)
        new_devices = registry.discover()
        assert len(new_devices) == 2
        assert registry.get("GOOD2") is not None
        assert registry.get("BAD2") is not None
        assert registry.get("BAD2").platform.value == "android"

    def test_discover_skips_personal_devices_and_never_persists_serials(
        self, tmp_path: Path, mock_nerve: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] personal devices in DevicePolicy are skipped during discover()
        and never written to fleet.yaml.
        """
        policy_file = tmp_path / "device-fleet.yaml"
        policy_file.write_text(
            yaml.safe_dump(
                {
                    "devices": [
                        {"serial": "R58N000TEST", "personal": True},
                        {"serial": "PERSONALWATCH01", "personal": True},
                        {"serial": "HWID000TEST", "personal": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(policy_file))

        mock_nerve.discover.return_value = [
            {
                "serial": "emulator-5554",
                "platform": "android",
                "device_type": "phone",
                "model": "Test Emulator",
            },
            {
                "serial": "R58N000TEST",
                "platform": "android",
                "device_type": "phone",
                "model": "Personal Phone",
            },
            {
                "serial": "192.0.2.10:5555",
                "id": "personalwatch01",
                "platform": "android",
                "device_type": "watch",
                "model": "Personal Watch",
            },
            {
                "serial": "192.0.2.11:5555",
                "hardware_id": "HWID000TEST",
                "platform": "android",
                "device_type": "phone",
                "model": "Personal HWID Phone",
            },
        ]

        fleet_file = tmp_path / "fleet.yaml"
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        new_devices = registry.discover()

        assert [d.serial for d in new_devices] == ["emulator-5554"]
        assert registry.last_discover_skipped_personal == 3
        assert registry.get("R58N000TEST") is None
        assert registry.get("192.0.2.10:5555") is None
        assert registry.get("192.0.2.11:5555") is None

        persisted = fleet_file.read_text(encoding="utf-8")
        for forbidden in (
            "R58N000TEST",
            "PERSONALWATCH01",
            "personalwatch01",
            "HWID000TEST",
            "192.0.2.10:5555",
            "192.0.2.11:5555",
        ):
            assert forbidden not in persisted

    def test_discover_only_personal_devices_writes_no_personal_serial_to_fleet_yaml(
        self, tmp_path: Path, mock_nerve: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] when only personal devices are discovered, grep of fleet.yaml is empty."""
        policy_file = tmp_path / "device-fleet.yaml"
        policy_file.write_text(
            yaml.safe_dump({"devices": [{"serial": "R58N000TEST", "personal": True}]}),
            encoding="utf-8",
        )
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(policy_file))

        mock_nerve.discover.return_value = [
            {
                "serial": "R58N000TEST",
                "platform": "android",
                "device_type": "phone",
                "model": "Personal Phone",
            }
        ]
        fleet_file = tmp_path / "fleet.yaml"
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)
        new_devices = registry.discover()

        assert new_devices == []
        assert registry.last_discover_skipped_personal == 1
        assert not fleet_file.exists() or "R58N000TEST" not in fleet_file.read_text(
            encoding="utf-8"
        )

    def test_discover_unreadable_policy_persists_nothing_and_logs_warning(
        self, tmp_path: Path, mock_nerve: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] unreadable policy persists nothing and emits
        fleet_discover_policy_unreadable.
        """
        import structlog.testing

        bad_policy = tmp_path / "device-fleet.yaml"
        bad_policy.write_text("devices: [unclosed", encoding="utf-8")
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(bad_policy))

        mock_nerve.discover.return_value = [
            {
                "serial": "emulator-5554",
                "platform": "android",
                "device_type": "phone",
                "model": "Test Emulator",
            }
        ]
        fleet_file = tmp_path / "fleet.yaml"
        registry = FleetRegistry(fleet_path=fleet_file, nerve_client=mock_nerve)

        with structlog.testing.capture_logs() as logs:
            new_devices = registry.discover()

        assert new_devices == []
        assert registry.list_devices() == []
        assert not fleet_file.exists()
        mock_nerve.discover.assert_not_called()
        warnings = [e for e in logs if e.get("event") == "fleet_discover_policy_unreadable"]
        assert len(warnings) == 1

    def test_cli_fleet_discover_prints_skipped_personal_count_without_serial(
        self, tmp_path: Path, mock_nerve: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] CLI prints 'skipped N personal device(s)' with no serial."""
        from click.testing import CliRunner

        from linc_crucible.cli import cli

        policy_file = tmp_path / "device-fleet.yaml"
        policy_file.write_text(
            yaml.safe_dump(
                {
                    "devices": [
                        {"serial": "R58N000TEST", "personal": True},
                        {"serial": "PERSONALWATCH01", "personal": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        fleet_file = tmp_path / "fleet.yaml"
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(policy_file))
        monkeypatch.setenv("CRUCIBLE_FLEET_FILE", str(fleet_file))

        mock_nerve.discover.return_value = [
            {
                "serial": "emulator-5554",
                "platform": "android",
                "device_type": "phone",
                "model": "Test Emulator",
            },
            {
                "serial": "R58N000TEST",
                "platform": "android",
                "device_type": "phone",
                "model": "Personal Phone",
            },
            {
                "serial": "PERSONALWATCH01",
                "platform": "android",
                "device_type": "watch",
                "model": "Personal Watch",
            },
        ]
        monkeypatch.setattr("linc_crucible.core.fleet_registry.NerveClient", lambda: mock_nerve)

        runner = CliRunner()
        result = runner.invoke(cli, ["fleet", "discover"])

        assert result.exit_code == 0
        assert "skipped 2 personal device(s)" in result.stdout
        assert "R58N000TEST" not in result.stdout
        assert "PERSONALWATCH01" not in result.stdout


class TestReturnMutationContracts:
    def test_assayer_run_a11y_tree_return_fields(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from linc_crucible.core import assayer as assayer_mod
        from linc_crucible.core.assayer import A11yTreeResult, Assayer

        forge = MagicMock()
        assayer_none = Assayer(forge_reader=forge, nerve_client=None)
        res_none = assayer_none._run_a11y_tree("proj", "")
        assert res_none.error == "nerve client not configured"
        assert res_none.output == "nerve client not configured"

        nerve = MagicMock()
        assayer = Assayer(forge_reader=forge, nerve_client=nerve)
        with patch.object(assayer, "_resolve_a11y_policy", return_value=([], None)):
            res_empty = assayer._run_a11y_tree("proj", "")
            assert res_empty.output == "no devices available"
            assert res_empty.attempted is True

        with (
            patch.object(
                assayer,
                "_resolve_a11y_policy",
                return_value=([{"id": "DEV1", "platform": "android"}], None),
            ),
            patch.object(assayer, "_capture_a11y_elements", return_value="tree error"),
            patch.object(assayer_mod, "A11yTreeResult", wraps=A11yTreeResult) as mock_a11y,
        ):
            res_str = assayer._run_a11y_tree("proj", "")
            assert res_str.error == "tree error"
            assert res_str.configured is True
            assert mock_a11y.call_args.kwargs.get("configured") is True

        with (
            patch.object(
                assayer,
                "_resolve_a11y_policy",
                return_value=([{"id": "DEV1", "platform": "android"}], None),
            ),
            patch.object(
                assayer,
                "_capture_a11y_elements",
                return_value=[{"text": "OK", "bounds": "[0,0][100,100]", "clickable": False}],
            ),
        ):
            res_ok = assayer._run_a11y_tree("proj", "")
            assert res_ok.error == ""
            assert res_ok.output == ""

        with patch.object(assayer, "_resolve_a11y_policy", side_effect=RuntimeError("boom")):
            res_exc = assayer._run_a11y_tree("proj", "")
            assert res_exc.output == "boom"

    def test_bvt_do_dismiss_dialog_return_fields(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from linc_crucible.core.bvt.runner import BVTRunner
        from linc_crucible.core.bvt.schema import BVTStep

        nerve = MagicMock()
        runner = BVTRunner(nerve=nerve, output_dir=tmp_path)
        step = BVTStep(action="dismiss_dialog", target="Allow", params={"timeout": "1s"})
        with (
            patch.object(
                runner,
                "_elements_or_harness",
                side_effect=[[{"text": "Allow"}], []],
            ),
            patch("linc_crucible.core.bvt.step_handlers.time.sleep"),
        ):
            res = runner._do_dismiss_dialog(step, "emulator-5554", "com.example.app", "test")
            assert res.action == "dismiss_dialog"
            assert res.passed is True

    def test_deployer_pour_forwards_launch_and_launch_intent(self) -> None:
        from unittest.mock import patch

        from linc_crucible.core.deployer import Deployer, PourResult

        forge = MagicMock()
        nerve = MagicMock()
        deployer = Deployer(forge_reader=forge, nerve_client=nerve)

        proj_multi = MagicMock()
        proj_multi.is_multi_target = True
        forge.get.return_value = proj_multi
        with patch.object(
            deployer, "_pour_multi_target", return_value=PourResult(project="p")
        ) as mock_multi:
            deployer._pour(
                project_name="p",
                device="emulator-5554",
                platform="android",
                launch=False,
                skip_build=True,
                module="",
                device_type="",
                launch_intent="myapp://open",
            )
            assert mock_multi.call_args.kwargs["launch"] is False
            assert mock_multi.call_args.kwargs["launch_intent"] == "myapp://open"

        proj_single = MagicMock()
        proj_single.is_multi_target = False
        forge.get.return_value = proj_single
        with patch.object(
            deployer, "_pour_single_module", return_value=PourResult(project="p")
        ) as mock_mod:
            deployer._pour(
                project_name="p",
                device="emulator-5554",
                platform="android",
                launch=False,
                skip_build=True,
                module="app",
                device_type="",
                launch_intent="myapp://open",
            )
            assert mock_mod.call_args.kwargs["launch"] is False
            assert mock_mod.call_args.kwargs["launch_intent"] == "myapp://open"

    def test_scorer_compute_and_load_score_return_fields(self, tmp_path: Path) -> None:
        import json
        from unittest.mock import patch

        from linc_crucible.core import scorer as scorer_mod
        from linc_crucible.core.scorer import QualityScore, QualityScorer

        scorer = QualityScorer(scores_dir=tmp_path)
        with patch.object(
            scorer_mod, "QualityScore", wraps=QualityScore
        ) as mock_qs:
            res = scorer.compute("proj", test_pass_rate=None, lint_issues=None)
            assert res.dimensions == []
            assert mock_qs.call_args.kwargs["dimensions"] == []

        score_file = tmp_path / "manual.json"
        score_file.write_text(
            json.dumps(
                {
                    "project": "proj",
                    "dimensions": [
                        {
                            "name": "lint",
                            "raw_value": 90.0,
                            "score": 90.0,
                            "weight": 1.0,
                            "weighted_score": 90.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        loaded = scorer._load_score(score_file)
        assert loaded is not None
        assert len(loaded.dimensions) == 1
        assert loaded.dimensions[0].name == "lint"
        assert loaded.timestamp == ""
        assert loaded.ship_gate is False

