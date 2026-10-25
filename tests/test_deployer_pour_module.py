"""Characterization tests for Deployer._pour_single_module and
Deployer._deploy_module_to_device (crucible#115).

Both functions were baked into .crap-baseline.json at their 0%-coverage
CRAP scores (_pour_single_module: complexity 14, score 210.0;
_deploy_module_to_device: complexity 6, score 42.0) on the assumption they
require real device hardware. In fact all device I/O routes through the
injected NerveClient and all builds through Deployer._run_build /
Deployer._build, so both are fully mockable — mirroring the fixture style
already established in tests/test_deployer.py.

Pure characterization: these tests assert today's literal error strings
and result fields. No src/ changes are made here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from linc_crucible.core.deployer import Deployer, DeployResult
from linc_crucible.core.forge_integration import ForgeProject, ForgeProjectReader, ModuleConfig
from linc_crucible.core.nerve_client import NerveClient

_PHONE_DICT = {
    "serial": "PIXEL8PRO",
    "platform": "android",
    "device_type": "phone",
    "model": "Pixel 8 Pro",
    "state": "online",
}

_IOS_SIM_DICT = {
    "serial": "IPHONE15-SIM",
    "platform": "ios",
    "device_type": "emulator",
    "model": "iPhone 15",
    "state": "online",
}


@pytest.fixture
def mock_nerve() -> MagicMock:
    nerve = MagicMock(spec=NerveClient)
    nerve.discover.return_value = [_PHONE_DICT]
    nerve.install.return_value = {"status": "installed"}
    nerve.launch.return_value = {"status": "launched"}
    return nerve


@pytest.fixture
def app_module() -> ModuleConfig:
    return ModuleConfig(name="app", device_types=["phone"])


@pytest.fixture
def watch_module() -> ModuleConfig:
    return ModuleConfig(name="watchApp", device_types=["watch"])


@pytest.fixture
def project(app_module: ModuleConfig) -> ForgeProject:
    return ForgeProject(
        name="TestProj",
        path="/tmp/TestProj",
        modules=[app_module],
    )


@pytest.fixture
def mock_forge(project: ForgeProject) -> MagicMock:
    forge = MagicMock(spec=ForgeProjectReader)
    forge.get.return_value = project
    forge.find_module_apks.return_value = {
        "app": "/tmp/TestProj/app/build/outputs/apk/debug/app-debug.apk"
    }
    forge.find_module_ios_apps.return_value = {}
    forge.find_apk.return_value = None
    forge.find_ios_app.return_value = None
    return forge


@pytest.fixture(name="deployer")
def make_deployer(mock_forge: MagicMock, mock_nerve: MagicMock) -> Deployer:
    return Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)



class TestPourSingleModule:
    def test_unknown_module_not_found(self, deployer: Deployer, project: ForgeProject) -> None:
        result = deployer._pour_single_module(
            project,
            module="nope",
            device="",
            platform="",
            launch=False,
            skip_build=False,
            device_type="",
        )

        assert result.build_output == "Module 'nope' not found in project 'TestProj'"
        assert result.results == []

    def test_build_failure_short_circuits(
        self, deployer: Deployer, project: ForgeProject, app_module: ModuleConfig
    ) -> None:
        app_module.build_command = "./gradlew :app:assembleDebug"

        with patch.object(deployer, "_run_build", return_value=(False, "boom")) as mock_run_build:
            result = deployer._pour_single_module(
                project,
                module="app",
                device="",
                platform="",
                launch=False,
                skip_build=False,
                device_type="",
            )

        mock_run_build.assert_called_once_with(app_module.build_command, project.path)
        assert result.results == []
        assert result.build_output == "boom"

    def test_no_matching_devices(
        self, deployer: Deployer, project: ForgeProject, mock_nerve: MagicMock
    ) -> None:
        mock_nerve.discover.return_value = []

        result = deployer._pour_single_module(
            project,
            module="app",
            device="",
            platform="",
            launch=False,
            skip_build=True,
            device_type="",
        )

        assert "No matching devices found for module." in result.build_output
        assert result.results == []

    def test_skip_build_never_calls_run_build(
        self, deployer: Deployer, project: ForgeProject, app_module: ModuleConfig
    ) -> None:
        app_module.build_command = "./gradlew :app:assembleDebug"

        with (
            patch.object(deployer, "_run_build") as mock_run_build,
            patch.object(deployer, "_build") as mock_build,
        ):
            deployer._pour_single_module(
                project,
                module="app",
                device="",
                platform="",
                launch=False,
                skip_build=True,
                device_type="",
            )

        mock_run_build.assert_not_called()
        mock_build.assert_not_called()


class TestDeployModuleToDevice:
    def test_android_happy_path(
        self,
        deployer: Deployer,
        project: ForgeProject,
        app_module: ModuleConfig,
        mock_nerve: MagicMock,
    ) -> None:
        apk_path = "/tmp/TestProj/app/build/outputs/apk/debug/app-debug.apk"

        result = deployer._deploy_module_to_device(
            _PHONE_DICT,
            app_module,
            project,
            {"app": apk_path},
            {},
            False,
            "",
        )

        mock_nerve.install.assert_called_once_with(_PHONE_DICT["serial"], apk_path)
        assert isinstance(result, DeployResult)
        assert result.module == "app"
        assert result.install_success is True
        assert result.artifact_path == apk_path
        assert result.error == ""

    def test_android_missing_apk(
        self, deployer: Deployer, project: ForgeProject, app_module: ModuleConfig
    ) -> None:
        result = deployer._deploy_module_to_device(
            _PHONE_DICT,
            app_module,
            project,
            {},
            {},
            False,
            "",
        )

        assert result.error == "No APK found for module 'app'"
        assert result.install_success is False

    def test_ios_missing_app(
        self, deployer: Deployer, project: ForgeProject, watch_module: ModuleConfig
    ) -> None:
        result = deployer._deploy_module_to_device(
            _IOS_SIM_DICT,
            watch_module,
            project,
            {},
            {},
            False,
            "",
        )

        assert result.error == "No iOS .app found for module 'watchApp'"

    def test_launch_triggers_after_install(
        self,
        deployer: Deployer,
        project: ForgeProject,
        app_module: ModuleConfig,
    ) -> None:
        apk_path = "/tmp/TestProj/app/build/outputs/apk/debug/app-debug.apk"

        with patch.object(deployer, "_launch_android") as mock_launch:
            result = deployer._deploy_module_to_device(
                _PHONE_DICT,
                app_module,
                project,
                {"app": apk_path},
                {},
                True,
                "",
            )

        mock_launch.assert_called_once_with(project, _PHONE_DICT["serial"], result, "")
