"""Tests for deployer orchestrator."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog
import yaml

from linc_crucible.core.deployer import (
    _BUILD_ONLY_NO_COMMAND_MSG,
    _NO_BUILD_COMMAND_MSG,
    Deployer,
    DeployResult,
    PourResult,
    _aggregate_staleness,
    _append_lines,
    _artifact_mtime,
    _artifact_provenance,
    _attach_provenance,
    _build_only_output,
    _build_only_platforms,
    _device_section,
    _finalize_pour_provenance,
    _head_commit_iso,
    _is_older,
    _parse_commit_time,
    _pour_succeeded,
    _provenance_fields,
    _stale_artifact_warning,
    artifacts_older_than_head,
)
from linc_crucible.core.device import Platform
from linc_crucible.core.forge_integration import ForgeProject, ForgeProjectReader, ModuleConfig
from linc_crucible.core.nerve_client import NerveClient
from tests.test_subprocess_utils import bridge_safe_run_to_subprocess_run


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


@pytest.fixture(autouse=True)
def _safe_run_via_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module fakes builds with @patch("subprocess.run"); safe_subprocess_run now
    spawns via Popen (linc-crucible-64q.5), so bridge its spawn back to that seam."""
    bridge_safe_run_to_subprocess_run(monkeypatch)


@pytest.fixture(autouse=True)
def _stub_resolve_java_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub resolve_java_home across this module so @patch("subprocess.run") tests do
    not double-count /usr/libexec/java_home on darwin (REVIEW B9)."""
    monkeypatch.setattr("linc_crucible.core.deployer.resolve_java_home", lambda: None)


# Canonical device dicts returned by NerveClient.discover()
_EMULATOR_DICT = {
    "serial": "emulator-5554",
    "platform": "android",
    "device_type": "emulator",
    "model": "sdk_gphone64",
    "state": "online",
}

_PHONE_DICT = {
    "serial": "PIXEL8PRO",
    "platform": "android",
    "device_type": "phone",
    "model": "Pixel 8 Pro",
    "state": "online",
}

_WATCH_DICT = {
    "serial": "PIXELWATCH2",
    "platform": "android",
    "device_type": "watch",
    "model": "Pixel Watch 2",
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
def mock_forge() -> MagicMock:
    forge = MagicMock(spec=ForgeProjectReader)
    forge.get.return_value = ForgeProject(
        name="PrecisionLevel",
        path="/tmp/PrecisionLevel",
        package="com.example.level",
        build_command="./gradlew assembleDebug",
        framework="gradle_kmp",
    )
    forge.find_apk.return_value = "/tmp/PrecisionLevel/app/build/outputs/apk/debug/app-debug.apk"
    forge.find_ios_app.return_value = None
    return forge


@pytest.fixture
def mock_nerve() -> MagicMock:
    nerve = MagicMock(spec=NerveClient)
    nerve.discover.return_value = [_EMULATOR_DICT]
    nerve.install.return_value = {"status": "installed"}
    nerve.launch.return_value = {"status": "launched"}
    return nerve


class TestDeployerPour:
    @patch("subprocess.run")
    def test_pour_project_not_found(
        self,
        mock_subprocess: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = None

        deployer = Deployer(
            forge_reader=forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("NonExistent")

        assert not result.success
        assert "not found" in result.build_output

    @patch("subprocess.run")
    def test_pour_build_fails(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(returncode=1, stdout="", stderr="BUILD FAILED")

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel")

        assert not result.success
        assert "BUILD FAILED" in result.build_output

    @patch("subprocess.run")
    def test_pour_skip_build(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel", skip_build=True)

        # subprocess not called for build
        mock_subprocess.assert_not_called()
        assert len(result.results) == 1

    @patch("subprocess.run")
    def test_pour_android_success(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="BUILD SUCCESSFUL", stderr=""
        )

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel")

        assert len(result.results) == 1
        assert result.results[0].install_success is True
        mock_nerve.install.assert_called_once()

    @patch("subprocess.run")
    def test_pour_with_launch(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="BUILD SUCCESSFUL", stderr=""
        )

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel", launch=True)

        assert result.results[0].install_success is True
        assert result.results[0].launch_success is True
        mock_nerve.launch.assert_called_once()

    @patch("subprocess.run")
    def test_pour_specific_device(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        mock_nerve.discover.return_value = [_EMULATOR_DICT]

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel", device="emulator-5554")

        assert len(result.results) == 1
        assert result.results[0].device_serial == "emulator-5554"

    @patch("subprocess.run")
    def test_pour_no_devices(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        mock_nerve.discover.return_value = []

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel", device="nonexistent")

        assert not result.success
        assert len(result.results) == 0

    @patch("subprocess.run")
    def test_pour_no_apk(
        self,
        mock_subprocess: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="NoApk", path="/tmp/noapk", build_command="build"
        )
        forge.find_apk.return_value = None

        deployer = Deployer(
            forge_reader=forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("NoApk")

        assert result.results[0].install_success is False
        assert "No APK" in result.results[0].error


class TestDeployerPourBuildOnly:
    """Acceptance tests for Deployer.pour(build_only=True)."""

    @patch("subprocess.run")
    def test_build_only_no_devices_succeeds(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """[test] Deployer.pour(project, build_only=True) with nerve discover returning
        [] and a fake build command that succeeds -> success True, build_success
        True, results == [], _resolve_targets never called (fails pre-fix: "No
        matching devices found", success False, no build).
        """
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="BUILD SUCCESSFUL", stderr=""
        )
        mock_nerve.discover.return_value = []

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        with patch.object(
            deployer, "_resolve_targets", wraps=deployer._resolve_targets
        ) as spy_resolve:
            result = deployer.pour("PrecisionLevel", build_only=True)

            assert result.success is True
            assert result.build_success is True
            assert result.results == []
            spy_resolve.assert_not_called()
            mock_subprocess.assert_called_once()
            assert mock_subprocess.call_args_list[0].args[0] == [
                "./gradlew",
                "assembleDebug",
            ]

    @patch("subprocess.run")
    def test_build_only_online_devices_not_deployed(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """[test] build_only with two online devices -> _deploy_to_device never called
        (fails pre-fix: installs on both).
        """
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="BUILD SUCCESSFUL", stderr=""
        )
        mock_nerve.discover.return_value = [_PHONE_DICT, _WATCH_DICT]

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        with patch.object(
            deployer, "_deploy_to_device", wraps=deployer._deploy_to_device
        ) as spy_deploy:
            result = deployer.pour("PrecisionLevel", build_only=True)

            assert result.success is True
            assert result.build_success is True
            assert result.results == []
            spy_deploy.assert_not_called()
            mock_nerve.install.assert_not_called()

    @patch("subprocess.run")
    def test_build_only_failing_build(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """[test] build_only with a failing build -> success False and the build output in
        the result.
        """
        mock_subprocess.return_value = MagicMock(
            returncode=1, stdout="", stderr="BUILD FAILED: compilation error"
        )
        mock_nerve.discover.return_value = []

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel", build_only=True)

        assert result.success is False
        assert result.build_success is False
        assert "BUILD FAILED: compilation error" in result.build_output


class TestDeployerReservationGuard:
    """Issue #110 Part 1 — refuse to install onto a device another nerve session holds."""

    @patch("subprocess.run")
    def test_pour_reserved_device_blocks_install(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="BUILD SUCCESSFUL", stderr=""
        )
        mock_nerve.pool.return_value = {
            "reservations": [
                {
                    "device_serial": "emulator-5554",
                    "session_id": "other-sess",
                    "purpose": "bench-grid",
                }
            ],
            "count": 1,
        }

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel")

        assert len(result.results) == 1
        assert result.results[0].install_success is False
        assert "reserved by session" in result.results[0].error
        assert "other-sess" in result.results[0].error
        mock_nerve.install.assert_not_called()

    @patch("subprocess.run")
    def test_pour_unreserved_device_installs_normally(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="BUILD SUCCESSFUL", stderr=""
        )
        mock_nerve.pool.return_value = {"reservations": [], "count": 0}

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel")

        assert result.results[0].install_success is True
        mock_nerve.install.assert_called_once()

    @patch("subprocess.run")
    def test_pour_pool_connection_error_degrades_gracefully(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """nerve_pool unreachable must not block a deploy that would otherwise succeed."""
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="BUILD SUCCESSFUL", stderr=""
        )
        mock_nerve.pool.side_effect = ConnectionError("nerve unreachable")

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel")

        assert result.results[0].install_success is True
        mock_nerve.install.assert_called_once()

    @patch("subprocess.run")
    def test_pour_reserved_ios_device_blocks_install(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """iOS twin of test_pour_reserved_device_blocks_install — the reservation
        guard lives in _deploy_ios_artifact too, so it needs its own coverage
        rather than riding on the Android branch (issue #110 repair)."""
        mock_forge.find_ios_app.return_value = "/tmp/PrecisionLevel/build/PrecisionLevel.app"
        mock_nerve.discover.return_value = [_IOS_SIM_DICT]
        mock_nerve.pool.return_value = {
            "reservations": [
                {
                    "device_serial": "IPHONE15-SIM",
                    "session_id": "other-sess",
                    "purpose": "bench-grid",
                }
            ],
            "count": 1,
        }

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel", skip_build=True)

        assert len(result.results) == 1
        assert result.results[0].install_success is False
        assert "reserved by session" in result.results[0].error
        assert "other-sess" in result.results[0].error
        mock_nerve.install.assert_not_called()


class TestRunBuildShellMetacharRejection:
    """A metachar-containing build_command degrades to a failed build, not a crash."""

    def test_run_build_rejects_metachar_command(
        self, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        """A compound build_command returns (False, actionable error), no shell run."""
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        success, output = deployer._run_build(
            "./gradlew assembleDebug && ./publish.sh", cwd="/tmp"
        )

        assert success is False
        assert "metacharacters" in output


class TestRunBuildGate:
    """_run_build serializes through linc-build-gate only when LINC_BUILD_GATE=1."""

    def test_run_build_prefixes_build_gate_when_enabled(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        seen = {}

        def fake_run(command, *, cwd, timeout, env=None):
            seen["command"] = command

            class R:  # minimal CompletedProcess stand-in
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", fake_run)
        monkeypatch.setenv("LINC_BUILD_GATE", "1")
        deployer._run_build("./gradlew test", cwd="/tmp")
        assert seen["command"].startswith("linc-build-gate ")
        assert seen["command"].endswith("-- ./gradlew test")

    def test_run_build_unchanged_when_disabled(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        seen = {}

        def fake_run(command, *, cwd, timeout, env=None):
            seen["command"] = command

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", fake_run)
        monkeypatch.delenv("LINC_BUILD_GATE", raising=False)
        deployer._run_build("./gradlew test", cwd="/tmp")
        assert seen["command"] == "./gradlew test"


class TestRunBuildJavaHomeInjection:
    """_run_build injects a resolved JAVA_HOME so launchd-spawned builds find the JDK.

    launchd strips the interactive-shell PATH/JAVA_HOME (#76), so builds spawned by
    the gateway daemon can't find a keg-only Homebrew JDK that works fine in a
    terminal. Precedence: caller/pre-existing env JAVA_HOME > resolved JAVA_HOME.
    """

    def _capture_env(self, monkeypatch, mock_forge, mock_nerve) -> dict:
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        seen: dict = {}

        def fake_run(command, *, cwd, timeout, env=None):
            seen["env"] = env

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", fake_run)
        monkeypatch.delenv("LINC_BUILD_GATE", raising=False)
        deployer._run_build("./gradlew test", cwd="/tmp")
        return seen

    def test_injects_resolved_java_home_when_absent(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        monkeypatch.delenv("JAVA_HOME", raising=False)
        monkeypatch.setattr(
            "linc_crucible.core.deployer.resolve_java_home", lambda: "/resolved/jdk-17"
        )
        seen = self._capture_env(monkeypatch, mock_forge, mock_nerve)
        assert seen["env"]["JAVA_HOME"] == "/resolved/jdk-17"

    def test_does_not_override_preexisting_java_home(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        monkeypatch.setenv("JAVA_HOME", "/already/set/jdk")
        monkeypatch.setattr(
            "linc_crucible.core.deployer.resolve_java_home", lambda: "/resolved/jdk-17"
        )
        seen = self._capture_env(monkeypatch, mock_forge, mock_nerve)
        assert seen["env"]["JAVA_HOME"] == "/already/set/jdk"

    def test_no_java_home_key_when_resolution_fails_and_none_preexisting(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        monkeypatch.delenv("JAVA_HOME", raising=False)
        monkeypatch.setattr("linc_crucible.core.deployer.resolve_java_home", lambda: None)
        seen = self._capture_env(monkeypatch, mock_forge, mock_nerve)
        assert "JAVA_HOME" not in seen["env"]

    def test_logs_structured_event_before_build_when_resolution_fails(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        import structlog

        monkeypatch.delenv("JAVA_HOME", raising=False)
        monkeypatch.setattr("linc_crucible.core.deployer.resolve_java_home", lambda: None)

        def fake_run(command, *, cwd, timeout, env=None):
            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", fake_run)
        monkeypatch.delenv("LINC_BUILD_GATE", raising=False)
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        with structlog.testing.capture_logs() as captured:
            deployer._run_build("./gradlew test", cwd="/tmp")

        events = [c["event"] for c in captured]
        assert "java_home_resolution_failed" in events

    def test_does_not_log_when_resolution_succeeds(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        import structlog

        monkeypatch.delenv("JAVA_HOME", raising=False)
        monkeypatch.setattr(
            "linc_crucible.core.deployer.resolve_java_home", lambda: "/resolved/jdk-17"
        )

        def fake_run(command, *, cwd, timeout, env=None):
            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", fake_run)
        monkeypatch.delenv("LINC_BUILD_GATE", raising=False)
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        with structlog.testing.capture_logs() as captured:
            deployer._run_build("./gradlew test", cwd="/tmp")

        events = [c["event"] for c in captured]
        assert "java_home_resolution_failed" not in events


class TestBuildTimeoutConfig:
    """Build timeout is configurable via LINC_CRUCIBLE_BUILD_TIMEOUT, default 600s."""

    def _capture_timeout(self, monkeypatch, mock_forge, mock_nerve) -> dict:
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        seen: dict = {}

        def fake_run(command, *, cwd, timeout, env=None):
            seen["timeout"] = timeout

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", fake_run)
        monkeypatch.delenv("LINC_BUILD_GATE", raising=False)
        deployer._run_build("./gradlew test", cwd="/tmp")
        return seen

    def test_default_timeout_is_600(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        monkeypatch.delenv("LINC_CRUCIBLE_BUILD_TIMEOUT", raising=False)
        seen = self._capture_timeout(monkeypatch, mock_forge, mock_nerve)
        assert seen["timeout"] == 600

    def test_env_var_overrides_timeout(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        monkeypatch.setenv("LINC_CRUCIBLE_BUILD_TIMEOUT", "1800")
        seen = self._capture_timeout(monkeypatch, mock_forge, mock_nerve)
        assert seen["timeout"] == 1800

    @pytest.mark.parametrize("bad", ["banana", "0", "-5", ""])
    def test_invalid_env_var_falls_back_to_default(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock, bad: str
    ) -> None:
        monkeypatch.setenv("LINC_CRUCIBLE_BUILD_TIMEOUT", bad)
        seen = self._capture_timeout(monkeypatch, mock_forge, mock_nerve)
        assert seen["timeout"] == 600


class TestBuildTimeoutPayload:
    """A timed-out build is machine-distinguishable from a compile failure."""

    def _timeout_run(self, command, *, cwd, timeout, env=None):
        import subprocess as _sp

        raise _sp.TimeoutExpired(cmd=command, timeout=timeout)

    def test_timed_out_pour_payload_has_build_timeout_reason(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", self._timeout_run)
        monkeypatch.delenv("LINC_BUILD_GATE", raising=False)
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("PrecisionLevel")
        d = result.to_dict()

        assert d["success"] is False
        assert d["build_error"] is True
        assert d["reason"] == "build-timeout"

    def test_timeout_message_reflects_configured_timeout(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        monkeypatch.setenv("LINC_CRUCIBLE_BUILD_TIMEOUT", "1234")
        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", self._timeout_run)
        monkeypatch.delenv("LINC_BUILD_GATE", raising=False)
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("PrecisionLevel")

        assert "1234" in result.build_output

    @patch("subprocess.run")
    def test_compile_failure_payload_has_no_timeout_reason(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        mock_subprocess.return_value = MagicMock(
            returncode=1, stdout="", stderr="e: Foo.kt:1 unresolved reference"
        )
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("PrecisionLevel")
        d = result.to_dict()

        assert d["success"] is False
        assert d["build_error"] is True
        assert "reason" not in d
        assert "unresolved reference" in result.build_output

    def test_timeout_flag_resets_between_pours(
        self, monkeypatch, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        """A timeout in one pour must not leak into the next pour's payload."""
        monkeypatch.delenv("LINC_BUILD_GATE", raising=False)
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", self._timeout_run)
        first = deployer.pour("PrecisionLevel")
        assert first.to_dict()["reason"] == "build-timeout"

        def ok_run(command, *, cwd, timeout, env=None):
            class R:
                returncode = 0
                stdout = "BUILD SUCCESSFUL"
                stderr = ""

            return R()

        monkeypatch.setattr("linc_crucible.core.deployer.safe_subprocess_run", ok_run)
        second = deployer.pour("PrecisionLevel")
        assert "reason" not in second.to_dict()


class TestDeployResult:
    def test_to_dict(self) -> None:
        r = DeployResult(
            device_serial="ABC",
            device_model="Pixel",
            platform="android",
            install_success=True,
            launch_success=True,
        )
        d = r.to_dict()
        assert d["device_serial"] == "ABC"
        assert d["install_success"] is True


class TestPourResult:
    def test_success_true(self) -> None:
        r = PourResult(
            project="Test",
            results=[
                DeployResult(device_serial="A", install_success=True),
                DeployResult(device_serial="B", install_success=False),
            ],
        )
        assert r.success is True

    def test_success_false(self) -> None:
        r = PourResult(
            project="Test",
            results=[
                DeployResult(device_serial="A", install_success=False),
            ],
        )
        assert r.success is False

    def test_success_empty(self) -> None:
        r = PourResult(project="Test")
        assert r.success is False

    def test_to_dict(self) -> None:
        r = PourResult(
            project="Test",
            results=[
                DeployResult(device_serial="A", install_success=True),
            ],
        )
        d = r.to_dict()
        assert d["project"] == "Test"
        assert d["success"] is True
        assert len(d["devices"]) == 1

    def test_to_dict_with_artifacts(self) -> None:
        r = PourResult(
            project="Test",
            results=[DeployResult(device_serial="A", install_success=True)],
            artifacts={"android": "/tmp/app-debug.apk", "ios": "/tmp/MyApp.app"},
        )
        d = r.to_dict()
        assert d["artifacts"] == {"android": "/tmp/app-debug.apk", "ios": "/tmp/MyApp.app"}

    def test_to_dict_without_artifacts_omits_key(self) -> None:
        r = PourResult(
            project="Test",
            results=[DeployResult(device_serial="A", install_success=True)],
        )
        d = r.to_dict()
        assert "artifacts" not in d

    def test_pin_normal_pour_success_is_any_install_success(self) -> None:
        """[test] pin: a normal pour's success is still any(install_success)."""
        r_success = PourResult(
            project="Test",
            results=[
                DeployResult(device_serial="A", install_success=False),
                DeployResult(device_serial="B", install_success=True),
            ],
        )
        assert r_success.success is True
        assert _pour_succeeded(r_success) is True

        r_failure = PourResult(
            project="Test",
            results=[
                DeployResult(device_serial="A", install_success=False),
                DeployResult(device_serial="B", install_success=False),
            ],
        )
        assert r_failure.success is False
        assert _pour_succeeded(r_failure) is False

    def test_build_only_success_reflects_build_success(self) -> None:
        """build-only success reflects build_success."""
        r_ok = PourResult(project="Test", build_only=True, build_success=True)
        assert r_ok.success is True
        assert _pour_succeeded(r_ok) is True

        r_bad = PourResult(project="Test", build_only=True, build_success=False)
        assert r_bad.success is False
        assert _pour_succeeded(r_bad) is False


class TestPourSucceededBoundary:
    """_pour_succeeded keys on build_success for build-only, install_success otherwise."""

    def test_build_only_ignores_install_success(self) -> None:
        r = PourResult(
            project="T",
            build_only=True,
            build_success=False,
            results=[DeployResult(device_serial="A", install_success=True)],
        )
        assert _pour_succeeded(r) is False

    def test_build_only_success_with_no_results(self) -> None:
        r = PourResult(project="T", build_only=True, build_success=True, results=[])
        assert _pour_succeeded(r) is True

    def test_normal_pour_ignores_build_success(self) -> None:
        r = PourResult(project="T", build_only=False, build_success=True, results=[])
        assert _pour_succeeded(r) is False


class TestDeviceSection:
    """_device_section owns the build-only vs devices payload split."""

    def test_build_only_section(self) -> None:
        r = PourResult(
            project="T",
            build_only=True,
            results=[DeployResult(device_serial="A", install_success=True)],
        )
        assert _device_section(r) == {"build_only": True}

    def test_normal_section_lists_devices(self) -> None:
        dr = DeployResult(device_serial="A", install_success=True)
        r = PourResult(project="T", results=[dr])
        assert _device_section(r) == {"devices": [dr.to_dict()]}

    def test_normal_section_empty_devices(self) -> None:
        assert _device_section(PourResult(project="T")) == {"devices": []}


class TestPourResultToDictExact:
    """Exact payload shapes for every to_dict branch."""

    def test_build_only_success_exact(self) -> None:
        r = PourResult(
            project="P",
            build_only=True,
            build_success=True,
            build_output="BUILD SUCCESSFUL",
            artifacts={"android": "/a.apk"},
        )
        assert r.to_dict() == {
            "project": "P",
            "success": True,
            "build_only": True,
            "artifacts": {"android": "/a.apk"},
        }

    def test_build_only_failure_exact(self) -> None:
        r = PourResult(
            project="P",
            build_only=True,
            build_success=False,
            build_output="  BUILD FAILED  \n",
        )
        assert r.to_dict() == {
            "project": "P",
            "success": False,
            "build_only": True,
            "error": "BUILD FAILED",
            "build_error": True,
        }

    def test_build_only_timeout_exact(self) -> None:
        r = PourResult(
            project="P",
            build_only=True,
            build_success=False,
            build_output="Build timed out after 600s",
            timed_out=True,
        )
        assert r.to_dict() == {
            "project": "P",
            "success": False,
            "build_only": True,
            "error": "Build timed out after 600s",
            "build_error": True,
            "reason": "build-timeout",
        }

    def test_normal_success_exact(self) -> None:
        dr = DeployResult(device_serial="A", install_success=True)
        r = PourResult(project="P", results=[dr])
        assert r.to_dict() == {
            "project": "P",
            "success": True,
            "devices": [dr.to_dict()],
        }

    def test_normal_failure_with_device_error_exact(self) -> None:
        dr = DeployResult(device_serial="A", install_success=False, error="boom")
        r = PourResult(project="P", results=[dr], build_output="ignored")
        assert r.to_dict() == {
            "project": "P",
            "success": False,
            "devices": [dr.to_dict()],
            "error": "boom",
        }

    def test_normal_failure_without_error_exact(self) -> None:
        r = PourResult(project="P")
        assert r.to_dict() == {"project": "P", "success": False, "devices": []}


class TestBuildOnlyPlatforms:
    """_build_only_platforms picks what to build when no devices are resolved."""

    def _project(self, build: str = "", ios: str = "") -> ForgeProject:
        return ForgeProject(name="P", path="/tmp/P", build_command=build, ios_build_command=ios)

    def test_explicit_android(self) -> None:
        proj = self._project(build="g", ios="x")
        assert _build_only_platforms(proj, "android") == {Platform.ANDROID}

    def test_explicit_ios_case_insensitive(self) -> None:
        proj = self._project(build="g", ios="x")
        assert _build_only_platforms(proj, "IOS") == {Platform.IOS}

    def test_unknown_platform_falls_back_to_project_commands(self) -> None:
        proj = self._project(ios="x")
        assert _build_only_platforms(proj, "windows") == {Platform.IOS}

    def test_both_commands(self) -> None:
        proj = self._project(build="g", ios="x")
        assert _build_only_platforms(proj, "") == {Platform.ANDROID, Platform.IOS}

    def test_android_command_only(self) -> None:
        proj = self._project(build="g")
        assert _build_only_platforms(proj, "") == {Platform.ANDROID}

    def test_no_commands_defaults_android(self) -> None:
        assert _build_only_platforms(self._project(), "") == {Platform.ANDROID}


class TestPourBuildOnlyPayload:
    """Exact build-only result content from Deployer.pour."""

    @patch("subprocess.run")
    def test_success_carries_exact_output_and_artifacts(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="BUILD SUCCESSFUL", stderr=""
        )
        mock_forge.find_ios_app.return_value = "/tmp/PrecisionLevel/My.app"
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("PrecisionLevel", build_only=True)

        assert result.build_only is True
        assert result.build_output == "BUILD SUCCESSFUL"
        assert result.artifacts == {
            "android": "/tmp/PrecisionLevel/app/build/outputs/apk/debug/app-debug.apk",
            "ios": "/tmp/PrecisionLevel/My.app",
        }
        mock_nerve.discover.assert_not_called()

    @patch("subprocess.run")
    def test_no_artifacts_found_leaves_empty(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        mock_forge.find_apk.return_value = None
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("PrecisionLevel", build_only=True)

        assert result.success is True
        assert result.artifacts == {}

    @patch("subprocess.run")
    def test_failure_skips_artifact_lookup(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        mock_subprocess.return_value = MagicMock(returncode=1, stdout="out:", stderr="err")
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("PrecisionLevel", build_only=True)

        assert result.build_output == "out:err"
        assert result.artifacts == {}
        mock_forge.find_apk.assert_not_called()
        mock_forge.find_ios_app.assert_not_called()
        assert result.to_dict() == {
            "project": "PrecisionLevel",
            "success": False,
            "build_only": True,
            "error": "out:err",
            "build_error": True,
        }

    @patch("subprocess.run")
    def test_platform_ios_builds_only_ios(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        mock_forge.get.return_value = ForgeProject(
            name="PrecisionLevel",
            path="/tmp/PrecisionLevel",
            build_command="./gradlew assembleDebug",
            ios_build_command="xcodebuild -scheme App build",
        )
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        with patch.object(
            deployer, "_build_for_platforms", return_value=(True, "OK")
        ) as spy_build:
            deployer.pour("PrecisionLevel", platform="ios", build_only=True)

        spy_build.assert_called_once()
        assert spy_build.call_args.args[1] == {Platform.IOS}
        assert spy_build.call_args.kwargs == {"targets": None}


def _build_only_deployer(project: ForgeProject, devices: list[dict]) -> Deployer:
    """Deployer over *project* with *devices* online and a fake artifact finder."""
    forge = MagicMock(spec=ForgeProjectReader)
    forge.get.return_value = project
    forge.find_apk.return_value = f"{project.path}/app-debug.apk"
    forge.find_ios_app.return_value = None
    nerve = MagicMock(spec=NerveClient)
    nerve.discover.return_value = devices
    nerve.install.return_value = {"status": "installed"}
    return Deployer(forge_reader=forge, nerve_client=nerve)


class TestBuildOnlyRouting:
    """build_only is dispatched before multi-target/module routing (audit F5 bounce)."""

    def test_multi_target_build_only_never_deploys(self) -> None:
        """A project with modules and two online devices builds and installs nowhere."""
        deployer = _build_only_deployer(
            _multi_target_project(), [_phone_device(), _watch_device()]
        )
        with (
            patch.object(deployer, "_run_build", return_value=(True, "BUILD OK")) as run,
            patch.object(deployer, "_deploy_to_device") as deploy,
            patch.object(deployer, "_deploy_module_to_target") as deploy_target,
            patch.object(deployer, "_deploy_module_to_device") as deploy_device,
        ):
            result = deployer.pour("SniperPulseApp", build_only=True)

        deploy.assert_not_called()
        deploy_target.assert_not_called()
        deploy_device.assert_not_called()
        deployer._nerve.install.assert_not_called()
        deployer._nerve.discover.assert_not_called()
        assert [c.args for c in run.call_args_list] == [
            ("./gradlew :composeApp:assembleDebug", "/tmp/SniperPulseApp"),
            ("./gradlew :watchApp:assembleDebug", "/tmp/SniperPulseApp"),
        ]
        assert result.to_dict() == {
            "project": "SniperPulseApp",
            "success": True,
            "build_only": True,
            "artifacts": {"android": "/tmp/SniperPulseApp/app-debug.apk"},
        }

    def test_multi_target_build_only_failure(self) -> None:
        """A failed module build is a failed build-only pour with no artifact lookup."""
        deployer = _build_only_deployer(_multi_target_project(), [_phone_device()])
        with patch.object(deployer, "_run_build", return_value=(False, "e: boom")):
            result = deployer.pour("SniperPulseApp", build_only=True)

        deployer._forge.find_apk.assert_not_called()
        assert result.to_dict() == {
            "project": "SniperPulseApp",
            "success": False,
            "build_only": True,
            "error": "[composeApp] e: boom",
            "build_error": True,
        }

    def test_module_build_only_runs_module_command_only(self) -> None:
        """module= with build_only runs that module's build and never resolves targets."""
        deployer = _build_only_deployer(
            _multi_target_project(), [_phone_device(), _watch_device()]
        )
        with (
            patch.object(deployer, "_run_build", return_value=(True, "OK")) as run,
            patch.object(deployer, "_resolve_targets_for_module") as resolve,
            patch.object(deployer, "_deploy_module_to_device") as deploy_device,
        ):
            result = deployer.pour("SniperPulseApp", module="watchApp", build_only=True)

        run.assert_called_once_with("./gradlew :watchApp:assembleDebug", "/tmp/SniperPulseApp")
        resolve.assert_not_called()
        deploy_device.assert_not_called()
        deployer._nerve.install.assert_not_called()
        assert result.to_dict() == {
            "project": "SniperPulseApp",
            "success": True,
            "build_only": True,
            "artifacts": {"android": "/tmp/SniperPulseApp/app-debug.apk"},
        }

    def test_module_build_only_build_type_reaches_build(self) -> None:
        """An auto-derived module command honours build_type under build_only."""
        project = _multi_target_project()
        project.modules[1].build_command = ""
        deployer = _build_only_deployer(project, [])
        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run:
            deployer.pour(
                "SniperPulseApp",
                module="watchApp",
                platform="android",
                build_type="release",
                build_only=True,
            )

        run.assert_called_once_with("./gradlew :watchApp:assembleRelease", "/tmp/SniperPulseApp")

    def test_module_build_only_without_module_command_uses_project_build(self) -> None:
        """No module command and no android narrowing falls back to the project build."""
        project = _multi_target_project()
        project.modules[1].build_command = ""
        deployer = _build_only_deployer(project, [])
        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run:
            deployer.pour("SniperPulseApp", module="watchApp", build_only=True)

        run.assert_called_once_with("./gradlew build", "/tmp/SniperPulseApp")

    def test_module_build_only_unknown_module(self) -> None:
        """An unknown module is a failed build-only pour that builds nothing."""
        deployer = _build_only_deployer(_multi_target_project(), [_phone_device()])
        with patch.object(deployer, "_run_build") as run:
            result = deployer.pour("SniperPulseApp", module="nope", build_only=True)

        run.assert_not_called()
        assert result.to_dict() == {
            "project": "SniperPulseApp",
            "success": False,
            "build_only": True,
            "error": "Module 'nope' not found in project 'SniperPulseApp'",
            "build_error": True,
        }

    def test_build_only_project_not_found(self) -> None:
        """An unknown project under build_only keeps the build-only payload shape."""
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = None
        deployer = Deployer(forge_reader=forge, nerve_client=MagicMock(spec=NerveClient))

        result = deployer.pour("Ghost", build_only=True)

        assert result.to_dict() == {
            "project": "Ghost",
            "success": False,
            "build_only": True,
            "error": "Project 'Ghost' not found in forge registry",
            "build_error": True,
        }

    def test_build_only_without_build_command_has_build_only_message(self) -> None:
        """No build command: build_only explains itself instead of suggesting skip_build."""
        project = ForgeProject(name="Bare", path="/tmp/Bare", build_command="")
        deployer = _build_only_deployer(project, [])

        result = deployer.pour("Bare", build_only=True)

        assert result.success is False
        assert "skip_build" not in result.build_output
        assert result.build_output == _BUILD_ONLY_NO_COMMAND_MSG

    def test_build_only_ignores_skip_build(self) -> None:
        """build_only always builds — skip_build has nothing to skip to."""
        deployer = _build_only_deployer(
            ForgeProject(name="P", path="/tmp/P", build_command="./gradlew assembleDebug"), []
        )
        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run:
            result = deployer.pour("P", skip_build=True, build_only=True)

        run.assert_called_once_with("./gradlew assembleDebug", "/tmp/P")
        assert result.success is True

    def test_build_only_looks_up_artifacts_by_project_name(self) -> None:
        """Artifact lookup is keyed on the resolved project's name."""
        deployer = _build_only_deployer(
            ForgeProject(name="P", path="/tmp/P", build_command="./gradlew assembleDebug"), []
        )
        with patch.object(deployer, "_run_build", return_value=(True, "OK")):
            deployer.pour("P", build_only=True)

        deployer._forge.find_apk.assert_called_once_with("P")
        deployer._forge.find_ios_app.assert_called_once_with("P")

    def test_default_pour_is_not_build_only(self) -> None:
        """pour() without build_only resolves targets and deploys (default stays False)."""
        deployer = _build_only_deployer(
            ForgeProject(name="P", path="/tmp/P", build_command="./gradlew assembleDebug"),
            [_EMULATOR_DICT],
        )
        with patch.object(deployer, "_run_build", return_value=(True, "OK")):
            result = deployer.pour("P")

        assert result.build_only is False
        deployer._nerve.discover.assert_called()
        deployer._nerve.install.assert_called_once()
        assert "devices" in result.to_dict()


class TestBuildOnlyOutput:
    """_build_only_output rewrites only the deploy-oriented no-command hint."""

    def test_rewrites_no_build_command_hint(self) -> None:
        assert _build_only_output(_NO_BUILD_COMMAND_MSG) == _BUILD_ONLY_NO_COMMAND_MSG

    def test_rewrites_hint_inside_module_output(self) -> None:
        text = f"[app] {_NO_BUILD_COMMAND_MSG}"
        assert _build_only_output(text) == f"[app] {_BUILD_ONLY_NO_COMMAND_MSG}"

    def test_leaves_other_output_alone(self) -> None:
        assert _build_only_output("BUILD FAILED") == "BUILD FAILED"


# --- Multi-target deployment tests ---


def _multi_target_project() -> ForgeProject:
    """Create a ForgeProject with phone + watch modules."""
    return ForgeProject(
        name="SniperPulseApp",
        path="/tmp/SniperPulseApp",
        build_command="./gradlew build",
        framework="gradle_kmp",
        modules=[
            ModuleConfig(
                name="composeApp",
                device_types=["phone", "tablet", "emulator"],
                build_command="./gradlew :composeApp:assembleDebug",
                apk_pattern="composeApp/build/outputs/apk/debug/*.apk",
            ),
            ModuleConfig(
                name="watchApp",
                device_types=["watch"],
                build_command="./gradlew :watchApp:assembleDebug",
                apk_pattern="watchApp/build/outputs/apk/debug/*.apk",
            ),
        ],
    )


def _phone_device() -> dict:
    return _PHONE_DICT.copy()


def _watch_device() -> dict:
    return {
        "serial": "PIXELWATCH2",
        "platform": "android",
        "device_type": "watch",
        "model": "Pixel Watch 2",
        "state": "online",
    }


def _emulator_device() -> dict:
    return _EMULATOR_DICT.copy()


class TestVerifySchemeIsWatchos:
    """SDKROOT detection across Xcode output formats."""

    def _verify(self, build_settings_stdout: str) -> bool:
        deployer = Deployer(forge_reader=MagicMock(spec=ForgeProjectReader))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=build_settings_stdout)
            return deployer._verify_scheme_is_watchos(
                Path("/tmp/x"), is_workspace=False, name="MyApp", scheme="MyAppWatch"
            )

    def test_short_form_sdkroot(self) -> None:
        assert self._verify("    SDKROOT = watchos\n") is True

    def test_full_path_sdkroot(self) -> None:
        """Modern Xcode emits the full SDK path, not the short name."""
        assert (
            self._verify(
                "    SDKROOT = /Applications/Xcode.app/Contents/Developer/Platforms/"
                "WatchOS.platform/Developer/SDKs/WatchOS26.4.sdk\n"
            )
            is True
        )

    def test_iphoneos_sdkroot_rejected(self) -> None:
        assert (
            self._verify(
                "    SDKROOT = /Applications/Xcode.app/Contents/Developer/Platforms/"
                "iPhoneOS.platform/Developer/SDKs/iPhoneOS26.4.sdk\n"
            )
            is False
        )

    def test_no_sdkroot_rejected(self) -> None:
        assert self._verify("    PRODUCT_NAME = MyApp\n") is False


def _physical_iphone() -> dict:
    return {
        "serial": "00008110-AAA",
        "platform": "ios",
        "device_type": "phone",
        "model": "Bench iPhone",
        "state": "online",
    }


def _physical_apple_watch() -> dict:
    return {
        "serial": "00008310-BBB",
        "platform": "ios",
        "device_type": "watch",
        "model": "Bench Apple Watch",
        "state": "online",
    }


def _ios_simulator() -> dict:
    return {
        "serial": "4C215E09-3AAF-498F-B8AF-FE9F8B3B5177",
        "platform": "ios",
        "device_type": "simulator",
        "model": "iPhone 16",
        "state": "online",
    }


class TestMultiTargetPhysicalIos:
    """Physical iOS devices in the multi-target pour path (spec 2026-06-09).

    Module build_commands are Gradle/Android-only; physical iOS targets must
    route through the platform-aware iOS build (signing) and device-SDK
    artifact discovery instead of the simulator-only module glob.
    """

    def _make_deployer(self, forge: MagicMock, fleet_devices: list[dict]) -> Deployer:
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = fleet_devices
        nerve.install.return_value = {"status": "installed"}
        nerve.launch.return_value = {"status": "launched"}
        return Deployer(forge_reader=forge, nerve_client=nerve)

    def _forge_with(self, project: ForgeProject) -> MagicMock:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {}
        forge.find_module_ios_apps.return_value = {}
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = None
        return forge

    def test_physical_iphone_builds_ios_and_deploys_device_app(self) -> None:
        project = _multi_target_project()
        forge = self._forge_with(project)
        deployer = self._make_deployer(forge, [_physical_iphone()])

        with (
            patch.object(deployer, "_build_modules", return_value=(True, "gradle ok")),
            patch.object(
                deployer, "_build_for_platforms", return_value=(True, "ios built")
            ) as mock_build,
        ):
            forge.find_ios_app.return_value = "/tmp/Debug-iphoneos/MyApp.app"
            result = deployer.pour("SniperPulseApp", device="00008110-AAA")

        mock_build.assert_called_once()
        assert mock_build.call_args[0][1] == {Platform.IOS}
        forge.find_ios_app.assert_called_with(
            "SniperPulseApp", device_type="phone", is_simulator=False
        )
        ios_results = [r for r in result.results if r.platform == "ios"]
        assert len(ios_results) == 1
        assert ios_results[0].device_serial == "00008110-AAA"
        assert ios_results[0].install_success

    def test_physical_watch_uses_watch_device_type(self) -> None:
        project = _multi_target_project()
        forge = self._forge_with(project)
        deployer = self._make_deployer(forge, [_physical_apple_watch()])

        with (
            patch.object(deployer, "_build_modules", return_value=(True, "gradle ok")),
            patch.object(deployer, "_build_for_platforms", return_value=(True, "ios built")),
        ):
            forge.find_ios_app.return_value = "/tmp/Debug-watchos/MyWatch.app"
            result = deployer.pour("SniperPulseApp", device="00008310-BBB")

        forge.find_ios_app.assert_called_with(
            "SniperPulseApp", device_type="watch", is_simulator=False
        )
        assert [r.device_serial for r in result.results if r.platform == "ios"] == ["00008310-BBB"]

    def test_ios_build_failure_stops_pour(self) -> None:
        project = _multi_target_project()
        forge = self._forge_with(project)
        deployer = self._make_deployer(forge, [_physical_iphone()])

        with (
            patch.object(deployer, "_build_modules", return_value=(True, "gradle ok")),
            patch.object(deployer, "_build_for_platforms", return_value=(False, "signing error")),
        ):
            result = deployer.pour("SniperPulseApp", device="00008110-AAA")

        assert result.results == []
        assert "signing error" in result.build_output

    def test_skip_build_still_deploys_existing_device_app(self) -> None:
        project = _multi_target_project()
        forge = self._forge_with(project)
        deployer = self._make_deployer(forge, [_physical_iphone()])

        with patch.object(deployer, "_build_for_platforms") as mock_build:
            forge.find_ios_app.return_value = "/tmp/Debug-iphoneos/MyApp.app"
            result = deployer.pour("SniperPulseApp", device="00008110-AAA", skip_build=True)

        mock_build.assert_not_called()
        assert [r.install_success for r in result.results if r.platform == "ios"] == [True]

    def test_missing_device_app_reports_error(self) -> None:
        project = _multi_target_project()
        forge = self._forge_with(project)
        deployer = self._make_deployer(forge, [_physical_iphone()])

        with (
            patch.object(deployer, "_build_modules", return_value=(True, "gradle ok")),
            patch.object(deployer, "_build_for_platforms", return_value=(True, "built")),
        ):
            forge.find_ios_app.return_value = None
            result = deployer.pour("SniperPulseApp", device="00008110-AAA")

        ios_results = [r for r in result.results if r.platform == "ios"]
        assert len(ios_results) == 1
        assert "iphoneos" in (ios_results[0].error or "")

    def test_simulator_targets_not_routed_through_device_path(self) -> None:
        """Simulators keep the existing module-artifact flow."""
        project = _multi_target_project()
        forge = self._forge_with(project)
        deployer = self._make_deployer(forge, [_ios_simulator()])

        with patch.object(deployer, "_build_for_platforms") as mock_build:
            deployer.pour("SniperPulseApp", skip_build=True)

        mock_build.assert_not_called()
        for call in forge.find_ios_app.call_args_list:
            assert call.kwargs.get("is_simulator", True) is True

    def test_android_platform_filter_skips_ios_device_path(self) -> None:
        project = _multi_target_project()
        forge = self._forge_with(project)
        forge.find_module_apks.return_value = {"composeApp": "/tmp/c.apk"}
        deployer = self._make_deployer(forge, [_phone_device(), _physical_iphone()])

        with patch.object(deployer, "_build_for_platforms") as mock_build:
            result = deployer.pour("SniperPulseApp", platform="android", skip_build=True)

        mock_build.assert_not_called()
        assert all(r.platform != "ios" for r in result.results)

    def test_physical_ios_not_double_deployed_by_module_loop(self) -> None:
        """A serial handled by the device path is skipped by the module loop."""
        project = _multi_target_project()
        forge = self._forge_with(project)
        # Stale simulator artifact exists for the phone module
        forge.find_module_ios_apps.return_value = {
            "composeApp": "/tmp/Debug-iphonesimulator/MyApp.app"
        }
        deployer = self._make_deployer(forge, [_physical_iphone()])

        with (
            patch.object(deployer, "_build_modules", return_value=(True, "gradle ok")),
            patch.object(deployer, "_build_for_platforms", return_value=(True, "built")),
        ):
            forge.find_ios_app.return_value = "/tmp/Debug-iphoneos/MyApp.app"
            result = deployer.pour("SniperPulseApp", device="00008110-AAA")

        ios_results = [r for r in result.results if r.device_serial == "00008110-AAA"]
        assert len(ios_results) == 1
        assert ios_results[0].artifact_path != "/tmp/Debug-iphonesimulator/MyApp.app"


class TestDeployResultModule:
    def test_to_dict_with_module(self) -> None:
        r = DeployResult(device_serial="ABC", install_success=True, module="watchApp")
        d = r.to_dict()
        assert d["module"] == "watchApp"

    def test_to_dict_without_module(self) -> None:
        r = DeployResult(device_serial="ABC", install_success=True)
        d = r.to_dict()
        assert "module" not in d


class TestDeployerMultiTarget:
    """Tests for multi-target pour routing."""

    def _make_deployer(
        self,
        forge: MagicMock,
        fleet_devices: list[dict],
    ) -> Deployer:
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = fleet_devices
        nerve.install.return_value = {"status": "installed"}
        nerve.launch.return_value = {"status": "launched"}
        return Deployer(
            forge_reader=forge,
            nerve_client=nerve,
        )

    def test_routes_phone_apk_to_phone_watch_apk_to_watch(self) -> None:
        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/composeApp-debug.apk",
            "watchApp": "/tmp/watchApp-debug.apk",
        }

        deployer = self._make_deployer(forge, [_phone_device(), _watch_device()])
        result = deployer.pour("SniperPulseApp", skip_build=True)

        assert len(result.results) == 2

        phone_results = [r for r in result.results if r.module == "composeApp"]
        watch_results = [r for r in result.results if r.module == "watchApp"]
        assert len(phone_results) == 1
        assert phone_results[0].device_serial == "PIXEL8PRO"
        assert phone_results[0].artifact_path == "/tmp/composeApp-debug.apk"
        assert len(watch_results) == 1
        assert watch_results[0].device_serial == "PIXELWATCH2"
        assert watch_results[0].artifact_path == "/tmp/watchApp-debug.apk"

    def test_emulator_matches_phone_module(self) -> None:
        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/composeApp-debug.apk",
            "watchApp": "/tmp/watchApp-debug.apk",
        }

        deployer = self._make_deployer(forge, [_emulator_device()])
        result = deployer.pour("SniperPulseApp", skip_build=True)

        assert len(result.results) == 1
        assert result.results[0].module == "composeApp"
        assert result.results[0].device_serial == "emulator-5554"

    def test_skip_build_no_subprocess(self) -> None:
        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {"composeApp": "/tmp/a.apk"}

        with patch("subprocess.run") as mock_subprocess:
            deployer = self._make_deployer(forge, [_phone_device()])
            deployer.pour("SniperPulseApp", skip_build=True)
            mock_subprocess.assert_not_called()

    def test_no_matching_devices(self) -> None:
        """Watch-only fleet, but only phone module APK found."""
        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/composeApp-debug.apk",
        }

        # Only a watch device, but composeApp targets phone/tablet/emulator
        deployer = self._make_deployer(forge, [_watch_device()])
        result = deployer.pour("SniperPulseApp", skip_build=True)

        # composeApp won't match watch, watchApp has no APK
        assert len(result.results) == 0

    @patch("subprocess.run")
    def test_build_per_module(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/a.apk",
            "watchApp": "/tmp/b.apk",
        }

        deployer = self._make_deployer(forge, [_phone_device(), _watch_device()])
        deployer.pour("SniperPulseApp")

        # Two module builds
        assert mock_subprocess.call_count == 2
        commands = [
            mock_subprocess.call_args_list[i].args[0] for i in range(mock_subprocess.call_count)
        ]
        assert any(":composeApp:" in str(c) for c in commands)
        assert any(":watchApp:" in str(c) for c in commands)

    @patch("subprocess.run")
    def test_build_first_module_fails(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.return_value = MagicMock(returncode=1, stdout="", stderr="BUILD FAILED")

        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project

        deployer = self._make_deployer(forge, [_phone_device()])
        result = deployer.pour("SniperPulseApp")

        assert not result.success
        assert "BUILD FAILED" in result.build_output

    def test_module_filter(self) -> None:
        """--module watchApp should only deploy watchApp."""
        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/composeApp.apk",
            "watchApp": "/tmp/watchApp.apk",
        }

        deployer = self._make_deployer(forge, [_phone_device(), _watch_device()])
        result = deployer.pour("SniperPulseApp", skip_build=True, module="watchApp")

        assert len(result.results) == 1
        assert result.results[0].module == "watchApp"
        assert result.results[0].device_serial == "PIXELWATCH2"

    def test_module_filter_not_found(self) -> None:
        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project

        deployer = self._make_deployer(forge, [_phone_device()])
        result = deployer.pour("SniperPulseApp", skip_build=True, module="nonexistent")

        assert not result.success
        assert "not found" in result.build_output

    def test_device_type_filter(self) -> None:
        """--device-type watch should only target watch devices."""
        project = _multi_target_project()
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/composeApp.apk",
            "watchApp": "/tmp/watchApp.apk",
        }

        deployer = self._make_deployer(forge, [_phone_device(), _watch_device()])
        result = deployer.pour("SniperPulseApp", skip_build=True, device_type="watch")

        # Only watchApp module matches watch device_type
        assert len(result.results) == 1
        assert result.results[0].module == "watchApp"
        assert result.results[0].device_serial == "PIXELWATCH2"


class TestExplicitDeviceFormFactorRouting:
    """A single explicit device serial routes only form-factor-matching modules.

    Regression for inbox 2e5895b6: pouring SniperPulseApp at one phone serial
    built + installed BOTH composeApp and watchApp onto that phone. The watch
    module (shared applicationId) overwrote the phone app. Expected: a phone
    serial gets composeApp only; a watch serial gets watchApp only.
    """

    def _make_deployer(self, forge: MagicMock, fleet_devices: list[dict]) -> Deployer:
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = fleet_devices
        nerve.install.return_value = {"status": "installed"}
        nerve.launch.return_value = {"status": "launched"}
        return Deployer(forge_reader=forge, nerve_client=nerve)

    def _forge(self) -> MagicMock:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = _multi_target_project()
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/composeApp-debug.apk",
            "watchApp": "/tmp/watchApp-debug.apk",
        }
        forge.find_module_ios_apps.return_value = {}
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = None
        return forge

    def test_explicit_phone_serial_installs_only_phone_module(self) -> None:
        forge = self._forge()
        # Fleet still has both devices, but the pour names the phone serial.
        deployer = self._make_deployer(forge, [_phone_device(), _watch_device()])

        result = deployer.pour("SniperPulseApp", device="PIXEL8PRO", skip_build=True)

        modules = sorted(r.module for r in result.results)
        assert modules == ["composeApp"]
        assert result.results[0].device_serial == "PIXEL8PRO"
        # watchApp APK must never be installed on the phone.
        installed = [c.args[1] for c in deployer._nerve.install.call_args_list]
        assert "/tmp/watchApp-debug.apk" not in installed

    def test_explicit_watch_serial_installs_only_watch_module(self) -> None:
        forge = self._forge()
        deployer = self._make_deployer(forge, [_phone_device(), _watch_device()])

        result = deployer.pour("SniperPulseApp", device="PIXELWATCH2", skip_build=True)

        modules = sorted(r.module for r in result.results)
        assert modules == ["watchApp"]
        assert result.results[0].device_serial == "PIXELWATCH2"

    def test_multi_device_no_explicit_serial_unchanged(self) -> None:
        """AC3: with no explicit serial both modules still route by form factor."""
        forge = self._forge()
        deployer = self._make_deployer(forge, [_phone_device(), _watch_device()])

        result = deployer.pour("SniperPulseApp", skip_build=True)

        routing = {r.module: r.device_serial for r in result.results}
        assert routing == {"composeApp": "PIXEL8PRO", "watchApp": "PIXELWATCH2"}


def _shared_app_id_project(root: Path, app_id: str = "com.sniperpulse.app") -> ForgeProject:
    """A two-module project whose modules share one applicationId on disk.

    Both modules declare ``device_types=["phone"]`` so they both resolve to a
    single phone device — the misconfiguration the collision guard must catch
    even after form-factor routing.
    """
    for module in ("composeApp", "watchApp"):
        gradle = root / module / "build.gradle.kts"
        gradle.parent.mkdir(parents=True, exist_ok=True)
        gradle.write_text(f'android {{\n  applicationId = "{app_id}"\n}}\n')
    return ForgeProject(
        name="SniperPulseApp",
        path=str(root),
        build_command="./gradlew build",
        framework="gradle_kmp",
        modules=[
            ModuleConfig(
                name="composeApp",
                device_types=["phone"],
                apk_pattern="composeApp/build/outputs/apk/debug/*.apk",
            ),
            ModuleConfig(
                name="watchApp",
                device_types=["phone"],
                apk_pattern="watchApp/build/outputs/apk/debug/*.apk",
            ),
        ],
    )


class TestSharedApplicationIdCollision:
    """AC2: refuse installing two modules sharing an applicationId on one device."""

    def _make_deployer(self, forge: MagicMock, fleet_devices: list[dict]) -> Deployer:
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = fleet_devices
        nerve.install.return_value = {"status": "installed"}
        nerve.launch.return_value = {"status": "launched"}
        return Deployer(forge_reader=forge, nerve_client=nerve)

    def _forge(self, project: ForgeProject) -> MagicMock:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/composeApp-debug.apk",
            "watchApp": "/tmp/watchApp-debug.apk",
        }
        forge.find_module_ios_apps.return_value = {}
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = None
        return forge

    def test_second_module_with_shared_app_id_is_refused(self, tmp_path: Path) -> None:
        project = _shared_app_id_project(tmp_path)
        forge = self._forge(project)
        deployer = self._make_deployer(forge, [_phone_device()])

        result = deployer.pour("SniperPulseApp", skip_build=True)

        installed = [r for r in result.results if r.install_success]
        refused = [r for r in result.results if not r.install_success]
        assert len(installed) == 1
        assert len(refused) == 1
        assert "applicationId" in refused[0].error
        # Only the first module's APK was actually installed.
        assert deployer._nerve.install.call_count == 1

    def test_distinct_app_ids_both_install(self, tmp_path: Path) -> None:
        project = _shared_app_id_project(tmp_path)
        # Give watchApp a distinct applicationId — no collision.
        (tmp_path / "watchApp" / "build.gradle.kts").write_text(
            'android {\n  applicationId = "com.sniperpulse.watch"\n}\n'
        )
        forge = self._forge(project)
        deployer = self._make_deployer(forge, [_phone_device()])

        result = deployer.pour("SniperPulseApp", skip_build=True)

        assert sorted(r.module for r in result.results if r.install_success) == [
            "composeApp",
            "watchApp",
        ]
        assert deployer._nerve.install.call_count == 2


def _ios_simulator_device() -> dict:
    return _IOS_SIM_DICT.copy()


class TestDeployerIosBuild:
    """Tests for iOS build command routing."""

    def _make_deployer(
        self,
        forge: MagicMock,
        fleet_devices: list[dict],
    ) -> Deployer:
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = fleet_devices
        nerve.install.return_value = {"status": "installed"}
        nerve.launch.return_value = {"status": "launched"}
        return Deployer(
            forge_reader=forge,
            nerve_client=nerve,
        )

    @patch("subprocess.run")
    def test_pour_ios_uses_ios_build_command(self, mock_subprocess: MagicMock) -> None:
        """When target is iOS and ios_build_command is set, use it."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = ForgeProject(
            name="KMPApp",
            path="/tmp/KMPApp",
            build_command="./gradlew assembleDebug",
            ios_build_command=(
                "xcodebuild -project iosApp/iosApp.xcodeproj "
                "-scheme iosApp -sdk iphonesimulator build"
            ),
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_ios_app.return_value = (
            "/tmp/KMPApp/build/Build/Products/Debug-iphonesimulator/KMPApp.app"
        )

        ios_sim = _ios_simulator_device()
        deployer = self._make_deployer(forge, [ios_sim])
        deployer.pour("KMPApp")

        # Should call xcodebuild, not gradlew
        assert mock_subprocess.call_count == 1
        cmd_str = " ".join(mock_subprocess.call_args_list[0].args[0])
        assert "xcodebuild" in cmd_str
        assert "gradlew" not in cmd_str

    @patch("subprocess.run")
    def test_pour_ios_auto_detects_xcodebuild(
        self,
        mock_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No ios_build_command, but iosApp/iosApp.xcodeproj exists."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        # Create iosApp directory with xcodeproj
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "iosApp.xcodeproj").mkdir()

        project = ForgeProject(
            name="AutoDetect",
            path=str(tmp_path),
            build_command="./gradlew assembleDebug",
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_ios_app.return_value = "/tmp/auto.app"

        deployer = self._make_deployer(forge, [_ios_simulator_device()])
        deployer.pour("AutoDetect")

        assert mock_subprocess.call_count == 1
        cmd_str = " ".join(mock_subprocess.call_args_list[0].args[0])
        assert "xcodebuild" in cmd_str
        assert "-project iosApp/iosApp.xcodeproj" in cmd_str
        assert "-scheme iosApp" in cmd_str

    @patch("subprocess.run")
    def test_pour_android_ignores_ios_build(self, mock_subprocess: MagicMock) -> None:
        """When all targets are Android, only run Gradle."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = ForgeProject(
            name="AndroidOnly",
            path="/tmp/AndroidOnly",
            build_command="./gradlew assembleDebug",
            ios_build_command="xcodebuild -scheme test build",
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_apk.return_value = "/tmp/app.apk"

        deployer = self._make_deployer(forge, [_phone_device()])
        deployer.pour("AndroidOnly")

        assert mock_subprocess.call_count == 1
        cmd_str = " ".join(mock_subprocess.call_args_list[0].args[0])
        assert "gradlew" in cmd_str
        assert "xcodebuild" not in cmd_str

    @patch("subprocess.run")
    def test_pour_mixed_targets_builds_both(self, mock_subprocess: MagicMock) -> None:
        """Fleet with Android + iOS devices should trigger both build commands."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = ForgeProject(
            name="MixedApp",
            path="/tmp/MixedApp",
            build_command="./gradlew assembleDebug",
            ios_build_command="xcodebuild -scheme MixedApp build",
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_apk.return_value = "/tmp/app.apk"
        forge.find_ios_app.return_value = "/tmp/app.app"

        deployer = self._make_deployer(forge, [_phone_device(), _ios_simulator_device()])
        deployer.pour("MixedApp")

        # Both Gradle and xcodebuild should be called
        assert mock_subprocess.call_count == 2
        commands = [
            " ".join(mock_subprocess.call_args_list[i].args[0])
            for i in range(mock_subprocess.call_count)
        ]
        assert any("gradlew" in c for c in commands)
        assert any("xcodebuild" in c for c in commands)

    @patch("subprocess.run")
    def test_multi_target_ios_module(self, mock_subprocess: MagicMock) -> None:
        """Multi-target project with iOS module deploys .app to iOS device."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = ForgeProject(
            name="MultiKMP",
            path="/tmp/MultiKMP",
            build_command="./gradlew build",
            modules=[
                ModuleConfig(name="composeApp", device_types=["phone"]),
                ModuleConfig(name="iosApp", device_types=["phone"]),
            ],
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {"composeApp": "/tmp/compose.apk"}
        forge.find_module_ios_apps.return_value = {"iosApp": "/tmp/ios.app"}

        ios_sim = _ios_simulator_device()
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [_phone_device(), ios_sim]
        nerve.install.return_value = {"status": "installed"}
        nerve.launch.return_value = {"status": "launched"}

        deployer = Deployer(
            forge_reader=forge,
            nerve_client=nerve,
        )
        result = deployer.pour("MultiKMP", skip_build=True)

        # iOS device should get iOS .app
        ios_results = [r for r in result.results if r.platform == "ios"]
        assert len(ios_results) == 1
        assert ios_results[0].artifact_path == "/tmp/ios.app"
        # Verify nerve.install was called with the iOS sim serial and app path
        nerve.install.assert_any_call("IPHONE15-SIM", "/tmp/ios.app")

    @patch("subprocess.run")
    def test_single_module_ios(self, mock_subprocess: MagicMock) -> None:
        """Single module pour targeting iOS device."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = ForgeProject(
            name="SingleIOS",
            path="/tmp/SingleIOS",
            build_command="./gradlew build",
            modules=[
                ModuleConfig(name="iosApp", device_types=["phone"]),
            ],
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {}
        forge.find_module_ios_apps.return_value = {"iosApp": "/tmp/ios.app"}

        ios_sim = _ios_simulator_device()
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [ios_sim]
        nerve.install.return_value = {"status": "installed"}

        deployer = Deployer(
            forge_reader=forge,
            nerve_client=nerve,
        )
        result = deployer.pour("SingleIOS", skip_build=True, module="iosApp")

        assert len(result.results) == 1
        assert result.results[0].artifact_path == "/tmp/ios.app"
        assert result.results[0].install_success is True


class TestBuildModules:
    """Tests for _build_modules()."""

    @patch("subprocess.run")
    def test_per_module_commands(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = _multi_target_project()
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        ok, output = deployer._build_modules(project)
        assert ok is True
        assert "[composeApp]" in output
        assert "[watchApp]" in output

    @patch("subprocess.run")
    def test_first_fails_stops(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.return_value = MagicMock(returncode=1, stdout="", stderr="FAIL")

        project = _multi_target_project()
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        ok, _output = deployer._build_modules(project)
        assert ok is False
        # Only first module attempted
        assert mock_subprocess.call_count == 1
        assert mock_subprocess.call_args_list[0].args[0] == [
            "./gradlew",
            ":composeApp:assembleDebug",
        ]

    @patch("subprocess.run")
    def test_fallback_to_project_build(self, mock_subprocess: MagicMock) -> None:
        """Modules without build_command fall back to project-level build."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = ForgeProject(
            name="Test",
            path="/tmp/test",
            build_command="./gradlew build",
            modules=[
                ModuleConfig(name="app", device_types=["phone"]),
                ModuleConfig(name="watchApp", device_types=["watch"]),
            ],
        )

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        ok, _output = deployer._build_modules(project)
        assert ok is True
        # Falls back to single project-level build
        assert mock_subprocess.call_count == 1
        assert mock_subprocess.call_args_list[0].args[0] == ["./gradlew", "build"]


class TestDeployerBackwardsCompat:
    """Verify projects without modules behave exactly as before."""

    @patch("subprocess.run")
    def test_no_modules_unchanged(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        deployer = Deployer(
            forge_reader=mock_forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("PrecisionLevel")

        assert len(result.results) == 1
        assert result.results[0].install_success is True
        assert result.results[0].module == ""
        # find_apk called (not find_module_apks); called twice: once for artifact
        # collection and once during deployment
        assert mock_forge.find_apk.call_count >= 1
        mock_forge.find_module_apks.assert_not_called()

    @patch("subprocess.run")
    def test_no_modules_with_device_type_filter(
        self,
        mock_subprocess: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """device_type filter works even without modules."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="Simple",
            path="/tmp/simple",
            build_command="build",
        )
        forge.find_apk.return_value = "/tmp/simple.apk"

        mock_nerve.discover.return_value = [_phone_device(), _watch_device()]

        deployer = Deployer(
            forge_reader=forge,
            nerve_client=mock_nerve,
        )
        result = deployer.pour("Simple", device_type="phone")

        assert len(result.results) == 1
        assert result.results[0].device_serial == "PIXEL8PRO"


class TestDeployerNerveDiscovery:
    """Tests for nerve-based device discovery in Deployer."""

    @patch("subprocess.run")
    def test_pour_discovers_via_nerve(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """Deployer calls nerve.discover() to find devices."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="BUILD SUCCESSFUL",
            stderr="",
        )
        mock_nerve.discover.return_value = [_phone_device()]

        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        result = deployer.pour("PrecisionLevel")

        mock_nerve.discover.assert_called_once()
        assert result.success

    @patch("subprocess.run")
    def test_pour_helpful_message_when_no_devices(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """When nerve returns no devices, show helpful message."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        mock_nerve.discover.return_value = []

        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        result = deployer.pour("PrecisionLevel")

        assert not result.success
        assert "linc-nerve" in result.build_output or "devices" in result.build_output

    @patch("subprocess.run")
    def test_pour_specific_device_matched_by_serial(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """When requesting a specific serial, nerve.discover() is called and filtered."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        emu = _emulator_device()
        mock_nerve.discover.return_value = [emu]

        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        result = deployer.pour("PrecisionLevel", device="emulator-5554")

        mock_nerve.discover.assert_called_once()
        assert result.success

    def test_resolve_targets_excludes_offline_device(
        self,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """A device reporting state='offline' is filtered out."""
        offline = {**_phone_device(), "state": "offline"}
        mock_nerve.discover.return_value = [offline]

        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        assert deployer._resolve_targets(device="", platform="", device_type="") == []

    def test_resolve_targets_excludes_device_missing_state(
        self,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """A device dict with no `state` key is treated as offline (blocklist).

        Mirrors nerve's ios.py blocklist principle — unknown/absent state must
        default to offline, never online (improvement d07eef8f). A permissive
        default here silently disabled the online filter over HTTP.
        """
        stateless = {k: v for k, v in _phone_device().items() if k != "state"}
        mock_nerve.discover.return_value = [stateless]

        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        assert deployer._resolve_targets(device="", platform="", device_type="") == []

    def test_resolve_targets_includes_online_device(
        self,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """An online device passes the filter."""
        mock_nerve.discover.return_value = [_phone_device()]

        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)
        result = deployer._resolve_targets(device="", platform="", device_type="")
        assert [d["serial"] for d in result] == ["PIXEL8PRO"]


class TestIosDestinationFlag:
    """Tests for xcodebuild -destination flag (IMP-022)."""

    def test_resolve_ios_build_command_adds_destination(self, tmp_path: Path) -> None:
        """When device_id is provided, use -destination instead of -sdk."""
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "MyApp.xcodeproj").mkdir()

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        cmd = deployer._resolve_ios_build_command(project, device_id="ABCD-1234")
        assert "-destination 'platform=iOS Simulator,id=ABCD-1234'" in cmd
        assert "-sdk iphonesimulator" not in cmd

    def test_resolve_ios_build_command_falls_back_to_sdk(self, tmp_path: Path) -> None:
        """When no device_id, fall back to -sdk iphonesimulator."""
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "MyApp.xcodeproj").mkdir()

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        cmd = deployer._resolve_ios_build_command(project)
        assert "-sdk iphonesimulator" in cmd
        assert "-destination" not in cmd

    def test_explicit_command_gets_destination_appended(self) -> None:
        """When ios_build_command is set and device_id provided, append -destination."""
        project = ForgeProject(
            name="MyApp",
            path="/tmp/MyApp",
            ios_build_command="xcodebuild -scheme MyApp -sdk iphonesimulator build",
        )
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        cmd = deployer._resolve_ios_build_command(project, device_id="SIM-999")
        assert "-destination 'platform=iOS Simulator,id=SIM-999'" in cmd

    def test_explicit_command_with_destination_not_duplicated(self) -> None:
        """When ios_build_command already has -destination, don't append another."""
        project = ForgeProject(
            name="MyApp",
            path="/tmp/MyApp",
            ios_build_command="xcodebuild -destination 'platform=iOS Simulator,id=EXISTING' build",
        )
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        cmd = deployer._resolve_ios_build_command(project, device_id="SIM-999")
        assert cmd.count("-destination") == 1
        assert "EXISTING" in cmd


class TestIosDeviceSigning:
    """Signing-flag injection for physical iOS builds (spec 2026-06-09)."""

    def _deployer_and_project(self, tmp_path: Path) -> tuple[Deployer, ForgeProject]:
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "MyApp.xcodeproj").mkdir()
        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = Deployer(forge_reader=MagicMock(spec=ForgeProjectReader))
        return deployer, project

    def test_simulator_build_has_no_signing_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CRUCIBLE_IOS_TEAM_ID", "TESTTEAM01")
        deployer, project = self._deployer_and_project(tmp_path)
        cmd = deployer._resolve_ios_build_command(
            project, device_id="SIM-UDID", device_type="phone", is_simulator=True
        )
        assert "DEVELOPMENT_TEAM" not in cmd
        assert "-allowProvisioningUpdates" not in cmd
        assert "CODE_SIGN_IDENTITY" not in cmd

    def test_device_build_injects_signing_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deployer, project = self._deployer_and_project(tmp_path)
        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="00008110-AAA",
            device_type="phone",
            is_simulator=False,
            signing_team="TESTTEAM01",
        )
        assert "-allowProvisioningUpdates" in cmd
        assert "DEVELOPMENT_TEAM=TESTTEAM01" in cmd
        assert 'CODE_SIGN_IDENTITY="Apple Development"' in cmd
        assert "platform=iOS," in cmd

    def test_watch_device_build_injects_signing_flags(self, tmp_path: Path) -> None:
        deployer, project = self._deployer_and_project(tmp_path)
        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="00008310-BBB",
            device_type="watch",
            is_simulator=False,
            signing_team="TESTTEAM01",
        )
        assert "platform=watchOS," in cmd
        assert "DEVELOPMENT_TEAM=TESTTEAM01" in cmd

    def test_explicit_command_flags_not_duplicated(self, tmp_path: Path) -> None:
        """Explicit ios_build_command with its own signing flags wins."""
        deployer, project = self._deployer_and_project(tmp_path)
        project.ios_build_command = (
            "xcodebuild -project iosApp/MyApp.xcodeproj -scheme MyApp "
            "DEVELOPMENT_TEAM=EXPLICIT01 build"
        )
        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="00008110-AAA",
            device_type="phone",
            is_simulator=False,
            signing_team="TESTTEAM01",
        )
        assert cmd.count("DEVELOPMENT_TEAM") == 1
        assert "DEVELOPMENT_TEAM=EXPLICIT01" in cmd
        assert "-allowProvisioningUpdates" in cmd  # still appended (absent before)

    def test_device_build_without_team_fails_prebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_IOS_TEAM_ID", raising=False)
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(tmp_path / "absent.yaml"))
        deployer, project = self._deployer_and_project(tmp_path)
        targets = [{"platform": "ios", "serial": "00008110-AAA", "device_type": "phone"}]
        ok, output = deployer._build_for_platforms(project, {Platform.IOS}, targets)
        assert ok is False
        assert "ios_signing.team_id" in output
        assert "CRUCIBLE_IOS_TEAM_ID" in output


class TestRootLevelXcodeproj:
    """Tests for root-level xcodeproj/xcworkspace detection (IMP-022)."""

    def test_resolve_ios_build_command_root_level_xcodeproj(self, tmp_path: Path) -> None:
        """xcodeproj at project root (no iosApp/) generates valid command."""
        (tmp_path / "RootApp.xcodeproj").mkdir()

        project = ForgeProject(name="RootApp", path=str(tmp_path))
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        cmd = deployer._resolve_ios_build_command(project)
        assert "xcodebuild" in cmd
        assert "-project RootApp.xcodeproj" in cmd
        assert "-scheme RootApp" in cmd
        assert "iosApp/" not in cmd

    def test_resolve_ios_build_command_root_level_xcworkspace(self, tmp_path: Path) -> None:
        """xcworkspace at project root is detected."""
        (tmp_path / "RootApp.xcworkspace").mkdir()

        project = ForgeProject(name="RootApp", path=str(tmp_path))
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        cmd = deployer._resolve_ios_build_command(project)
        assert "-workspace RootApp.xcworkspace" in cmd
        assert "-scheme RootApp" in cmd

    def test_iosapp_preferred_over_root(self, tmp_path: Path) -> None:
        """iosApp/ takes priority over root-level xcodeproj."""
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "Inner.xcodeproj").mkdir()
        (tmp_path / "Root.xcodeproj").mkdir()

        project = ForgeProject(name="Test", path=str(tmp_path))
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

        cmd = deployer._resolve_ios_build_command(project)
        assert "iosApp/Inner.xcodeproj" in cmd

    def test_xcodebuild_workspace_and_scheme_with_spaces_quoted(self, tmp_path: Path) -> None:
        """Workspace 'My App.xcworkspace' + scheme 'My Scheme' stay single argv elements."""
        import shlex

        (tmp_path / "My App.xcworkspace").mkdir()
        project = ForgeProject(
            name="My App",
            path=str(tmp_path),
            ios_watch_scheme="My Scheme",
        )
        deployer = Deployer(forge_reader=MagicMock(spec=ForgeProjectReader))
        cmd = deployer._resolve_ios_build_command(project, device_type="watch")
        argv = shlex.split(cmd)
        ws_idx = argv.index("-workspace")
        scheme_idx = argv.index("-scheme")
        assert argv[ws_idx + 1] == "My App.xcworkspace"
        assert argv[scheme_idx + 1] == "My Scheme"

    def test_xcodebuild_project_and_scheme_with_spaces_quoted(self, tmp_path: Path) -> None:
        """Project 'My App.xcodeproj' + scheme 'My Scheme' stay single argv elements."""
        import shlex

        (tmp_path / "My App.xcodeproj").mkdir()
        project = ForgeProject(
            name="My App",
            path=str(tmp_path),
            ios_watch_scheme="My Scheme",
        )
        deployer = Deployer(forge_reader=MagicMock(spec=ForgeProjectReader))
        cmd = deployer._resolve_ios_build_command(project, device_type="watch")
        argv = shlex.split(cmd)
        proj_idx = argv.index("-project")
        scheme_idx = argv.index("-scheme")
        assert argv[proj_idx + 1] == "My App.xcodeproj"
        assert argv[scheme_idx + 1] == "My Scheme"

    def test_xcodebuild_destination_with_single_quote_in_device_id_is_quoted(
        self, tmp_path: Path
    ) -> None:
        """device_id containing a single quote is safely quoted via shlex.quote, not broken."""
        import shlex

        (tmp_path / "MyApp.xcodeproj").mkdir()
        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = Deployer(forge_reader=MagicMock(spec=ForgeProjectReader))
        cmd = deployer._resolve_ios_build_command(project, device_id="SIM'123")
        argv = shlex.split(cmd)
        dest_idx = argv.index("-destination")
        assert argv[dest_idx + 1] == "platform=iOS Simulator,id=SIM'123"

        explicit = ForgeProject(
            name="MyApp",
            path=str(tmp_path),
            ios_build_command="xcodebuild -scheme MyApp build",
        )
        cmd_explicit = deployer._resolve_ios_build_command(explicit, device_id="SIM'123")
        argv_explicit = shlex.split(cmd_explicit)
        dest_explicit_idx = argv_explicit.index("-destination")
        assert argv_explicit[dest_explicit_idx + 1] == "platform=iOS Simulator,id=SIM'123"

    def test_xcodebuild_scheme_with_semicolon_still_refused_by_safe_subprocess_run(
        self, tmp_path: Path
    ) -> None:
        """shlex.quote does not bypass the metacharacter denylist in safe_subprocess_run."""
        from linc_crucible.core.subprocess_utils import ShellMetacharError, safe_subprocess_run

        (tmp_path / "MyApp.xcworkspace").mkdir()
        project = ForgeProject(
            name="MyApp",
            path=str(tmp_path),
            ios_watch_scheme="MyScheme;id",
        )
        deployer = Deployer(forge_reader=MagicMock(spec=ForgeProjectReader))
        cmd = deployer._resolve_ios_build_command(project, device_type="watch")
        with pytest.raises(ShellMetacharError):
            safe_subprocess_run(cmd, cwd=str(tmp_path), timeout=5)


class TestIosDeployViaNerve:
    """Tests for iOS deploy via NerveClient (nerve handles boot internally)."""

    def test_deploy_ios_simulator_calls_nerve_install(self) -> None:
        """NerveClient.install called for simulator device (nerve handles boot)."""
        nerve = MagicMock(spec=NerveClient)
        nerve.install.return_value = {"status": "installed"}

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=nerve,
        )

        device = {
            "serial": "IPHONE15-SIM",
            "platform": "ios",
            "device_type": "emulator",
            "model": "iPhone 15",
            "state": "offline",
        }
        project = ForgeProject(name="App", path="/tmp/App")
        result = DeployResult(device_serial=device["serial"])

        deployer._deploy_ios_artifact(
            "/tmp/App.app",
            device,
            project,
            False,
            result,
        )

        nerve.install.assert_called_once_with("IPHONE15-SIM", "/tmp/App.app")
        assert result.install_success is True

    def test_deploy_ios_physical_calls_nerve_install(self) -> None:
        """NerveClient.install called for physical device too."""
        nerve = MagicMock(spec=NerveClient)
        nerve.install.return_value = {"status": "installed"}

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=nerve,
        )

        device = {
            "serial": "00008120AABBCCDD",
            "platform": "ios",
            "device_type": "phone",
            "model": "iPhone 15 Pro",
            "state": "online",
        }
        project = ForgeProject(name="App", path="/tmp/App")
        result = DeployResult(device_serial=device["serial"])

        deployer._deploy_ios_artifact(
            "/tmp/App.app",
            device,
            project,
            False,
            result,
        )

        nerve.install.assert_called_once_with("00008120AABBCCDD", "/tmp/App.app")
        assert result.install_success is True

    def test_deploy_ios_install_failure(self) -> None:
        """When nerve install fails, sets error."""
        nerve = MagicMock(spec=NerveClient)
        nerve.install.side_effect = ConnectionError("Nerve unreachable")

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=nerve,
        )

        device = {
            "serial": "IPHONE15-SIM",
            "platform": "ios",
            "device_type": "emulator",
            "model": "iPhone 15",
            "state": "offline",
        }
        project = ForgeProject(name="App", path="/tmp/App")
        result = DeployResult(device_serial=device["serial"])

        deployer._deploy_ios_artifact(
            "/tmp/App.app",
            device,
            project,
            False,
            result,
        )

        assert result.install_success is False
        assert "Install failed" in result.error


class TestWatchOSSchemeRouting:
    """Tests for watchOS xcodebuild scheme routing."""

    def _make_deployer(self) -> Deployer:
        return Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

    def test_resolve_ios_build_command_watch_destination(self, tmp_path: Path) -> None:
        """When device_type=watch and device_id provided, use watchOS Simulator destination."""
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "MyApp.xcodeproj").mkdir()

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="WATCH-SIM-1234",
            device_type="watch",
        )
        assert "-destination 'platform=watchOS Simulator,id=WATCH-SIM-1234'" in cmd
        assert "-sdk" not in cmd

    def test_resolve_ios_build_command_watch_sdk_fallback(self, tmp_path: Path) -> None:
        """When device_type=watch and no device_id, use -sdk watchsimulator."""
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "MyApp.xcodeproj").mkdir()

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(project, device_type="watch")
        assert "-sdk watchsimulator" in cmd
        assert "-destination" not in cmd

    @patch("subprocess.run")
    def test_resolve_ios_build_command_watch_scheme_detection(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """When device_type=watch, detect Watch-suffixed scheme from xcscheme files."""
        ios_dir = tmp_path / "iosApp"
        proj_dir = ios_dir / "MyApp.xcodeproj"
        proj_dir.mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        mock_json = """{"project": {"schemes": ["MyApp", "MyAppWatch"]}}"""
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_json)

        with patch.object(deployer, "_verify_scheme_is_watchos", return_value=True):
            cmd = deployer._resolve_ios_build_command(project, device_type="watch")

        assert "-scheme MyAppWatch" in cmd
        assert "-sdk watchsimulator" in cmd

    def test_resolve_ios_build_command_watch_no_scheme_fallback(self, tmp_path: Path) -> None:
        """When no Watch scheme exists, fall back to default scheme with watchOS destination."""
        ios_dir = tmp_path / "iosApp"
        proj_dir = ios_dir / "MyApp.xcodeproj"
        schemes_dir = proj_dir / "xcshareddata" / "xcschemes"
        schemes_dir.mkdir(parents=True)
        (schemes_dir / "MyApp.xcscheme").write_text("<scheme/>")

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="WATCH-SIM-999",
            device_type="watch",
        )
        # Falls back to default scheme name (from xcodeproj name)
        assert "-scheme MyApp" in cmd
        assert "-destination 'platform=watchOS Simulator,id=WATCH-SIM-999'" in cmd

    def test_resolve_ios_build_command_phone_unchanged(self, tmp_path: Path) -> None:
        """Verify device_type='' (default) produces iOS Simulator destination — no regression."""
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(project, device_id="PHONE-SIM-123")
        assert "-destination 'platform=iOS Simulator,id=PHONE-SIM-123'" in cmd
        assert "watchOS" not in cmd

    def test_resolve_ios_build_command_explicit_watch_destination(self) -> None:
        """When ios_build_command is set and device_type=watch, append watchOS destination."""
        project = ForgeProject(
            name="MyApp",
            path="/tmp/MyApp",
            ios_build_command="xcodebuild -scheme MyAppWatch build",
        )
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="WATCH-SIM-1",
            device_type="watch",
        )
        assert "-destination 'platform=watchOS Simulator,id=WATCH-SIM-1'" in cmd

    @patch("subprocess.run")
    def test_build_for_platforms_passes_device_type_watch(
        self,
        mock_subprocess: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_build_for_platforms passes device_type=watch to _resolve_ios_build_command."""
        # Physical target → signing team required; keep hermetic (no live fleet file)
        monkeypatch.setenv("CRUCIBLE_IOS_TEAM_ID", "TESTTEAM01")
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        # A physical watch device: device_type="watch" (not emulator/simulator)
        watch_target = {
            "serial": "WATCH-SIM-1",
            "platform": "ios",
            "device_type": "watch",
            "model": "Apple Watch",
            "state": "online",
        }

        ok, _output = deployer._build_for_platforms(
            project,
            {Platform.IOS},
            targets=[watch_target],
        )
        assert ok is True
        # The build command should have used watchOS (physical) destination
        # subprocess.run receives a list from shlex.split, so join to check
        build_cmd = " ".join(mock_subprocess.call_args[0][0])
        assert "watchOS" in build_cmd

    @patch("subprocess.run")
    def test_build_modules_ios_build_command(self, mock_subprocess: MagicMock) -> None:
        """Module with ios_build_command uses it over build_command."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

        project = ForgeProject(
            name="Test",
            path="/tmp/test",
            build_command="./gradlew build",
            modules=[
                ModuleConfig(
                    name="watchApp",
                    device_types=["watch"],
                    ios_build_command="xcodebuild -scheme WatchApp -sdk watchsimulator build",
                ),
            ],
        )

        deployer = self._make_deployer()
        ok, _output = deployer._build_modules(project)
        assert ok is True
        build_cmd = mock_subprocess.call_args[0][0]
        assert "xcodebuild" in build_cmd
        assert "watchsimulator" in build_cmd

    @patch("subprocess.run")
    def test_resolve_ios_build_command_watch_workspace(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """watchOS scheme detection works with xcworkspace too."""
        ios_dir = tmp_path / "iosApp"
        ws_dir = ios_dir / "MyApp.xcworkspace"
        ws_dir.mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        mock_json = """{"workspace": {"schemes": ["MyApp", "MyAppWatch"]}}"""
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_json)

        with patch.object(deployer, "_verify_scheme_is_watchos", return_value=True):
            cmd = deployer._resolve_ios_build_command(
                project,
                device_id="WATCH-SIM-1",
                device_type="watch",
            )
        assert "-workspace iosApp/MyApp.xcworkspace" in cmd
        assert "-scheme MyAppWatch" in cmd
        assert "watchOS Simulator" in cmd

    @patch("subprocess.run")
    def test_declared_watch_scheme_skips_xcodebuild_probe(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """A manifest-declared ios_watch_scheme is used directly, skipping the
        xcodebuild -list / -showBuildSettings discovery probes entirely."""
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path), ios_watch_scheme="MyAppWatch")
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project, device_id="WATCH-SIM-1", device_type="watch"
        )

        assert "-scheme MyAppWatch" in cmd
        assert "-destination 'platform=watchOS Simulator,id=WATCH-SIM-1'" in cmd
        # No xcodebuild probe ran — scheme came from the manifest, not discovery.
        mock_run.assert_not_called()

    def test_declared_watch_scheme_ignored_for_phone(self, tmp_path: Path) -> None:
        """ios_watch_scheme only applies to watch pours; phone keeps default scheme."""
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path), ios_watch_scheme="MyAppWatch")
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(project, device_id="PHONE-SIM-1")
        assert "-scheme MyApp " in cmd
        assert "MyAppWatch" not in cmd


class TestPhysicalDevicePlatformNames:
    """Tests for physical iOS device platform/SDK selection."""

    def _make_deployer(self) -> Deployer:
        return Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
        )

    def test_resolve_ios_build_command_physical_destination(
        self,
        tmp_path: Path,
    ) -> None:
        """Physical device uses platform=iOS (not iOS Simulator)."""
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="00008120AABBCCDD",
            is_simulator=False,
        )
        assert "-destination platform=iOS,id=00008120AABBCCDD" in cmd
        assert "iOS Simulator" not in cmd

    def test_resolve_ios_build_command_physical_sdk_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        """Physical device without device_id uses -sdk iphoneos."""
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project,
            is_simulator=False,
        )
        assert "-sdk iphoneos" in cmd
        assert "iphonesimulator" not in cmd

    def test_explicit_command_physical_destination(self) -> None:
        """Explicit ios_build_command with is_simulator=False uses platform=iOS."""
        project = ForgeProject(
            name="MyApp",
            path="/tmp/MyApp",
            ios_build_command="xcodebuild -scheme MyApp build",
        )
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="00008120AABBCCDD",
            is_simulator=False,
        )
        assert "-destination platform=iOS,id=00008120AABBCCDD" in cmd
        assert "iOS Simulator" not in cmd

    def test_resolve_ios_build_command_watch_physical_destination(
        self,
        tmp_path: Path,
    ) -> None:
        """Physical watch uses platform=watchOS (not watchOS Simulator)."""
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project,
            device_id="00008120EEFF1122",
            device_type="watch",
            is_simulator=False,
        )
        assert "-destination platform=watchOS,id=00008120EEFF1122" in cmd
        assert "watchOS Simulator" not in cmd

    def test_resolve_ios_build_command_watch_physical_sdk_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        """Physical watch without device_id uses -sdk watchos."""
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        cmd = deployer._resolve_ios_build_command(
            project,
            device_type="watch",
            is_simulator=False,
        )
        assert "-sdk watchos" in cmd
        assert "watchsimulator" not in cmd

    @patch("subprocess.run")
    def test_build_for_platforms_passes_is_simulator_false(
        self,
        mock_subprocess: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_build_for_platforms passes is_simulator from the target device."""
        # Physical target → signing team required; keep hermetic (no live fleet file)
        monkeypatch.setenv("CRUCIBLE_IOS_TEAM_ID", "TESTTEAM01")
        ios_dir = tmp_path / "iosApp"
        (ios_dir / "MyApp.xcodeproj").mkdir(parents=True)
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="OK",
            stderr="",
        )

        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        physical_target = {
            "serial": "00008120AABBCCDD",
            "platform": "ios",
            "device_type": "phone",
            "model": "iPhone 15 Pro",
            "state": "online",
        }

        ok, _output = deployer._build_for_platforms(
            project,
            {Platform.IOS},
            targets=[physical_target],
        )
        assert ok is True
        build_cmd = " ".join(mock_subprocess.call_args[0][0])
        assert "platform=iOS," in build_cmd
        assert "iOS Simulator" not in build_cmd


class TestWatchSimulatorFormFactorAxis:
    """Watch SIMULATOR routing (#84): device_type ("phone"/"watch" form factor)
    and virtual-vs-physical are separate axes. A watch simulator reports
    device_type="watch" (same as a physical Apple Watch) but must still be
    classified as a simulator and routed to the watchOS-simulator scheme +
    destination — not silently treated as a signing-gated physical build.
    """

    def _make_deployer(self) -> Deployer:
        return Deployer(forge_reader=MagicMock(spec=ForgeProjectReader))

    def test_watch_simulator_target_classified_as_simulator_with_watch_destination(
        self, tmp_path: Path
    ) -> None:
        """AC1: a watch SIMULATOR target (form factor watch, virtual) is
        classified is_sim=True and gets a watchOS-simulator destination —
        no signing team required, no physical watchOS destination."""
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "MyApp.xcodeproj").mkdir()

        # Simulator UDIDs are UUID-format (dashes); this is how a watch
        # simulator is distinguished from a physical Apple Watch, which also
        # reports device_type="watch" (mirrors nerve's ios_discovery.py:
        # watch simulators are matched by name, not a distinct device_type).
        targets = [
            {
                "platform": "ios",
                "serial": "4C215E09-3AAF-498F-B8AF-FE9F8B3B5177",
                "device_type": "watch",
                "model": "Apple Watch Simulator",
                "state": "online",
            }
        ]
        project = ForgeProject(name="MyApp", path=str(tmp_path), ios_watch_scheme="MyAppWatch")
        deployer = self._make_deployer()

        with patch.object(deployer, "_run_build", return_value=(True, "ok")) as run_build:
            ok, _output = deployer._build_ios_platform(project, targets)

        assert ok is True
        run_build.assert_called_once()
        cmd = run_build.call_args[0][0]
        assert "-scheme MyAppWatch" in cmd
        assert (
            "-destination 'platform=watchOS Simulator,id=4C215E09-3AAF-498F-B8AF-FE9F8B3B5177'"
            in cmd
        )
        assert "watchOS Simulator" in cmd
        assert "platform=watchOS," not in cmd  # not the physical destination
        assert "DEVELOPMENT_TEAM" not in cmd  # simulator builds never sign

    def test_standalone_watch_app_pour_resolves_watch_scheme_and_simulator_artifact(
        self, tmp_path: Path
    ) -> None:
        """AC2: a standalone watch app (single-target project, no separate
        iosApp/watchApp module declared — e.g. SniperPulseApp) pours to a
        watch simulator by resolving the watch scheme and looking up the
        watchOS-simulator build output — not the physical one, and without
        requiring a signing team."""
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "SniperPulse.xcodeproj").mkdir()

        project = ForgeProject(
            name="SniperPulseApp",
            path=str(tmp_path),
            ios_watch_scheme="SniperPulseWatch",
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = "/tmp/Debug-watchsimulator/SniperPulseWatch.app"

        nerve = MagicMock(spec=NerveClient)
        watch_sim = {
            "serial": "5D326F1A-4BBF-598F-C8BF-FE9F8B3B5188",
            "platform": "ios",
            "device_type": "watch",
            "model": "Apple Watch Simulator",
            "state": "online",
        }
        nerve.discover.return_value = [watch_sim]
        nerve.install.return_value = {"status": "installed"}

        deployer = Deployer(forge_reader=forge, nerve_client=nerve)

        with patch.object(deployer, "_run_build", return_value=(True, "built")) as run_build:
            result = deployer.pour("SniperPulseApp")

        # Build routed through the watch scheme with a simulator destination,
        # not a signing-gated physical one.
        build_cmd = run_build.call_args[0][0]
        assert "-scheme SniperPulseWatch" in build_cmd
        assert "watchOS Simulator" in build_cmd
        assert "DEVELOPMENT_TEAM" not in build_cmd

        # Artifact lookup asked for the watchOS-simulator output, not physical.
        forge.find_ios_app.assert_called_with(
            "SniperPulseApp", device_type="watch", is_simulator=True
        )
        ios_results = [r for r in result.results if r.platform == "ios"]
        assert len(ios_results) == 1
        assert ios_results[0].install_success is True
        assert ios_results[0].artifact_path == "/tmp/Debug-watchsimulator/SniperPulseWatch.app"

    def test_iphone_simulator_pour_unchanged(self, tmp_path: Path) -> None:
        """AC3 (regression): an iPhone simulator target still resolves
        is_sim=True with an iOS Simulator destination — no watchOS anywhere."""
        ios_dir = tmp_path / "iosApp"
        ios_dir.mkdir()
        (ios_dir / "MyApp.xcodeproj").mkdir()

        targets = [
            {
                "platform": "ios",
                "serial": "4C215E09-3AAF-498F-B8AF-FE9F8B3B5177",
                "device_type": "emulator",
                "model": "iPhone 16",
                "state": "online",
            }
        ]
        project = ForgeProject(name="MyApp", path=str(tmp_path))
        deployer = self._make_deployer()

        with patch.object(deployer, "_run_build", return_value=(True, "ok")) as run_build:
            ok, _output = deployer._build_ios_platform(project, targets)

        assert ok is True
        cmd = run_build.call_args[0][0]
        assert (
            "-destination 'platform=iOS Simulator,id=4C215E09-3AAF-498F-B8AF-FE9F8B3B5177'" in cmd
        )
        assert "watchOS" not in cmd
        assert "DEVELOPMENT_TEAM" not in cmd


def test_find_watch_scheme_from_json() -> None:
    deployer = Deployer()
    mock_json = """
    {
      "project": {
        "schemes": [
          "SniperPulse",
          "SniperPulseWatch"
        ]
      }
    }
    """
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = mock_json

        with patch.object(deployer, "_verify_scheme_is_watchos", return_value=True):
            scheme = deployer._find_watch_scheme_via_xcodebuild(
                Path("/some/project"), is_workspace=False, name="Project"
            )
            assert scheme == "SniperPulseWatch"


class TestDeployerNerveClient:
    """Tests verifying Deployer uses NerveClient for install/launch."""

    def test_deploy_android_uses_nerve_install(self) -> None:
        """NerveClient.install is called for Android APK install."""
        mock_client = MagicMock(spec=NerveClient)
        mock_client.install.return_value = {"status": "installed"}

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=mock_client,
        )

        result = DeployResult(device_serial="emulator-5554")
        deployer._install_android_apk("/tmp/app.apk", "emulator-5554", result)

        mock_client.install.assert_called_once_with("emulator-5554", "/tmp/app.apk")
        assert result.install_success is True

    def test_deploy_android_uses_nerve_launch(self) -> None:
        """NerveClient.launch is called for Android app launch."""
        mock_client = MagicMock(spec=NerveClient)
        mock_client.launch.return_value = {"status": "launched"}

        project = ForgeProject(
            name="App",
            path="/tmp/App",
            package="com.example.app",
            build_command="build",
        )

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=mock_client,
        )

        result = DeployResult(device_serial="emulator-5554")
        deployer._launch_android(project, "emulator-5554", result)

        mock_client.launch.assert_called_once_with("emulator-5554", "com.example.app")
        assert result.launch_success is True

    def test_launch_android_no_resolvable_identity_skips_launch_and_names_identity_package(
        self, tmp_path: Path
    ) -> None:
        """[contract] no resolvable identity -> no launch; error names identity.package."""
        mock_client = MagicMock(spec=NerveClient)
        project = ForgeProject(name="StrangerApp", path=str(tmp_path))

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=mock_client,
        )

        result = DeployResult(device_serial="emulator-5554")
        deployer._launch_android(project, "emulator-5554", result)

        mock_client.launch.assert_not_called()
        assert result.launch_success is False
        assert result.error == (
            "Could not infer package name for launch; set identity.package in linc-manifest.yaml"
        )

    def test_launch_android_passes_uri_launch_intent_to_nerve(self) -> None:
        """[contract] launch_intent 'myapp://settings' -> nerve launch payload carries uri."""
        client = NerveClient(base_url="http://localhost:8101")
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"status": "launched"}}
        mock_response.raise_for_status = MagicMock()

        project = ForgeProject(
            name="App",
            path="/tmp/App",
            package="com.example.app",
        )
        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=client,
        )
        result = DeployResult(device_serial="emulator-5554")

        with patch.object(client._http, "post", return_value=mock_response) as mock_post:
            deployer._launch_android(
                project, "emulator-5554", result, launch_intent="myapp://settings"
            )

        assert result.launch_success is True
        mock_post.assert_called_once_with(
            "http://localhost:8101/tools/nerve_launch",
            json={
                "device": "emulator-5554",
                "package": "com.example.app",
                "uri": "myapp://settings",
            },
            timeout=30,
        )

    @pytest.mark.parametrize(
        "bad_intent",
        [
            ".SettingsActivity",
            "com.example.app/.SettingsActivity",
            "SettingsActivity",
        ],
    )
    def test_crucible_pour_activity_launch_intent_returns_validation_error_with_zero_calls(
        self, bad_intent: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] .SettingsActivity -> validation error, no build, no nerve call."""
        import asyncio

        from click.testing import CliRunner

        import linc_crucible.server_context as ctx
        from linc_crucible.cli import cli
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer = MagicMock(spec=Deployer)
        mock_nerve = MagicMock(spec=NerveClient)
        monkeypatch.setattr(ctx, "_ensure_initialized", lambda: None)
        monkeypatch.setattr(ctx, "_deployer", mock_deployer)
        monkeypatch.setattr(ctx, "_nerve_client", mock_nerve)

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("subprocess.run") as mock_run,
        ):
            res = asyncio.run(
                crucible_pour(
                    project="PrecisionLevel",
                    launch_intent=bad_intent,
                )
            )
            mock_popen.assert_not_called()
            mock_run.assert_not_called()
            mock_deployer.pour.assert_not_called()
            mock_nerve.launch.assert_not_called()

        assert res.get("validation_error") is True
        assert res.get("error_category") == "validation"
        assert res.get("error") == (
            "launch_intent must be a deep-link URI such as myapp://settings; "
            "activity launch is not supported by the nerve daemon"
        )

        runner = CliRunner()
        with (
            patch("subprocess.Popen") as cli_popen,
            patch("subprocess.run") as cli_run,
        ):
            cli_res = runner.invoke(
                cli,
                ["pour", "--project", "PrecisionLevel", "--launch-intent", bad_intent],
            )
            cli_popen.assert_not_called()
            cli_run.assert_not_called()

        assert cli_res.exit_code == 2

    def test_deploy_ios_uses_nerve_install(self) -> None:
        """NerveClient.install is called for iOS .app install."""
        mock_client = MagicMock(spec=NerveClient)
        mock_client.install.return_value = {"status": "installed"}

        project = ForgeProject(name="App", path="/tmp/App")
        device = {
            "serial": "IPHONE15-SIM",
            "platform": "ios",
            "device_type": "emulator",
            "model": "iPhone 15",
            "state": "online",
        }

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=mock_client,
        )

        result = DeployResult(device_serial=device["serial"])
        deployer._deploy_ios_artifact("/tmp/App.app", device, project, False, result)

        mock_client.install.assert_called_once_with("IPHONE15-SIM", "/tmp/App.app")
        assert result.install_success is True

    def test_deploy_ios_with_launch_uses_nerve(self) -> None:
        """NerveClient.launch is called for iOS app launch."""
        mock_client = MagicMock(spec=NerveClient)
        mock_client.install.return_value = {"status": "installed"}
        mock_client.launch.return_value = {"status": "launched"}

        project = ForgeProject(name="App", path="/tmp/App", bundle_id="com.example.app")
        device = {
            "serial": "IPHONE15-SIM",
            "platform": "ios",
            "device_type": "emulator",
            "model": "iPhone 15",
            "state": "online",
        }

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=mock_client,
        )

        result = DeployResult(device_serial=device["serial"])
        deployer._deploy_ios_artifact("/tmp/App.app", device, project, True, result)

        mock_client.install.assert_called_once()
        mock_client.launch.assert_called_once_with("IPHONE15-SIM", "com.example.app")
        assert result.launch_success is True

    def test_deploy_ios_no_resolvable_identity_skips_launch_and_names_identity_bundle_id(
        self, tmp_path: Path
    ) -> None:
        """When iOS bundle ID cannot be inferred, no launch is attempted and error is set."""
        mock_client = MagicMock(spec=NerveClient)
        mock_client.install.return_value = {"status": "installed"}

        project = ForgeProject(name="StrangerIosApp", path=str(tmp_path))
        device = {
            "serial": "IPHONE15-SIM",
            "platform": "ios",
            "device_type": "emulator",
            "model": "iPhone 15",
            "state": "online",
        }

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=mock_client,
        )

        result = DeployResult(device_serial=device["serial"])
        deployer._deploy_ios_artifact("/tmp/App.app", device, project, True, result)

        mock_client.install.assert_called_once_with("IPHONE15-SIM", "/tmp/App.app")
        mock_client.launch.assert_not_called()
        assert result.launch_success is False
        assert result.error == (
            "Could not infer bundle ID for launch; set identity.bundle_id in linc-manifest.yaml"
        )

    def test_nerve_connection_error_sets_error(self) -> None:
        """ConnectionError from NerveClient is caught and sets result.error."""
        mock_client = MagicMock(spec=NerveClient)
        mock_client.install.side_effect = ConnectionError("Nerve unreachable")

        deployer = Deployer(
            forge_reader=MagicMock(spec=ForgeProjectReader),
            nerve_client=mock_client,
        )

        result = DeployResult(device_serial="emulator-5554")
        deployer._install_android_apk("/tmp/app.apk", "emulator-5554", result)

        assert result.install_success is False
        assert "Install failed" in result.error


class TestModuleScopedAndroidBuild:
    """Single-module Android pours scope the Gradle build to the module (issue #82).

    A single-module Android deploy (e.g. DawnCup :watchApp android) must not run
    the project-wide ``build_command`` — that compiles composite-build KMP Apple
    metadata and lets an unrelated Apple-side error block an Android install.
    """

    def _project_without_module_build_commands(self) -> ForgeProject:
        """Multi-target project whose modules have NO explicit build_command."""
        return ForgeProject(
            name="DawnCup",
            path="/tmp/DawnCup",
            build_command="./gradlew build",
            framework="gradle_kmp",
            modules=[
                ModuleConfig(
                    name="app",
                    device_types=["phone", "emulator"],
                    apk_pattern="app/build/outputs/apk/debug/*.apk",
                ),
                ModuleConfig(
                    name="watchApp",
                    device_types=["watch"],
                    apk_pattern="watchApp/build/outputs/apk/debug/*.apk",
                ),
            ],
        )

    def _make_deployer(self, project: ForgeProject, devices: list[dict]) -> Deployer:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_module_apks.return_value = {"watchApp": "/tmp/watchApp.apk"}
        forge.find_module_ios_apps.return_value = {}
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = None
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = devices
        nerve.install.return_value = {"status": "installed"}
        nerve.launch.return_value = {"status": "launched"}
        return Deployer(forge_reader=forge, nerve_client=nerve)

    def test_module_android_pour_constructs_module_scoped_task(self) -> None:
        """pour(module=watchApp, platform=android) builds :watchApp:assembleDebug,
        NOT the project-wide build_command (AC1)."""
        project = self._project_without_module_build_commands()
        deployer = self._make_deployer(project, [_watch_device()])

        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run_build:
            deployer.pour("DawnCup", module="watchApp", platform="android")

        run_build.assert_called_once_with("./gradlew :watchApp:assembleDebug", "/tmp/DawnCup")

    def test_explicit_module_build_command_takes_precedence(self) -> None:
        """A module's explicit build_command still wins over auto-scoping."""
        project = self._project_without_module_build_commands()
        project.modules[1].build_command = "./gradlew :watchApp:assembleRelease"
        deployer = self._make_deployer(project, [_watch_device()])

        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run_build:
            deployer.pour("DawnCup", module="watchApp", platform="android")

        run_build.assert_called_once_with("./gradlew :watchApp:assembleRelease", "/tmp/DawnCup")

    def test_module_pour_without_platform_falls_back_to_project_build(self) -> None:
        """No platform narrowing → project-level build_command as today (AC2)."""
        project = self._project_without_module_build_commands()
        deployer = self._make_deployer(project, [_watch_device()])

        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run_build:
            deployer.pour("DawnCup", module="watchApp")

        run_build.assert_called_once_with("./gradlew build", "/tmp/DawnCup")

    def test_module_pour_ios_platform_not_gradle_scoped(self) -> None:
        """platform=ios must not construct an Android assemble task."""
        project = self._project_without_module_build_commands()
        deployer = self._make_deployer(project, [_watch_device()])

        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run_build:
            deployer.pour("DawnCup", module="watchApp", platform="ios")

        for call in run_build.call_args_list:
            assert ":watchApp:assembleDebug" not in call.args[0]

    def test_module_android_pour_threads_build_type(self) -> None:
        """pour(build_type=release) constructs :watchApp:assembleRelease (C5)."""
        project = self._project_without_module_build_commands()
        deployer = self._make_deployer(project, [_watch_device()])

        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run_build:
            deployer.pour("DawnCup", module="watchApp", platform="android", build_type="release")

        run_build.assert_called_once_with("./gradlew :watchApp:assembleRelease", "/tmp/DawnCup")

    def test_module_android_pour_defaults_to_debug_build_type(self) -> None:
        """Omitting build_type keeps today's assembleDebug (behavior-preserving default)."""
        project = self._project_without_module_build_commands()
        deployer = self._make_deployer(project, [_watch_device()])

        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run_build:
            deployer.pour("DawnCup", module="watchApp", platform="android")

        run_build.assert_called_once_with("./gradlew :watchApp:assembleDebug", "/tmp/DawnCup")

    def test_full_project_pour_uses_project_build_command(self) -> None:
        """Single-target pour with no module/platform narrowing is unchanged (AC2)."""
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="PrecisionLevel",
            path="/tmp/PrecisionLevel",
            build_command="./gradlew assembleDebug",
            framework="gradle_kmp",
        )
        forge.find_apk.return_value = "/tmp/app-debug.apk"
        forge.find_ios_app.return_value = None
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [_phone_device()]
        nerve.install.return_value = {"status": "installed"}
        deployer = Deployer(forge_reader=forge, nerve_client=nerve)

        with patch.object(deployer, "_run_build", return_value=(True, "OK")) as run_build:
            deployer.pour("PrecisionLevel")

        run_build.assert_called_once_with("./gradlew assembleDebug", "/tmp/PrecisionLevel")


class TestModuleFilterOnSingleTargetProject:
    """pour(module=X) issue #109 — module must not be silently ignored.

    Regression: when a project has zero declared modules (single-target, or
    a manifest-discovery gap before deployment.modules was parsed), passing
    --module used to fall through to the single-target path and silently
    ignore the module argument, installing whatever the wildcard artifact
    glob happened to match instead of erroring or targeting the module.
    """

    def test_module_on_project_with_no_modules_errors_without_installing(self) -> None:
        """AC3: pour(module='app') against a project with NO modules returns
        an explicit 'Module not found' error and installs nothing."""
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="Simple",
            path="/tmp/Simple",
            build_command="./gradlew assembleDebug",
        )
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [_phone_device()]

        deployer = Deployer(forge_reader=forge, nerve_client=nerve)
        result = deployer.pour("Simple", module="app", skip_build=True)

        assert not result.success
        assert "Module 'app' not found" in result.build_output
        nerve.install.assert_not_called()


class TestManifestDiscoveredModulesRouteThroughPour:
    """AC1+AC2 (issue #109): manifest deployment.modules end-to-end through pour().

    Uses a real ForgeProjectReader (not mocked) against a tmp project dir
    shaped like PrecisionLevel's linc-manifest.yaml, with real APK files on
    disk, to prove the flavored `app` module artifact is installed and the
    `watchApp` artifact is never picked up by the phone deploy.
    """

    @staticmethod
    def _write_manifest(project_dir: Path) -> None:
        project_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "identity": {
                "name": "PrecisionLevel",
                "package": "com.lincmobile.level",
                "display_name": "Precision Level",
                "app_type": "phone_and_watch",
            },
            "deployment": {
                "modules": [
                    {
                        "name": "app",
                        "device_types": ["phone", "tablet", "emulator"],
                        "build_command": "./gradlew :app:assemblePlaystoreDebug",
                        "apk_pattern": "app/build/outputs/apk/playstore/debug/*.apk",
                    },
                    {
                        "name": "watchApp",
                        "device_types": ["watch"],
                        "build_command": "./gradlew :watchApp:assembleDebug",
                        "apk_pattern": "watchApp/build/outputs/apk/debug/*.apk",
                    },
                ],
            },
        }
        with open(project_dir / "linc-manifest.yaml", "w") as f:
            yaml.safe_dump(manifest, f)

    def test_module_app_installs_flavored_apk_never_watch_apk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "PrecisionLevel"
        self._write_manifest(proj_dir)

        app_apk_dir = proj_dir / "app" / "build" / "outputs" / "apk" / "playstore" / "debug"
        app_apk_dir.mkdir(parents=True)
        (app_apk_dir / "app-playstore-debug.apk").write_text("fake")

        watch_apk_dir = proj_dir / "watchApp" / "build" / "outputs" / "apk" / "debug"
        watch_apk_dir.mkdir(parents=True)
        (watch_apk_dir / "watchApp-debug.apk").write_text("fake")

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        project = reader.get("PrecisionLevel")
        assert project is not None
        assert project.is_multi_target is True

        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [_phone_device()]
        nerve.install.return_value = {"status": "installed"}

        deployer = Deployer(forge_reader=reader, nerve_client=nerve)
        result = deployer.pour(
            "PrecisionLevel",
            device=_phone_device()["serial"],
            module="app",
            skip_build=True,
        )

        assert len(result.results) == 1
        assert result.results[0].artifact_path.endswith(
            "app/build/outputs/apk/playstore/debug/app-playstore-debug.apk"
        )
        assert not result.results[0].artifact_path.endswith("watchApp-debug.apk")

        installed_paths = [call.args[1] for call in nerve.install.call_args_list]
        assert installed_paths
        assert all(not p.endswith("watchApp-debug.apk") for p in installed_paths)


class TestNoBuildCommandAndArtifactProvenance:
    def test_pour_no_build_command_stale_apk_fails_without_skip_build(
        self, tmp_path: Path
    ) -> None:
        proj_dir = tmp_path / "NoBuildProj"
        proj_dir.mkdir()
        apk_file = proj_dir / "app-debug.apk"
        apk_file.write_bytes(b"stale-apk")

        project = ForgeProject(
            name="NoBuildProj",
            path=str(proj_dir),
            build_command="",
            framework="gradle_kmp",
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_apk.return_value = str(apk_file)

        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [_phone_device()]

        deployer = Deployer(forge_reader=forge, nerve_client=nerve)
        with patch.object(
            deployer, "_deploy_to_device", wraps=deployer._deploy_to_device
        ) as mock_deploy:
            result = deployer.pour("NoBuildProj")
            assert result.build_success is False
            assert result.success is False
            mock_deploy.assert_not_called()
            assert result.build_output == (
                "No build command configured; pass skip_build=True "
                "to install the existing artifact"
            )

    @patch("subprocess.run")
    def test_pour_no_build_command_stale_apk_with_skip_build_installs_and_reports_older_than_head(
        self, mock_subprocess: MagicMock, tmp_path: Path
    ) -> None:
        proj_dir = tmp_path / "NoBuildProj"
        proj_dir.mkdir()
        apk_file = proj_dir / "app-debug.apk"
        apk_file.write_bytes(b"stale-apk")

        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="2026-10-01T12:00:00+00:00\n",
            stderr="",
        )
        older_mtime = 1790850000.0
        os.utime(apk_file, (older_mtime, older_mtime))

        project = ForgeProject(
            name="NoBuildProj",
            path=str(proj_dir),
            build_command="",
            framework="gradle_kmp",
        )
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        forge.find_apk.return_value = str(apk_file)

        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [_phone_device()]
        nerve.install.return_value = {"status": "installed"}

        deployer = Deployer(forge_reader=forge, nerve_client=nerve)
        result = deployer.pour("NoBuildProj", skip_build=True)

        assert result.success is True
        assert len(result.results) == 1
        assert result.results[0].install_success is True
        assert result.artifact_older_than_head is True
        assert result.results[0].artifact_older_than_head is True
        nerve.install.assert_called_once()
        assert result.build_output == (
            "Warning: installed artifact predates HEAD commit "
            f"({apk_file}, artifact_older_than_head=True)"
        )
        assert result.provenance["head_committed_at"] == "2026-10-01T12:00:00+00:00"
        assert result.provenance["artifact_older_than_head"] is True

    @patch("subprocess.run")
    def test_artifact_provenance_git_unavailable(
        self, mock_subprocess: MagicMock, tmp_path: Path
    ) -> None:
        from linc_crucible.core.deployer import _artifact_provenance

        apk_file = tmp_path / "app-debug.apk"
        apk_file.write_bytes(b"content")

        mock_subprocess.side_effect = FileNotFoundError("git")
        prov = _artifact_provenance(str(apk_file), str(tmp_path))

        assert prov["head_committed_at"] is None
        assert prov["artifact_older_than_head"] is None
        assert prov["artifact_mtime"] is not None
        assert isinstance(prov["artifact_age_s"], float)

    @patch("subprocess.run")
    def test_artifact_provenance_branches(
        self, mock_subprocess: MagicMock, tmp_path: Path
    ) -> None:
        from linc_crucible.core.deployer import _artifact_provenance

        apk_file = tmp_path / "app-debug.apk"
        apk_file.write_bytes(b"content")

        # Branch 1: Artifact older than HEAD
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="2026-10-01T12:00:00+00:00\n",
            stderr="",
        )
        os.utime(apk_file, (1790850000.0, 1790850000.0))
        prov1 = _artifact_provenance(str(apk_file), str(tmp_path))
        assert prov1["artifact_older_than_head"] is True
        assert prov1["head_committed_at"] == "2026-10-01T12:00:00+00:00"

        # Branch 2: Artifact newer than HEAD
        os.utime(apk_file, (1790860000.0, 1790860000.0))
        prov2 = _artifact_provenance(str(apk_file), str(tmp_path))
        assert prov2["artifact_older_than_head"] is False

        # Branch 3: Path does not exist
        missing_file = tmp_path / "missing.apk"
        prov3 = _artifact_provenance(str(missing_file), str(tmp_path))
        assert prov3["artifact_mtime"] is None
        assert prov3["artifact_age_s"] is None
        assert prov3["artifact_older_than_head"] is None

        # Branch 4: Git fails with non-zero exit code
        mock_subprocess.return_value = MagicMock(
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )
        prov4 = _artifact_provenance(str(apk_file), str(tmp_path))
        assert prov4["head_committed_at"] is None
        assert prov4["artifact_older_than_head"] is None

        # Branch 5: Git stdout is unparseable
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="invalid-date\n",
            stderr="",
        )
        prov5 = _artifact_provenance(str(apk_file), str(tmp_path))
        assert prov5["head_committed_at"] is None
        assert prov5["artifact_older_than_head"] is None

        # Branch 6: Empty path or project_dir
        prov6 = _artifact_provenance("", "")
        assert prov6["artifact_mtime"] is None
        assert prov6["artifact_older_than_head"] is None


# HEAD commit instant used by the provenance helper tests below.
_HEAD_ISO = "2026-10-01T12:00:00+00:00"
_HEAD_TS = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC).timestamp()


def _git_ok(stdout: str = f"{_HEAD_ISO}\n") -> MagicMock:
    return MagicMock(returncode=0, stdout=stdout, stderr="")


class TestArtifactMtime:
    def test_returns_exact_mtime(self, tmp_path: Path) -> None:
        apk = tmp_path / "a.apk"
        apk.write_bytes(b"x")
        os.utime(apk, (1_700_000_000.0, 1_700_000_000.0))
        assert _artifact_mtime(str(apk)) == 1_700_000_000.0

    def test_missing_path_returns_none(self, tmp_path: Path) -> None:
        assert _artifact_mtime(str(tmp_path / "missing.apk")) is None

    def test_empty_path_returns_none(self) -> None:
        assert _artifact_mtime("") is None


class TestHeadCommitIso:
    @patch("subprocess.run")
    def test_runs_exact_git_command_and_strips(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = _git_ok(f"  {_HEAD_ISO}\n")
        assert _head_commit_iso(str(tmp_path)) == _HEAD_ISO
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == [
            "git",
            "-C",
            str(tmp_path),
            "log",
            "-1",
            "--format=%cI",
        ]
        assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)
        assert mock_run.call_args.kwargs["timeout"] == 10

    @patch("subprocess.run")
    def test_nonzero_exit_returns_none(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=128, stdout=_HEAD_ISO, stderr="fatal")
        assert _head_commit_iso(str(tmp_path)) is None

    @patch("subprocess.run")
    def test_blank_stdout_returns_none(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = _git_ok("   \n")
        assert _head_commit_iso(str(tmp_path)) is None

    @patch("subprocess.run")
    def test_none_stdout_returns_none(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout=None, stderr="")
        assert _head_commit_iso(str(tmp_path)) is None

    @patch("subprocess.run")
    def test_git_missing_returns_none(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = FileNotFoundError("git")
        assert _head_commit_iso(str(tmp_path)) is None

    @patch("subprocess.run")
    def test_timeout_returns_none(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=10)
        assert _head_commit_iso(str(tmp_path)) is None

    @patch("subprocess.run")
    def test_empty_project_dir_skips_git(self, mock_run: MagicMock) -> None:
        assert _head_commit_iso("") is None
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_missing_project_dir_skips_git(self, mock_run: MagicMock, tmp_path: Path) -> None:
        assert _head_commit_iso(str(tmp_path / "nope")) is None
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_metachar_project_dir_is_refused(self, mock_run: MagicMock, tmp_path: Path) -> None:
        odd = tmp_path / "a;b"
        odd.mkdir()
        assert _head_commit_iso(str(odd)) is None
        mock_run.assert_not_called()


class TestParseCommitTime:
    def test_none_returns_none(self) -> None:
        assert _parse_commit_time(None) is None

    def test_garbage_returns_none(self) -> None:
        assert _parse_commit_time("invalid-date") is None

    def test_aware_timestamp(self) -> None:
        assert _parse_commit_time(_HEAD_ISO) == _HEAD_TS

    def test_naive_timestamp_treated_as_utc(self) -> None:
        assert _parse_commit_time("2026-10-01T12:00:00") == _HEAD_TS

    def test_offset_is_honoured(self) -> None:
        assert _parse_commit_time("2026-10-01T14:00:00+02:00") == _HEAD_TS


class TestIsOlder:
    def test_strictly_older_is_true(self) -> None:
        assert _is_older(_HEAD_TS - 1, _HEAD_TS) is True

    def test_strictly_newer_is_false(self) -> None:
        assert _is_older(_HEAD_TS + 1, _HEAD_TS) is False

    def test_equal_is_not_older(self) -> None:
        assert _is_older(_HEAD_TS, _HEAD_TS) is False

    def test_unknown_artifact_time_is_none(self) -> None:
        assert _is_older(None, _HEAD_TS) is None

    def test_unknown_head_time_is_none(self) -> None:
        assert _is_older(_HEAD_TS, None) is None


class TestArtifactProvenanceExact:
    @patch("linc_crucible.core.deployer.time")
    @patch("subprocess.run")
    def test_exact_dict_for_stale_artifact(
        self, mock_run: MagicMock, mock_time: MagicMock, tmp_path: Path
    ) -> None:
        apk = tmp_path / "a.apk"
        apk.write_bytes(b"x")
        mtime = _HEAD_TS - 3600
        os.utime(apk, (mtime, mtime))
        mock_time.time.return_value = _HEAD_TS + 100
        mock_run.return_value = _git_ok()

        assert _artifact_provenance(str(apk), str(tmp_path)) == {
            "artifact_mtime": "2026-10-01T11:00:00+00:00",
            "artifact_age_s": 3700.0,
            "head_committed_at": _HEAD_ISO,
            "artifact_older_than_head": True,
        }

    @patch("subprocess.run")
    def test_unparseable_head_reports_none(self, mock_run: MagicMock, tmp_path: Path) -> None:
        apk = tmp_path / "a.apk"
        apk.write_bytes(b"x")
        mock_run.return_value = _git_ok("not-a-date\n")
        prov = _artifact_provenance(str(apk), str(tmp_path))
        assert prov["head_committed_at"] is None
        assert prov["artifact_older_than_head"] is None

    @patch("subprocess.run")
    def test_missing_artifact_is_all_none_without_git(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        prov = _artifact_provenance(str(tmp_path / "missing.apk"), str(tmp_path))
        assert prov == {
            "artifact_mtime": None,
            "artifact_age_s": None,
            "head_committed_at": None,
            "artifact_older_than_head": None,
        }
        mock_run.assert_not_called()


class TestProvenanceFields:
    def test_nothing_set_is_empty(self) -> None:
        assert _provenance_fields({}, None) == {}

    def test_both_set(self) -> None:
        prov = {"artifact_older_than_head": True}
        assert _provenance_fields(prov, True) == {
            "provenance": prov,
            "artifact_older_than_head": True,
        }

    def test_false_flag_is_kept(self) -> None:
        assert _provenance_fields({}, False) == {"artifact_older_than_head": False}

    def test_provenance_only(self) -> None:
        assert _provenance_fields({"k": 1}, None) == {"provenance": {"k": 1}}

    def test_deploy_result_to_dict_carries_fields(self) -> None:
        r = DeployResult(device_serial="S", provenance={"k": 1}, artifact_older_than_head=False)
        d = r.to_dict()
        assert d["provenance"] == {"k": 1}
        assert d["artifact_older_than_head"] is False

    def test_deploy_result_to_dict_omits_unset_fields(self) -> None:
        d = DeployResult(device_serial="S").to_dict()
        assert "provenance" not in d
        assert "artifact_older_than_head" not in d

    def test_pour_result_to_dict_carries_fields(self) -> None:
        r = PourResult(project="P", provenance={"k": 1}, artifact_older_than_head=True)
        d = r.to_dict()
        assert d["provenance"] == {"k": 1}
        assert d["artifact_older_than_head"] is True

    def test_pour_result_to_dict_omits_unset_fields(self) -> None:
        d = PourResult(project="P").to_dict()
        assert "provenance" not in d
        assert "artifact_older_than_head" not in d


class TestAttachProvenance:
    @patch("linc_crucible.core.deployer._artifact_provenance")
    def test_no_artifact_path_leaves_result_untouched(self, mock_prov: MagicMock) -> None:
        r = DeployResult(device_serial="S")
        _attach_provenance(r, "/proj")
        assert r.provenance == {}
        assert r.artifact_older_than_head is None
        mock_prov.assert_not_called()

    @patch("linc_crucible.core.deployer._artifact_provenance")
    def test_artifact_path_attaches_provenance(self, mock_prov: MagicMock) -> None:
        prov = {"artifact_older_than_head": True, "head_committed_at": _HEAD_ISO}
        mock_prov.return_value = prov
        r = DeployResult(device_serial="S", artifact_path="/proj/a.apk")
        _attach_provenance(r, "/proj")
        mock_prov.assert_called_once_with("/proj/a.apk", "/proj")
        assert r.provenance == prov
        assert r.artifact_older_than_head is True


class TestStalenessSummary:
    def test_warning_text_is_exact(self) -> None:
        assert _stale_artifact_warning("/p/a.apk") == (
            "Warning: installed artifact predates HEAD commit "
            "(/p/a.apk, artifact_older_than_head=True)"
        )

    @pytest.mark.parametrize(
        ("text", "lines", "expected"),
        [
            ("", [], ""),
            ("built", [], "built"),
            ("", ["w"], "w"),
            ("built", ["w1", "w2"], "built\nw1\nw2"),
        ],
    )
    def test_append_lines(self, text: str, lines: list[str], expected: str) -> None:
        assert _append_lines(text, lines) == expected

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            ([], None),
            ([None], None),
            ([None, False], False),
            ([False, True], True),
            ([True, None], True),
        ],
    )
    def test_aggregate_staleness(self, flags: list[bool | None], expected: bool | None) -> None:
        assert _aggregate_staleness(flags) is expected

    @patch("linc_crucible.core.deployer._artifact_provenance")
    def test_artifacts_older_than_head(self, mock_prov: MagicMock) -> None:
        mock_prov.side_effect = [
            {"artifact_older_than_head": False},
            {"artifact_older_than_head": True},
        ]
        assert artifacts_older_than_head(["/a", "/b"], "/proj") is True
        assert [c.args for c in mock_prov.call_args_list] == [("/a", "/proj"), ("/b", "/proj")]

    def test_artifacts_older_than_head_no_paths(self) -> None:
        assert artifacts_older_than_head([], "/proj") is None

    def test_finalize_with_stale_and_fresh_results(self) -> None:
        stale = DeployResult(
            device_serial="A",
            artifact_path="/p/a.apk",
            provenance={"id": "a"},
            artifact_older_than_head=True,
        )
        fresh = DeployResult(
            device_serial="B",
            artifact_path="/p/b.apk",
            provenance={"id": "b"},
            artifact_older_than_head=False,
        )
        result = PourResult(project="P", build_output="built", results=[fresh, stale])
        _finalize_pour_provenance(result)
        assert result.build_output == (
            "built\nWarning: installed artifact predates HEAD commit "
            "(/p/a.apk, artifact_older_than_head=True)"
        )
        assert result.artifact_older_than_head is True
        assert result.provenance == {"id": "b"}

    def test_finalize_skips_empty_provenance(self) -> None:
        bare = DeployResult(device_serial="A")
        fresh = DeployResult(
            device_serial="B", provenance={"id": "b"}, artifact_older_than_head=False
        )
        result = PourResult(project="P", results=[bare, fresh])
        _finalize_pour_provenance(result)
        assert result.build_output == ""
        assert result.artifact_older_than_head is False
        assert result.provenance == {"id": "b"}

    def test_finalize_without_results_is_noop(self) -> None:
        result = PourResult(project="P", build_output="out")
        _finalize_pour_provenance(result)
        assert result.build_output == "out"
        assert result.artifact_older_than_head is None
        assert result.provenance == {}


# ── Personal-device policy (epic linc-crucible-6jf, bead 6jf.2) ───────
_PERSONAL_WATCH_SERIAL = "PERSONALWATCH01"
_BENCH_PHONE_SERIAL = "R58N000TEST"

_BENCH_PHONE = {
    "serial": _BENCH_PHONE_SERIAL,
    "platform": "android",
    "device_type": "phone",
    "model": "Pixel 8",
    "state": "online",
}
_PERSONAL_WATCH = {
    "serial": _PERSONAL_WATCH_SERIAL,
    "platform": "android",
    "device_type": "watch",
    "model": "Pixel Watch 3",
    "state": "online",
}


def _personal_fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Temp device-fleet.yaml: one personal watch, one bench phone."""
    from linc_crucible.core.device_policy import load_device_policy

    fleet = tmp_path / "device-fleet.yaml"
    fleet.write_text(
        yaml.safe_dump(
            {
                "devices": [
                    {"name": "test-watch", "serial": _PERSONAL_WATCH_SERIAL, "personal": True},
                    {"name": "bench-phone", "serial": _BENCH_PHONE_SERIAL, "personal": False},
                ]
            }
        )
    )
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(fleet))
    return load_device_policy()


def _unloaded_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from linc_crucible.core.device_policy import load_device_policy

    bad = tmp_path / "absent-fleet.yaml"
    bad.write_text("::: unparseable :::\n")
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(bad))
    # The malformed registry logs device_fleet_policy_read_failed; capture it so the
    # warning never reaches structlog's PrintLogger (closed under mutmut's runner).
    with structlog.testing.capture_logs():
        return load_device_policy()


def _policy_deployer(mock_forge: MagicMock, devices: list[dict]) -> tuple[Deployer, MagicMock]:
    nerve = MagicMock(spec=NerveClient)
    nerve.discover.return_value = devices
    nerve.install.return_value = {"status": "installed"}
    nerve.launch.return_value = {"status": "launched"}
    return Deployer(forge_reader=mock_forge, nerve_client=nerve), nerve


def _installed_serials(nerve: MagicMock) -> list[str]:
    serials = []
    for call in nerve.install.call_args_list:
        serials.append(call.kwargs.get("device") or call.args[0])
    return serials


class TestApplyDevicePolicy:
    """`_apply_device_policy` — one test per branch."""

    def test_no_policy_passes_everything_through(self) -> None:
        from linc_crucible.core.deployer import _apply_device_policy

        assert _apply_device_policy([_PERSONAL_WATCH], "", None, False) == (
            [_PERSONAL_WATCH],
            [],
            None,
        )

    def test_allow_personal_passes_everything_through(self, tmp_path, monkeypatch) -> None:
        from linc_crucible.core.deployer import _apply_device_policy

        policy = _personal_fleet(tmp_path, monkeypatch)
        devices = [_BENCH_PHONE, _PERSONAL_WATCH]
        assert _apply_device_policy(devices, _PERSONAL_WATCH_SERIAL, policy, True) == (
            devices,
            [],
            None,
        )

    def test_explicit_personal_is_refused(self, tmp_path, monkeypatch) -> None:
        from linc_crucible.core.deployer import _apply_device_policy

        policy = _personal_fleet(tmp_path, monkeypatch)
        allowed, excluded, refusal = _apply_device_policy(
            [_PERSONAL_WATCH], _PERSONAL_WATCH_SERIAL, policy, False
        )
        assert allowed == []
        assert excluded == []
        assert refusal is not None
        assert _PERSONAL_WATCH_SERIAL in refusal
        assert "allow_personal=True" in refusal

    def test_explicit_bench_is_allowed(self, tmp_path, monkeypatch) -> None:
        from linc_crucible.core.deployer import _apply_device_policy

        policy = _personal_fleet(tmp_path, monkeypatch)
        assert _apply_device_policy([_BENCH_PHONE], _BENCH_PHONE_SERIAL, policy, False) == (
            [_BENCH_PHONE],
            [],
            None,
        )

    def test_fan_out_drops_and_reports_personal(self, tmp_path, monkeypatch) -> None:
        from linc_crucible.core.deployer import _apply_device_policy

        policy = _personal_fleet(tmp_path, monkeypatch)
        assert _apply_device_policy([_BENCH_PHONE, _PERSONAL_WATCH], "", policy, False) == (
            [_BENCH_PHONE],
            [_PERSONAL_WATCH_SERIAL],
            None,
        )

    def test_fan_out_matches_hardware_id(self, tmp_path, monkeypatch) -> None:
        from linc_crucible.core.deployer import _apply_device_policy

        policy = _personal_fleet(tmp_path, monkeypatch)
        ios_watch = {"serial": "UDID-1", "hardware_id": _PERSONAL_WATCH_SERIAL}
        assert _apply_device_policy([ios_watch], "", policy, False) == (
            [],
            ["UDID-1"],
            None,
        )

    def test_unloaded_policy_refuses_fan_out_and_explicit(self, tmp_path, monkeypatch) -> None:
        from linc_crucible.core.deployer import _apply_device_policy

        policy = _unloaded_policy(tmp_path, monkeypatch)
        for explicit in ("", _BENCH_PHONE_SERIAL):
            allowed, excluded, refusal = _apply_device_policy(
                [_BENCH_PHONE], explicit, policy, False
            )
            assert (allowed, excluded) == ([], [])
            assert refusal is not None
            assert "absent-fleet.yaml" in refusal

    def test_resolve_policy_target_helper(self, tmp_path, monkeypatch) -> None:
        from linc_crucible.core.deployer import resolve_policy_target

        policy = _personal_fleet(tmp_path, monkeypatch)
        # Auto-pick skips personal device and selects first online bench device
        target, refusal = resolve_policy_target([_PERSONAL_WATCH, _BENCH_PHONE], "", policy, False)
        assert refusal is None
        assert target == _BENCH_PHONE

        # Auto-pick with only personal devices refuses
        target, refusal = resolve_policy_target([_PERSONAL_WATCH], "", policy, False)
        assert target is None
        assert refusal is not None
        assert _PERSONAL_WATCH_SERIAL in refusal

        # Auto-pick with empty devices returns (None, None)
        assert resolve_policy_target([], "", policy, False) == (None, None)

        # Explicit alias matched by hardware_id is refused
        alias_rec = {"serial": "192.0.2.10:5555", "hardware_id": _PERSONAL_WATCH_SERIAL}
        target, refusal = resolve_policy_target([alias_rec], "192.0.2.10:5555", policy, False)
        assert target is None
        assert refusal is not None
        assert "192.0.2.10:5555" in refusal


class TestPourPersonalDevicePolicy:
    """crucible_pour never targets a `personal: true` device unless allowed."""

    def test_fan_out_skips_personal_watch_and_reports_it(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [_BENCH_PHONE, _PERSONAL_WATCH])

        result = deployer.pour("PrecisionLevel", skip_build=True, policy=policy)

        assert _installed_serials(nerve) == [_BENCH_PHONE_SERIAL]
        assert [r.device_serial for r in result.results] == [_BENCH_PHONE_SERIAL]
        assert result.excluded_personal == [_PERSONAL_WATCH_SERIAL]
        d = result.to_dict()
        assert d["excluded_personal"] == [_PERSONAL_WATCH_SERIAL]
        assert d["success"] is True

    def test_watch_fan_out_with_only_personal_watch_installs_nothing(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [_BENCH_PHONE, _PERSONAL_WATCH])

        result = deployer.pour(
            "PrecisionLevel", skip_build=True, device_type="watch", policy=policy
        )

        nerve.install.assert_not_called()
        assert result.results == []
        assert result.excluded_personal == [_PERSONAL_WATCH_SERIAL]
        d = result.to_dict()
        assert d["success"] is False
        assert d["excluded_personal"] == [_PERSONAL_WATCH_SERIAL]
        assert _PERSONAL_WATCH_SERIAL in d["error"]
        assert "personal" in d["error"]

    def test_explicit_personal_serial_is_refused_before_build(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [_BENCH_PHONE, _PERSONAL_WATCH])

        with (
            patch.object(deployer, "_build") as build,
            patch.object(deployer, "_build_for_platforms") as build_platforms,
            patch.object(deployer, "_deploy_to_device") as deploy,
        ):
            result = deployer.pour("PrecisionLevel", device=_PERSONAL_WATCH_SERIAL, policy=policy)

        build.assert_not_called()
        build_platforms.assert_not_called()
        deploy.assert_not_called()
        nerve.install.assert_not_called()
        d = result.to_dict()
        assert d["success"] is False
        assert d["error_category"] == "personal_device"
        assert _PERSONAL_WATCH_SERIAL in d["error"]
        assert "allow_personal=True" in d["error"]

    def test_explicit_personal_serial_refused_on_multi_target_before_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _personal_fleet(tmp_path, monkeypatch)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = _multi_target_project()
        deployer, nerve = _policy_deployer(forge, [_PERSONAL_WATCH])

        with patch.object(deployer, "_build_modules") as build_modules:
            result = deployer.pour("SniperPulseApp", device=_PERSONAL_WATCH_SERIAL, policy=policy)

        build_modules.assert_not_called()
        nerve.install.assert_not_called()
        assert result.to_dict()["error_category"] == "personal_device"

    def test_explicit_personal_with_allow_personal_proceeds_as_today(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin: allow_personal=True yields exactly the pre-policy result shape."""
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [_PERSONAL_WATCH])
        plain, _ = _policy_deployer(mock_forge, [_PERSONAL_WATCH])

        allowed = deployer.pour(
            "PrecisionLevel",
            device=_PERSONAL_WATCH_SERIAL,
            skip_build=True,
            policy=policy,
            allow_personal=True,
        )
        baseline = plain.pour("PrecisionLevel", device=_PERSONAL_WATCH_SERIAL, skip_build=True)

        assert _installed_serials(nerve) == [_PERSONAL_WATCH_SERIAL]
        assert allowed.to_dict() == baseline.to_dict()
        assert not {"excluded_personal", "error_category", "error"} & set(allowed.to_dict())
        assert allowed.to_dict()["success"] is True

    def test_unloaded_policy_refuses_fan_out(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _unloaded_policy(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [_BENCH_PHONE])

        with patch.object(deployer, "_build_for_platforms") as build:
            result = deployer.pour("PrecisionLevel", policy=policy)

        build.assert_not_called()
        nerve.install.assert_not_called()
        d = result.to_dict()
        assert d["success"] is False
        assert d["error_category"] == "personal_device"
        assert "absent-fleet.yaml" in d["error"]

    def test_unloaded_policy_refuses_explicit(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _unloaded_policy(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [_BENCH_PHONE])

        result = deployer.pour(
            "PrecisionLevel", device=_BENCH_PHONE_SERIAL, skip_build=True, policy=policy
        )

        nerve.install.assert_not_called()
        d = result.to_dict()
        assert d["error_category"] == "personal_device"
        assert "absent-fleet.yaml" in d["error"]

    def test_multi_target_fan_out_skips_personal_watch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _personal_fleet(tmp_path, monkeypatch)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = _multi_target_project()
        forge.find_module_apks.return_value = {
            "composeApp": "/tmp/composeApp-debug.apk",
            "watchApp": "/tmp/watchApp-debug.apk",
        }
        forge.find_module_ios_apps.return_value = {}
        deployer, nerve = _policy_deployer(forge, [_BENCH_PHONE, _PERSONAL_WATCH])

        result = deployer.pour("SniperPulseApp", skip_build=True, policy=policy)

        assert _installed_serials(nerve) == [_BENCH_PHONE_SERIAL]
        assert result.excluded_personal == [_PERSONAL_WATCH_SERIAL]

    def test_policy_context_does_not_leak_into_next_call(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer, _ = _policy_deployer(mock_forge, [_BENCH_PHONE, _PERSONAL_WATCH])

        deployer.pour("PrecisionLevel", skip_build=True, policy=policy)
        targets = deployer._resolve_targets("", "")

        assert {t["serial"] for t in targets} == {_BENCH_PHONE_SERIAL, _PERSONAL_WATCH_SERIAL}

    def test_explicit_alias_with_personal_hardware_id_is_refused_before_build(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] pour with device 192.0.2.10:5555 whose discover record carries a
        personal hardware_id is refused before any build or install."""
        policy = _personal_fleet(tmp_path, monkeypatch)
        alias_device = {
            "serial": "192.0.2.10:5555",
            "id": "192.0.2.10:5555",
            "hardware_id": _PERSONAL_WATCH_SERIAL,
            "platform": "android",
            "device_type": "watch",
            "state": "online",
        }
        deployer, nerve = _policy_deployer(mock_forge, [_BENCH_PHONE, alias_device])

        with (
            patch.object(deployer, "_build") as build,
            patch.object(deployer, "_build_for_platforms") as build_platforms,
            patch.object(deployer, "_deploy_to_device") as deploy,
        ):
            result = deployer.pour("PrecisionLevel", device="192.0.2.10:5555", policy=policy)

        build.assert_not_called()
        build_platforms.assert_not_called()
        deploy.assert_not_called()
        nerve.install.assert_not_called()
        d = result.to_dict()
        assert d["success"] is False
        assert d["error_category"] == "personal_device"
        assert "192.0.2.10:5555" in d["error"]

    @pytest.mark.parametrize(
        "raw_serial",
        [
            _PERSONAL_WATCH_SERIAL.lower(),
            f" {_PERSONAL_WATCH_SERIAL} ",
            f"{_PERSONAL_WATCH_SERIAL}\n",
        ],
    )
    def test_explicit_case_and_whitespace_variants_refused_before_build(
        self,
        mock_forge: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        raw_serial: str,
    ) -> None:
        """[contract] Casefold and whitespace variants of a personal serial are refused."""
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [_BENCH_PHONE, _PERSONAL_WATCH])

        with patch.object(deployer, "_build_for_platforms") as build_platforms:
            result = deployer.pour("PrecisionLevel", device=raw_serial, policy=policy)

        build_platforms.assert_not_called()
        nerve.install.assert_not_called()
        assert result.to_dict()["error_category"] == "personal_device"

    def test_nerve_unreachable_during_precheck_still_refuses_exact_personal_serial(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[test] When nerve.discover() raises ConnectionError during pre-build check,
        the string-only check still refuses an exact personal serial."""
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [])
        nerve.discover.side_effect = ConnectionError("nerve daemon down")

        with patch.object(deployer, "_build_for_platforms") as build_platforms:
            result = deployer.pour(
                "PrecisionLevel", device=_PERSONAL_WATCH_SERIAL.lower(), policy=policy
            )

        build_platforms.assert_not_called()
        nerve.install.assert_not_called()
        d = result.to_dict()
        assert d["success"] is False
        assert d["error_category"] == "personal_device"

    def test_resolve_targets_matches_case_insensitively_across_serial_id_hardware_id(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] _resolve_targets matches explicit device case-insensitively across
        serial, id, or hardware_id."""
        from linc_crucible.core.deployer import (
            _device_matches,
            discover_for_policy,
            resolve_policy_target,
        )

        assert _device_matches({"serial": "R58N000TEST"}, "") is False
        assert _device_matches({"serial": ""}, "   ") is False
        dev = {
            "serial": "192.0.2.10:5555",
            "id": "watch-wifi-1",
            "hardware_id": "R58N000BENCH",
            "platform": "android",
            "state": "online",
        }
        deployer, nerve = _policy_deployer(mock_forge, [dev, _PERSONAL_WATCH])

        assert deployer._resolve_targets("192.0.2.10:5555", "") == [dev]
        assert deployer._resolve_targets("WATCH-WIFI-1", "") == [dev]
        assert deployer._resolve_targets("r58n000bench", "") == [dev]

        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer._policy_ctx.begin(policy, False)
        try:
            assert deployer._resolve_targets(_PERSONAL_WATCH_SERIAL, "") == []
        finally:
            assert deployer._policy_ctx.end() == []

        assert resolve_policy_target([], "explicit-1", None, False) == (
            {"serial": "explicit-1", "id": "explicit-1"},
            None,
        )
        assert resolve_policy_target([], "", policy, False) == (None, None)
        _, ref = resolve_policy_target(
            [{"serial": _PERSONAL_WATCH_SERIAL}, {"serial": _PERSONAL_WATCH_SERIAL.lower()}],
            "",
            policy,
            False,
        )
        assert ref == (
            f"No non-personal device available (excluded personal: "
            f"{_PERSONAL_WATCH_SERIAL}, {_PERSONAL_WATCH_SERIAL.lower()}); "
            "pass allow_personal=True to override"
        )
        nerve.discover.return_value = "not-a-list"
        assert discover_for_policy(nerve) == []

    def test_a11y_baseline_capture_refuses_alias_with_personal_hardware_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] a11y baseline capture with device 192.0.2.10:5555 whose discover
        record carries a personal hardware_id is refused before any elements call."""
        import linc_crucible.server_context as ctx
        from linc_crucible.baseline_tools import _a11y_capture, _a11y_target

        policy = _personal_fleet(tmp_path, monkeypatch)
        monkeypatch.setattr(ctx, "_cached_device_policy", policy)
        nerve = MagicMock(spec=NerveClient)
        nerve.discover.return_value = [
            {
                "serial": "192.0.2.10:5555",
                "id": "192.0.2.10:5555",
                "hardware_id": _PERSONAL_WATCH_SERIAL,
            }
        ]
        monkeypatch.setattr(ctx, "_nerve_client", nerve)
        tree_capture = MagicMock()

        result = _a11y_capture(
            tree_capture,
            "TestProject",
            "home",
            "192.0.2.10:5555",
            "",
            "",
            False,
            allow_personal=False,
        )

        assert result["success"] is False
        assert result["error_category"] == "personal_device"
        assert result["error"] == (
            f"device 192.0.2.10:5555 is registered personal: true in {policy.source}; "
            "pass allow_personal=True to override"
        )
        nerve.elements.assert_not_called()

        serial, err = _a11y_target(nerve, "", False)
        assert serial == ""
        assert err is not None
        assert err["excluded_personal"] == ["192.0.2.10:5555"]
        nerve.discover.return_value = [{"serial": "SAFE_1"}]
        assert _a11y_target(nerve, "", False) == ("SAFE_1", None)
        nerve.discover.side_effect = ConnectionError("down")
        serial2, err2 = _a11y_target(nerve, _PERSONAL_WATCH_SERIAL, False)
        assert serial2 == ""
        assert err2 is not None
        assert "excluded_personal" not in err2


class TestBuildOnlyUnderDevicePolicy:
    """build_only touches no device, so it is dispatched ahead of the personal-device
    gate (#174) — and must therefore never reach target resolution or nerve."""

    def _multi_target_deployer(self, devices: list[dict]) -> Deployer:
        return _build_only_deployer(_multi_target_project(), devices)

    def test_explicit_personal_serial_builds_without_refusal_or_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer = self._multi_target_deployer([_BENCH_PHONE, _PERSONAL_WATCH])

        with (
            patch.object(deployer, "_run_build", return_value=(True, "BUILD OK")) as run,
            patch.object(deployer, "_resolve_targets") as resolve,
            patch.object(deployer, "_deploy_to_device") as deploy,
        ):
            result = deployer.pour(
                "SniperPulseApp", device=_PERSONAL_WATCH_SERIAL, build_only=True, policy=policy
            )

        assert run.call_count == 2
        resolve.assert_not_called()
        deploy.assert_not_called()
        deployer._nerve.discover.assert_not_called()
        deployer._nerve.install.assert_not_called()
        d = result.to_dict()
        assert d["success"] is True
        assert d["build_only"] is True
        assert not {"devices", "error", "error_category", "excluded_personal"} & set(d)

    def test_unloaded_registry_does_not_refuse_a_build_only_pour(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail-closed guards device access; a build-only pour has none to guard."""
        policy = _unloaded_policy(tmp_path, monkeypatch)
        deployer = self._multi_target_deployer([_BENCH_PHONE])

        with patch.object(deployer, "_run_build", return_value=(True, "BUILD OK")):
            result = deployer.pour("SniperPulseApp", build_only=True, policy=policy)

        deployer._nerve.install.assert_not_called()
        assert result.refusal == ""
        assert result.to_dict()["success"] is True

    def test_fan_out_build_only_reports_no_exclusions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer = self._multi_target_deployer([_BENCH_PHONE, _PERSONAL_WATCH])

        with patch.object(deployer, "_run_build", return_value=(True, "BUILD OK")):
            result = deployer.pour("SniperPulseApp", build_only=True, policy=policy)

        deployer._nerve.discover.assert_not_called()
        deployer._nerve.install.assert_not_called()
        assert result.excluded_personal == []
        assert "excluded_personal" not in result.to_dict()

    def test_build_only_leaves_the_policy_gate_armed_for_the_next_pour(
        self, mock_forge: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An allow_personal build-only pour must not open the gate for a later deploy."""
        policy = _personal_fleet(tmp_path, monkeypatch)
        deployer, nerve = _policy_deployer(mock_forge, [_BENCH_PHONE, _PERSONAL_WATCH])

        with patch.object(deployer, "_run_build", return_value=(True, "BUILD OK")):
            deployer.pour("PrecisionLevel", build_only=True, policy=policy, allow_personal=True)
        result = deployer.pour("PrecisionLevel", skip_build=True, policy=policy)

        assert _installed_serials(nerve) == [_BENCH_PHONE_SERIAL]
        assert result.excluded_personal == [_PERSONAL_WATCH_SERIAL]


class TestDarwinJavaHomeStubAndNoEnvAssertionContract:
    """[contract] REVIEW B9 guard: autouse resolve_java_home stub + no env= in mock assertions."""

    def test_build_env_with_java_home_issues_no_java_home_subprocess_on_darwin(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_forge: MagicMock,
        mock_nerve: MagicMock,
    ) -> None:
        """[contract] With sys.platform patched to 'darwin' and subprocess.run patched,
        Deployer._build_env_with_java_home issues no /usr/libexec/java_home call when
        resolve_java_home is the module autouse stub."""
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        monkeypatch.delenv("JAVA_HOME", raising=False)
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        with patch("subprocess.run") as mock_subprocess:
            build_env = deployer._build_env_with_java_home(
                None, command="./gradlew test", cwd="/tmp"
            )

        mock_subprocess.assert_not_called()
        assert "JAVA_HOME" not in build_env

    def test_no_mock_call_assertion_includes_env_kwarg(self) -> None:
        """[contract] No mock call assertion in this module may include env in
        expected kwargs, so the secret sink cannot come back."""
        import ast
        import inspect
        import sys

        guarded = {f"assert_called{s}_with" for s in ("", "_once")}
        source = inspect.getsource(sys.modules[__name__])
        offenders = [
            ast.unparse(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr in guarded)
                or (isinstance(node.func, ast.Name) and node.func.id in guarded)
            )
            and any(kw.arg == "env" for kw in node.keywords)
        ]
        assert offenders == []


class TestPourNerveUnreachable:
    """[test] pour with discover raising ConnectionError -> result error names unreachable."""

    def test_single_target_pour_discover_connection_error(
        self, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        from linc_crucible.core.deployer import NERVE_UNREACHABLE_MSG

        mock_nerve.discover.side_effect = ConnectionError(
            "Nerve unreachable at http://localhost:8111"
        )
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("PrecisionLevel")

        assert result.project == "PrecisionLevel"
        assert result.success is False
        assert "Nerve unreachable" in result.error
        assert result.error == NERVE_UNREACHABLE_MSG
        assert result.build_output == NERVE_UNREACHABLE_MSG
        assert result.to_dict()["error"] == NERVE_UNREACHABLE_MSG

    def test_discover_or_raise_skips_health_when_devices_present(
        self, mock_nerve: MagicMock
    ) -> None:
        from linc_crucible.core.deployer import _discover_or_raise

        mock_nerve.discover.return_value = [{"serial": "D1", "state": "online"}]
        mock_nerve.health.side_effect = ConnectionError("should not be called")

        assert _discover_or_raise(mock_nerve) == [{"serial": "D1", "state": "online"}]
        mock_nerve.health.assert_not_called()

    def test_multi_target_pour_discover_connection_error(
        self, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        from linc_crucible.core.deployer import NERVE_UNREACHABLE_MSG

        mock_forge.get.return_value = ForgeProject(
            name="MultiApp",
            path="/tmp/MultiApp",
            modules=[ModuleConfig(name="phoneApp", device_types=["phone"])],
        )
        mock_nerve.discover.side_effect = ConnectionError(
            "Nerve unreachable at http://localhost:8111"
        )
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("MultiApp", skip_build=True)

        assert result.success is False
        assert result.error == NERVE_UNREACHABLE_MSG

    def test_single_module_pour_discover_connection_error(
        self, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        from linc_crucible.core.deployer import NERVE_UNREACHABLE_MSG

        mock_forge.get.return_value = ForgeProject(
            name="MultiApp",
            path="/tmp/MultiApp",
            modules=[ModuleConfig(name="phoneApp", device_types=["phone"])],
        )
        mock_nerve.discover.side_effect = ConnectionError(
            "Nerve unreachable at http://localhost:8111"
        )
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("MultiApp", module="phoneApp", skip_build=True)

        assert result.success is False
        assert result.error == NERVE_UNREACHABLE_MSG

    def test_pour_with_empty_discover_and_unhealthy_nerve_reports_unreachable(
        self, mock_forge: MagicMock, mock_nerve: MagicMock
    ) -> None:
        from linc_crucible.core.deployer import NERVE_UNREACHABLE_MSG

        mock_nerve.discover.return_value = []
        mock_nerve.health.side_effect = ConnectionError(
            "Nerve unreachable at http://localhost:8111"
        )
        deployer = Deployer(forge_reader=mock_forge, nerve_client=mock_nerve)

        result = deployer.pour("PrecisionLevel")

        assert result.success is False
        assert result.error == NERVE_UNREACHABLE_MSG
