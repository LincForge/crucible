"""Characterization tests for CRAP warning sweep — zero/low-cov functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Assayer._run_security ────────────────────────────────────────────
class TestRunSecurity:
    def _make_assayer(self, *, api_key: str | None = None):
        from linc_crucible.core.assayer import Assayer

        a = Assayer.__new__(Assayer)
        a._mobsf_api_key = api_key
        a._mobsf_url = "http://localhost:8000"
        a._forge = MagicMock()
        return a

    def test_returns_failure_when_no_api_key(self) -> None:
        a = self._make_assayer(api_key=None)
        project = MagicMock()
        project.name = "PrecisionLevel"

        result = a._run_security(project)
        assert result.success is False
        assert "API key" in result.output

    def test_returns_failure_when_no_apk_found(self) -> None:
        a = self._make_assayer(api_key="fake-key")
        a._forge.find_apk.return_value = None
        project = MagicMock()
        project.name = "PrecisionLevel"

        result = a._run_security(project)
        assert result.success is False
        assert "No APK found" in result.output

    def test_success_path_uploads_scans_and_reports(self, tmp_path) -> None:
        a = self._make_assayer(api_key="fake-key")
        apk_path = tmp_path / "app.apk"
        apk_path.write_bytes(b"fake-apk")
        a._forge.find_apk.return_value = str(apk_path)
        project = MagicMock()
        project.name = "PrecisionLevel"

        upload_resp = MagicMock()
        upload_resp.raise_for_status = MagicMock()
        upload_resp.json.return_value = {"hash": "abc123"}

        scan_resp = MagicMock()
        scan_resp.raise_for_status = MagicMock()

        report_resp = MagicMock()
        report_resp.raise_for_status = MagicMock()
        report_resp.json.return_value = {"security_score": 87.5}

        with patch(
            "linc_crucible.core.assayer.httpx.Client.post",
            side_effect=[upload_resp, scan_resp, report_resp],
        ):
            result = a._run_security(project)
        assert result.success is True
        assert result.score == 87.5

    def test_returns_failure_on_http_error(self, tmp_path) -> None:
        import httpx

        a = self._make_assayer(api_key="fake-key")
        apk_path = tmp_path / "app.apk"
        apk_path.write_bytes(b"fake")
        a._forge.find_apk.return_value = str(apk_path)
        project = MagicMock()
        project.name = "PrecisionLevel"

        with patch(
            "linc_crucible.core.assayer.httpx.Client.post",
            side_effect=httpx.HTTPError("connection refused"),
        ):
            result = a._run_security(project)
        assert result.success is False
        assert "connection refused" in result.output


# ── PreflightChecker.__init__ ────────────────────────────────────────
class TestPreflightInit:
    def test_defaults_use_default_workspace(self) -> None:
        import os

        from linc_crucible.core.preflight import PreflightChecker, _default_linc_workspace

        # Remove only the LINC env vars preflight reads, preserving the rest of
        # os.environ. clear=True would also strip mutmut 3.x's MUTANT_UNDER_TEST,
        # whose instrumentation trampoline reads it via os.environ[...] (KeyError
        # under the mutation harness). patch.dict still restores on exit.
        with patch.dict(os.environ, clear=False):
            os.environ.pop("LINC_WORKSPACE", None)
            os.environ.pop("LINC_HOME", None)
            p = PreflightChecker()
            # Capture expected INSIDE the controlled env. Computing it after the
            # with-block would see a restored LINC_HOME (set under the sandboxed
            # HOME used by the autonomy wrappers / CI), diverging from the
            # home-based path p resolved while LINC_HOME was popped (inbox 2cbfc285).
            expected = _default_linc_workspace()
        assert p._linc_workspace == expected
        assert p._forge is not None
        assert p._scorer is not None

    def test_honors_env_var(self, tmp_path) -> None:
        import os

        from linc_crucible.core.preflight import PreflightChecker

        with patch.dict(os.environ, {"LINC_WORKSPACE": str(tmp_path)}):
            p = PreflightChecker()
        assert p._linc_workspace == Path(str(tmp_path))

    def test_explicit_workspace_wins_over_env(self, tmp_path) -> None:
        import os

        from linc_crucible.core.preflight import PreflightChecker

        with patch.dict(os.environ, {"LINC_WORKSPACE": "/tmp/env"}):
            p = PreflightChecker(linc_workspace=tmp_path)
        assert p._linc_workspace == tmp_path


# ── AssayReport.to_dict ──────────────────────────────────────────────
class TestAssayReportToDict:
    def test_minimal_report_serializes(self) -> None:
        from linc_crucible.core.assayer import AssayReport, LintResult, RunResult

        r = AssayReport(
            project="PL",
            mode="quick",
            lint=LintResult(issues=3, success=False),
            tests=RunResult(total=10, passed=9, failed=1, skipped=0, success=False),
        )
        d = r.to_dict()
        assert d["project"] == "PL"
        assert d["mode"] == "quick"
        assert d["lint"]["issues"] == 3
        assert d["tests"]["passed"] == 9
        assert d["tests"]["failed"] == 1
        assert "security" not in d

    def test_full_report_includes_all_optional_sections(self) -> None:
        from linc_crucible.core.assayer import (
            A11yTreeResult,
            AccessibilityResult,
            AssayReport,
            LintResult,
            PerformanceResult,
            RunResult,
            SecurityResult,
        )

        r = AssayReport(
            project="PL",
            mode="full",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=5, passed=5, failed=0, skipped=0, success=True),
            security=SecurityResult(score=90.0, success=True),
            performance=PerformanceResult(apk_size_mb=12.345, ipa_size_mb=20.0, success=True),
            accessibility=AccessibilityResult(
                critical=0, major=1, minor=2, score=95.0, success=True, details=[]
            ),
            a11y_tree=A11yTreeResult(
                structural_score=1.0, rules_score=1.0, success=True, error=""
            ),
        )
        d = r.to_dict()
        assert d["security"]["score"] == 90.0
        assert d["performance"]["apk_size_mb"] == 12.35
        assert d["accessibility"]["critical"] == 0
        assert d["a11y_tree"]["structural_score"] == 1.0


# ── asset_extractor.filter_frames ───────────────────────────────────
class TestFilterFrames:
    def test_unreadable_images_are_skipped(self, tmp_path) -> None:
        from linc_crucible.core import asset_extractor

        path = tmp_path / "bad.png"
        path.write_bytes(b"not an image")
        survivors = asset_extractor.filter_frames([path], blur_threshold=50.0, phash_threshold=5)
        assert survivors == []

    def test_blurry_frames_dropped(self, tmp_path) -> None:
        import numpy as np
        from PIL import Image

        from linc_crucible.core import asset_extractor

        path = tmp_path / "blur.png"
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path)

        with patch("linc_crucible.core.asset_extractor._laplacian_variance", return_value=10.0):
            survivors = asset_extractor.filter_frames(
                [path], blur_threshold=50.0, phash_threshold=5
            )
        assert survivors == []

    def test_sharp_unique_frame_survives(self, tmp_path) -> None:
        import numpy as np
        from PIL import Image

        from linc_crucible.core import asset_extractor

        path = tmp_path / "sharp.png"
        Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 255).save(path)
        fake_hash = MagicMock()
        fake_hash.__sub__ = lambda self, other: 10  # large distance → unique

        with (
            patch("linc_crucible.core.asset_extractor._laplacian_variance", return_value=200.0),
            patch("linc_crucible.core.asset_extractor.imagehash.phash", return_value=fake_hash),
        ):
            survivors = asset_extractor.filter_frames(
                [path], blur_threshold=50.0, phash_threshold=5
            )
        assert survivors == [path]


# ── FleetRegistry._discover_via_nerve ────────────────────────────────
class TestDiscoverViaNerve:
    def _make_registry(self):
        from linc_crucible.core.fleet_registry import FleetRegistry

        r = FleetRegistry.__new__(FleetRegistry)
        r._devices = {}
        r._nerve = MagicMock()
        return r

    def test_adds_new_device(self) -> None:
        r = self._make_registry()
        r._nerve.discover.return_value = [
            {
                "serial": "ABC123",
                "platform": "android",
                "device_type": "phone",
                "model": "Pixel 8 Pro",
                "os_version": "14",
            }
        ]
        new = r._discover_via_nerve()
        assert len(new) == 1
        assert "ABC123" in r._devices
        assert r._devices["ABC123"].model == "Pixel 8 Pro"

    def test_skips_devices_without_serial(self) -> None:
        r = self._make_registry()
        r._nerve.discover.return_value = [{"serial": "", "platform": "android"}]
        new = r._discover_via_nerve()
        assert new == []
        assert r._devices == {}

    def test_updates_existing_device_state(self) -> None:
        from linc_crucible.core.device import DeviceType, FleetDevice, Platform

        r = self._make_registry()
        existing = FleetDevice(
            serial="ABC",
            platform=Platform.ANDROID,
            device_type=DeviceType.PHONE,
            model="Pixel 8",
            os_version="14",
            state="offline",
            is_simulator=False,
        )
        r._devices["ABC"] = existing
        r._nerve.discover.return_value = [
            {"serial": "ABC", "platform": "android", "device_type": "phone"}
        ]
        new = r._discover_via_nerve()
        assert new == []  # not new, just state update
        assert r._devices["ABC"].state == "online"

    def test_marks_simulators_via_device_type(self) -> None:
        r = self._make_registry()
        r._nerve.discover.return_value = [
            {
                "serial": "SIM-1",
                "platform": "ios",
                "device_type": "simulator",
                "model": "iPhone 15",
                "os_version": "17.0",
            }
        ]
        r._discover_via_nerve()
        assert r._devices["SIM-1"].is_simulator is True


# ── Deployer._resolve_targets_for_module ──────────────────────────────
class TestResolveTargetsForModule:
    def _make_deployer(self):
        from linc_crucible.core.deployer import Deployer

        d = Deployer.__new__(Deployer)
        d._forge = MagicMock()
        d._nerve = MagicMock()
        return d

    def test_explicit_device_override_still_filters_by_form_factor(self) -> None:
        """An explicit serial is honored but still gated by the module's
        device_types — a phone serial must not receive a watch-only module
        (inbox 2e5895b6)."""
        d = self._make_deployer()
        phone_module = MagicMock(device_types=["phone"])
        watch_module = MagicMock(device_types=["watch"])
        phone = {"serial": "EXPLICIT", "device_type": "phone", "platform": "android"}
        with patch.object(d, "_resolve_targets", return_value=[phone]) as mock_resolve:
            kept = d._resolve_targets_for_module(
                phone_module, device="EXPLICIT", platform="android", device_type=""
            )
            dropped = d._resolve_targets_for_module(
                watch_module, device="EXPLICIT", platform="android", device_type=""
            )
        assert kept == [phone]
        assert dropped == []
        mock_resolve.assert_called_with("EXPLICIT", "android")

    def test_filters_by_module_device_types(self) -> None:
        d = self._make_deployer()
        module = MagicMock(device_types=["phone"])
        candidates = [
            {"serial": "a", "device_type": "phone", "platform": "android"},
            {"serial": "b", "device_type": "tablet", "platform": "android"},
        ]
        with patch.object(d, "_resolve_targets", return_value=candidates):
            result = d._resolve_targets_for_module(
                module, device="", platform="android", device_type=""
            )
        serials = [r["serial"] for r in result]
        assert "a" in serials
        assert "b" not in serials

    def test_ios_platform_match_passes_even_when_device_type_mismatches(self) -> None:
        from linc_crucible.core.device import Platform

        d = self._make_deployer()
        module = MagicMock(device_types=["phone"])
        candidates = [
            {"serial": "sim", "device_type": "emulator", "platform": "ios"},
            {"serial": "android-tab", "device_type": "tablet", "platform": "android"},
        ]
        with patch.object(d, "_resolve_targets", return_value=candidates):
            result = d._resolve_targets_for_module(
                module,
                device="",
                platform="",
                device_type="",
                artifact_platforms={Platform.IOS},
            )
        serials = [r["serial"] for r in result]
        assert "sim" in serials
        assert "android-tab" not in serials
