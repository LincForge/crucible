"""Tests for pure APK static audit core (archive, xmltree manifest, signing, assetlinks)."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linc_crucible.core import apk_audit as aa
from linc_crucible.core.apk_audit import (
    _MAX_ZIP_ENTRIES,
    ApkAuditError,
    _default_run_cmd,
    audit_apk,
    check_assetlinks,
    find_apksigner,
    inspect_apk_archive,
    normalize_sha256_fingerprint,
    parse_aapt2_xmltree,
    parse_apksigner_certs,
    summarize_findings,
)


def _write_zip(
    path: Path,
    entries: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    dirs: tuple[str, ...] = (),
) -> Path:
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for d in dirs:
            zf.writestr(d, b"")
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _incompressible_bytes(length: int, seed: int = 12345) -> bytes:
    """Deterministic pseudo-random bytes that do not deflate much below original size."""
    rng = random.Random(seed)
    chunk = rng.randbytes(65536)
    repeats, rem = divmod(length, len(chunk))
    return chunk * repeats + chunk[:rem]


def _format_digest(digest_bytes: bytes, style: str) -> str:
    hex_lower = digest_bytes.hex()
    if style == "raw_lower":
        return hex_lower
    if style == "raw_upper":
        return hex_lower.upper()
    pairs = [hex_lower[i : i + 2] for i in range(0, 64, 2)]
    if style == "colon_lower":
        return ":".join(pairs)
    return ":".join(p.upper() for p in pairs)


_NS = "http://schemas.android.com/apk/res/android"

CLEAN_XMLTREE = (
    f"N: android={_NS} (line=2)\n"
    "  E: manifest (line=2)\n"
    f"    A: {_NS}:versionCode(0x0101021b)=42\n"
    f'    A: {_NS}:versionName(0x0101021c)="1.2.3" (Raw: "1.2.3")\n'
    '    A: package="com.example.demo" (Raw: "com.example.demo")\n'
    "      E: uses-sdk (line=6)\n"
    f"        A: {_NS}:minSdkVersion(0x0101020c)=26\n"
    f"        A: {_NS}:targetSdkVersion(0x01010270)=35\n"
    "      E: application (line=7)\n"
    f"        A: {_NS}:debuggable(0x0101000f)=false\n"
    f"        A: {_NS}:allowBackup(0x01010280)=false\n"
    f"        A: {_NS}:usesCleartextTraffic(0x010104ec)=false\n"
    "          E: activity (line=10)\n"
    f'            A: {_NS}:name(0x01010003)=".MainActivity" (Raw: ".MainActivity")\n'
    f"            A: {_NS}:exported(0x01010010)=true\n"
    f"            A: {_NS}:configChanges(0x0101001f)=0x000006b0\n"
    "              E: intent-filter (line=14)\n"
    f"                A: {_NS}:autoVerify(0x010104ee)=true\n"
    "                  E: action (line=15)\n"
    f'                    A: {_NS}:name(0x01010003)="android.intent.action.MAIN"\n'
    "                  E: category (line=16)\n"
    f'                    A: {_NS}:name(0x01010003)="android.intent.category.LAUNCHER"\n'
    "                  E: data (line=17)\n"
    f'                    A: {_NS}:scheme(0x01010027)="https"\n'
    f'                    A: {_NS}:host(0x01010028)="app.example.com"\n'
)

HEX_FLAG_XMLTREE = (
    f"N: ns0={_NS}\n"
    "  E: manifest (line=2)\n"
    "    A: ns0:versionCode(0x0101021b)=(type 0x10)0x7\n"
    '    A: ns0:versionName(0x0101021c)="2.0.0" (Raw: "2.0.0")\n'
    '    A: package="com.example.hexapp" (Raw: "com.example.hexapp")\n'
    "    E: uses-sdk (line=5)\n"
    "      A: android:minSdkVersion(0x0101020c)=(type 0x10)0x18\n"
    "      A: android:targetSdkVersion(0x01010270)=(type 0x10)0x23\n"
    "    E: application (line=6)\n"
    "      A: android:debuggable(0x0101000f)=(type 0x12)0xffffffff\n"
    "      A: android:usesCleartextTraffic(0x010104ec)=0x1\n"
    "      A: android:allowBackup(0x01010280)=(type 0x12)0x0\n"
    "      E: activity (line=10)\n"
    '        A: android:name(0x01010003)=".LauncherNoConfig" (Raw: ".LauncherNoConfig")\n'
    "        A: android:configChanges(0x0101001f)=(type 0x11)0x4b0\n"
    "        E: intent-filter (line=13)\n"
    "          A: android:autoVerify(0x010104ee)=0xffffffff\n"
    "          E: action (line=14)\n"
    '            A: android:name(0x01010003)="android.intent.action.MAIN"\n'
    "          E: category (line=15)\n"
    '            A: android:name(0x01010003)="android.intent.category.LAUNCHER"\n'
    "          E: data (line=16)\n"
    '            A: android:scheme(0x01010027)="https"\n'
    "          E: data (line=17)\n"
    '            A: android:host(0x01010028)="links.example.com"\n'
    "          E: data (line=18)\n"
    '            A: android:host(0x01010028)="app.example.com"\n'
    "      E: activity (line=20)\n"
    '        A: android:name(0x01010003)=".UiModeSafe" (Raw: ".UiModeSafe")\n'
    "        A: android:exported(0x01010010)=0x1\n"
    "        A: android:configChanges(0x0101001f)=0x6b0\n"
    "      E: activity (line=23)\n"
    '        A: android:name(0x01010003)=".InternalActivity" (Raw: ".InternalActivity")\n'
    "        A: android:exported(0x01010010)=0x0\n"
    "        A: android:configChanges(0x0101001f)=0x4b0\n"
    "        E: intent-filter (line=26)\n"
    "          A: android:autoVerify(0x010104ee)=0x1\n"
    "          E: data (line=27)\n"
    '            A: android:scheme(0x01010027)="http"\n'
    '            A: android:host(0x01010028)="insecure.example.com"\n'
)

SAMPLE_HEX = "aa11bb22cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233"
APKSIGNER_SAMPLE = (
    "Signer #1 certificate DN: CN=Example, O=Demo, C=US\n"
    f"Signer #1 certificate SHA-256 digest: {SAMPLE_HEX}\n"
    "Signer #1 certificate SHA-1 digest: 00112233445566778899aabbccddeeff00112233\n"
    "Signer #1 certificate MD5 digest: 00112233445566778899aabbccddeeff\n"
)

SAMPLE_FINGERPRINT_COLON = (
    "AA:11:BB:22:CC:33:DD:44:EE:55:FF:66:00:11:22:33:"
    "44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33"
)


def _assert_summary_matches_findings(report: dict) -> None:
    findings = report["findings"]
    summary = report["summary"]
    for sev in ("high", "medium", "low", "info"):
        assert summary[sev] == len([f for f in findings if f["severity"] == sev])
    assert summary["high"] + summary["medium"] + summary["low"] + summary["info"] == len(findings)


def test_archive_clean_vs_bloated_dex_and_sourcemaps(tmp_path: Path) -> None:
    """[contract] Clean small APK vs bloated 6 MB zip with 12 MB classes.dex + sourcemaps."""
    clean_apk = _write_zip(
        tmp_path / "clean.apk",
        {
            "AndroidManifest.xml": b"binary-xml",
            "classes.dex": b"dex\n035\x00" + b"\x01" * 1024,
            "lib/arm64-v8a/libcore.so": b"elf",
            "lib/armeabi-v7a/libcore.so": b"elf",
        },
        dirs=("lib/", "assets/"),
    )
    clean_report = audit_apk(clean_apk, aapt_bin=None, apksigner_bin=None)
    assert clean_report["archive"]["dex_count"] == 1
    assert clean_report["archive"]["dex_stored"] is False
    assert clean_report["archive"]["abis"] == ["arm64-v8a", "armeabi-v7a"]
    assert clean_report["archive"]["sourcemap_files"] == []
    assert clean_report["findings"] == []
    assert clean_report["summary"]["measured"] is False
    assert clean_report["summary"]["passed"] is False
    _assert_summary_matches_findings(clean_report)

    bloated_dex = _incompressible_bytes(6_200_000) + b"\x00" * 5_800_000
    bloated_apk = _write_zip(
        tmp_path / "bloated.apk",
        {
            "AndroidManifest.xml": b"binary-xml",
            "classes.dex": bloated_dex,
            "classes2.dex": b"dex\n035\x00" + b"\x02" * 128,
            "assets/index.bundle.map": b"{}",
            "res/raw/app.js.map": b"{}",
            "lib/x86_64/libcore.so": b"elf",
        },
    )
    bloated_report = audit_apk(bloated_apk, aapt_bin=None, apksigner_bin=None)
    arc = bloated_report["archive"]
    assert arc["apk_bytes"] >= 5_000_000
    assert arc["dex_count"] == 2
    assert arc["dex_uncompressed_bytes"] >= 10_000_000
    assert arc["dex_compressed_ratio"] == round(arc["dex_compressed_bytes"] / arc["apk_bytes"], 4)
    assert arc["dex_stored"] is False
    assert arc["abis"] == ["x86_64"]
    assert arc["sourcemap_files"] == ["assets/index.bundle.map", "res/raw/app.js.map"]

    by_code = {f["code"]: f for f in bloated_report["findings"]}
    assert by_code["unminified_dex_bloat"]["severity"] == "medium"
    assert by_code["bundled_sourcemaps"]["severity"] == "low"
    _assert_summary_matches_findings(bloated_report)
    assert bloated_report["summary"]["medium"] == 1
    assert bloated_report["summary"]["low"] == 1
    assert bloated_report["summary"]["measured"] is False
    assert bloated_report["summary"]["passed"] is False


def test_archive_bloat_ratio_trigger_and_zero_apk_bytes_guard(tmp_path: Path) -> None:
    """[contract] Ratio >= 0.65 triggers bloat under 10 MB DEX; apk_bytes == 0 is guarded."""
    ratio_apk = _write_zip(
        tmp_path / "ratio_bloat.apk",
        {
            "classes.dex": _incompressible_bytes(5_200_000, seed=99),
            "assets/small.txt": b"hi",
        },
    )
    report = audit_apk(ratio_apk, aapt_bin=None, apksigner_bin=None)
    assert report["archive"]["apk_bytes"] >= 5_000_000
    assert report["archive"]["dex_uncompressed_bytes"] < 10_000_000
    assert report["archive"]["dex_compressed_ratio"] >= 0.65
    assert [f["code"] for f in report["findings"]] == ["unminified_dex_bloat"]

    small_apk = _write_zip(tmp_path / "small.apk", {"classes.dex": b"abc"})
    real_stat = Path.stat

    def _zero_size_stat(self: Path, *args, **kwargs):
        st = real_stat(self, *args, **kwargs)
        if self == small_apk:
            vals = list(st)
            vals[6] = 0
            return type(st)(vals)
        return st

    with patch.object(Path, "stat", _zero_size_stat):
        arc = inspect_apk_archive(small_apk)
    assert arc["apk_bytes"] == 0
    assert arc["dex_compressed_ratio"] == 0.0


def test_parse_aapt2_xmltree_hex_and_boolean_forms() -> None:
    """[contract] Parses hex (0x1/0xffffffff/0x0, configChanges 0x4b0 vs 0x6b0) & boolean forms."""
    clean = parse_aapt2_xmltree(CLEAN_XMLTREE)
    assert clean["package"] == "com.example.demo"
    assert clean["version_code"] == 42
    assert clean["version_name"] == "1.2.3"
    assert clean["min_sdk"] == 26
    assert clean["target_sdk"] == 35
    assert clean["debuggable"] is False
    assert clean["uses_cleartext_traffic"] is False
    assert clean["allow_backup"] is False
    assert clean["activities_missing_uimode_config"] == []
    assert clean["autoverify_hosts"] == ["app.example.com"]

    hex_manifest = parse_aapt2_xmltree(HEX_FLAG_XMLTREE)
    assert hex_manifest["package"] == "com.example.hexapp"
    assert hex_manifest["version_code"] == 7
    assert hex_manifest["version_name"] == "2.0.0"
    assert hex_manifest["min_sdk"] == 24
    assert hex_manifest["target_sdk"] == 35
    assert hex_manifest["debuggable"] is True
    assert hex_manifest["uses_cleartext_traffic"] is True
    assert hex_manifest["allow_backup"] is False
    assert hex_manifest["activities_missing_uimode_config"] == [".LauncherNoConfig"]
    assert hex_manifest["autoverify_hosts"] == ["app.example.com", "links.example.com"]


def test_parse_aapt2_xmltree_edge_cases() -> None:
    """Covers escaped quotes, codename SDK, missing configChanges, string uiMode, empty lines."""
    edge_xml = (
        "\n"
        "N: app=http://schemas.android.com/apk/res-auto\n"
        "A: orphan=ignored\n"
        "E:\n"
        "  E: manifest\n"
        '    A: package="com.example.edge"\n'
        '    A: android:versionName="1.0\\"beta\\\\rc"\n'
        "    A: android:versionCode=not-a-number\n"
        "    A: app:debuggable=true\n"
        "    A: malformed_attr_without_equals\n"
        "    E: permission\n"
        '      A: android:name="com.example.PERM"\n'
        "    E: uses-sdk\n"
        '      A: android:minSdkVersion="Baklava"\n'
        '      A: android:targetSdkVersion=""\n'
        "    E: application\n"
        "      A: android:debuggable=0xZZZZ\n"
        "      A: android:usesCleartextTraffic=maybe\n"
        "      E: service\n"
        '        A: android:name=".BgService"\n'
        "      E: activity\n"
        "        A: android:exported=true\n"
        "      E: activity\n"
        '        A: android:name=".StringUiMode\n'
        "        A: android:exported=true\n"
        '        A: android:configChanges="orientation|uiMode"\n'
        "        E: intent-filter\n"
        "          A: android:autoVerify=true\n"
        "          E: data\n"
        '            A: android:scheme="https"\n'
    )
    parsed = parse_aapt2_xmltree(edge_xml)
    assert parsed["package"] == "com.example.edge"
    assert parsed["version_name"] == '1.0"beta\\rc'
    assert parsed["version_code"] is None
    assert parsed["min_sdk"] == "Baklava"
    assert parsed["target_sdk"] is None
    assert parsed["debuggable"] is False
    assert parsed["uses_cleartext_traffic"] is False
    assert parsed["allow_backup"] is None
    assert parsed["activities_missing_uimode_config"] == ["<unnamed>"]
    assert parsed["autoverify_hosts"] == []

    empty_parsed = parse_aapt2_xmltree("")
    assert empty_parsed["package"] is None
    assert empty_parsed["activities_missing_uimode_config"] == []

    minimal_parsed = parse_aapt2_xmltree('E: manifest\n  A: package="com.example.min"\n')
    assert minimal_parsed["package"] == "com.example.min"
    assert minimal_parsed["min_sdk"] is None
    assert minimal_parsed["debuggable"] is False


def test_audit_apk_manifest_findings_and_backup_true(tmp_path: Path) -> None:
    """[contract] Emits debuggable_enabled, cleartext_traffic_enabled, backup_enabled, uimode."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    xmltree_with_backup = HEX_FLAG_XMLTREE.replace(
        "android:allowBackup(0x01010280)=(type 0x12)0x0",
        "android:allowBackup(0x01010280)=0x1",
    )
    seen_cmds: list[list[str]] = []

    def _runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(cmd)
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=xmltree_with_backup, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=APKSIGNER_SAMPLE, stderr="")

    report = audit_apk(
        apk,
        aapt_bin="/usr/bin/aapt",
        apksigner_bin="/usr/bin/apksigner",
        run_cmd=_runner,
    )
    assert seen_cmds[0] == ["/usr/bin/aapt", "dump", "xmltree", str(apk), "AndroidManifest.xml"]
    assert seen_cmds[1] == ["/usr/bin/apksigner", "verify", "--print-certs", str(apk)]
    codes = {f["code"]: f["severity"] for f in report["findings"]}
    assert codes == {
        "debuggable_enabled": "high",
        "cleartext_traffic_enabled": "high",
        "backup_enabled": "low",
        "activity_recreates_on_uimode": "info",
    }
    assert report["manifest"]["allow_backup"] is True
    assert report["summary"]["high"] == 2
    assert report["summary"]["low"] == 1
    assert report["summary"]["info"] == 1
    assert report["summary"]["passed"] is False
    _assert_summary_matches_findings(report)


def test_signing_and_assetlinks_match_and_mismatch(tmp_path: Path) -> None:
    """[contract] Colon-hex & lowercase raw-hex match; mismatch emits high & passed=False."""
    apk = _write_zip(tmp_path / "signed.apk", {"classes.dex": b"dex"})

    def _runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=CLEAN_XMLTREE, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=APKSIGNER_SAMPLE, stderr="")

    certs = parse_apksigner_certs(
        APKSIGNER_SAMPLE
        + f"Signer #2 certificate SHA-256 digest: {SAMPLE_HEX.upper()}\n"
        + "Signer #3 certificate SHA-256 digest: deadbeef\n"
    )
    assert certs == [SAMPLE_FINGERPRINT_COLON]

    al_match = tmp_path / "assetlinks_match.json"
    al_match.write_text(
        json.dumps(
            [
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [
                            SAMPLE_FINGERPRINT_COLON.replace(":", "").lower()
                        ],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    report_ok = audit_apk(
        apk,
        assetlinks_path=al_match,
        aapt_bin="/usr/bin/aapt2",
        apksigner_bin="/usr/bin/apksigner",
        run_cmd=_runner,
    )
    assert report_ok["signing"]["status"] == "ok"
    assert report_ok["signing"]["cert_sha256s"] == [SAMPLE_FINGERPRINT_COLON]
    assert report_ok["assetlinks"]["status"] == "matched"
    assert report_ok["findings"] == []
    assert report_ok["summary"]["passed"] is True
    json.dumps(report_ok)

    al_mismatch = tmp_path / "assetlinks_mismatch.json"
    al_mismatch.write_text(
        json.dumps(
            [
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": ["FF:" * 31 + "FF"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    report_bad = audit_apk(
        apk,
        assetlinks_path=al_mismatch,
        aapt_bin="/usr/bin/aapt2",
        apksigner_bin="/usr/bin/apksigner",
        run_cmd=_runner,
    )
    assert report_bad["assetlinks"]["status"] == "mismatch"
    assert report_bad["findings"] == [
        {
            "severity": "high",
            "code": "assetlinks_sha256_mismatch",
            "detail": report_bad["findings"][0]["detail"],
        }
    ]
    assert report_bad["summary"]["high"] == 1
    assert report_bad["summary"]["passed"] is False

    al_missing_pkg = tmp_path / "assetlinks_other.json"
    al_missing_pkg.write_text(
        json.dumps(
            [
                {
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.other",
                        "sha256_cert_fingerprints": [SAMPLE_FINGERPRINT_COLON],
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    report_pkg_missing = audit_apk(
        apk,
        assetlinks_path=al_missing_pkg,
        aapt_bin="/usr/bin/aapt2",
        apksigner_bin="/usr/bin/apksigner",
        run_cmd=_runner,
    )
    assert report_pkg_missing["assetlinks"]["status"] == "package_missing"
    assert [f["code"] for f in report_pkg_missing["findings"]] == ["assetlinks_package_missing"]
    assert report_pkg_missing["summary"]["passed"] is False


def test_missing_and_failing_sdk_tools_are_honest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] Missing aapt2/apksigner reports unavailable without fabricating clean flags."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    al = tmp_path / "al.json"
    al.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    assert find_apksigner() is None

    monkeypatch.setenv("ANDROID_HOME", str(tmp_path / "empty_sdk"))
    assert find_apksigner() is None
    report = audit_apk(apk, assetlinks_path=al)
    assert report["manifest"]["status"] == "unavailable"
    assert report["manifest"]["debuggable"] is None
    assert report["manifest"]["uses_cleartext_traffic"] is None
    assert report["manifest"]["allow_backup"] is None
    assert report["signing"]["status"] == "unavailable"
    assert report["signing"]["cert_sha256s"] == []
    assert report["assetlinks"]["status"] == "unavailable"
    assert report["findings"] == [
        {
            "severity": "high",
            "code": "assetlinks_unavailable",
            "detail": _CANNOT_VERIFY,
        }
    ]
    assert report["summary"]["measured"] is False
    assert report["summary"]["passed"] is False

    report_no_al = audit_apk(apk)
    assert report_no_al["findings"] == []
    assert report_no_al["summary"]["measured"] is False
    assert report_no_al["summary"]["passed"] is False

    # Non-zero exit codes set status="error" and fail summary["passed"]
    def _failing_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed to verify")

    err_report = audit_apk(
        apk,
        aapt_bin="/usr/bin/aapt2",
        apksigner_bin="/usr/bin/apksigner",
        run_cmd=_failing_runner,
    )
    assert err_report["manifest"]["status"] == "error"
    assert err_report["manifest"]["debuggable"] is None
    assert err_report["signing"]["status"] == "error"
    assert err_report["summary"]["passed"] is False

    # Subprocess OSError and zero-exit empty output are both treated as errors
    def _raising_runner(_cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError("binary missing")

    os_err_report = audit_apk(
        apk,
        aapt_bin="/usr/bin/aapt2",
        apksigner_bin="/usr/bin/apksigner",
        run_cmd=_raising_runner,
    )
    assert os_err_report["manifest"]["status"] == "error"
    assert os_err_report["signing"]["status"] == "error"
    assert os_err_report["summary"]["passed"] is False

    def _empty_ok_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="nothing useful", stderr="")

    empty_report = audit_apk(
        apk,
        aapt_bin="/usr/bin/aapt2",
        apksigner_bin="/usr/bin/apksigner",
        run_cmd=_empty_ok_runner,
    )
    assert empty_report["manifest"]["status"] == "error"
    assert empty_report["signing"]["status"] == "error"
    assert empty_report["summary"]["passed"] is False

    # Default subprocess runner works cleanly with bounded args
    cp = _default_run_cmd([sys.executable, "-c", "print('ok')"])
    assert cp.returncode == 0
    assert cp.stdout.strip() == "ok"


def test_find_apksigner_prefers_which_then_sorted_sdk_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "shutil.which", lambda name: "/opt/bin/apksigner" if name == "apksigner" else None
    )
    assert find_apksigner() == "/opt/bin/apksigner"

    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.setenv("ANDROID_SDK_ROOT", str(tmp_path))
    for ver in ("33.0.2", "35.0.0", "34.0.0"):
        bt = tmp_path / "build-tools" / ver
        bt.mkdir(parents=True)
        (bt / "apksigner").write_text("", encoding="utf-8")
    assert find_apksigner() == str(tmp_path / "build-tools" / "35.0.0" / "apksigner")


def test_contract_find_apksigner_picks_highest_version_tuple_over_string_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] build-tools 9.0.0 and 35.0.0 present -> 35.0.0 chosen."""
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path))

    for ver in ("9.0.0", "35.0.0"):
        bt = tmp_path / "build-tools" / ver
        bt.mkdir(parents=True)
        (bt / "apksigner").write_text("", encoding="utf-8")

    assert find_apksigner() == str(tmp_path / "build-tools" / "35.0.0" / "apksigner")


def test_find_apksigner_version_edge_and_hostile_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pre-releases, non-numeric dirs, and non-file apksigner entries are handled safely."""
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path))

    (tmp_path / "build-tools" / "99.0.0" / "apksigner").mkdir(parents=True)
    for ver in ("custom", "9.0.0", "35.0.0-rc1", "35.0.0", "100.0.0"):
        bt = tmp_path / "build-tools" / ver
        bt.mkdir(parents=True)
        (bt / "apksigner").write_text("", encoding="utf-8")
    assert find_apksigner() == str(tmp_path / "build-tools" / "100.0.0" / "apksigner")


def test_hostile_apk_and_assetlinks_inputs(tmp_path: Path) -> None:
    """[contract] Hostile APK and assetlinks.json inputs raise ApkAuditError or skip safely."""
    with pytest.raises(ApkAuditError):
        audit_apk("")

    with pytest.raises(ApkAuditError):
        audit_apk(tmp_path / "does_not_exist.apk")

    with pytest.raises(ApkAuditError):
        audit_apk(tmp_path)

    broken_link = tmp_path / "broken_link.apk"
    broken_link.symlink_to(tmp_path / "missing_target.apk")
    with pytest.raises(ApkAuditError):
        audit_apk(broken_link)

    fifo_path = tmp_path / "pipe.apk"
    os.mkfifo(fifo_path)
    with pytest.raises(ApkAuditError):
        audit_apk(fifo_path)

    non_zip = tmp_path / "bad.apk"
    non_zip.write_bytes(b"not a zip archive at all")
    with pytest.raises(ApkAuditError):
        inspect_apk_archive(non_zip)

    fake_entries = [zipfile.ZipInfo(f"f{i}.txt") for i in range(_MAX_ZIP_ENTRIES + 1)]
    small_zip = _write_zip(tmp_path / "many.apk", {"classes.dex": b"dex"})
    with (
        patch("zipfile.ZipFile.infolist", return_value=fake_entries),
        pytest.raises(ApkAuditError, match=r"50,000|50000"),
    ):
        inspect_apk_archive(small_zip)

    bad_info = zipfile.ZipInfo("classes.dex")
    bad_info.file_size = -1
    with (
        patch("zipfile.ZipFile.infolist", return_value=[bad_info]),
        pytest.raises(ApkAuditError),
    ):
        inspect_apk_archive(small_zip)

    with (
        patch.object(Path, "open", side_effect=OSError("disk read failed")),
        pytest.raises(ApkAuditError),
    ):
        audit_apk(small_zip, aapt_bin=None, apksigner_bin=None)

    for bad_path in ("", tmp_path, tmp_path / "missing.json"):
        with pytest.raises(ApkAuditError):
            check_assetlinks([SAMPLE_FINGERPRINT_COLON], "com.example.demo", bad_path)

    non_json = tmp_path / "non_json.json"
    non_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ApkAuditError):
        check_assetlinks([SAMPLE_FINGERPRINT_COLON], "com.example.demo", non_json)

    obj_json = tmp_path / "obj.json"
    obj_json.write_text('{"target": {}}', encoding="utf-8")
    with pytest.raises(ApkAuditError):
        check_assetlinks([SAMPLE_FINGERPRINT_COLON], "com.example.demo", obj_json)

    oversized = tmp_path / "huge.json"
    oversized.write_bytes(b"[" + b" " * (1024 * 1024 + 16) + b"]")
    with pytest.raises(ApkAuditError):
        check_assetlinks([SAMPLE_FINGERPRINT_COLON], "com.example.demo", oversized)

    hostile_entries = tmp_path / "hostile_entries.json"
    hostile_entries.write_text(
        json.dumps(
            [
                "not-a-dict",
                42,
                {"target": "not-a-dict"},
                {"target": {"namespace": "web", "package_name": "com.example.demo"}},
                {
                    "target": {
                        "namespace": "android_app",
                        "package_name": 123,
                        "sha256_cert_fingerprints": [SAMPLE_FINGERPRINT_COLON],
                    }
                },
                {
                    "relation": "not-a-list",
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_B],
                    },
                },
                {
                    "relation": [None, 123, {"rel": "delegate_permission/common.handle_all_urls"}],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_B],
                    },
                },
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": "not-a-list",
                    },
                },
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [
                            None,
                            12345,
                            {"hex": "deadbeef"},
                            "not-hex",
                            "AA:BB",
                            SAMPLE_FINGERPRINT_COLON.lower(),
                        ],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    res_no_pkg = check_assetlinks([SAMPLE_FINGERPRINT_COLON], None, hostile_entries)
    assert res_no_pkg["status"] == "unavailable"

    res = check_assetlinks([SAMPLE_FINGERPRINT_COLON], "com.example.demo", hostile_entries)
    assert res["status"] == "matched"
    assert res["matched_cert_sha256s"] == [SAMPLE_FINGERPRINT_COLON]


def _check_digest_and_summary_invariants(
    digest_a: bytes,
    digest_b: bytes,
    fmt_apksigner: str,
    fmt_assetlinks: str,
    tmp_dir: Path,
) -> None:
    s_apksigner = _format_digest(digest_a, fmt_apksigner)
    s_assetlinks = _format_digest(digest_b, fmt_assetlinks)

    norm_a = normalize_sha256_fingerprint(s_apksigner)
    norm_b = normalize_sha256_fingerprint(s_assetlinks)
    assert norm_a is not None
    assert norm_b is not None
    assert normalize_sha256_fingerprint(norm_a) == norm_a
    assert normalize_sha256_fingerprint(norm_b) == norm_b
    assert (norm_a == norm_b) == (digest_a == digest_b)

    parsed_certs = parse_apksigner_certs(f"Signer #1 certificate SHA-256 digest: {s_apksigner}\n")
    assert parsed_certs == [norm_a]

    al_path = tmp_dir / "assetlinks.json"
    al_path.write_text(
        json.dumps(
            [
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.prop",
                        "sha256_cert_fingerprints": [s_assetlinks],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    al_res = check_assetlinks(parsed_certs, "com.example.prop", al_path)
    if digest_a == digest_b:
        assert al_res["status"] == "matched"
    else:
        assert al_res["status"] == "mismatch"

    apk_path = _write_zip(tmp_dir / "prop.apk", {"classes.dex": b"dex"})
    xmltree = CLEAN_XMLTREE.replace("com.example.demo", "com.example.prop")

    def _runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=xmltree, stderr="")
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"Signer #1 certificate SHA-256 digest: {s_apksigner}\n", stderr=""
        )

    report = audit_apk(
        apk_path,
        assetlinks_path=al_path,
        aapt_bin="/usr/bin/aapt2",
        apksigner_bin="/usr/bin/apksigner",
        run_cmd=_runner,
    )
    _assert_summary_matches_findings(report)
    assert report["summary"]["passed"] is (digest_a == digest_b)


def test_property_sha256_fingerprint_normalisation_and_summary(tmp_path: Path) -> None:
    """[property] Fingerprint normalisation is idempotent, agrees iff bytes match, summary sums."""
    formats = ("colon_upper", "colon_lower", "raw_lower")

    @settings(max_examples=50, deadline=None)
    @given(
        digest_a=st.binary(min_size=32, max_size=32),
        digest_b=st.binary(min_size=32, max_size=32),
        same=st.booleans(),
        fmt_apk=st.sampled_from(formats),
        fmt_al=st.sampled_from(formats),
    )
    def _prop(
        digest_a: bytes,
        digest_b: bytes,
        same: bool,
        fmt_apk: str,
        fmt_al: str,
    ) -> None:
        b_val = digest_a if same else digest_b
        _check_digest_and_summary_invariants(digest_a, b_val, fmt_apk, fmt_al, tmp_path)

    _prop()


# ---------------------------------------------------------------------------
# Precise behaviour pins (exact dicts, exact messages, exact argv) for each helper.
# ---------------------------------------------------------------------------

_APKSIGNER_MISSING = "apksigner binary not found on PATH or in ANDROID_HOME/ANDROID_SDK_ROOT"
_AAPT_MISSING = "aapt2/aapt binary not found on PATH or in ANDROID_HOME/ANDROID_SDK_ROOT"
_CANNOT_VERIFY = "Cannot verify assetlinks without both package name and signing certificates"
_FP_A = ":".join(["AA", "BB"] * 16)
_FP_B = ":".join(["CC", "DD"] * 16)


def _cp(
    cmd: list[str], rc: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)


def _manifest_shell(status: str, detail: str | None) -> dict:
    return {
        "status": status,
        "detail": detail,
        "package": None,
        "version_code": None,
        "version_name": None,
        "min_sdk": None,
        "target_sdk": None,
        "debuggable": None,
        "uses_cleartext_traffic": None,
        "allow_backup": None,
        "activities_missing_uimode_config": [],
        "autoverify_hosts": [],
    }


def _write_assetlinks(path: Path, package: str, fingerprints: list) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": package,
                        "sha256_cert_fingerprints": fingerprints,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def debug(self, event: str, **kwargs: object) -> None:
        self.calls.append((event, kwargs))


def test_default_run_cmd_passes_exact_bounded_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """[contract] The default runner is bounded: capture, utf-8 replace, 30 s timeout, no check."""
    seen: list[tuple[tuple, dict]] = []
    sentinel = _cp(["x"], 0)

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        seen.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(aa.subprocess, "run", _fake_run)
    assert _default_run_cmd(["/sdk/aapt2", "dump"]) is sentinel
    assert seen == [
        (
            (["/sdk/aapt2", "dump"],),
            {
                "capture_output": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": 30,
                "check": False,
            },
        )
    ]


def test_find_apksigner_reads_android_home_before_sdk_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    home = tmp_path / "home_sdk"
    root = tmp_path / "root_sdk"
    for base in (home, root):
        (base / "build-tools" / "34.0.0").mkdir(parents=True)
        (base / "build-tools" / "34.0.0" / "apksigner").write_text("", encoding="utf-8")
    monkeypatch.setenv("ANDROID_HOME", str(home))
    monkeypatch.setenv("ANDROID_SDK_ROOT", str(root))
    assert find_apksigner() == str(home / "build-tools" / "34.0.0" / "apksigner")
    monkeypatch.delenv("ANDROID_SDK_ROOT")
    assert find_apksigner() == str(home / "build-tools" / "34.0.0" / "apksigner")


def test_validate_regular_file_exact_messages(tmp_path: Path) -> None:
    """[contract] Each rejected path kind raises with an exact, labelled message."""
    fifo = tmp_path / "pipe.apk"
    os.mkfifo(fifo)
    cases = [
        ("", "APK path must not be empty"),
        ("   ", "APK path must not be empty"),
        (tmp_path, f"APK path is a directory: {tmp_path}"),
        (tmp_path / "nope.apk", f"APK file does not exist: {tmp_path / 'nope.apk'}"),
        (fifo, f"APK path is not a regular file: {fifo}"),
    ]
    for raw, message in cases:
        with pytest.raises(ApkAuditError) as exc_info:
            inspect_apk_archive(raw)
        assert str(exc_info.value) == message
        with pytest.raises(ApkAuditError) as exc_info:
            audit_apk(raw, aapt_bin=None, apksigner_bin=None)
        assert str(exc_info.value) == message

    missing_al = tmp_path / "missing.json"
    with pytest.raises(ApkAuditError) as exc_info:
        check_assetlinks([_FP_A], "com.example.demo", missing_al)
    assert str(exc_info.value) == f"assetlinks file does not exist: {missing_al}"
    with pytest.raises(ApkAuditError) as exc_info:
        check_assetlinks([_FP_A], "com.example.demo", tmp_path)
    assert str(exc_info.value) == f"assetlinks path is a directory: {tmp_path}"


def test_file_sha256_and_read_failure_message(tmp_path: Path) -> None:
    """[contract] sha256 is the real file digest; read failure is a labelled ApkAuditError."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    report = audit_apk(apk, aapt_bin=None, apksigner_bin=None)
    assert report["sha256"] == hashlib.sha256(apk.read_bytes()).hexdigest()
    assert report["apk_path"] == str(apk)

    with (
        patch.object(Path, "open", side_effect=OSError("disk read failed")),
        pytest.raises(ApkAuditError) as exc_info,
    ):
        aa._file_sha256(apk)
    assert str(exc_info.value) == f"Failed to read APK file {apk}: disk read failed"
    assert isinstance(exc_info.value.__cause__, OSError)


def test_dex_ratio_abi_and_root_dex_helpers() -> None:
    assert aa._dex_compressed_ratio(1, 3) == 0.3333
    assert aa._dex_compressed_ratio(2, 3) == 0.6667
    assert aa._dex_compressed_ratio(1, 1) == 1.0
    assert aa._dex_compressed_ratio(5, 0) == 0.0
    assert aa._dex_compressed_ratio(5, -1) == 0.0

    assert aa._extract_abi("lib/arm64-v8a/libcore.so") == "arm64-v8a"
    for name in (
        "lib//libcore.so",
        "lib/arm64-v8a/sub/libcore.so",
        "libs/arm64-v8a/libcore.so",
        "lib/arm64-v8a/libcore.txt",
        "lib/libcore.so",
        "lib/x86/",
    ):
        assert aa._extract_abi(name) is None, name

    for name in ("classes.dex", "classes2.dex", "classes10.dex"):
        assert aa._is_root_dex(name) is True, name
    for name in (
        "sub/classes.dex",
        "classes/inner.dex",
        "foo.dex",
        "classes.txt",
        "classes",
        "assets/classes2.dex",
    ):
        assert aa._is_root_dex(name) is False, name


def test_accumulate_zip_entry_exact_tuples_and_bounds() -> None:
    """[contract] Entry accounting is exact; negative sizes are rejected, zero sizes are not."""
    abis: set[str] = set()
    maps: set[str] = set()

    dex = zipfile.ZipInfo("classes.dex")
    dex.file_size, dex.compress_size = 100, 40
    assert aa._accumulate_zip_entry(dex, abis, maps) == (1, 100, 40)

    empty_dex = zipfile.ZipInfo("classes2.dex")
    empty_dex.file_size, empty_dex.compress_size = 0, 0
    assert aa._accumulate_zip_entry(empty_dex, abis, maps) == (1, 0, 0)

    for name in ("lib/x86/libfoo.so", "assets/app.js.map", "res/raw/x.map", "other/x.map"):
        info = zipfile.ZipInfo(name)
        info.file_size, info.compress_size = 7, 5
        assert aa._accumulate_zip_entry(info, abis, maps) == (0, 0, 0)

    folder = zipfile.ZipInfo("assets/")
    assert aa._accumulate_zip_entry(folder, abis, maps) == (0, 0, 0)

    assert abis == {"x86"}
    assert maps == {"assets/app.js.map", "res/raw/x.map"}

    for field_name in ("file_size", "compress_size"):
        bad = zipfile.ZipInfo("classes.dex")
        bad.file_size, bad.compress_size = 1, 1
        setattr(bad, field_name, -1)
        with pytest.raises(ApkAuditError) as exc_info:
            aa._accumulate_zip_entry(bad, set(), set())
        assert str(exc_info.value) == "Corrupt ZIP entry size for 'classes.dex'"


def test_inspect_apk_archive_exact_stored_metrics(tmp_path: Path) -> None:
    """[contract] Stored (uncompressed) entries give exact, fully pinned archive metrics."""
    apk = _write_zip(
        tmp_path / "stored.apk",
        {
            "classes.dex": b"d" * 100,
            "classes2.dex": b"e" * 50,
            "sub/classes3.dex": b"f" * 10,
            "lib/arm64-v8a/a.so": b"elf",
            "lib/x86/a.so": b"elf",
            "assets/b.map": b"{}",
            "res/a.map": b"{}",
        },
        compression=zipfile.ZIP_STORED,
        dirs=("lib/",),
    )
    size = apk.stat().st_size
    assert inspect_apk_archive(str(apk)) == {
        "apk_bytes": size,
        "dex_count": 2,
        "dex_uncompressed_bytes": 150,
        "dex_compressed_bytes": 150,
        "dex_compressed_ratio": round(150 / size, 4),
        "dex_stored": True,
        "abis": ["arm64-v8a", "x86"],
        "sourcemap_files": ["assets/b.map", "res/a.map"],
    }


def test_read_zip_entries_messages_and_entry_cap_boundary(tmp_path: Path) -> None:
    non_zip = tmp_path / "bad.apk"
    non_zip.write_bytes(b"not a zip archive at all")
    with pytest.raises(ApkAuditError) as exc_info:
        inspect_apk_archive(non_zip)
    assert str(exc_info.value).startswith(f"Invalid or corrupt APK ZIP archive {non_zip}: ")
    assert isinstance(exc_info.value.__cause__, zipfile.BadZipFile)

    small_zip = _write_zip(tmp_path / "many.apk", {"classes.dex": b"dex"})
    at_cap = [zipfile.ZipInfo(f"f{i}.txt") for i in range(_MAX_ZIP_ENTRIES)]
    with patch("zipfile.ZipFile.infolist", return_value=at_cap):
        assert inspect_apk_archive(small_zip)["dex_count"] == 0
    over_cap = [*at_cap, zipfile.ZipInfo("extra.txt")]
    with (
        patch("zipfile.ZipFile.infolist", return_value=over_cap),
        pytest.raises(ApkAuditError) as exc_info,
    ):
        inspect_apk_archive(small_zip)
    assert str(exc_info.value) == "APK archive exceeds max ZIP entry count (50001 > 50000)"


def _archive(
    apk: int,
    dex: int,
    ratio: float,
    maps: list[str] | None = None,
    *,
    dex_stored: bool = False,
) -> dict:
    return {
        "apk_bytes": apk,
        "dex_uncompressed_bytes": dex,
        "dex_compressed_ratio": ratio,
        "dex_stored": dex_stored,
        "sourcemap_files": maps or [],
    }


def test_archive_findings_thresholds_and_exact_details() -> None:
    """[contract] Bloat needs apk >= 5 MB AND (dex >= 10 MB OR deflated ratio >= 0.65)."""
    bloat = {
        "severity": "medium",
        "code": "unminified_dex_bloat",
        "detail": "APK size 5000000 B with 10000000 B uncompressed DEX (compressed ratio 0.1)",
    }
    assert aa._archive_findings(_archive(5_000_000, 10_000_000, 0.1)) == [bloat]
    assert aa._archive_findings(_archive(4_999_999, 20_000_000, 0.9)) == []
    assert aa._archive_findings(_archive(5_000_000, 9_999_999, 0.6499)) == []
    assert aa._archive_findings(_archive(5_000_000, 9_999_999, 0.99, dex_stored=True)) == []
    stored_bloat = aa._archive_findings(_archive(5_000_000, 10_000_000, 0.99, dex_stored=True))
    assert [f["code"] for f in stored_bloat] == ["unminified_dex_bloat"]
    assert [f["code"] for f in aa._archive_findings(_archive(5_000_000, 0, 0.65))] == [
        "unminified_dex_bloat"
    ]
    assert aa._archive_findings(_archive(10, 0, 0.0, ["assets/a.map", "res/b.map"])) == [
        {
            "severity": "low",
            "code": "bundled_sourcemaps",
            "detail": "Bundled sourcemap files found: assets/a.map, res/b.map",
        }
    ]


def test_attribute_value_and_name_parsing_helpers() -> None:
    assert aa._read_quoted('"a\\"b\\\\c" (Raw: "x")') == 'a"b\\c'
    assert aa._read_quoted('"unterminated') == "unterminated"
    assert aa._read_quoted('""') == ""
    assert aa._read_quoted('"\\\\"') == "\\"
    assert aa._extract_attr_value(" (type 0x10)0x18  trailing") == "0x18"
    assert aa._extract_attr_value('"com.example.demo" (Raw: "com.example.demo")') == (
        "com.example.demo"
    )
    assert aa._extract_attr_value("   ") == ""

    prefixes = {"android", "ns0"}
    assert aa._parse_attr_name(f"{_NS}:name(0x01010003)", set()) == "name"
    assert aa._parse_attr_name("ns0:exported(0x01010010)", prefixes) == "exported"
    assert aa._parse_attr_name("android:a:b", prefixes) == "a:b"
    assert aa._parse_attr_name("app:debuggable", prefixes) is None
    assert aa._parse_attr_name(" package ", prefixes) == "package"

    assert aa._parse_attr_line("no_equals_here", prefixes) is None
    assert aa._parse_attr_line('="value"', prefixes) is None
    assert aa._parse_attr_line("app:x=1", prefixes) is None
    assert aa._parse_attr_line('android:name="a=b"', prefixes) == ("name", "a=b")


def test_xmltree_prefixes_without_space_are_honoured() -> None:
    """Record markers are 'N:'/'E:'/'A:' — the separator space is optional."""
    text = (
        "N: ???\n"
        f"N:ns0={_NS}\n"
        "E:manifest\n"
        ' A:package="com.example.tight"\n'
        " E:application\n"
        "  A:ns0:debuggable=true\n"
    )
    parsed = parse_aapt2_xmltree(text)
    assert parsed["package"] == "com.example.tight"
    assert parsed["debuggable"] is True


def test_scalar_parsers_cover_every_literal() -> None:
    for raw in ("true", "TRUE ", "1", "-1", "0x1", "0xffffffff"):
        assert aa._parse_bool(raw) is True, raw
    for raw in ("false", " False", "0", "0x0"):
        assert aa._parse_bool(raw) is False, raw
    for raw in (None, "maybe", "0xZZ", "", "2"):
        assert aa._parse_bool(raw) is None, raw

    assert aa._parse_int("42") == 42
    assert aa._parse_int(" 0x10 ") == 16
    assert aa._parse_int("0X1A") == 26
    for raw in (None, "", "  ", "abc", "0xZZ"):
        assert aa._parse_int(raw) is None, raw

    assert aa._parse_sdk_version(None) is None
    assert aa._parse_sdk_version("26") == 26
    assert aa._parse_sdk_version(" Baklava ") == "Baklava"
    assert aa._parse_sdk_version("  ") is None


def _node(tag: str, attrs: dict | None = None, children: list | None = None) -> aa._XmlNode:
    return aa._XmlNode(tag=tag, indent=0, attrs=attrs or {}, children=children or [])


def test_activity_classifiers_and_uimode_mask() -> None:
    main = _node("action", {"name": "android.intent.action.MAIN"})
    launcher = _node("category", {"name": "android.intent.category.LAUNCHER"})
    assert aa._is_launcher_filter(_node("intent-filter", children=[main, launcher])) is True
    assert aa._is_launcher_filter(_node("intent-filter", children=[main])) is False
    assert aa._is_launcher_filter(_node("intent-filter", children=[launcher])) is False
    swapped = [
        _node("category", {"name": "android.intent.action.MAIN"}),
        _node("action", {"name": "android.intent.category.LAUNCHER"}),
    ]
    assert aa._is_launcher_filter(_node("intent-filter", children=swapped)) is False

    assert aa._has_uimode_config(None) is False
    assert aa._has_uimode_config("0x200") is True
    assert aa._has_uimode_config("512") is True
    assert aa._has_uimode_config("0x4b0") is False
    assert aa._has_uimode_config("orientation| UIMODE ") is True
    assert aa._has_uimode_config("orientation|keyboard") is False

    launch_filter = _node("intent-filter", children=[main, launcher])
    assert aa._is_exported_or_launcher(_node("activity", {"exported": "true"})) is True
    assert aa._is_exported_or_launcher(_node("activity", {"exported": "false"})) is False
    assert (
        aa._is_exported_or_launcher(_node("activity", {"exported": "false"}, [launch_filter]))
        is True
    )
    assert aa._is_exported_or_launcher(_node("activity", {}, [_node("meta-data")])) is False

    app = _node(
        "application",
        children=[
            _node("service", {"exported": "true"}),
            _node("activity", {"exported": "true"}),
            _node("activity", {"name": ".A", "exported": "true", "configChanges": "0x200"}),
            _node("activity", {"name": ".B"}, [launch_filter]),
            _node("activity", {"name": ".C"}),
        ],
    )
    assert aa._collect_activities_missing_uimode(app) == ["<unnamed>", ".B"]
    assert aa._collect_activities_missing_uimode(None) == []


def test_autoverify_hosts_honour_per_data_scheme() -> None:
    """[contract] A host only counts when its own scheme (or the filter's) is https."""
    flt = _node(
        "intent-filter",
        {"autoVerify": "true"},
        [
            _node("data", {"scheme": "HTTPS", "host": "secure.example.com"}),
            _node("data", {"scheme": "http", "host": "plain.example.com"}),
            _node("data", {"host": "inherit.example.com"}),
            _node("data", {"scheme": "https"}),
            _node("action", {"host": "not-data.example.com", "scheme": "https"}),
        ],
    )
    assert aa._autoverify_hosts_for_filter(flt) == {"secure.example.com", "inherit.example.com"}
    http_only = _node(
        "intent-filter",
        {"autoVerify": "true"},
        [_node("data", {"scheme": "http"}), _node("data", {"host": "a.example.com"})],
    )
    assert aa._autoverify_hosts_for_filter(http_only) == set()

    not_verified = _node(
        "intent-filter", {"autoVerify": "false"}, [_node("data", {"scheme": "https", "host": "x"})]
    )
    nested = _node("manifest", children=[_node("application", children=[flt, not_verified])])
    assert aa._extract_autoverify_hosts([nested]) == ["inherit.example.com", "secure.example.com"]
    assert {n.tag for n in aa._iter_elements([nested])} == {
        "manifest",
        "application",
        "intent-filter",
        "data",
        "action",
    }
    assert len(aa._iter_elements([nested])) == 10


def test_parse_aapt2_xmltree_full_dict_equality() -> None:
    assert parse_aapt2_xmltree(CLEAN_XMLTREE) == {
        "package": "com.example.demo",
        "version_code": 42,
        "version_name": "1.2.3",
        "min_sdk": 26,
        "target_sdk": 35,
        "debuggable": False,
        "uses_cleartext_traffic": False,
        "allow_backup": False,
        "activities_missing_uimode_config": [],
        "autoverify_hosts": ["app.example.com"],
    }
    assert parse_aapt2_xmltree("E: other\nE: manifest\n  A: package=com.example.second\n") == {
        "package": "com.example.second",
        "version_code": None,
        "version_name": None,
        "min_sdk": None,
        "target_sdk": None,
        "debuggable": False,
        "uses_cleartext_traffic": False,
        "allow_backup": None,
        "activities_missing_uimode_config": [],
        "autoverify_hosts": [],
    }


def test_manifest_findings_exact_list() -> None:
    manifest = {
        "status": "ok",
        "debuggable": True,
        "uses_cleartext_traffic": True,
        "allow_backup": True,
        "activities_missing_uimode_config": [".A", ".B"],
    }
    assert aa._manifest_findings(manifest) == [
        {
            "severity": "high",
            "code": "debuggable_enabled",
            "detail": "AndroidManifest enables android:debuggable=true",
        },
        {
            "severity": "high",
            "code": "cleartext_traffic_enabled",
            "detail": "AndroidManifest enables android:usesCleartextTraffic=true",
        },
        {
            "severity": "low",
            "code": "backup_enabled",
            "detail": "AndroidManifest explicitly sets android:allowBackup=true",
        },
        {
            "severity": "info",
            "code": "activity_recreates_on_uimode",
            "detail": (
                "Exported/launcher activities missing uiMode (0x200) in "
                "android:configChanges: .A, .B"
            ),
        },
    ]
    assert aa._manifest_findings({**manifest, "status": "error"}) == []
    assert aa._manifest_findings({"status": "ok", "allow_backup": False}) == []


def test_build_aapt_cmd_argv_per_binary() -> None:
    apk = Path("/tmp/app.apk")
    assert aa._build_aapt_cmd("/sdk/aapt", apk) == [
        "/sdk/aapt",
        "dump",
        "xmltree",
        str(apk.resolve()),
        "AndroidManifest.xml",
    ]
    assert aa._build_aapt_cmd("/sdk/aapt2", apk) == [
        "/sdk/aapt2",
        "dump",
        "xmltree",
        str(apk.resolve()),
        "--file",
        "AndroidManifest.xml",
    ]


def test_run_manifest_audit_exact_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """[contract] Every manifest-audit outcome returns an exact, honest status dict."""
    apk = Path("/tmp/app.apk")
    expected_argv = [
        "/sdk/aapt2",
        "dump",
        "xmltree",
        str(apk.resolve()),
        "--file",
        "AndroidManifest.xml",
    ]
    assert aa._run_manifest_audit(apk, None, _default_run_cmd) == _manifest_shell(
        "unavailable", _AAPT_MISSING
    )
    assert aa._run_manifest_audit(apk, "", _default_run_cmd) == _manifest_shell(
        "unavailable", _AAPT_MISSING
    )

    cases = [
        (_cp([], 2, stdout="out\n", stderr="err\n"), "err"),
        (_cp([], 2, stdout="out only\n", stderr=""), "out only"),
        (_cp([], 3, stdout="", stderr=""), "aapt2 exited with code 3"),
        (_cp([], 4, stdout="", stderr="  \n"), "aapt2 exited with code 4"),
    ]
    for result, detail in cases:
        seen: list[list[str]] = []

        def _runner(cmd: list[str], _r=result, _seen=seen) -> subprocess.CompletedProcess:
            _seen.append(cmd)
            return _r

        assert aa._run_manifest_audit(apk, "/sdk/aapt2", _runner) == _manifest_shell(
            "error", detail
        )
        assert seen == [expected_argv]

    no_manifest = aa._run_manifest_audit(apk, "/sdk/aapt2", lambda c: _cp(c, 0, "E: other\n"))
    assert no_manifest == _manifest_shell(
        "error", "aapt2 xmltree output did not contain a <manifest> element"
    )

    ok = aa._run_manifest_audit(apk, "/sdk/aapt2", lambda c: _cp(c, 0, CLEAN_XMLTREE))
    assert ok == {"status": "ok", "detail": None, **parse_aapt2_xmltree(CLEAN_XMLTREE)}

    recorder = _RecordingLogger()
    monkeypatch.setattr(aa, "logger", recorder)
    for exc in (OSError("no exec"), subprocess.TimeoutExpired(["aapt2"], 30)):

        def _raiser(_cmd: list[str], _e=exc) -> subprocess.CompletedProcess:
            raise _e

        assert aa._run_manifest_audit(apk, "/sdk/aapt2", _raiser) == _manifest_shell(
            "error", str(exc)
        )
    assert recorder.calls == [
        ("aapt2_xmltree_exec_failed", {"error": "no exec"}),
        ("aapt2_xmltree_exec_failed", {"error": str(subprocess.TimeoutExpired(["aapt2"], 30))}),
    ]


def test_fingerprint_normalisation_and_apksigner_parsing_exact() -> None:
    raw = "aabb" * 16
    assert normalize_sha256_fingerprint(raw) == _FP_A
    assert normalize_sha256_fingerprint(f"  {_FP_A.lower()}  ") == _FP_A
    assert normalize_sha256_fingerprint(raw.upper()) == _FP_A
    for bad in (None, 42, b"aa", "", "aabb", "aa" * 33, "zz" * 32, _FP_A + ":AA", _FP_A[:-1]):
        assert normalize_sha256_fingerprint(bad) is None, bad

    text = (
        f"Signer #1 certificate SHA-256 digest: {raw}\n"
        f"Signer #2 Certificate sha-256 Digest:{_FP_B.lower()}\n"
        f"Signer #3 certificate SHA-256 digest: {_FP_A}\n"
        "Signer #4 certificate SHA-256 digest: deadbeef\n"
        f"Signer #5 certificate SHA-1 digest: {raw}\n"
    )
    assert parse_apksigner_certs(text) == [_FP_A, _FP_B]
    assert parse_apksigner_certs("") == []


def test_run_signing_audit_exact_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """[contract] Every signing-audit outcome returns an exact, honest status dict."""
    apk = Path("/tmp/app.apk")
    expected_argv = ["/sdk/apksigner", "verify", "--print-certs", str(apk.resolve())]
    unavailable = {"status": "unavailable", "detail": _APKSIGNER_MISSING, "cert_sha256s": []}
    assert aa._run_signing_audit(apk, None, _default_run_cmd) == unavailable
    assert aa._run_signing_audit(apk, "", _default_run_cmd) == unavailable

    cases = [
        (_cp([], 2, stdout="out\n", stderr="err\n"), "err"),
        (_cp([], 2, stdout="out only\n", stderr=""), "out only"),
        (_cp([], 3, stdout="", stderr=""), "apksigner exited with code 3"),
        (_cp([], 4, stdout="", stderr=" \n"), "apksigner exited with code 4"),
        (
            _cp([], 0, stdout="Verifies\n"),
            "apksigner returned 0 without any SHA-256 certificate digest",
        ),
    ]
    for result, detail in cases:
        seen: list[list[str]] = []

        def _runner(cmd: list[str], _r=result, _seen=seen) -> subprocess.CompletedProcess:
            _seen.append(cmd)
            return _r

        assert aa._run_signing_audit(apk, "/sdk/apksigner", _runner) == {
            "status": "error",
            "detail": detail,
            "cert_sha256s": [],
        }
        assert seen == [expected_argv]

    signed = f"Signer #1 certificate SHA-256 digest: {_FP_A}\n"
    assert aa._run_signing_audit(apk, "/sdk/apksigner", lambda c: _cp(c, 0, signed)) == {
        "status": "ok",
        "detail": None,
        "cert_sha256s": [_FP_A],
    }

    recorder = _RecordingLogger()
    monkeypatch.setattr(aa, "logger", recorder)

    def _raiser(_cmd: list[str]) -> subprocess.CompletedProcess:
        raise OSError("no exec")

    assert aa._run_signing_audit(apk, "/sdk/apksigner", _raiser) == {
        "status": "error",
        "detail": "no exec",
        "cert_sha256s": [],
    }
    assert recorder.calls == [("apksigner_exec_failed", {"error": "no exec"})]


def test_load_assetlinks_exact_errors(tmp_path: Path) -> None:
    """[contract] Malformed assetlinks inputs raise exact, labelled ApkAuditError messages."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{nope", encoding="utf-8")
    bad_utf8 = tmp_path / "bad_utf8.json"
    bad_utf8.write_bytes(b"[\xff\xfe]")
    huge = tmp_path / "huge.json"
    huge.write_bytes(b"[" + b" " * (1024 * 1024 + 16) + b"]")
    for path in (bad_json, bad_utf8, huge):
        with pytest.raises(ApkAuditError) as exc_info:
            aa._load_assetlinks_statements(path)
        assert str(exc_info.value).startswith(f"Failed to load assetlinks JSON {path}: ")
        assert exc_info.value.__cause__ is not None

    obj = tmp_path / "obj.json"
    obj.write_text("{}", encoding="utf-8")
    with pytest.raises(ApkAuditError) as exc_info:
        aa._load_assetlinks_statements(obj)
    assert str(exc_info.value) == "assetlinks.json must be a JSON list of statements"

    empty = tmp_path / "empty_label.json"
    with pytest.raises(ApkAuditError) as exc_info:
        aa._load_assetlinks_statements("  ")
    assert str(exc_info.value) == "assetlinks path must not be empty"
    assert not empty.exists()

    ok = tmp_path / "ok.json"
    ok.write_text('[{"a": 1}, 2]', encoding="utf-8")
    assert aa._load_assetlinks_statements(str(ok)) == [{"a": 1}, 2]


def test_extract_matching_package_fingerprints_exact() -> None:
    target = {"namespace": "android_app", "package_name": "com.example.demo"}
    assert aa._android_app_target({"target": target}, "com.example.demo") is target
    assert aa._android_app_target({"target": target}, "com.example.other") is None
    assert aa._android_app_target({"target": {**target, "namespace": "web"}}, "x") is None
    assert aa._android_app_target(["target"], "com.example.demo") is None

    result = aa._extract_matching_package_fingerprints([], "com.example.demo")
    assert result == (False, False, [])
    assert result[0] is False

    handle_urls = ["delegate_permission/common.handle_all_urls"]
    statements = [
        {
            "relation": ["delegate_permission/common.get_login_creds"],
            "target": {**target, "sha256_cert_fingerprints": [_FP_A]},
        },
        {
            "relation": handle_urls,
            "target": {**target, "sha256_cert_fingerprints": "bad"},
        },
        {
            "relation": handle_urls,
            "target": {**target, "sha256_cert_fingerprints": [_FP_A.lower(), "junk", _FP_A]},
        },
        {
            "relation": handle_urls,
            "target": {**target, "sha256_cert_fingerprints": [_FP_B]},
        },
    ]
    found, rel_found, fps = aa._extract_matching_package_fingerprints(
        statements, "com.example.demo"
    )
    assert found is True
    assert rel_found is True
    assert fps == [_FP_A, _FP_B]
    only_other_rel = aa._extract_matching_package_fingerprints(statements[:1], "com.example.demo")
    assert only_other_rel == (True, False, [])
    only_bad_fps = aa._extract_matching_package_fingerprints(statements[1:2], "com.example.demo")
    assert only_bad_fps == (True, True, [])


def test_check_assetlinks_full_result_dicts(tmp_path: Path) -> None:
    """[contract] check_assetlinks returns fully pinned dicts for all four statuses."""
    al = _write_assetlinks(tmp_path / "al.json", "com.example.demo", [_FP_A.lower()])
    certs_in = [_FP_A.replace(":", "").lower(), _FP_A, "garbage", _FP_B]

    assert check_assetlinks(certs_in, None, al) == {
        "status": "unavailable",
        "assetlinks_path": str(al),
        "package": None,
        "apk_cert_sha256s": [_FP_A, _FP_B],
        "expected_cert_sha256s": [],
        "matched_cert_sha256s": [],
        "detail": "APK package name is unavailable; cannot match assetlinks.json",
    }
    assert check_assetlinks(certs_in, "", str(al))["status"] == "unavailable"
    assert check_assetlinks([], "com.example.demo", al) == {
        "status": "unavailable",
        "assetlinks_path": str(al),
        "package": "com.example.demo",
        "apk_cert_sha256s": [],
        "expected_cert_sha256s": [],
        "matched_cert_sha256s": [],
        "detail": "APK signing certificates are unavailable; cannot match assetlinks.json",
    }

    assert check_assetlinks(certs_in, "com.example.other", al) == {
        "status": "package_missing",
        "assetlinks_path": str(al),
        "package": "com.example.other",
        "apk_cert_sha256s": [_FP_A, _FP_B],
        "expected_cert_sha256s": [],
        "matched_cert_sha256s": [],
        "detail": "No android_app statement for package 'com.example.other' in assetlinks.json",
    }

    assert check_assetlinks(certs_in, "com.example.demo", al) == {
        "status": "matched",
        "assetlinks_path": str(al),
        "package": "com.example.demo",
        "apk_cert_sha256s": [_FP_A, _FP_B],
        "expected_cert_sha256s": [_FP_A],
        "matched_cert_sha256s": [_FP_A],
        "detail": None,
    }

    assert check_assetlinks([_FP_B], "com.example.demo", al) == {
        "status": "mismatch",
        "assetlinks_path": str(al),
        "package": "com.example.demo",
        "apk_cert_sha256s": [_FP_B],
        "expected_cert_sha256s": [_FP_A],
        "matched_cert_sha256s": [],
        "detail": (
            f"APK signing SHA-256 {[_FP_B]} does not match assetlinks.json "
            f"fingerprints {[_FP_A]} for 'com.example.demo'"
        ),
    }

    al_rel_missing = tmp_path / "al_rel_missing.json"
    al_rel_missing.write_text(
        json.dumps(
            [
                {
                    "relation": ["delegate_permission/common.get_login_creds"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_A],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    assert check_assetlinks([_FP_A], "com.example.demo", al_rel_missing) == {
        "status": "relation_missing",
        "assetlinks_path": str(al_rel_missing),
        "package": "com.example.demo",
        "apk_cert_sha256s": [_FP_A],
        "expected_cert_sha256s": [],
        "matched_cert_sha256s": [],
        "detail": (
            "No android_app statement for package 'com.example.demo' grants "
            "'delegate_permission/common.handle_all_urls'"
        ),
    }


def test_assetlinks_findings_exact_and_fallback_details() -> None:
    assert aa._assetlinks_findings({"status": "package_missing", "detail": "custom"}) == [
        {"severity": "high", "code": "assetlinks_package_missing", "detail": "custom"}
    ]
    assert aa._assetlinks_findings({"status": "package_missing"}) == [
        {
            "severity": "high",
            "code": "assetlinks_package_missing",
            "detail": "Package missing from assetlinks.json",
        }
    ]
    assert aa._assetlinks_findings({"status": "relation_missing", "detail": "custom"}) == [
        {"severity": "high", "code": "assetlinks_relation_missing", "detail": "custom"}
    ]
    assert aa._assetlinks_findings({"status": "relation_missing", "detail": None}) == [
        {
            "severity": "high",
            "code": "assetlinks_relation_missing",
            "detail": (
                "Missing delegate_permission/common.handle_all_urls relation in assetlinks.json"
            ),
        }
    ]
    assert aa._assetlinks_findings({"status": "mismatch", "detail": "custom"}) == [
        {"severity": "high", "code": "assetlinks_sha256_mismatch", "detail": "custom"}
    ]
    assert aa._assetlinks_findings({"status": "mismatch", "detail": None}) == [
        {
            "severity": "high",
            "code": "assetlinks_sha256_mismatch",
            "detail": "Signing SHA-256 mismatch in assetlinks.json",
        }
    ]
    assert aa._assetlinks_findings({"status": "unavailable", "detail": "custom"}) == [
        {"severity": "high", "code": "assetlinks_unavailable", "detail": "custom"}
    ]
    assert aa._assetlinks_findings({"status": "unavailable", "detail": None}) == [
        {"severity": "high", "code": "assetlinks_unavailable", "detail": _CANNOT_VERIFY}
    ]
    for status in ("matched", "skipped", None):
        assert aa._assetlinks_findings({"status": status, "detail": "x"}) == []


def test_summarize_findings_defaults_and_each_error_flips_passed() -> None:
    """[contract] passed requires measured, no high finding, and no error/failed layer."""
    low = [{"severity": "low"}, {"severity": "info"}, {"severity": "bogus"}]
    assert summarize_findings(
        low, manifest_status="ok", signing_status="ok", assetlinks_status="skipped"
    ) == {
        "high": 0,
        "medium": 0,
        "low": 1,
        "info": 1,
        "measured": True,
        "passed": True,
    }
    assert summarize_findings(
        [], manifest_status="ok", signing_status="ok", assetlinks_status="skipped"
    ) == {
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0,
        "measured": True,
        "passed": True,
    }
    assert summarize_findings(
        low, manifest_status="error", signing_status="ok", assetlinks_status="skipped"
    ) == {
        "high": 0,
        "medium": 0,
        "low": 1,
        "info": 1,
        "measured": True,
        "passed": False,
    }
    assert summarize_findings(
        low, manifest_status="ok", signing_status="error", assetlinks_status="skipped"
    ) == {
        "high": 0,
        "medium": 0,
        "low": 1,
        "info": 1,
        "measured": True,
        "passed": False,
    }
    assert summarize_findings(
        low, manifest_status="unavailable", signing_status="ok", assetlinks_status="skipped"
    ) == {
        "high": 0,
        "medium": 0,
        "low": 1,
        "info": 1,
        "measured": False,
        "passed": False,
    }
    assert summarize_findings(
        low, manifest_status="ok", signing_status="unavailable", assetlinks_status="skipped"
    ) == {
        "high": 0,
        "medium": 0,
        "low": 1,
        "info": 1,
        "measured": False,
        "passed": False,
    }
    assert summarize_findings(
        low, manifest_status="ok", signing_status="ok", assetlinks_status="unavailable"
    ) == {
        "high": 0,
        "medium": 0,
        "low": 1,
        "info": 1,
        "measured": False,
        "passed": False,
    }
    assert summarize_findings(
        [{"severity": "high"}, {"severity": "medium"}],
        manifest_status="ok",
        signing_status="ok",
        assetlinks_status="skipped",
    ) == {
        "high": 1,
        "medium": 1,
        "low": 0,
        "info": 0,
        "measured": True,
        "passed": False,
    }


def _signed_runner(xmltree: str, signer_stdout: str, signer_rc: int = 0, aapt_rc: int = 0):
    def _runner(cmd: list[str]) -> subprocess.CompletedProcess:
        if "xmltree" in cmd:
            return _cp(cmd, aapt_rc, xmltree, "aapt boom" if aapt_rc else "")
        return _cp(cmd, signer_rc, signer_stdout, "signer boom" if signer_rc else "")

    return _runner


def test_audit_apk_assetlinks_skipped_and_unavailable_shapes(tmp_path: Path) -> None:
    """[contract] audit_apk assetlinks sub-dict is fully pinned for skipped/unavailable."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    signed = f"Signer #1 certificate SHA-256 digest: {_FP_A}\n"

    skipped = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(CLEAN_XMLTREE, signed),
    )
    assert skipped["assetlinks"] == {
        "status": "skipped",
        "assetlinks_path": None,
        "package": "com.example.demo",
        "apk_cert_sha256s": [_FP_A],
        "expected_cert_sha256s": [],
        "matched_cert_sha256s": [],
        "detail": None,
    }
    assert skipped["summary"]["measured"] is True
    assert skipped["summary"]["passed"] is True

    al = _write_assetlinks(tmp_path / "al.json", "com.example.demo", [_FP_A])
    signer_error = audit_apk(
        apk,
        assetlinks_path=str(al),
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(CLEAN_XMLTREE, "", signer_rc=1),
    )
    assert signer_error["assetlinks"] == {
        "status": "unavailable",
        "assetlinks_path": str(al),
        "package": "com.example.demo",
        "apk_cert_sha256s": [],
        "expected_cert_sha256s": [],
        "matched_cert_sha256s": [],
        "detail": _CANNOT_VERIFY,
    }
    assert signer_error["signing"] == {
        "status": "error",
        "detail": "signer boom",
        "cert_sha256s": [],
    }
    assert signer_error["findings"] == [
        {
            "severity": "high",
            "code": "assetlinks_unavailable",
            "detail": _CANNOT_VERIFY,
        }
    ]
    assert signer_error["summary"]["measured"] is False
    assert signer_error["summary"]["passed"] is False

    no_package = audit_apk(
        apk,
        assetlinks_path=al,
        aapt_bin=None,
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(CLEAN_XMLTREE, signed),
    )
    assert no_package["assetlinks"] == {
        "status": "unavailable",
        "assetlinks_path": str(al),
        "package": None,
        "apk_cert_sha256s": [_FP_A],
        "expected_cert_sha256s": [],
        "matched_cert_sha256s": [],
        "detail": _CANNOT_VERIFY,
    }
    assert no_package["findings"] == [
        {
            "severity": "high",
            "code": "assetlinks_unavailable",
            "detail": _CANNOT_VERIFY,
        }
    ]
    assert no_package["summary"]["measured"] is False
    assert no_package["summary"]["passed"] is False

    bad_al = tmp_path / "bad_al.json"
    bad_al.write_text("{}", encoding="utf-8")
    with pytest.raises(ApkAuditError) as exc_info:
        audit_apk(apk, assetlinks_path=bad_al, aapt_bin=None, apksigner_bin=None)
    assert str(exc_info.value) == "assetlinks.json must be a JSON list of statements"


def test_audit_apk_each_tool_error_alone_fails_summary(tmp_path: Path) -> None:
    """[contract] A manifest error OR a signing error alone flips summary.passed to False."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    signed = f"Signer #1 certificate SHA-256 digest: {_FP_A}\n"

    manifest_err = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(CLEAN_XMLTREE, signed, aapt_rc=1),
    )
    assert manifest_err["manifest"] == _manifest_shell("error", "aapt boom")
    assert manifest_err["signing"]["status"] == "ok"
    assert manifest_err["findings"] == []
    assert manifest_err["summary"] == {
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0,
        "measured": True,
        "passed": False,
    }

    signing_err = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(CLEAN_XMLTREE, "", signer_rc=1),
    )
    assert signing_err["manifest"]["status"] == "ok"
    assert signing_err["findings"] == []
    assert signing_err["summary"]["measured"] is True
    assert signing_err["summary"]["passed"] is False


def test_audit_apk_full_report_and_tool_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset binaries are discovered; empty strings mean unavailable; default runner used."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    al = _write_assetlinks(tmp_path / "al.json", "com.example.demo", [_FP_A])
    signed = f"Signer #1 certificate SHA-256 digest: {_FP_A}\n"
    seen: list[list[str]] = []
    runner = _signed_runner(CLEAN_XMLTREE, signed)

    def _recording_default(cmd: list[str]) -> subprocess.CompletedProcess:
        seen.append(cmd)
        return runner(cmd)

    monkeypatch.setattr(aa, "find_aapt", lambda: "/found/aapt2")
    monkeypatch.setattr(aa, "find_apksigner", lambda: "/found/apksigner")
    monkeypatch.setattr(aa, "_default_run_cmd", _recording_default)

    report = audit_apk(apk, assetlinks_path=al)
    assert [cmd[0] for cmd in seen] == ["/found/aapt2", "/found/apksigner"]
    assert report == {
        "apk_path": str(apk),
        "sha256": hashlib.sha256(apk.read_bytes()).hexdigest(),
        "archive": inspect_apk_archive(apk),
        "manifest": {"status": "ok", "detail": None, **parse_aapt2_xmltree(CLEAN_XMLTREE)},
        "signing": {"status": "ok", "detail": None, "cert_sha256s": [_FP_A]},
        "assetlinks": check_assetlinks([_FP_A], "com.example.demo", al),
        "findings": [],
        "summary": {
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
            "measured": True,
            "passed": True,
        },
    }

    seen.clear()
    blank = audit_apk(apk, aapt_bin="", apksigner_bin="")
    assert seen == []
    assert blank["manifest"]["status"] == "unavailable"
    assert blank["signing"]["status"] == "unavailable"
    assert blank["summary"]["measured"] is False
    assert blank["summary"]["passed"] is False


# ---------------------------------------------------------------------------
# Hardening [contract] tests (linc-crucible-t3ul)
# ---------------------------------------------------------------------------


def _write_byte_emitter(path: Path, payload: bytes) -> Path:
    code = f"import sys; sys.stdout.buffer.write({payload!r})"
    path.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(code)}\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def test_assetlinks_deep_nesting_raises_apk_audit_error(tmp_path: Path) -> None:
    """[contract] 200,000 '[' in assetlinks.json raises ApkAuditError, not RecursionError."""
    deep = tmp_path / "deep.json"
    deep.write_bytes(b"[" * 200_000)
    with pytest.raises(ApkAuditError) as exc_info:
        check_assetlinks([_FP_A], "com.example.demo", deep)
    assert isinstance(exc_info.value.__cause__, RecursionError)
    assert str(exc_info.value).startswith(f"Failed to load assetlinks JSON {deep}: ")

    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    with pytest.raises(ApkAuditError):
        audit_apk(apk, assetlinks_path=deep, aapt_bin=None, apksigner_bin=None)


def test_default_run_cmd_handles_undecodable_tool_bytes(tmp_path: Path) -> None:
    """[contract] Undecodable bytes from aapt2/apksigner never escape audit_apk."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    bad_aapt2 = _write_byte_emitter(tmp_path / "aapt2", b"\xff\xfe")
    bad_apksigner = _write_byte_emitter(tmp_path / "apksigner", b"\xff\xfe")

    err_report = audit_apk(
        apk,
        aapt_bin=str(bad_aapt2),
        apksigner_bin=str(bad_apksigner),
    )
    assert err_report["manifest"]["status"] == "error"
    assert err_report["signing"]["status"] == "error"
    assert err_report["summary"]["passed"] is False

    best_aapt2 = _write_byte_emitter(
        tmp_path / "aapt2_best",
        CLEAN_XMLTREE.encode("utf-8") + b"\xff\xfe\n",
    )
    best_apksigner = _write_byte_emitter(
        tmp_path / "apksigner_best",
        b"Signer #1 certificate DN: CN=\xff\xfe\n"
        + f"Signer #1 certificate SHA-256 digest: {SAMPLE_HEX}\n".encode(),
    )
    best_report = audit_apk(
        apk,
        aapt_bin=str(best_aapt2),
        apksigner_bin=str(best_apksigner),
    )
    assert best_report["manifest"]["status"] == "ok"
    assert best_report["manifest"]["package"] == "com.example.demo"
    assert best_report["signing"]["status"] == "ok"
    assert best_report["signing"]["cert_sha256s"] == [SAMPLE_FINGERPRINT_COLON]


def test_assetlinks_requires_handle_all_urls_relation(tmp_path: Path) -> None:
    """[contract] Statement must grant handle_all_urls; non-list relations skip safely."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    runner = _signed_runner(CLEAN_XMLTREE, f"Signer #1 certificate SHA-256 digest: {_FP_A}\n")

    al_login = tmp_path / "al_login.json"
    al_login.write_text(
        json.dumps(
            [
                {
                    "relation": ["delegate_permission/common.get_login_creds"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_A],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    res_login = audit_apk(
        apk,
        assetlinks_path=al_login,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=runner,
    )
    assert res_login["assetlinks"]["status"] == "relation_missing"
    assert res_login["assetlinks"]["status"] != "matched"
    assert res_login["assetlinks"]["expected_cert_sha256s"] == []
    assert res_login["findings"] == [
        {
            "severity": "high",
            "code": "assetlinks_relation_missing",
            "detail": res_login["assetlinks"]["detail"],
        }
    ]
    assert res_login["summary"]["high"] == 1
    assert res_login["summary"]["passed"] is False

    al_urls = tmp_path / "al_urls.json"
    al_urls.write_text(
        json.dumps(
            [
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_A],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    res_urls = audit_apk(
        apk,
        assetlinks_path=al_urls,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=runner,
    )
    assert res_urls["assetlinks"]["status"] == "matched"
    assert res_urls["findings"] == []
    assert res_urls["summary"]["passed"] is True

    al_non_list = tmp_path / "al_non_list.json"
    al_non_list.write_text(
        json.dumps(
            [
                {
                    "relation": "delegate_permission/common.handle_all_urls",
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_A],
                    },
                },
                {
                    "relation": [123, None, {"rel": "delegate_permission/common.handle_all_urls"}],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_A],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    res_non_list = audit_apk(
        apk,
        assetlinks_path=al_non_list,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=runner,
    )
    assert res_non_list["assetlinks"]["status"] == "relation_missing"
    assert [f["code"] for f in res_non_list["findings"]] == ["assetlinks_relation_missing"]
    assert res_non_list["summary"]["passed"] is False

    al_mixed = tmp_path / "al_mixed.json"
    al_mixed.write_text(
        json.dumps(
            [
                {
                    "relation": "delegate_permission/common.handle_all_urls",
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_A],
                    },
                },
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [_FP_B],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    res_mixed = check_assetlinks([_FP_A], "com.example.demo", al_mixed)
    assert res_mixed["status"] == "mismatch"
    assert res_mixed["expected_cert_sha256s"] == [_FP_B]


def test_xmltree_android_namespace_precedence_regardless_of_order(tmp_path: Path) -> None:
    """[contract] android:-namespaced attrs always beat un-namespaced attrs in either order."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    xml_before = (
        f"N: android={_NS}\n"
        "  E: manifest\n"
        '    A: package="com.example.ns"\n'
        "    E: application\n"
        "      A: android:debuggable(0x0101000f)=true\n"
        '      A: debuggable="false"\n'
    )
    xml_after = (
        f"N: android={_NS}\n"
        "  E: manifest\n"
        '    A: package="com.example.ns"\n'
        "    E: application\n"
        '      A: debuggable="false"\n'
        "      A: android:debuggable(0x0101000f)=true\n"
    )
    for xml in (xml_before, xml_after):
        parsed = parse_aapt2_xmltree(xml)
        assert parsed["debuggable"] is True
        report = audit_apk(
            apk,
            aapt_bin="/sdk/aapt2",
            apksigner_bin=None,
            run_cmd=lambda c, _x=xml: _cp(c, 0, _x),
        )
        assert report["manifest"]["debuggable"] is True
        assert [f["code"] for f in report["findings"]] == ["debuggable_enabled"]
        assert report["summary"]["passed"] is False

    xml_ns_alias = (
        f"N: ns0={_NS}\n"
        "  E: manifest\n"
        '    A: package="com.example.ns"\n'
        "    E: application\n"
        "      A: ns0:debuggable=true\n"
        '      A: debuggable="false"\n'
        f"      A: {_NS}:usesCleartextTraffic=true\n"
        '      A: usesCleartextTraffic="false"\n'
    )
    parsed_alias = parse_aapt2_xmltree(xml_ns_alias)
    assert parsed_alias["debuggable"] is True
    assert parsed_alias["uses_cleartext_traffic"] is True


def test_archive_stored_dex_ratio_bypass_and_bloat_threshold(tmp_path: Path) -> None:
    """[contract] Stored 6 MB DEX emits no bloat; >=10 MB or deflated ratio still fires."""
    stored_6mb = tmp_path / "stored_6mb.apk"
    with zipfile.ZipFile(stored_6mb, "w") as zf:
        zf.writestr("classes.dex", b"dex\n" + b"a" * 5_999_996, compress_type=zipfile.ZIP_STORED)
        zf.writestr("assets/note.txt", b"hello" * 200, compress_type=zipfile.ZIP_DEFLATED)
    report_6mb = audit_apk(stored_6mb, aapt_bin=None, apksigner_bin=None)
    arc_6mb = report_6mb["archive"]
    assert arc_6mb["apk_bytes"] >= 5_000_000
    assert arc_6mb["dex_uncompressed_bytes"] == 6_000_000
    assert arc_6mb["dex_compressed_ratio"] >= 0.65
    assert arc_6mb["dex_stored"] is True
    assert report_6mb["findings"] == []

    stored_10mb = _write_zip(
        tmp_path / "stored_10mb.apk",
        {"classes.dex": b"dex\n" + b"a" * 9_999_996},
        compression=zipfile.ZIP_STORED,
    )
    report_10mb = audit_apk(stored_10mb, aapt_bin=None, apksigner_bin=None)
    assert report_10mb["archive"]["dex_stored"] is True
    assert [f["code"] for f in report_10mb["findings"]] == ["unminified_dex_bloat"]

    mixed_apk = tmp_path / "mixed.apk"
    with zipfile.ZipFile(mixed_apk, "w") as zf:
        zf.writestr("classes.dex", b"dex\n" + b"a" * 2_999_996, compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "classes2.dex",
            _incompressible_bytes(3_000_000, seed=7),
            compress_type=zipfile.ZIP_DEFLATED,
        )
    mixed_report = audit_apk(mixed_apk, aapt_bin=None, apksigner_bin=None)
    assert mixed_report["archive"]["dex_stored"] is False
    assert [f["code"] for f in mixed_report["findings"]] == ["unminified_dex_bloat"]

    no_dex_apk = _write_zip(tmp_path / "nodex.apk", {"assets/data.bin": b"x" * 100})
    assert inspect_apk_archive(no_dex_apk)["dex_stored"] is False


def test_parse_apksigner_certs_excludes_source_stamp_and_supports_single_signer() -> None:
    """[contract] Collects Signer #N / single-signer digests and excludes Source Stamp."""
    stamp_hex = "99" * 32
    stamp_colon = ":".join(["99"] * 32)
    text = (
        "Source Stamp Signer certificate DN: CN=Source Stamp\n"
        f"Source Stamp Signer certificate SHA-256 digest: {stamp_hex}\n"
        "Signer #1 certificate DN: CN=App\n"
        f"Signer #1 certificate SHA-256 digest: {_FP_A.replace(':', '').lower()}\n"
        f"Source Stamp Signer #1 certificate SHA-256 digest: {stamp_colon}\n"
        f"Signer #2 certificate SHA-256 digest: {_FP_B}\n"
    )
    assert parse_apksigner_certs(text) == [_FP_A, _FP_B]

    single = (
        f"Signer #1 certificate DN: CN=evil certificate SHA-256 digest: {stamp_hex}\n"
        "  certificate DN: CN=Single\n"
        f"  certificate SHA-256 digest: {_FP_A.lower()}\n"
        f"Source Stamp Signer certificate SHA-256 digest: {stamp_hex}\n"
    )
    assert parse_apksigner_certs(single) == [_FP_A]
    stamp_only = f"Source Stamp Signer certificate SHA-256 digest: {stamp_hex}\n"
    assert parse_apksigner_certs(stamp_only) == []


def test_audit_apk_passes_apk_path_to_apksigner_argv(tmp_path: Path) -> None:
    """[contract] audit_apk forwards str(apk_path) in apksigner verify --print-certs argv."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    seen: list[list[str]] = []

    def _runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        if "xmltree" in cmd:
            return _cp(cmd, 0, CLEAN_XMLTREE)
        return _cp(cmd, 0, f"Signer #1 certificate SHA-256 digest: {_FP_A}\n")

    report = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_runner,
    )
    assert report["signing"]["status"] == "ok"
    assert seen == [
        ["/sdk/aapt2", "dump", "xmltree", str(apk), "--file", "AndroidManifest.xml"],
        ["/sdk/apksigner", "verify", "--print-certs", str(apk)],
    ]


def test_hypothesis_is_installed_for_property_tests() -> None:
    """[contract] Real Hypothesis is installed and active (no fallback)."""
    import hypothesis

    assert callable(given)
    assert callable(settings)
    assert hasattr(st, "binary")
    assert hasattr(hypothesis, "__version__")


# ---------------------------------------------------------------------------
# Honest measured/passed semantics & property test (linc-crucible-8wbz)
# ---------------------------------------------------------------------------


def test_property_summarize_findings_passed_implies_measured_no_high_no_error() -> None:
    """[property] Across (manifest_status, signing_status, assetlinks_status, findings):
    passed implies measured and no high finding and no error."""
    finding_st = st.fixed_dictionaries(
        {
            "severity": st.sampled_from(["high", "medium", "low", "info", "bogus"]),
            "code": st.text(min_size=1, max_size=24),
            "detail": st.text(max_size=40),
        }
    )

    @settings(max_examples=250, deadline=None)
    @given(
        manifest_status=st.sampled_from(["ok", "unavailable", "error"]),
        signing_status=st.sampled_from(["ok", "unavailable", "error"]),
        assetlinks_status=st.sampled_from(
            [
                "skipped",
                "matched",
                "mismatch",
                "package_missing",
                "relation_missing",
                "unavailable",
                "error",
            ]
        ),
        findings=st.lists(finding_st, max_size=8),
    )
    def _prop(
        manifest_status: str,
        signing_status: str,
        assetlinks_status: str,
        findings: list[dict[str, str]],
    ) -> None:
        summary = summarize_findings(
            findings,
            manifest_status=manifest_status,
            signing_status=signing_status,
            assetlinks_status=assetlinks_status,
        )
        expected_measured = (
            manifest_status != "unavailable"
            and signing_status != "unavailable"
            and assetlinks_status
            in {"skipped", "matched", "mismatch", "package_missing", "relation_missing"}
        )
        assert summary["measured"] is expected_measured

        has_high = any(f["severity"] == "high" for f in findings)
        has_error = "error" in (manifest_status, signing_status, assetlinks_status)
        if summary["passed"]:
            assert summary["measured"] is True
            assert not has_high
            assert not has_error
            assert manifest_status == "ok"
            assert signing_status == "ok"
            assert assetlinks_status in {"skipped", "matched"}

        if (
            expected_measured
            and not has_high
            and manifest_status == "ok"
            and signing_status == "ok"
            and assetlinks_status in {"skipped", "matched"}
        ):
            assert summary["passed"] is True

    _prop()


def test_audit_apk_single_unavailable_tool_is_unmeasured(tmp_path: Path) -> None:
    """[contract] aapt2 OK + apksigner missing (and vice versa) -> measured=False, passed=False."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    signed = f"Signer #1 certificate SHA-256 digest: {_FP_A}\n"

    only_aapt = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin=None,
        run_cmd=_signed_runner(CLEAN_XMLTREE, signed),
    )
    assert only_aapt["manifest"]["status"] == "ok"
    assert only_aapt["signing"]["status"] == "unavailable"
    assert only_aapt["summary"]["measured"] is False
    assert only_aapt["summary"]["passed"] is False

    only_signer = audit_apk(
        apk,
        aapt_bin=None,
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(CLEAN_XMLTREE, signed),
    )
    assert only_signer["manifest"]["status"] == "unavailable"
    assert only_signer["signing"]["status"] == "ok"
    assert only_signer["summary"]["measured"] is False
    assert only_signer["summary"]["passed"] is False


# ---------------------------------------------------------------------------
# R4 findings 2-5 hardening tests (linc-crucible-pcin)
# ---------------------------------------------------------------------------


def _write_eocd64_apk(path: Path, entry_count: int) -> Path:
    import struct

    eocd64 = struct.pack(
        "<4sQHHLLQQQQ",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        entry_count,
        entry_count,
        0,
        0,
    )
    locator64 = struct.pack("<4sLQL", b"PK\x06\x07", 0, 0, 1)
    eocd32_count = min(entry_count, 0xFFFF)
    eocd32 = struct.pack(
        "<4sHHHHLLH",
        b"PK\x05\x06",
        0,
        0,
        eocd32_count,
        eocd32_count,
        0,
        0xFFFFFFFF,
        0,
    )
    path.write_bytes(eocd64 + locator64 + eocd32)
    return path


def test_audit_apk_resolves_relative_dash_path_for_aapt_and_apksigner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] Relative '-' path is resolved to an absolute path for both tools."""
    monkeypatch.chdir(tmp_path)
    apk = _write_zip(tmp_path / "--in=ok.apk", {"classes.dex": b"dex"})
    expected_abs = str(apk.resolve())
    seen: list[list[str]] = []

    def _runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        if "xmltree" in cmd:
            return _cp(cmd, 0, CLEAN_XMLTREE)
        return _cp(cmd, 0, f"Signer #1 certificate SHA-256 digest: {_FP_A}\n")

    report = audit_apk(
        "--in=ok.apk",
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_runner,
    )
    assert report["summary"]["passed"] is True
    assert seen == [
        ["/sdk/aapt2", "dump", "xmltree", expected_abs, "--file", "AndroidManifest.xml"],
        ["/sdk/apksigner", "verify", "--print-certs", expected_abs],
    ]
    assert aa._build_aapt_cmd("/sdk/aapt", Path("--in=ok.apk")) == [
        "/sdk/aapt",
        "dump",
        "xmltree",
        expected_abs,
        "AndroidManifest.xml",
    ]


def test_assetlinks_5000_digit_integer_raises_apk_audit_error(tmp_path: Path) -> None:
    """[contract] assetlinks with a 5000-digit integer raises ApkAuditError (from ValueError)."""
    huge_int = tmp_path / "huge_int.json"
    huge_int.write_text("[" + ("9" * 5000) + "]", encoding="utf-8")
    with pytest.raises(ApkAuditError) as exc_info:
        check_assetlinks([_FP_A], "com.example.demo", huge_int)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value).startswith(f"Failed to load assetlinks JSON {huge_int}: ")

    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    with pytest.raises(ApkAuditError):
        audit_apk(apk, assetlinks_path=huge_int, aapt_bin=None, apksigner_bin=None)


def test_million_entry_eocd_refused_before_zipfile_and_bounded_rss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] Generated APK declaring 1,000,000 entries is refused before ZipFile parses it."""
    import resource
    import struct

    million_apk = _write_eocd64_apk(tmp_path / "million.apk", 1_000_000)

    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with (
        patch.object(
            aa.zipfile,
            "ZipFile",
            side_effect=AssertionError("ZipFile must not be constructed"),
        ),
        pytest.raises(ApkAuditError) as exc_info,
    ):
        inspect_apk_archive(million_apk)
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert str(exc_info.value) == "APK archive exceeds max ZIP entry count (1000000 > 50000)"
    # ru_maxrss is in KiB on Linux (bytes on macOS); in either unit growth is < 100 MB
    assert (rss_after_kb - rss_before_kb) < 100 * 1024

    # 32-bit EOCD with 50,001 entries (disk vs total) also rejected before ZipFile
    eocd32_over = tmp_path / "eocd32_over.apk"
    eocd32_over.write_bytes(
        struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 0, _MAX_ZIP_ENTRIES + 1, 0, 0, 0)
    )
    with (
        patch.object(
            aa.zipfile,
            "ZipFile",
            side_effect=AssertionError("ZipFile must not be constructed"),
        ),
        pytest.raises(ApkAuditError) as exc_info_32,
    ):
        inspect_apk_archive(eocd32_over)
    assert str(exc_info_32.value) == "APK archive exceeds max ZIP entry count (50001 > 50000)"

    eocd32_disk_over = tmp_path / "eocd32_disk_over.apk"
    eocd32_disk_over.write_bytes(
        struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, _MAX_ZIP_ENTRIES + 2, 0, 0, 0, 0)
    )
    with pytest.raises(ApkAuditError) as exc_info_disk:
        inspect_apk_archive(eocd32_disk_over)
    assert str(exc_info_disk.value) == "APK archive exceeds max ZIP entry count (50002 > 50000)"

    # Exact boundary: 50,000 declared entries in EOCD is allowed to reach ZipFile
    eocd32_at_cap = tmp_path / "eocd32_at_cap.apk"
    eocd32_at_cap.write_bytes(
        struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, _MAX_ZIP_ENTRIES, _MAX_ZIP_ENTRIES, 0, 0, 0)
    )
    with patch("zipfile.ZipFile.infolist", return_value=[]):
        assert inspect_apk_archive(eocd32_at_cap)["dex_count"] == 0

    # Truncated EOCD (21 bytes after PK\x05\x06) raises BadZipFile -> ApkAuditError
    truncated_eocd = tmp_path / "truncated_eocd.apk"
    truncated_eocd.write_bytes(b"PK\x05\x06" + b"\x00" * 17)
    with pytest.raises(ApkAuditError) as exc_trunc:
        inspect_apk_archive(truncated_eocd)
    assert isinstance(exc_trunc.value.__cause__, zipfile.BadZipFile)
    assert (
        str(exc_trunc.value)
        == f"Invalid or corrupt APK ZIP archive {truncated_eocd}: File is not a zip file"
    )

    # Truncated ZIP64 header (< 56 bytes before EOCD) is ignored in favour of 32-bit EOCD
    short_z64 = tmp_path / "short_z64.apk"
    short_z64.write_bytes(
        b"PK\x06\x06"
        + b"\xff" * 51  # 55 bytes total (< 56)
        + struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 0, 0, 0, 0, 0)
    )
    with patch("zipfile.ZipFile.infolist", return_value=[]):
        assert inspect_apk_archive(short_z64)["dex_count"] == 0

    # Direct ZIP64 EOCD (56 bytes immediately before 22-byte EOCD, z64_pos == 0, distance == 56)
    tight_z64 = tmp_path / "tight_z64.apk"
    tight_z64.write_bytes(
        struct.pack("<4sQHHLLQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, 0, 60_000, 0, 0)
        + struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 1, 1, 0, 0, 0)
    )
    with pytest.raises(ApkAuditError) as exc_tight:
        inspect_apk_archive(tight_z64)
    assert str(exc_tight.value) == "APK archive exceeds max ZIP entry count (60000 > 50000)"

    # Non-zero z64_pos, disk64 > total64, and trailing fake PK\x06\x06 in comment after EOCD
    offset_z64 = tmp_path / "offset_z64.apk"
    offset_z64.write_bytes(
        b"prefix-bytes"
        + struct.pack("<4sQHHLLQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, 70_000, 0, 0, 0)
        + struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 1, 1, 0, 0, 56)
        + b"PK\x06\x06"
        + b"\x00" * 52
    )
    with pytest.raises(ApkAuditError) as exc_offset_z64:
        inspect_apk_archive(offset_z64)
    assert str(exc_offset_z64.value) == "APK archive exceeds max ZIP entry count (70000 > 50000)"

    # 32-bit declared count wins if larger than ZIP64 counts, and tail seek skips prefix > 65_633 B
    assert aa._EOCD_TAIL_BYTES == 65_633
    large_prefix_z64 = tmp_path / "large_prefix_z64.apk"
    tail_payload = struct.pack(
        "<4sQHHLLQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, 0, 0
    ) + struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 0, 55_000, 0, 0, 0)
    large_prefix_z64.write_bytes(
        b"PK\x06\x06"
        + struct.pack("<QHHLLQQQQ", 44, 45, 45, 0, 0, 999_999, 999_999, 0, 0)
        + b"\x00" * (aa._EOCD_TAIL_BYTES - len(tail_payload))
        + tail_payload
    )
    with pytest.raises(ApkAuditError) as exc_large_prefix:
        inspect_apk_archive(large_prefix_z64)
    assert str(exc_large_prefix.value) == "APK archive exceeds max ZIP entry count (55000 > 50000)"


def _zip_with(path: Path, entries: dict[str, bytes], *, stored: bool = False) -> Path:
    mode = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w", compression=mode) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_contract_eocd_precheck_boundary_fake_signature_and_truncated_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] the declared-entry pre-check allows exactly the cap and refuses one more,
    reads the LAST end-of-central-directory record (an EOCD-like signature inside an entry's
    stored bytes does not change the count), and a truncated EOCD is a corrupt archive."""
    import struct

    monkeypatch.setattr(aa, "_MAX_ZIP_ENTRIES", 3)
    at_cap = _zip_with(tmp_path / "three.apk", {f"f{i}": b"x" for i in range(3)})
    assert len(aa._read_zip_entries(at_cap)[1]) == 3
    over = _zip_with(tmp_path / "four.apk", {f"f{i}": b"x" for i in range(4)})
    with pytest.raises(aa.ApkAuditError, match=r"exceeds max ZIP entry count \(4 > 3\)"):
        aa._read_zip_entries(over)

    fake_eocd = b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, 60000, 60000, 0, 0, 0)
    decoy = _zip_with(tmp_path / "decoy.apk", {"a.bin": fake_eocd, "b": b"y"}, stored=True)
    assert aa._read_eocd_entry_count(decoy, decoy.stat().st_size) == 2

    truncated = tmp_path / "truncated.apk"
    truncated.write_bytes(b"\x00" * 64 + b"PK\x05\x06" + b"\x00" * 4)
    with pytest.raises(aa.ApkAuditError, match="Invalid or corrupt APK ZIP archive"):
        aa._read_zip_entries(truncated)


# ---------------------------------------------------------------------------
# R4b hardening [contract] tests (linc-crucible-ipxu)
# ---------------------------------------------------------------------------


def _write_lying_60000_cd_apk(path: Path, entry_count: int = 60_000) -> Path:
    """Create a real multi-entry ZIP whose EOCD lies and declares only 1 entry."""
    import io
    import struct

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for i in range(entry_count):
            zf.writestr(f"e{i:05d}", b"")
    data = bytearray(buf.getvalue())
    eocd = data.rfind(b"PK\x05\x06")
    assert eocd >= 0
    struct.pack_into("<HH", data, eocd + 8, 1, 1)
    path.write_bytes(data)
    return path


def test_contract_lying_eocd_with_60000_entry_cd_refused_by_parsed_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] An APK whose EOCD declares 1 entry but whose central directory holds
    60,000 entries is still refused (by the parsed entry count); a central directory over
    the byte cap is refused before ZipFile parses it, whatever count the EOCD declares."""
    import struct

    lying_apk = _write_lying_60000_cd_apk(tmp_path / "lying_60k.apk", 60_000)
    with pytest.raises(ApkAuditError, match=r"exceeds max ZIP entry count \(60000 > 50000\)"):
        inspect_apk_archive(lying_apk)

    # ZIP64 EOCD declaring 1 entry with a 2.76 MB cd_size64: over a lowered byte cap it is
    # refused before ZipFile is constructed.
    monkeypatch.setattr(aa, "_MAX_ZIP_CD_BYTES", 1_000_000)
    z64_lying = tmp_path / "z64_lying.apk"
    z64_lying.write_bytes(
        struct.pack("<4sQHHLLQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, 60_000 * 46, 0)
        + struct.pack("<4sLQL", b"PK\x06\x07", 0, 0, 1)
        + struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 1, 1, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    )
    with (
        patch.object(
            aa.zipfile,
            "ZipFile",
            side_effect=AssertionError("ZipFile must not be constructed"),
        ),
        pytest.raises(
            ApkAuditError,
            match=r"central directory exceeds max size \(2760000 > 1000000\)",
        ),
    ):
        inspect_apk_archive(z64_lying)
    monkeypatch.setattr(aa, "_MAX_ZIP_CD_BYTES", 16 * 1024 * 1024)

    # Valid small ZIP64 archive using 0xFFFFFFFF sentinel in 32-bit EOCD cd_size is NOT refused
    z64_base = _write_zip(tmp_path / "z64_base.apk", {"classes.dex": b"dex", "a.txt": b"hi"})
    raw = z64_base.read_bytes()
    eocd_pos = raw.rfind(b"PK\x05\x06")
    _, _, d, t, cd_sz, cd_off, _ = struct.unpack_from("<HHHHLLH", raw, eocd_pos + 4)
    z64_small = tmp_path / "z64_small.apk"
    z64_small.write_bytes(
        raw[:eocd_pos]
        + struct.pack("<4sQHHLLQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, d, t, cd_sz, cd_off)
        + struct.pack("<4sLQL", b"PK\x06\x07", 0, eocd_pos, 1)
        + struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, d, t, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    )
    assert inspect_apk_archive(z64_small)["dex_count"] == 1

    # Central directory exceeding the hard byte cap is refused before ZipFile
    cd_byte_cap_over = tmp_path / "cd_huge.apk"
    cd_byte_cap_over.write_bytes(
        struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 1, 1, 64 * 1024 * 1024, 0, 0)
    )
    with (
        patch.object(
            aa.zipfile,
            "ZipFile",
            side_effect=AssertionError("ZipFile must not be constructed"),
        ),
        pytest.raises(ApkAuditError, match="central directory exceeds max size"),
    ):
        inspect_apk_archive(cd_byte_cap_over)

    # Exact byte boundary: a CD of exactly _MAX_ZIP_CD_BYTES reaches ZipFile; one byte more
    # is refused. A CD-size-based entry estimate is no longer used.
    cd_at_cap = tmp_path / "cd_at_cap.apk"
    cd_at_cap.write_bytes(
        struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 1, 1, aa._MAX_ZIP_CD_BYTES, 0, 0)
    )
    reached_zipfile = False

    class _StubZipFile:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal reached_zipfile
            reached_zipfile = True

        def __enter__(self) -> _StubZipFile:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def infolist(self) -> list[zipfile.ZipInfo]:
            return []

    with patch.object(aa.zipfile, "ZipFile", _StubZipFile):
        assert inspect_apk_archive(cd_at_cap)["dex_count"] == 0
    assert reached_zipfile is True

    cd_over_cap = tmp_path / "cd_over_cap.apk"
    cd_over_cap.write_bytes(
        struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 1, 1, aa._MAX_ZIP_CD_BYTES + 1, 0, 0)
    )
    with (
        patch.object(
            aa.zipfile,
            "ZipFile",
            side_effect=AssertionError("ZipFile must not be constructed"),
        ),
        pytest.raises(
            ApkAuditError,
            match=rf"central directory exceeds max size \({aa._MAX_ZIP_CD_BYTES + 1} > "
            rf"{aa._MAX_ZIP_CD_BYTES}\)",
        ),
    ):
        inspect_apk_archive(cd_over_cap)


def test_contract_xmltree_u2028_cannot_hide_debuggable_and_crlf_supported(tmp_path: Path) -> None:
    """[contract] Canned aapt2 output with U+2028 (or other non-LF line separators) inside an
    attribute value cannot hide debuggable=true; CRLF line endings still parse cleanly."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    for sep in ("\u2028", "\u2029", "\x0b", "\x0c", "\x85", "\x1c"):
        xml = (
            f"N: android={_NS}\n"
            "  E: manifest\n"
            '    A: package="com.example.sep"\n'
            "    E: application\n"
            f'      A: android:label(0x01010001)="app{sep}E: decoy{sep}'
            f'A: android:debuggable=false"\n'
            "      A: android:debuggable(0x0101000f)=true\n"
        )
        parsed = parse_aapt2_xmltree(xml)
        assert parsed["debuggable"] is True, f"failed for {sep!r}"
        report = audit_apk(
            apk,
            aapt_bin="/sdk/aapt2",
            apksigner_bin="/sdk/apksigner",
            run_cmd=_signed_runner(xml, f"Signer #1 certificate SHA-256 digest: {_FP_A}\n"),
        )
        assert [f["code"] for f in report["findings"]] == ["debuggable_enabled"]
        assert report["summary"]["high"] == 1
        assert report["summary"]["passed"] is False

    crlf_parsed = parse_aapt2_xmltree(CLEAN_XMLTREE.replace("\n", "\r\n"))
    assert crlf_parsed == parse_aapt2_xmltree(CLEAN_XMLTREE)


def test_contract_duplicate_android_attributes_emit_high_finding(tmp_path: Path) -> None:
    """[contract] Duplicate android attributes on the same element emit a high finding."""
    apk = _write_zip(tmp_path / "app.apk", {"classes.dex": b"dex"})
    signed = f"Signer #1 certificate SHA-256 digest: {_FP_A}\n"

    # (a) debuggable=true followed by debuggable=false: not last-wins + duplicate high finding
    dup_true_first = (
        f"N: android={_NS}\n"
        "  E: manifest\n"
        '    A: package="com.example.dup"\n'
        "    E: application\n"
        "      A: android:debuggable(0x0101000f)=true\n"
        "      A: android:debuggable(0x0101000f)=false\n"
    )
    report_tf = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(dup_true_first, signed),
    )
    assert report_tf["manifest"]["debuggable"] is True
    assert any(
        f["severity"] == "high" and f["code"] == "duplicate_android_attribute"
        for f in report_tf["findings"]
    )
    assert any(f["code"] == "debuggable_enabled" for f in report_tf["findings"])
    assert report_tf["summary"]["high"] >= 2
    assert report_tf["summary"]["passed"] is False

    # (b) debuggable=false followed by debuggable=true -> high finding & passed=False
    dup_false_first = (
        f"N: android={_NS}\n"
        "  E: manifest\n"
        '    A: package="com.example.dup"\n'
        "    E: application\n"
        "      A: android:debuggable(0x0101000f)=false\n"
        "      A: android:debuggable(0x0101000f)=true\n"
    )
    report_ft = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(dup_false_first, signed),
    )
    assert any(
        f["severity"] == "high" and f["code"] == "duplicate_android_attribute"
        for f in report_ft["findings"]
    )
    assert report_ft["summary"]["passed"] is False

    # (c) namespace alias + android prefix duplicating exported on <activity>
    dup_alias = (
        f"N: android={_NS}\n"
        f"N: ns0={_NS}\n"
        "  E: manifest\n"
        '    A: package="com.example.dup"\n'
        "    E: application\n"
        "      E: activity\n"
        '        A: android:name=".Main"\n'
        "        A: android:exported=false\n"
        "        A: ns0:exported=true\n"
    )
    report_alias = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(dup_alias, signed),
    )
    assert any(
        f["severity"] == "high" and f["code"] == "duplicate_android_attribute"
        for f in report_alias["findings"]
    )
    assert report_alias["summary"]["passed"] is False

    # (d) same attribute name on TWO separate elements is NOT a duplicate
    separate_elements = (
        f"N: android={_NS}\n"
        "  E: manifest\n"
        '    A: package="com.example.ok"\n'
        "    E: application\n"
        "      E: activity\n"
        '        A: android:name=".First"\n'
        "        A: android:exported=false\n"
        "      E: activity\n"
        '        A: android:name=".Second"\n'
        "        A: android:exported=false\n"
    )
    report_ok = audit_apk(
        apk,
        aapt_bin="/sdk/aapt2",
        apksigner_bin="/sdk/apksigner",
        run_cmd=_signed_runner(separate_elements, signed),
    )
    assert report_ok["findings"] == []
    assert report_ok["summary"]["passed"] is True


def test_contract_legitimate_apk_with_long_entry_names_is_not_refused(tmp_path: Path) -> None:
    """[contract] a well-formed APK with 25,000 entries and 60-character names (a central
    directory of about 2.6 MB, i.e. far more than 25,000 x 46 bytes) is read, not refused:
    the entry cap counts real entries, it is not estimated from the CD size."""
    apk = tmp_path / "big_but_legit.apk"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("classes.dex", b"dex\n035\x00" + b"\x01" * 64)
        for i in range(25_000):
            zf.writestr(f"res/drawable-xxhdpi/generated_resource_name_{i:06d}_long.png", b"")
    result = inspect_apk_archive(apk)
    assert result["dex_count"] == 1


def test_contract_declared_entry_count_boundary() -> None:
    """[contract] exactly _MAX_ZIP_ENTRIES declared entries pass the EOCD bound; one more is
    refused, and the message reports the declared count."""
    assert aa._validate_eocd_bounds(aa._MAX_ZIP_ENTRIES, 0) is None
    with pytest.raises(ApkAuditError) as exc:
        aa._validate_eocd_bounds(aa._MAX_ZIP_ENTRIES + 1, 0)
    assert str(exc.value) == "APK archive exceeds max ZIP entry count (50001 > 50000)"
