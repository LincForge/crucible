"""Contract tests for `crucible apk-audit` CLI command."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from linc_crucible.cli import cli

_NS = "http://schemas.android.com/apk/res/android"

_CLEAN_XMLTREE = (
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
)

_SAMPLE_HEX = "aa11bb22cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233"
_SAMPLE_COLON = (
    "AA:11:BB:22:CC:33:DD:44:EE:55:FF:66:00:11:22:33:"
    "44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33"
)
_APKSIGNER_STDOUT = (
    "Signer #1 certificate DN: CN=Example, O=Demo, C=US\n"
    f"Signer #1 certificate SHA-256 digest: {_SAMPLE_HEX}\n"
)


def _write_apk(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("AndroidManifest.xml", b"binary-xml")
        zf.writestr("classes.dex", b"dex\n035\x00" + b"\x01" * 1024)
    return path


def _write_assetlinks(path: Path, fingerprint: str) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "relation": ["delegate_permission/common.handle_all_urls"],
                    "target": {
                        "namespace": "android_app",
                        "package_name": "com.example.demo",
                        "sha256_cert_fingerprints": [fingerprint],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def stub_sdk_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=_CLEAN_XMLTREE, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_APKSIGNER_STDOUT, stderr="")

    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr(
        "linc_crucible.core.apk_audit.find_apksigner", lambda: "/usr/bin/apksigner"
    )
    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _fake_run_cmd)


def test_contract_apk_audit_default_json_and_human_output(
    runner: CliRunner, tmp_path: Path, stub_sdk_tools: None
) -> None:
    """[contract] (1) Default invocation emits valid JSON on stdout, empty stderr, exits 0;
    --human emits plain text (non-JSON, no ANSI) showing package, SHA-256, DEX size,
    signing status, and findings."""
    apk = _write_apk(tmp_path / "demo.apk")

    res = runner.invoke(cli, ["apk-audit", str(apk)])
    assert res.exit_code == 0, res.output
    assert res.stderr == ""
    payload = json.loads(res.stdout)
    assert payload["apk_path"] == str(apk)
    assert len(payload["sha256"]) == 64
    assert payload["manifest"]["package"] == "com.example.demo"
    assert payload["archive"]["dex_count"] == 1
    assert payload["archive"]["dex_uncompressed_bytes"] == 1032
    assert payload["signing"]["status"] == "ok"
    assert payload["signing"]["cert_sha256s"] == [_SAMPLE_COLON]
    assert payload["findings"] == []
    assert payload["summary"]["passed"] is True

    human_res = runner.invoke(cli, ["--human", "apk-audit", str(apk)])
    assert human_res.exit_code == 0, human_res.output
    assert human_res.stderr == ""
    with pytest.raises(json.JSONDecodeError):
        json.loads(human_res.stdout)
    assert "\x1b" not in human_res.stdout
    assert "com.example.demo" in human_res.stdout
    assert payload["sha256"] in human_res.stdout
    assert "1032 B uncompressed" in human_res.stdout
    assert "Signing: ok" in human_res.stdout
    assert "Findings: none" in human_res.stdout


def test_contract_apk_audit_assetlinks_and_strict_exit_codes(
    runner: CliRunner, tmp_path: Path, stub_sdk_tools: None
) -> None:
    """[contract] (2) --assetlinks <mismatch.json> --strict emits full report and exits 1
    when summary['passed'] is False, and exits 0 when summary['passed'] is True."""
    apk = _write_apk(tmp_path / "demo.apk")
    mismatch = _write_assetlinks(tmp_path / "mismatch.json", "FF:" * 31 + "FF")
    match = _write_assetlinks(tmp_path / "match.json", _SAMPLE_COLON)

    # Without --strict, completed report with high finding still exits 0 for JSON inspection
    non_strict_res = runner.invoke(cli, ["apk-audit", str(apk), "--assetlinks", str(mismatch)])
    assert non_strict_res.exit_code == 0, non_strict_res.output
    assert non_strict_res.stderr == ""
    non_strict_payload = json.loads(non_strict_res.stdout)
    assert non_strict_payload["summary"]["passed"] is False
    assert non_strict_payload["assetlinks"]["status"] == "mismatch"

    # With --strict and passed=False, emits full report first and exits 1
    strict_fail_res = runner.invoke(
        cli, ["apk-audit", str(apk), "--assetlinks", str(mismatch), "--strict"]
    )
    assert strict_fail_res.exit_code == 1, strict_fail_res.output
    assert strict_fail_res.stderr == ""
    fail_payload = json.loads(strict_fail_res.stdout)
    assert fail_payload["summary"]["passed"] is False
    assert fail_payload["summary"]["high"] == 1
    assert fail_payload["findings"][0]["code"] == "assetlinks_sha256_mismatch"

    # --human with findings and --strict renders findings table and exits 1
    human_fail_res = runner.invoke(
        cli, ["--human", "apk-audit", str(apk), "--assetlinks", str(mismatch), "--strict"]
    )
    assert human_fail_res.exit_code == 1, human_fail_res.output
    assert human_fail_res.stderr == ""
    assert "\x1b" not in human_fail_res.stdout
    assert "assetlinks_sha256_mismatch" in human_fail_res.stdout
    assert "[FAIL]" in human_fail_res.stdout

    # With --strict and passed=True, exits 0
    strict_pass_res = runner.invoke(
        cli, ["apk-audit", str(apk), "--assetlinks", str(match), "--strict"]
    )
    assert strict_pass_res.exit_code == 0, strict_pass_res.output
    assert strict_pass_res.stderr == ""
    pass_payload = json.loads(strict_pass_res.stdout)
    assert pass_payload["summary"]["passed"] is True
    assert pass_payload["assetlinks"]["status"] == "matched"


def test_contract_apk_audit_hostile_inputs_exit_2_with_validation_envelope(
    runner: CliRunner, tmp_path: Path, stub_sdk_tools: None
) -> None:
    """[contract] (3) Hostile/missing APK or corrupt --assetlinks path exits 2 with empty
    stdout and a valid JSON validation error envelope on stderr."""
    apk = _write_apk(tmp_path / "valid.apk")
    corrupt_apk = tmp_path / "corrupt.apk"
    corrupt_apk.write_bytes(b"not a valid zip archive")
    corrupt_assetlinks = tmp_path / "corrupt_assetlinks.json"
    corrupt_assetlinks.write_text("{not valid json", encoding="utf-8")
    non_list_assetlinks = tmp_path / "obj_assetlinks.json"
    non_list_assetlinks.write_text('{"target": {}}', encoding="utf-8")

    invalid_invocations = [
        ["apk-audit", str(tmp_path / "does_not_exist.apk")],
        ["apk-audit", str(tmp_path)],
        ["apk-audit", str(corrupt_apk)],
        ["apk-audit", str(apk), "--assetlinks", str(tmp_path / "missing_assetlinks.json")],
        ["apk-audit", str(apk), "--assetlinks", str(corrupt_assetlinks)],
        ["apk-audit", str(apk), "--assetlinks", str(non_list_assetlinks)],
    ]

    for argv in invalid_invocations:
        res = runner.invoke(cli, argv)
        assert res.exit_code == 2, f"argv={argv} output={res.output!r}"
        assert res.stdout == "", f"argv={argv} stdout={res.stdout!r}"
        envelope = json.loads(res.stderr)
        assert envelope["error"]["category"] == "validation"
        assert envelope["error"]["code"] == "USAGE_ERROR"
        assert envelope["error"]["message"]
        assert "No such command" not in envelope["error"]["message"]


def test_apk_audit_help_documents_options(runner: CliRunner) -> None:
    res = runner.invoke(cli, ["apk-audit", "--help"])
    assert res.exit_code == 0
    assert "--assetlinks" in res.stdout
    assert "--strict" in res.stdout
    flat = " ".join(res.stdout.split()).lower()
    assert "unmeasured" in flat
    assert "tool error" in flat
    assert "without --strict" in flat
    assert "0" in flat


def test_apk_audit_human_without_sdk_tools_and_human_error(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: None)
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_apksigner", lambda: None)
    apk = _write_apk(tmp_path / "no_sdk.apk")

    res = runner.invoke(cli, ["--human", "apk-audit", str(apk)])
    assert res.exit_code == 0, res.output
    assert res.stderr == ""
    assert "[UNMEASURED]" in res.stdout
    assert "[PASS]" not in res.stdout
    assert "Package: unavailable" in res.stdout
    assert "aapt2/aapt binary not found" in res.stdout
    assert "Signing: unavailable (certs: none)" in res.stdout
    assert "apksigner binary not found" in res.stdout
    assert "Findings: none" in res.stdout

    err_res = runner.invoke(cli, ["--human", "apk-audit", str(tmp_path / "missing.apk")])
    assert err_res.exit_code == 2
    assert err_res.stdout == ""
    assert "does not exist" in err_res.stderr


def test_contract_apk_audit_unmeasured_without_sdk_tools_strict_exits_1(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] With no aapt2/apksigner resolvable, --strict exits 1 and JSON shows
    measured: false, passed: false; without --strict exit code stays 0."""
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: None)
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_apksigner", lambda: None)
    apk = _write_apk(tmp_path / "ok.apk")

    strict_res = runner.invoke(cli, ["apk-audit", str(apk), "--strict"])
    assert strict_res.exit_code == 1, strict_res.output
    assert strict_res.stderr == ""
    strict_payload = json.loads(strict_res.stdout)
    assert strict_payload["summary"]["measured"] is False
    assert strict_payload["summary"]["passed"] is False

    non_strict_res = runner.invoke(cli, ["apk-audit", str(apk)])
    assert non_strict_res.exit_code == 0, non_strict_res.output
    assert non_strict_res.stderr == ""
    non_strict_payload = json.loads(non_strict_res.stdout)
    assert non_strict_payload["summary"]["measured"] is False
    assert non_strict_payload["summary"]["passed"] is False


def test_contract_apk_audit_partial_sdk_tools_unmeasured(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] aapt2 OK but apksigner missing (and vice versa) -> unmeasured."""
    apk = _write_apk(tmp_path / "partial.apk")

    def _fake_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=_CLEAN_XMLTREE, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_APKSIGNER_STDOUT, stderr="")

    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _fake_run_cmd)

    # aapt2 OK, apksigner missing
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_apksigner", lambda: None)

    res = runner.invoke(cli, ["apk-audit", str(apk), "--strict"])
    assert res.exit_code == 1, res.output
    payload = json.loads(res.stdout)
    assert payload["manifest"]["status"] == "ok"
    assert payload["signing"]["status"] == "unavailable"
    assert payload["summary"]["measured"] is False
    assert payload["summary"]["passed"] is False

    human_res = runner.invoke(cli, ["--human", "apk-audit", str(apk)])
    assert human_res.exit_code == 0, human_res.output
    assert "[UNMEASURED]" in human_res.stdout
    assert "[PASS]" not in human_res.stdout
    assert "apksigner binary not found" in human_res.stdout

    # aapt2 missing, apksigner OK
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: None)
    monkeypatch.setattr(
        "linc_crucible.core.apk_audit.find_apksigner", lambda: "/usr/bin/apksigner"
    )

    res_no_aapt = runner.invoke(cli, ["apk-audit", str(apk), "--strict"])
    assert res_no_aapt.exit_code == 1, res_no_aapt.output
    payload_no_aapt = json.loads(res_no_aapt.stdout)
    assert payload_no_aapt["manifest"]["status"] == "unavailable"
    assert payload_no_aapt["signing"]["status"] == "ok"
    assert payload_no_aapt["summary"]["measured"] is False
    assert payload_no_aapt["summary"]["passed"] is False

    human_no_aapt = runner.invoke(cli, ["--human", "apk-audit", str(apk)])
    assert human_no_aapt.exit_code == 0, human_no_aapt.output
    assert "[UNMEASURED]" in human_no_aapt.stdout
    assert "aapt2/aapt binary not found" in human_no_aapt.stdout


def test_contract_apk_audit_assetlinks_unverifiable_adds_high_finding_and_strict_exits_1(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] --assetlinks with a fingerprint that cannot be checked -> passed: false
    and a high finding; --strict exits 1."""
    apk = _write_apk(tmp_path / "demo.apk")
    al = _write_assetlinks(tmp_path / "al.json", _SAMPLE_COLON)

    def _fake_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=_CLEAN_XMLTREE, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_APKSIGNER_STDOUT, stderr="")

    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _fake_run_cmd)
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr("linc_crucible.core.apk_audit.find_apksigner", lambda: None)

    non_strict = runner.invoke(cli, ["apk-audit", str(apk), "--assetlinks", str(al)])
    assert non_strict.exit_code == 0, non_strict.output
    assert non_strict.stderr == ""
    payload = json.loads(non_strict.stdout)
    assert payload["assetlinks"]["status"] == "unavailable"
    assert payload["summary"]["measured"] is False
    assert payload["summary"]["passed"] is False
    assert payload["summary"]["high"] == 1
    assert payload["findings"] == [
        {
            "severity": "high",
            "code": "assetlinks_unavailable",
            "detail": payload["assetlinks"]["detail"],
        }
    ]

    strict = runner.invoke(cli, ["apk-audit", str(apk), "--assetlinks", str(al), "--strict"])
    assert strict.exit_code == 1, strict.output
    assert json.loads(strict.stdout) == payload

    human = runner.invoke(cli, ["--human", "apk-audit", str(apk), "--assetlinks", str(al)])
    assert human.exit_code == 0, human.output
    assert "[UNMEASURED]" in human.stdout
    assert "Assetlinks: unavailable" in human.stdout
    assert "Cannot verify assetlinks" in human.stdout
    assert "assetlinks_unavailable" in human.stdout


def test_contract_apk_audit_tool_error_shows_human_detail_and_strict_exits_1(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] Tool execution error keeps measured=True, passed=False, --strict exits 1,
    and --human prints [FAIL] along with each failing layer's detail."""
    apk = _write_apk(tmp_path / "broken.apk")

    def _failing_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="corrupt xmltree output")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="DOES NOT VERIFY")

    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr(
        "linc_crucible.core.apk_audit.find_apksigner", lambda: "/usr/bin/apksigner"
    )
    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _failing_run_cmd)

    res = runner.invoke(cli, ["apk-audit", str(apk), "--strict"])
    assert res.exit_code == 1, res.output
    payload = json.loads(res.stdout)
    assert payload["manifest"]["status"] == "error"
    assert payload["signing"]["status"] == "error"
    assert payload["summary"]["measured"] is True
    assert payload["summary"]["passed"] is False

    human = runner.invoke(cli, ["--human", "apk-audit", str(apk)])
    assert human.exit_code == 0, human.output
    assert "[FAIL]" in human.stdout
    assert "corrupt xmltree output" in human.stdout
    assert "DOES NOT VERIFY" in human.stdout


def test_contract_apk_audit_human_renders_untrusted_markup_literally(
    runner: CliRunner, tmp_path: Path, stub_sdk_tools: None
) -> None:
    """[contract] Strings taken from the APK (ZIP entry names, manifest values) and the
    APK path are untrusted input: --human must print them verbatim, never parse them as
    Rich markup (a stray closing tag raised MarkupError; an open tag swallowed text)."""
    (tmp_path / "evil[").mkdir()
    apk = tmp_path / "evil[" / "bold].apk"  # path text contains "[/bold]"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("AndroidManifest.xml", b"binary-xml")
        zf.writestr("classes.dex", b"dex\n035\x00" + b"\x01" * 1024)
        zf.writestr("assets/x[/bold].js.map", b"{}")
        zf.writestr("assets/[red]hidden[/red].js.map", b"{}")

    res = runner.invoke(cli, ["--human", "apk-audit", str(apk)])
    assert res.exit_code == 0, res.output
    assert res.exception is None
    flat = " ".join(res.stdout.split())
    assert "evil[/bold].apk" in flat
    assert "x[/bold].js.map" in flat
    assert "[red]hidden[/red].js.map" in flat


def test_contract_apk_audit_human_verdict_detail_and_summary_lines(
    runner: CliRunner, tmp_path: Path, stub_sdk_tools: None
) -> None:
    """[contract] --human prints the exact verdict token, the assetlinks detail line, and the
    summary line including measured, so a reader sees why a check failed."""
    apk = _write_apk(tmp_path / "demo.apk")
    match = _write_assetlinks(tmp_path / "match.json", _SAMPLE_COLON)
    mismatch = _write_assetlinks(tmp_path / "mismatch.json", "FF:" * 31 + "FF")

    ok = runner.invoke(cli, ["--human", "apk-audit", str(apk), "--assetlinks", str(match)])
    assert ok.exit_code == 0, ok.output
    first = ok.stdout.splitlines()[0]
    assert first.endswith(" [PASS]"), first
    assert "  Summary: high=0, medium=0, low=0, info=0, measured=True" in ok.stdout

    bad = runner.invoke(cli, ["--human", "apk-audit", str(apk), "--assetlinks", str(mismatch)])
    assert bad.exit_code == 0, bad.output
    lines = bad.stdout.splitlines()
    assert lines[0].endswith(" [FAIL]"), lines[0]
    i = next(n for n, line in enumerate(lines) if line.startswith("  Assetlinks: mismatch"))
    assert lines[i + 1].startswith("    detail: APK signing SHA-256"), lines[i : i + 2]
    assert "measured=True" in bad.stdout


def _write_million_entry_apk(path: Path, entry_count: int = 1_000_000) -> Path:
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


def test_contract_apk_audit_human_escapes_terminal_and_bidi_controls(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] an APK with entry assets/\\x1b[31mevil\\x1b[0m.map and a bidi U+202E name:
    --human output contains no raw ESC or U+202E byte (fails on 2143889)."""
    apk = tmp_path / "bidi_\u202e_esc.apk"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("AndroidManifest.xml", b"binary-xml")
        zf.writestr("classes.dex", b"dex\n035\x00" + b"\x01" * 1024)
        zf.writestr("assets/\x1b[31mevil\x1b[0m.map", b"{}")
        zf.writestr("assets/spoof_\u202ekpa.map", b"{}")
        zf.writestr("assets/controls_\r_\x07_\x9d_\u202a_\u2066_\u2069.map", b"{}")

    hostile_xmltree = _CLEAN_XMLTREE.replace(
        'package="com.example.demo"',
        'package="com.example.\x1b[31mdemo\u202e\x07\x9d"',
    )

    def _fake_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=hostile_xmltree, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_APKSIGNER_STDOUT, stderr="")

    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr(
        "linc_crucible.core.apk_audit.find_apksigner", lambda: "/usr/bin/apksigner"
    )
    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _fake_run_cmd)

    res = runner.invoke(cli, ["--human", "apk-audit", str(apk)])
    assert res.exit_code == 0, res.output
    assert "\x1b" not in res.stdout
    assert "\u202e" not in res.stdout
    assert b"\xe2\x80\xae" not in res.stdout.encode("utf-8")
    assert "\x07" not in res.stdout
    assert "\x9d" not in res.stdout
    assert "\r" not in res.stdout
    for bidi_cp in (*range(0x202A, 0x202F), *range(0x2066, 0x206A)):
        assert chr(bidi_cp) not in res.stdout
    assert r"\x1b[31mevil\x1b[0m.map" in res.stdout
    assert r"\u202e" in res.stdout


def test_contract_apk_audit_dash_prefixed_relative_path_uses_resolved_abspath(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] crucible apk-audit -- --in=ok.apk (relative path starting with -):
    the shim records an absolute path argument, never --in=ok.apk as an option."""
    monkeypatch.chdir(tmp_path)
    apk = _write_apk(tmp_path / "--in=ok.apk")
    expected_abs = str(apk.resolve())
    seen_cmds: list[list[str]] = []

    def _recording_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(cmd)
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=_CLEAN_XMLTREE, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_APKSIGNER_STDOUT, stderr="")

    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr(
        "linc_crucible.core.apk_audit.find_apksigner", lambda: "/usr/bin/apksigner"
    )
    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _recording_run_cmd)

    res = runner.invoke(cli, ["apk-audit", "--", "--in=ok.apk"])
    assert res.exit_code == 0, res.output
    assert len(seen_cmds) == 2
    for cmd in seen_cmds:
        assert "--in=ok.apk" not in cmd
        assert expected_abs in cmd
    assert seen_cmds[0] == [
        "/usr/bin/aapt2",
        "dump",
        "xmltree",
        expected_abs,
        "--file",
        "AndroidManifest.xml",
    ]
    assert seen_cmds[1] == ["/usr/bin/apksigner", "verify", "--print-certs", expected_abs]


def test_contract_apk_audit_assetlinks_5000_digit_integer_exits_2(
    runner: CliRunner, tmp_path: Path, stub_sdk_tools: None
) -> None:
    """[contract] assetlinks with a 5000-digit integer -> exit 2 validation envelope."""
    apk = _write_apk(tmp_path / "ok.apk")
    huge_int_al = tmp_path / "huge_int_assetlinks.json"
    huge_int_al.write_text("[" + ("9" * 5000) + "]", encoding="utf-8")

    res = runner.invoke(cli, ["apk-audit", str(apk), "--assetlinks", str(huge_int_al)])
    assert res.exit_code == 2, res.output
    assert res.stdout == ""
    envelope = json.loads(res.stderr)
    assert envelope["error"]["category"] == "validation"
    assert envelope["error"]["code"] == "USAGE_ERROR"
    assert "Failed to load assetlinks JSON" in envelope["error"]["message"]


def test_contract_apk_audit_million_entry_apk_refused_before_zipfile(
    runner: CliRunner, tmp_path: Path, stub_sdk_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] Generated APK declaring 1,000,000 entries is refused before ZipFile parses it."""
    million_apk = _write_million_entry_apk(tmp_path / "million.apk", 1_000_000)

    def _forbid_zipfile(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("zipfile.ZipFile must not be constructed for over-cap EOCD")

    monkeypatch.setattr("linc_crucible.core.apk_audit.zipfile.ZipFile", _forbid_zipfile)

    res = runner.invoke(cli, ["apk-audit", str(million_apk)])
    assert res.exit_code == 2, res.output
    assert res.stdout == ""
    envelope = json.loads(res.stderr)
    assert envelope["error"]["category"] == "validation"
    assert envelope["error"]["code"] == "USAGE_ERROR"
    assert "1000000 > 50000" in envelope["error"]["message"]


def test_contract_escape_terminal_text_exact_forms() -> None:
    """[contract] each control class is shown as one exact visible escape; other text is
    unchanged."""
    from linc_crucible.cli import _escape_terminal_text

    assert _escape_terminal_text("a\x1bb") == "a\\x1bb"
    assert _escape_terminal_text("\x00\x07\x7f\x9d\x9f") == "\\x00\\x07\\x7f\\x9d\\x9f"
    assert _escape_terminal_text("\t\n\r") == "\\t\\n\\r"
    assert _escape_terminal_text("‪‮⁦⁩") == "\\u202a\\u202e\\u2066\\u2069"
    assert _escape_terminal_text("plain é ✓") == "plain é ✓"
    assert _escape_terminal_text(42) == "42"


def test_contract_apk_audit_lying_eocd_60000_entry_cd_exits_2(
    runner: CliRunner, tmp_path: Path, stub_sdk_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] Generated APK declaring 1 entry in EOCD whose central directory holds 60,000
    entries is refused with exit 2 (by its parsed entry count)."""
    import io
    import struct

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for i in range(60_000):
            zf.writestr(f"e{i:05d}", b"")
    data = bytearray(buf.getvalue())
    eocd = data.rfind(b"PK\x05\x06")
    assert eocd >= 0
    struct.pack_into("<HH", data, eocd + 8, 1, 1)
    lying_apk = tmp_path / "lying_60k.apk"
    lying_apk.write_bytes(data)

    res = runner.invoke(cli, ["apk-audit", str(lying_apk)])
    assert res.exit_code == 2, res.output
    assert res.stdout == ""
    envelope = json.loads(res.stderr)
    assert envelope["error"]["category"] == "validation"
    assert envelope["error"]["code"] == "USAGE_ERROR"
    assert "50000" in envelope["error"]["message"]


def test_contract_apk_audit_u2028_in_attr_cannot_hide_debuggable_strict_exits_1(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] Canned aapt2 output with U+2028 inside an attribute value cannot hide
    debuggable=true (debuggable finding still raised; --strict exits 1)."""
    apk = _write_apk(tmp_path / "u2028.apk")
    xmltree = (
        f"N: android={_NS} (line=2)\n"
        "  E: manifest (line=2)\n"
        '    A: package="com.example.u2028" (Raw: "com.example.u2028")\n'
        "      E: application (line=5)\n"
        '        A: android:label(0x01010001)="bad\u2028E: decoy\u2028'
        'A: android:debuggable=false"\n'
        "        A: android:debuggable(0x0101000f)=true\n"
    )

    def _fake_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=xmltree, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_APKSIGNER_STDOUT, stderr="")

    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr(
        "linc_crucible.core.apk_audit.find_apksigner", lambda: "/usr/bin/apksigner"
    )
    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _fake_run_cmd)

    res = runner.invoke(cli, ["apk-audit", str(apk), "--strict"])
    assert res.exit_code == 1, res.output
    assert res.stderr == ""
    payload = json.loads(res.stdout)
    assert payload["manifest"]["debuggable"] is True
    assert any(f["code"] == "debuggable_enabled" for f in payload["findings"])
    assert payload["summary"]["passed"] is False


def test_contract_apk_audit_duplicate_debuggable_emits_high_finding_strict_exits_1(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] Duplicate debuggable attributes -> high finding and --strict exits 1."""
    apk = _write_apk(tmp_path / "dup.apk")
    xmltree = (
        f"N: android={_NS} (line=2)\n"
        "  E: manifest (line=2)\n"
        '    A: package="com.example.dup" (Raw: "com.example.dup")\n'
        "      E: application (line=5)\n"
        "        A: android:debuggable(0x0101000f)=true\n"
        "        A: android:debuggable(0x0101000f)=false\n"
    )

    def _fake_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=xmltree, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_APKSIGNER_STDOUT, stderr="")

    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr(
        "linc_crucible.core.apk_audit.find_apksigner", lambda: "/usr/bin/apksigner"
    )
    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _fake_run_cmd)

    res = runner.invoke(cli, ["apk-audit", str(apk), "--strict"])
    assert res.exit_code == 1, res.output
    payload = json.loads(res.stdout)
    assert any(
        f["severity"] == "high" and f["code"] == "duplicate_android_attribute"
        for f in payload["findings"]
    )
    assert payload["summary"]["high"] >= 1
    assert payload["summary"]["passed"] is False


def test_contract_apk_audit_human_escapes_unicode_format_and_separators(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] U+2028/U+200B/U+FEFF (and all Cf/Zl/Zp chars) are shown escaped in --human."""
    import sys
    import unicodedata

    from linc_crucible.cli import _escape_terminal_text

    apk = _write_apk(tmp_path / "unicode_\u200b_\u2028_\ufeff.apk")
    xmltree = _CLEAN_XMLTREE.replace(
        'package="com.example.demo"',
        'package="com.example.\u200b\u200f\u2028\u2029\u2060\ufeff\u061c"',
    )

    def _fake_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if "xmltree" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=xmltree, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_APKSIGNER_STDOUT, stderr="")

    monkeypatch.setattr("linc_crucible.core.apk_audit.find_aapt", lambda: "/usr/bin/aapt2")
    monkeypatch.setattr(
        "linc_crucible.core.apk_audit.find_apksigner", lambda: "/usr/bin/apksigner"
    )
    monkeypatch.setattr("linc_crucible.core.apk_audit._default_run_cmd", _fake_run_cmd)

    res = runner.invoke(cli, ["--human", "apk-audit", str(apk)])
    assert res.exit_code == 0, res.output
    for raw_ch in ("\u200b", "\u200f", "\u2028", "\u2029", "\u2060", "\ufeff", "\u061c"):
        assert raw_ch not in res.stdout
    for escaped in (r"\u2028", r"\u200b", r"\ufeff", r"\u2029", r"\u2060", r"\u061c"):
        assert escaped in res.stdout

    # Also verify every Unicode Cf/Zl/Zp character across the full Unicode range is escaped
    for cp in range(sys.maxunicode + 1):
        ch = chr(cp)
        if unicodedata.category(ch) in {"Cf", "Zl", "Zp"}:
            expected = f"\\u{cp:04x}" if cp <= 0xFFFF else f"\\U{cp:08x}"
            assert _escape_terminal_text(ch) == expected
