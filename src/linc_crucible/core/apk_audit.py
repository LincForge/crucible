"""Offline static audit for standalone Android APK binaries.

Inspects three layers without requiring a connected device or Gradle checkout:
1. Pure-stdlib ZIP/DEX archive structure (DEX footprint, ABIs, bundled sourcemaps).
2. Compiled AndroidManifest.xml via ``aapt2 dump xmltree`` (security & lifecycle flags).
3. Signing certificate SHA-256 digests via ``apksigner verify --print-certs`` and
   optional offline verification against a local Digital Asset Links ``assetlinks.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from .._bounded_read import _MAX_REGISTRY_BYTES, RegistryTooLargeError, read_capped
from .package_resolver import find_aapt, find_build_tools_binary

logger = structlog.get_logger().bind(domain="apk_audit")

_MAX_ZIP_ENTRIES = 50_000
_MAX_ZIP_CD_BYTES = 16 * 1024 * 1024
_ZIP64_SENTINEL32 = 0xFFFFFFFF
_EOCD_SIG = b"PK\x05\x06"
_EOCD_SIZE = 22
_EOCD64_SIG = b"PK\x06\x06"
_EOCD64_SIZE = 56
_EOCD_TAIL_BYTES = 65_633
_MAX_ASSETLINKS_BYTES = _MAX_REGISTRY_BYTES  # 1 MiB
_SUBPROCESS_TIMEOUT_S = 30

_DEX_BLOAT_MIN_APK_BYTES = 5_000_000
_DEX_BLOAT_MIN_UNCOMPRESSED_BYTES = 10_000_000
_DEX_BLOAT_MIN_RATIO = 0.65
_CONFIG_UI_MODE = 0x200

_ANDROID_NS = "http://schemas.android.com/apk/res/android"
_HANDLE_ALL_URLS_RELATION = "delegate_permission/common.handle_all_urls"
_SEVERITIES = ("high", "medium", "low", "info")
_UNSET: Any = object()

_RES_ID_SUFFIX_RE = re.compile(r"\(0x[0-9a-fA-F]+\)$")
_TYPE_PREFIX_RE = re.compile(r"^\(type\s+0x[0-9a-fA-F]+\)\s*")
_NS_DECL_RE = re.compile(r"^([A-Za-z0-9_.-]+)=([^\s(]+)")
_APKSIGNER_SHA256_RE = re.compile(
    r"^[ \t]*(?:Signer\s+#\d+\s+)?certificate\s+SHA-256\s+digest:\s*([0-9a-fA-F:]+)",
    re.IGNORECASE | re.MULTILINE,
)
_VALID_FINGERPRINT_RE = re.compile(r"^[0-9a-fA-F]{64}$|^(?:[0-9a-fA-F]{2}:){31}[0-9a-fA-F]{2}$")

RunCmd = Callable[[list[str]], subprocess.CompletedProcess[str]]


class ApkAuditError(ValueError):
    """Raised when an APK or assetlinks input is missing, invalid, corrupt, or over cap."""


@dataclass
class _XmlNode:
    tag: str
    indent: int
    attrs: dict[str, str] = field(default_factory=dict)
    android_keys: set[str] = field(default_factory=set)
    duplicate_android_keys: set[str] = field(default_factory=set)
    children: list[_XmlNode] = field(default_factory=list)


def _default_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
    )


def find_apksigner() -> str | None:
    """Locate ``apksigner`` on PATH first, then ``build-tools/*/apksigner``."""
    return find_build_tools_binary("apksigner")


def _validate_regular_file(raw_path: str | Path, label: str) -> Path:
    if isinstance(raw_path, str) and not raw_path.strip():
        raise ApkAuditError(f"{label} path must not be empty")
    path = Path(raw_path)
    if path.is_dir():
        raise ApkAuditError(f"{label} path is a directory: {path}")
    if not path.exists():
        raise ApkAuditError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise ApkAuditError(f"{label} path is not a regular file: {path}")
    return path


def _file_sha256(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()
    except OSError as exc:
        raise ApkAuditError(f"Failed to read APK file {path}: {exc}") from exc


def _dex_compressed_ratio(dex_compressed_bytes: int, apk_bytes: int) -> float:
    if apk_bytes <= 0:
        return 0.0
    return round(dex_compressed_bytes / apk_bytes, 4)


def _extract_abi(entry_name: str) -> str | None:
    parts = entry_name.split("/")
    if len(parts) == 3 and parts[0] == "lib" and parts[1] and parts[2].endswith(".so"):
        return parts[1]
    return None


def _read_eocd_bounds(path: Path, apk_bytes: int) -> tuple[int, int]:
    """Return ``(declared_entries, central_directory_bytes)`` from EOCD/ZIP64 EOCD."""
    with path.open("rb") as fh:
        fh.seek(max(apk_bytes - _EOCD_TAIL_BYTES, 0))
        tail = fh.read()
    eocd_pos = tail.rfind(_EOCD_SIG)
    if eocd_pos < 0 or len(tail) - eocd_pos < _EOCD_SIZE:
        raise zipfile.BadZipFile("File is not a zip file")
    disk_entries, total_entries, cd_size = struct.unpack_from("<HHL", tail, eocd_pos + 8)
    declared = max(disk_entries, total_entries)
    z64_pos = tail.rfind(_EOCD64_SIG, 0, eocd_pos)
    if z64_pos >= 0 and eocd_pos - z64_pos >= _EOCD64_SIZE:
        disk64, total64, cd_size64 = struct.unpack_from("<QQQ", tail, z64_pos + 24)
        declared = max(declared, disk64, total64)
        cd_size = cd_size64 if cd_size == _ZIP64_SENTINEL32 else max(cd_size, cd_size64)
    return declared, cd_size


def _read_eocd_entry_count(path: Path, apk_bytes: int) -> int:
    return _read_eocd_bounds(path, apk_bytes)[0]


def _validate_eocd_bounds(declared_entries: int, cd_size: int) -> None:
    if cd_size > _MAX_ZIP_CD_BYTES:
        raise ApkAuditError(
            f"APK archive central directory exceeds max size ({cd_size} > {_MAX_ZIP_CD_BYTES})"
        )
    # The byte cap bounds parsing memory. An entry count estimated from the CD size
    # would refuse legitimate APKs with long entry names, so only the declared count
    # is checked here; the parsed count is checked after ZipFile reads the index.
    if declared_entries > _MAX_ZIP_ENTRIES:
        raise ApkAuditError(
            f"APK archive exceeds max ZIP entry count ({declared_entries} > {_MAX_ZIP_ENTRIES})"
        )


def _read_zip_entries(path: Path) -> tuple[int, list[zipfile.ZipInfo]]:
    try:
        apk_bytes = path.stat().st_size
        declared_entries, cd_size = _read_eocd_bounds(path, apk_bytes)
        _validate_eocd_bounds(declared_entries, cd_size)
        with zipfile.ZipFile(path, "r") as zf:
            entries = zf.infolist()
    except ApkAuditError:
        raise
    except (zipfile.BadZipFile, EOFError, zlib.error, OSError, ValueError) as exc:
        raise ApkAuditError(f"Invalid or corrupt APK ZIP archive {path}: {exc}") from exc
    if len(entries) > _MAX_ZIP_ENTRIES:
        raise ApkAuditError(
            f"APK archive exceeds max ZIP entry count ({len(entries)} > {_MAX_ZIP_ENTRIES})"
        )
    return apk_bytes, entries


def _is_root_dex(name: str) -> bool:
    return "/" not in name and name.startswith("classes") and name.endswith(".dex")


def _accumulate_zip_entry(
    info: zipfile.ZipInfo,
    abis: set[str],
    sourcemaps: set[str],
) -> tuple[int, int, int]:
    if info.file_size < 0 or info.compress_size < 0:
        raise ApkAuditError(f"Corrupt ZIP entry size for {info.filename!r}")
    name = info.filename
    dex = (1, info.file_size, info.compress_size) if _is_root_dex(name) else (0, 0, 0)
    abi = _extract_abi(name)
    if abi is not None:
        abis.add(abi)
    if name.startswith(("assets/", "res/")) and name.endswith(".map"):
        sourcemaps.add(name)
    return dex


def inspect_apk_archive(apk_path: str | Path) -> dict[str, Any]:
    """Inspect an APK as a pure ZIP archive and compute DEX, ABI, and sourcemap metrics."""
    path = _validate_regular_file(apk_path, "APK")
    apk_bytes, entries = _read_zip_entries(path)

    dex_count = 0
    dex_uncompressed = 0
    dex_compressed = 0
    all_dex_stored = True
    abis: set[str] = set()
    sourcemaps: set[str] = set()

    for info in entries:
        dc, du, dcmp = _accumulate_zip_entry(info, abis, sourcemaps)
        if dc:
            dex_count += dc
            dex_uncompressed += du
            dex_compressed += dcmp
            all_dex_stored = all_dex_stored and info.compress_type == zipfile.ZIP_STORED

    dex_stored = dex_count > 0 and all_dex_stored
    return {
        "apk_bytes": apk_bytes,
        "dex_count": dex_count,
        "dex_uncompressed_bytes": dex_uncompressed,
        "dex_compressed_bytes": dex_compressed,
        "dex_compressed_ratio": _dex_compressed_ratio(dex_compressed, apk_bytes),
        "dex_stored": dex_stored,
        "abis": sorted(abis),
        "sourcemap_files": sorted(sourcemaps),
    }


def _archive_findings(archive: Mapping[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    apk_bytes = archive["apk_bytes"]
    dex_uncompressed = archive["dex_uncompressed_bytes"]
    dex_ratio = archive["dex_compressed_ratio"]
    ratio_bloated = not archive["dex_stored"] and dex_ratio >= _DEX_BLOAT_MIN_RATIO
    if apk_bytes >= _DEX_BLOAT_MIN_APK_BYTES and (
        dex_uncompressed >= _DEX_BLOAT_MIN_UNCOMPRESSED_BYTES or ratio_bloated
    ):
        findings.append(
            {
                "severity": "medium",
                "code": "unminified_dex_bloat",
                "detail": (
                    f"APK size {apk_bytes} B with {dex_uncompressed} B uncompressed DEX "
                    f"(compressed ratio {dex_ratio})"
                ),
            }
        )
    sourcemaps = archive["sourcemap_files"]
    if sourcemaps:
        findings.append(
            {
                "severity": "low",
                "code": "bundled_sourcemaps",
                "detail": f"Bundled sourcemap files found: {', '.join(sourcemaps)}",
            }
        )
    return findings


def _read_quoted(text: str) -> str:
    chars: list[str] = []
    escaped = False
    for ch in text[1:]:
        if escaped:
            chars.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            break
        else:
            chars.append(ch)
    return "".join(chars)


def _extract_attr_value(rhs: str) -> str:
    val = _TYPE_PREFIX_RE.sub("", rhs.strip())
    if val.startswith('"'):
        return _read_quoted(val)
    parts = val.split()
    return parts[0] if parts else ""


def _parse_attr_name_info(lhs: str, android_prefixes: set[str]) -> tuple[str, bool] | None:
    clean = _RES_ID_SUFFIX_RE.sub("", lhs.strip())
    if clean.startswith(f"{_ANDROID_NS}:"):
        return clean[len(_ANDROID_NS) + 1 :], True
    if ":" in clean:
        prefix, name = clean.split(":", 1)
        return (name, True) if prefix in android_prefixes else None
    return clean, False


def _parse_attr_name(lhs: str, android_prefixes: set[str]) -> str | None:
    info = _parse_attr_name_info(lhs, android_prefixes)
    return info[0] if info is not None else None


def _parse_attr_line_info(body: str, android_prefixes: set[str]) -> tuple[str, str, bool] | None:
    if "=" not in body:
        return None
    lhs, rhs = body.split("=", 1)
    info = _parse_attr_name_info(lhs, android_prefixes)
    if info is None or not info[0]:
        return None
    return info[0], _extract_attr_value(rhs), info[1]


def _parse_attr_line(body: str, android_prefixes: set[str]) -> tuple[str, str] | None:
    info = _parse_attr_line_info(body, android_prefixes)
    return (info[0], info[1]) if info is not None else None


def _set_node_attr(node: _XmlNode, name: str, value: str, is_android: bool) -> None:
    if is_android:
        if name in node.android_keys:
            node.duplicate_android_keys.add(name)
        else:
            node.attrs[name] = value
            node.android_keys.add(name)
    elif name not in node.android_keys:
        node.attrs[name] = value


def _pop_to_indent(stack: list[_XmlNode], indent: int) -> None:
    while stack and stack[-1].indent >= indent:
        stack.pop()


def _push_element_node(
    body: str, indent: int, roots: list[_XmlNode], stack: list[_XmlNode]
) -> None:
    tag_parts = body.split()
    if not tag_parts:
        return
    _pop_to_indent(stack, indent)
    node = _XmlNode(tag=tag_parts[0], indent=indent)
    (stack[-1].children if stack else roots).append(node)
    stack.append(node)


def _handle_xmltree_line(
    line: str,
    roots: list[_XmlNode],
    stack: list[_XmlNode],
    android_prefixes: set[str],
) -> None:
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if stripped.startswith("N:"):
        m = _NS_DECL_RE.match(stripped[2:].strip())
        if m and m.group(2) == _ANDROID_NS:
            android_prefixes.add(m.group(1))
    elif stripped.startswith("E:"):
        _push_element_node(stripped[2:], indent, roots, stack)
    elif stripped.startswith("A:"):
        _pop_to_indent(stack, indent)
        if stack and (parsed := _parse_attr_line_info(stripped[2:].strip(), android_prefixes)):
            _set_node_attr(stack[-1], *parsed)


def _parse_xmltree_nodes(text: str) -> list[_XmlNode]:
    roots: list[_XmlNode] = []
    stack: list[_XmlNode] = []
    android_prefixes: set[str] = {"android"}
    # Split strictly on LF (stripping optional CR) so U+2028/U+2029/NEL inside an
    # attribute value cannot forge a new xmltree line and escape the current element.
    for raw_line in text.split("\n"):
        _handle_xmltree_line(raw_line.removesuffix("\r"), roots, stack, android_prefixes)
    return roots


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in {"true", "1", "-1"}:
        return True
    if val in {"false", "0"}:
        return False
    if val.startswith("0x"):
        try:
            return int(val, 16) != 0
        except ValueError:
            return None
    return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    val = raw.strip()
    try:
        return int(val, 16) if val.lower().startswith("0x") else int(val, 10)
    except ValueError:
        return None


def _parse_sdk_version(raw: str | None) -> int | str | None:
    if raw is None:
        return None
    parsed = _parse_int(raw)
    if parsed is not None:
        return parsed
    cleaned = raw.strip()
    return cleaned or None


def _find_first_child(node: _XmlNode | None, tag: str) -> _XmlNode | None:
    if node is None:
        return None
    for child in node.children:
        if child.tag == tag:
            return child
    return None


def _is_launcher_filter(node: _XmlNode) -> bool:
    actions = {c.attrs.get("name") for c in node.children if c.tag == "action"}
    categories = {c.attrs.get("name") for c in node.children if c.tag == "category"}
    return (
        "android.intent.action.MAIN" in actions
        and "android.intent.category.LAUNCHER" in categories
    )


def _has_uimode_config(raw_config: str | None) -> bool:
    if raw_config is None:
        return False
    mask = _parse_int(raw_config)
    if mask is not None:
        return (mask & _CONFIG_UI_MODE) != 0
    return "uimode" in {part.strip().lower() for part in raw_config.split("|")}


def _is_exported_or_launcher(activity: _XmlNode) -> bool:
    if _parse_bool(activity.attrs.get("exported")) is True:
        return True
    return any(_is_launcher_filter(sub) for sub in activity.children if sub.tag == "intent-filter")


def _collect_activities_missing_uimode(application: _XmlNode | None) -> list[str]:
    if application is None:
        return []
    return [
        child.attrs.get("name") or "<unnamed>"
        for child in application.children
        if child.tag == "activity"
        and _is_exported_or_launcher(child)
        and not _has_uimode_config(child.attrs.get("configChanges"))
    ]


def _iter_elements(nodes: Sequence[_XmlNode]) -> list[_XmlNode]:
    out: list[_XmlNode] = []
    stack = list(nodes)
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children)
    return out


def _autoverify_hosts_for_filter(filter_node: _XmlNode) -> set[str]:
    data_nodes = [c for c in filter_node.children if c.tag == "data"]
    schemes = {scheme.lower() for d in data_nodes if (scheme := d.attrs.get("scheme"))}
    has_https = "https" in schemes
    hosts: set[str] = set()
    for data in data_nodes:
        host = data.attrs.get("host")
        scheme = data.attrs.get("scheme")
        if host and (scheme.lower() == "https" if scheme is not None else has_https):
            hosts.add(host)
    return hosts


def _extract_autoverify_hosts(nodes: Sequence[_XmlNode]) -> list[str]:
    hosts: set[str] = set()
    for node in _iter_elements(nodes):
        if node.tag == "intent-filter" and _parse_bool(node.attrs.get("autoVerify")) is True:
            hosts.update(_autoverify_hosts_for_filter(node))
    return sorted(hosts)


def _collect_duplicate_android_attrs(nodes: Sequence[_XmlNode]) -> list[str]:
    return sorted(
        f"<{node.tag}> android:{attr}"
        for node in _iter_elements(nodes)
        for attr in node.duplicate_android_keys
    )


def _parsed_manifest_from_nodes(nodes: Sequence[_XmlNode]) -> dict[str, Any]:
    manifest = next((n for n in nodes if n.tag == "manifest"), None)
    uses_sdk = _find_first_child(manifest, "uses-sdk")
    application = _find_first_child(manifest, "application")

    m_attrs = manifest.attrs if manifest else {}
    s_attrs = uses_sdk.attrs if uses_sdk else {}
    a_attrs = application.attrs if application else {}

    return {
        "package": m_attrs.get("package"),
        "version_code": _parse_int(m_attrs.get("versionCode")),
        "version_name": m_attrs.get("versionName"),
        "min_sdk": _parse_sdk_version(s_attrs.get("minSdkVersion")),
        "target_sdk": _parse_sdk_version(s_attrs.get("targetSdkVersion")),
        "debuggable": _parse_bool(a_attrs.get("debuggable")) is True,
        "uses_cleartext_traffic": _parse_bool(a_attrs.get("usesCleartextTraffic")) is True,
        "allow_backup": _parse_bool(a_attrs.get("allowBackup")),
        "activities_missing_uimode_config": _collect_activities_missing_uimode(application),
        "autoverify_hosts": _extract_autoverify_hosts(nodes),
    }


def parse_aapt2_xmltree(text: str) -> dict[str, Any]:
    """Parse ``aapt2 dump xmltree <apk> --file AndroidManifest.xml`` output."""
    return _parsed_manifest_from_nodes(_parse_xmltree_nodes(text))


def _manifest_findings(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    if manifest.get("status") != "ok":
        return []
    findings: list[dict[str, str]] = []
    if manifest.get("debuggable") is True:
        findings.append(
            {
                "severity": "high",
                "code": "debuggable_enabled",
                "detail": "AndroidManifest enables android:debuggable=true",
            }
        )
    if manifest.get("uses_cleartext_traffic") is True:
        findings.append(
            {
                "severity": "high",
                "code": "cleartext_traffic_enabled",
                "detail": "AndroidManifest enables android:usesCleartextTraffic=true",
            }
        )
    if manifest.get("allow_backup") is True:
        findings.append(
            {
                "severity": "low",
                "code": "backup_enabled",
                "detail": "AndroidManifest explicitly sets android:allowBackup=true",
            }
        )
    missing_uimode = manifest.get("activities_missing_uimode_config")
    if missing_uimode:
        findings.append(
            {
                "severity": "info",
                "code": "activity_recreates_on_uimode",
                "detail": (
                    "Exported/launcher activities missing uiMode (0x200) in "
                    f"android:configChanges: {', '.join(missing_uimode)}"
                ),
            }
        )
    duplicates = manifest.get("duplicate_attributes")
    if duplicates:
        findings.append(
            {
                "severity": "high",
                "code": "duplicate_android_attribute",
                "detail": (
                    "Manifest element(s) carry duplicate android attributes: "
                    f"{', '.join(duplicates)}"
                ),
            }
        )
    return findings


def _unavailable_or_error_manifest(status: str, detail: str) -> dict[str, Any]:
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


def _build_aapt_cmd(aapt_bin: str, apk_path: Path) -> list[str]:
    resolved_apk = str(apk_path.resolve())
    if Path(aapt_bin).name == "aapt":
        return [aapt_bin, "dump", "xmltree", resolved_apk, "AndroidManifest.xml"]
    return [aapt_bin, "dump", "xmltree", resolved_apk, "--file", "AndroidManifest.xml"]


def _run_manifest_audit(apk_path: Path, aapt_bin: str | None, run_cmd: RunCmd) -> dict[str, Any]:
    if not aapt_bin:
        return _unavailable_or_error_manifest(
            "unavailable",
            "aapt2/aapt binary not found on PATH or in ANDROID_HOME/ANDROID_SDK_ROOT",
        )
    try:
        cp = run_cmd(_build_aapt_cmd(aapt_bin, apk_path))
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("aapt2_xmltree_exec_failed", error=str(exc))
        return _unavailable_or_error_manifest("error", str(exc))

    if cp.returncode != 0:
        detail = (
            cp.stderr or cp.stdout or ""
        ).strip() or f"aapt2 exited with code {cp.returncode}"
        return _unavailable_or_error_manifest("error", detail)

    nodes = _parse_xmltree_nodes(cp.stdout)
    if not any(n.tag == "manifest" for n in nodes):
        return _unavailable_or_error_manifest(
            "error",
            "aapt2 xmltree output did not contain a <manifest> element",
        )

    result: dict[str, Any] = {
        "status": "ok",
        "detail": None,
        **_parsed_manifest_from_nodes(nodes),
    }
    duplicates = _collect_duplicate_android_attrs(nodes)
    if duplicates:
        result["duplicate_attributes"] = duplicates
    return result


def normalize_sha256_fingerprint(value: object) -> str | None:
    """Normalize a 32-byte SHA-256 hex digest to uppercase colon-hex (``AA:BB:...``)."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not _VALID_FINGERPRINT_RE.fullmatch(stripped):
        return None
    raw_hex = stripped.replace(":", "").upper()
    return ":".join(raw_hex[i : i + 2] for i in range(0, 64, 2))


def parse_apksigner_certs(text: str) -> list[str]:
    """Extract normalized uppercase colon-hex SHA-256 digests from apksigner output."""
    certs: list[str] = []
    seen: set[str] = set()
    for match in _APKSIGNER_SHA256_RE.finditer(text):
        norm = normalize_sha256_fingerprint(match.group(1))
        if norm is not None and norm not in seen:
            seen.add(norm)
            certs.append(norm)
    return certs


def _run_signing_audit(
    apk_path: Path, apksigner_bin: str | None, run_cmd: RunCmd
) -> dict[str, Any]:
    if not apksigner_bin:
        return {
            "status": "unavailable",
            "detail": "apksigner binary not found on PATH or in ANDROID_HOME/ANDROID_SDK_ROOT",
            "cert_sha256s": [],
        }
    try:
        cp = run_cmd([apksigner_bin, "verify", "--print-certs", str(apk_path.resolve())])
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("apksigner_exec_failed", error=str(exc))
        return {"status": "error", "detail": str(exc), "cert_sha256s": []}

    if cp.returncode != 0:
        detail = (
            cp.stderr or cp.stdout or ""
        ).strip() or f"apksigner exited with code {cp.returncode}"
        return {"status": "error", "detail": detail, "cert_sha256s": []}

    certs = parse_apksigner_certs(cp.stdout)
    if not certs:
        return {
            "status": "error",
            "detail": "apksigner returned 0 without any SHA-256 certificate digest",
            "cert_sha256s": [],
        }
    return {"status": "ok", "detail": None, "cert_sha256s": certs}


def _load_assetlinks_statements(assetlinks_path: str | Path) -> list[Any]:
    path = _validate_regular_file(assetlinks_path, "assetlinks")
    try:
        raw_bytes = read_capped(path, _MAX_ASSETLINKS_BYTES)
        data = json.loads(raw_bytes.decode("utf-8"))
    except (
        RegistryTooLargeError,
        OSError,
        ValueError,
        RecursionError,
    ) as exc:
        raise ApkAuditError(f"Failed to load assetlinks JSON {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ApkAuditError("assetlinks.json must be a JSON list of statements")
    return data


def _android_app_target(stmt: object, package: str) -> dict[str, Any] | None:
    if not isinstance(stmt, dict):
        return None
    target = stmt.get("target")
    if not isinstance(target, dict) or target.get("namespace") != "android_app":
        return None
    return target if target.get("package_name") == package else None


def _has_handle_all_urls_relation(stmt: Mapping[str, Any]) -> bool:
    relation = stmt.get("relation")
    return isinstance(relation, list) and _HANDLE_ALL_URLS_RELATION in relation


def _extract_matching_package_fingerprints(
    statements: Sequence[Any], package: str
) -> tuple[bool, bool, list[str]]:
    package_found = False
    relation_found = False
    expected: list[str] = []
    seen: set[str] = set()

    for stmt in statements:
        target = _android_app_target(stmt, package)
        if target is None:
            continue
        package_found = True
        if not _has_handle_all_urls_relation(stmt):
            continue
        relation_found = True
        fps = target.get("sha256_cert_fingerprints")
        if not isinstance(fps, list):
            continue
        for fp in fps:
            norm = normalize_sha256_fingerprint(fp)
            if norm is not None and norm not in seen:
                seen.add(norm)
                expected.append(norm)

    return package_found, relation_found, expected


def _assetlinks_result(
    status: str,
    assetlinks_path: str | Path | None,
    package: str | None,
    apk_certs: list[str],
    *,
    expected: list[str] | None = None,
    matched: list[str] | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "assetlinks_path": str(assetlinks_path) if assetlinks_path is not None else None,
        "package": package,
        "apk_cert_sha256s": apk_certs,
        "expected_cert_sha256s": expected if expected is not None else [],
        "matched_cert_sha256s": matched if matched is not None else [],
        "detail": detail,
    }


def check_assetlinks(
    cert_sha256s: Sequence[str],
    package: str | None,
    assetlinks_path: str | Path,
) -> dict[str, Any]:
    """Compare APK signing SHA-256 fingerprints against a local ``assetlinks.json``."""
    statements = _load_assetlinks_statements(assetlinks_path)
    apk_certs = list(
        dict.fromkeys(
            norm for raw in cert_sha256s if (norm := normalize_sha256_fingerprint(raw)) is not None
        )
    )

    if not package:
        return _assetlinks_result(
            "unavailable",
            assetlinks_path,
            None,
            apk_certs,
            detail="APK package name is unavailable; cannot match assetlinks.json",
        )
    if not apk_certs:
        return _assetlinks_result(
            "unavailable",
            assetlinks_path,
            package,
            [],
            detail="APK signing certificates are unavailable; cannot match assetlinks.json",
        )

    package_found, relation_found, expected = _extract_matching_package_fingerprints(
        statements, package
    )
    if not package_found:
        return _assetlinks_result(
            "package_missing",
            assetlinks_path,
            package,
            apk_certs,
            detail=f"No android_app statement for package {package!r} in assetlinks.json",
        )
    if not relation_found:
        return _assetlinks_result(
            "relation_missing",
            assetlinks_path,
            package,
            apk_certs,
            detail=(
                f"No android_app statement for package {package!r} grants "
                f"{_HANDLE_ALL_URLS_RELATION!r}"
            ),
        )

    expected_set = set(expected)
    matched = [cert for cert in apk_certs if cert in expected_set]
    status = "matched" if matched else "mismatch"
    detail = (
        None
        if matched
        else (
            f"APK signing SHA-256 {apk_certs} does not match assetlinks.json "
            f"fingerprints {expected} for {package!r}"
        )
    )
    return _assetlinks_result(
        status,
        assetlinks_path,
        package,
        apk_certs,
        expected=expected,
        matched=matched,
        detail=detail,
    )


_ASSETLINKS_FINDING_SPECS: dict[Any, tuple[str, str]] = {
    "package_missing": (
        "assetlinks_package_missing",
        "Package missing from assetlinks.json",
    ),
    "relation_missing": (
        "assetlinks_relation_missing",
        "Missing delegate_permission/common.handle_all_urls relation in assetlinks.json",
    ),
    "mismatch": (
        "assetlinks_sha256_mismatch",
        "Signing SHA-256 mismatch in assetlinks.json",
    ),
    "unavailable": (
        "assetlinks_unavailable",
        "Cannot verify assetlinks without both package name and signing certificates",
    ),
}

_TOOL_RAN_STATUSES = frozenset({"ok", "error"})
_ASSETLINKS_MEASURED_STATUSES = frozenset(
    {"skipped", "matched", "mismatch", "package_missing", "relation_missing"}
)
_ASSETLINKS_PASSING_STATUSES = frozenset({"skipped", "matched"})


def _assetlinks_findings(assetlinks: Mapping[str, Any]) -> list[dict[str, str]]:
    spec = _ASSETLINKS_FINDING_SPECS.get(assetlinks.get("status"))
    if spec is None:
        return []
    code, fallback = spec
    return [
        {
            "severity": "high",
            "code": code,
            "detail": str(assetlinks.get("detail") or fallback),
        }
    ]


def summarize_findings(
    findings: Sequence[Mapping[str, str]],
    *,
    manifest_status: str,
    signing_status: str,
    assetlinks_status: str,
) -> dict[str, int | bool]:
    """Count findings by severity and compute ``measured`` and ``passed``.

    ``measured`` is False when manifest or signing is ``"unavailable"`` (or not in
    ``{"ok", "error"}``), or when assetlinks was requested and is anything other than
    a completed check. ``passed`` is True iff ``measured`` is True, there are no
    ``"high"`` findings, and all evaluated layers succeeded without error.
    """
    counts = {
        sev: sum(1 for finding in findings if finding.get("severity") == sev)
        for sev in _SEVERITIES
    }
    measured = (
        manifest_status in _TOOL_RAN_STATUSES
        and signing_status in _TOOL_RAN_STATUSES
        and assetlinks_status in _ASSETLINKS_MEASURED_STATUSES
    )
    passed = (
        measured
        and counts["high"] == 0
        and manifest_status == "ok"
        and signing_status == "ok"
        and assetlinks_status in _ASSETLINKS_PASSING_STATUSES
    )
    return {**counts, "measured": measured, "passed": passed}


def audit_apk(
    apk_path: str | Path,
    *,
    assetlinks_path: str | Path | None = None,
    aapt_bin: str | None = _UNSET,
    apksigner_bin: str | None = _UNSET,
    run_cmd: RunCmd | None = None,
) -> dict[str, Any]:
    """Run the full static APK audit and return a JSON-serialisable report."""
    path = _validate_regular_file(apk_path, "APK")
    archive = inspect_apk_archive(path)
    sha256 = _file_sha256(path)

    resolved_aapt = find_aapt() if aapt_bin is _UNSET else aapt_bin
    resolved_apksigner = find_apksigner() if apksigner_bin is _UNSET else apksigner_bin
    runner = run_cmd if run_cmd is not None else _default_run_cmd

    manifest = _run_manifest_audit(path, resolved_aapt, runner)
    signing = _run_signing_audit(path, resolved_apksigner, runner)

    findings = [*_archive_findings(archive), *_manifest_findings(manifest)]

    if assetlinks_path is None:
        assetlinks = _assetlinks_result(
            "skipped",
            None,
            manifest.get("package"),
            signing["cert_sha256s"],
        )
    else:
        _load_assetlinks_statements(assetlinks_path)
        if signing["status"] != "ok" or not manifest.get("package"):
            assetlinks = _assetlinks_result(
                "unavailable",
                assetlinks_path,
                manifest.get("package"),
                signing["cert_sha256s"],
                detail=(
                    "Cannot verify assetlinks without both package name and signing certificates"
                ),
            )
        else:
            assetlinks = check_assetlinks(
                signing["cert_sha256s"],
                manifest["package"],
                assetlinks_path,
            )
        findings.extend(_assetlinks_findings(assetlinks))

    summary = summarize_findings(
        findings,
        manifest_status=manifest["status"],
        signing_status=signing["status"],
        assetlinks_status=assetlinks["status"],
    )
    return {
        "apk_path": str(path),
        "sha256": sha256,
        "archive": archive,
        "manifest": manifest,
        "signing": signing,
        "assetlinks": assetlinks,
        "findings": findings,
        "summary": summary,
    }
