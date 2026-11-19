"""Guard tests for uv.lock security floors and pyproject.toml Pillow minimum."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]

_MIN_PATCHED_LOCK_VERSIONS: dict[str, str] = {
    "anyio": "4.14.2",
    "certifi": "2026.10.18",
    "cryptography": "50.0.0",
    "filelock": "4.2.1",
    "idna": "3.15",
    "mcp": "2.3.0",
    "oauthlib": "4.0.0",
    "opentelemetry-api": "1.45.1",
    "pillow": "13.0.0",
    "pyasn1": "0.6.5",
    "pygments": "2.20.0",
    "pyjwt": "2.15.0",
    "pytest": "9.0.3",
    "python-multipart": "0.0.31",
    "starlette": "1.3.1",
    "urllib3": "2.8.0",
    "virtualenv": "21.14.9",
}
# pydantic-settings and python-dotenv left the lock with mcp 2.x (mcp no longer depends on
# them), so their floors were dropped with the 2.3 migration rather than kept as "missing".


def test_uv_lock_patched_releases() -> None:
    """Every package in _MIN_PATCHED_LOCK_VERSIONS is locked >= its floor (and pytest < 9.2)."""
    lock_path = REPO / "uv.lock"
    if not lock_path.is_file():
        pytest.skip("uv.lock not present")

    with lock_path.open("rb") as f:
        lock_data = tomllib.load(f)

    versions_by_pkg: dict[str, list[str]] = {}
    for pkg in lock_data.get("package", []):
        name = pkg.get("name", "").lower()
        ver = pkg.get("version")
        if name and ver:
            versions_by_pkg.setdefault(name, []).append(ver)

    unpatched: list[str] = []
    for pkg_name, min_ver_str in sorted(_MIN_PATCHED_LOCK_VERSIONS.items()):
        locked_versions = versions_by_pkg.get(pkg_name, [])
        if not locked_versions:
            unpatched.append(f"{pkg_name}: missing from uv.lock (need >={min_ver_str})")
            continue
        min_ver = Version(min_ver_str)
        for ver_str in locked_versions:
            if Version(ver_str) < min_ver:
                unpatched.append(f"{pkg_name}=={ver_str} < {min_ver_str}")

    assert not unpatched, f"Vulnerable package pins in uv.lock: {unpatched}"

    for ver_str in versions_by_pkg.get("pytest", []):
        assert Version(ver_str) < Version("9.2"), (
            f"pytest=={ver_str} must stay < 9.2 (boxed/mutmut#615)"
        )


def test_pyproject_pillow_minimum() -> None:
    """pyproject.toml declares Pillow minimum >= 13.0.0."""
    pyproject_path = REPO / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    deps = pyproject_data.get("project", {}).get("dependencies", [])
    pillow_reqs = [Requirement(d) for d in deps if Requirement(d).name.lower() == "pillow"]
    assert pillow_reqs, f"Pillow missing from [project].dependencies: {deps}"

    lower_bounds = [
        Version(spec.version)
        for req in pillow_reqs
        for spec in req.specifier
        if spec.operator in (">=", "==")
    ]
    assert lower_bounds, f"Pillow dependency has no >= or == floor: {pillow_reqs[0]}"
    min_floor = min(lower_bounds)
    assert min_floor >= Version("13.0.0"), (
        f"pyproject.toml Pillow minimum {min_floor} < 13.0.0 ({pillow_reqs[0]})"
    )


def test_public_docs_has_no_url_substring_check() -> None:
    """test_public_docs.py contains no bare URL `in` substring expression."""
    tree = ast.parse((REPO / "tests" / "test_public_docs.py").read_text(encoding="utf-8"))
    bad = [
        node.left.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
        and node.left.value.startswith(("http://", "https://"))
    ]
    assert not bad, f"bare URL substring checks in test_public_docs.py: {bad}"


def test_pyproject_declares_mcp_2_3_line() -> None:
    """pyproject declares exactly mcp>=2.3.0,<3 (the 2.x migration, LINC plan W2c)."""
    with (REPO / "pyproject.toml").open("rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    mcp_reqs = [Requirement(d) for d in deps if Requirement(d).name.lower() == "mcp"]
    assert len(mcp_reqs) == 1, deps
    assert str(mcp_reqs[0].specifier) == str(Requirement("mcp>=2.3.0,<3").specifier)


def test_uv_lock_resolves_mcp_2_3_line() -> None:
    """uv.lock resolves mcp inside >=2.3.0,<3."""
    lock_path = REPO / "uv.lock"
    if not lock_path.is_file():
        pytest.skip("uv.lock not present")  # e.g. mutmut's mutants/ copy
    with lock_path.open("rb") as f:
        locked = [p["version"] for p in tomllib.load(f)["package"] if p.get("name") == "mcp"]
    assert locked, "mcp missing from uv.lock"
    for ver in locked:
        assert Version("2.3.0") <= Version(ver) < Version("3"), f"mcp=={ver} outside >=2.3.0,<3"
