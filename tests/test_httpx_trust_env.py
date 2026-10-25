"""[contract] Every httpx client in src/ uses trust_env=False and no module-level calls exist.

Prevents HTTP_PROXY / HTTPS_PROXY / ALL_PROXY (or netrc / SSL env vars) from routing
loopback nerve (:8111) or local MobSF traffic (and its Authorization header) off-host
(linc-crucible-dq9, R4 Medium 4).
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

needs_src_tree = pytest.mark.skipif(
    not (REPO / "src").is_dir(), reason="no src/ tree beside tests/ (wheel-only test root)"
)

_MUTANT = re.compile(r"__mutmut_\d+$|^_mutmut_\w*trampoline$")

_CLIENT_CLASSES = frozenset({"Client", "AsyncClient"})
_MODULE_REQUEST_FUNCS = frozenset(
    {"get", "post", "put", "delete", "patch", "head", "options", "request", "stream"}
)


def _src_py_files(root: Path) -> list[Path]:
    """Return tracked ``src/**/*.py`` files via ``git ls-files``, falling back to ``rglob``."""
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", "src"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        files = [
            root / line
            for line in proc.stdout.splitlines()
            if line.endswith(".py") and (root / line).is_file()
        ]
        if files:
            return sorted(files)
    return sorted((root / "src").rglob("*.py"))


def _iter_non_mutant_nodes(tree: ast.AST) -> list[ast.AST]:
    """Return every AST node except mutmut's trampoline and ``*__mutmut_N`` copies."""
    out: list[ast.AST] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _MUTANT.search(node.name):
            continue
        out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _has_trust_env_false(call: ast.Call) -> bool:
    """True when ``call`` passes ``trust_env=False`` as an explicit keyword literal."""
    for kw in call.keywords:
        if (
            kw.arg == "trust_env"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is False
        ):
            return True
    return False


def scan_httpx_violations_in_source(source: str, rel: str) -> list[str]:
    """Scan one Python module's source for httpx client/request violations."""
    tree = ast.parse(source, filename=rel)
    nodes = _iter_non_mutant_nodes(tree)

    httpx_module_aliases: set[str] = set()
    imported_clients: dict[str, str] = {}
    imported_request_funcs: dict[str, str] = {}
    violations: list[str] = []

    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "httpx":
                    httpx_module_aliases.add(alias.asname or "httpx")
        elif isinstance(node, ast.ImportFrom) and node.module == "httpx":
            for alias in node.names:
                if alias.name == "*":
                    violations.append(f"{rel}:{node.lineno}: wildcard import from httpx")
                elif alias.name in _CLIENT_CLASSES:
                    imported_clients[alias.asname or alias.name] = alias.name
                elif alias.name in _MODULE_REQUEST_FUNCS:
                    imported_request_funcs[alias.asname or alias.name] = alias.name

    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        client_cls: str | None = None
        req_func: str | None = None

        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in httpx_module_aliases
        ):
            if func.attr in _CLIENT_CLASSES:
                client_cls = f"{func.value.id}.{func.attr}"
            elif func.attr in _MODULE_REQUEST_FUNCS:
                req_func = f"{func.value.id}.{func.attr}"
        elif isinstance(func, ast.Name):
            if func.id in imported_clients:
                client_cls = f"httpx.{imported_clients[func.id]}"
            elif func.id in imported_request_funcs:
                req_func = f"httpx.{imported_request_funcs[func.id]}"

        if client_cls is not None and not _has_trust_env_false(node):
            violations.append(
                f"{rel}:{node.lineno}: {client_cls}(...) without trust_env=False: "
                f"{ast.unparse(node)}"
            )
        elif req_func is not None:
            violations.append(
                f"{rel}:{node.lineno}: module-level {req_func}(...) call (use "
                f"httpx.Client(trust_env=False)): {ast.unparse(node)}"
            )

    return violations


def scan_httpx_violations(root: Path = REPO) -> list[str]:
    """Scan all Python files under ``root / 'src'`` for httpx trust_env violations."""
    violations: list[str] = []
    src_root = root / "src"
    for py_file in _src_py_files(root):
        rel = py_file.relative_to(src_root).as_posix()
        violations.extend(
            scan_httpx_violations_in_source(py_file.read_text(encoding="utf-8"), rel)
        )
    return violations


@needs_src_tree
def test_all_src_httpx_clients_disable_trust_env_and_no_module_level_calls() -> None:
    """[contract] Every httpx.Client/AsyncClient under src/ passes trust_env=False and
    no module-level httpx.get|post|put|request(...) call exists."""
    violations = scan_httpx_violations(REPO)
    assert violations == [], "httpx trust_env contract violations:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "import httpx\nc = httpx.Client(trust_env=False)\n",
        "import httpx\nc = httpx.Client(trust_env=False, timeout=30)\n",
        "import httpx\nc = httpx.AsyncClient(trust_env=False)\n",
        "from httpx import Client\nc = Client(trust_env=False)\n",
    ],
)
def test_scanner_accepts_explicit_trust_env_false(source: str) -> None:
    assert scan_httpx_violations_in_source(source, "pkg/ok.py") == []


@pytest.mark.parametrize(
    ("source", "expected_substring"),
    [
        ("import httpx\nc = httpx.Client()\n", "httpx.Client(...) without trust_env=False"),
        (
            "import httpx\nc = httpx.Client(trust_env=True)\n",
            "httpx.Client(...) without trust_env=False",
        ),
        (
            "import httpx\nc = httpx.AsyncClient(timeout=10)\n",
            "httpx.AsyncClient(...) without trust_env=False",
        ),
        (
            "from httpx import Client\nc = Client()\n",
            "httpx.Client(...) without trust_env=False",
        ),
        (
            'import httpx\nhttpx.post("http://localhost:8000/api/v1/upload")\n',
            "module-level httpx.post(...) call",
        ),
        (
            'import httpx\nhttpx.get("http://localhost:8111/health")\n',
            "module-level httpx.get(...) call",
        ),
        (
            'import httpx\nhttpx.put("http://localhost:8111/x")\n',
            "module-level httpx.put(...) call",
        ),
        (
            'import httpx\nhttpx.request("POST", "http://localhost:8111/x")\n',
            "module-level httpx.request(...) call",
        ),
        (
            'from httpx import post\npost("http://localhost:8000/api/v1/scan")\n',
            "module-level httpx.post(...) call",
        ),
    ],
)
def test_scanner_flags_missing_trust_env_and_module_level_calls(
    source: str, expected_substring: str
) -> None:
    violations = scan_httpx_violations_in_source(source, "pkg/bad.py")
    assert len(violations) == 1
    assert expected_substring in violations[0]
