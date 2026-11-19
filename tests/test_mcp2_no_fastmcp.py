"""[contract] No mcp 1.x server API survives the mcp 2.3 migration (LINC plan W2c).

mcp 2.x removed ``mcp.server.fastmcp`` (importing it raises ModuleNotFoundError) and turned
``JSONRPCMessage`` into a plain Union with no ``model_validate`` / ``model_validate_json``.
Walk every module's AST so the check also covers code held in string sources (the
``_STDIO_DRIVER`` child in tests/test_stdio.py) by parsing those string constants too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import linc_crucible

REPO = Path(__file__).resolve().parents[1]
_REMOVED_MODULE = "mcp.server.fastmcp"
_REMOVED_METHODS = {"model_validate", "model_validate_json"}


def _pkg_root() -> Path:
    return Path(linc_crucible.__file__).resolve().parent


def _python_files() -> list[Path]:
    # This file names the removed forms as scanner samples, so it is the one exclusion.
    me = Path(__file__).resolve()
    pkg_files = sorted(_pkg_root().rglob("*.py"))
    assert pkg_files, f"no Python files found under {_pkg_root()}"
    test_files = sorted(p for p in (REPO / "tests").rglob("*.py") if p.resolve() != me)
    assert test_files, f"no Python files found under {REPO / 'tests'}"
    return sorted({*pkg_files, *test_files})


def _violations(tree: ast.AST, where: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == _REMOVED_MODULE or node.module.startswith(_REMOVED_MODULE + "."):
                found.append(f"{where}:{node.lineno} from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _REMOVED_MODULE or alias.name.startswith(_REMOVED_MODULE + "."):
                    found.append(f"{where}:{node.lineno} import {alias.name}")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in _REMOVED_METHODS
            and isinstance(node.value, ast.Name)
            and node.value.id == "JSONRPCMessage"
        ):
            found.append(f"{where}:{node.lineno} JSONRPCMessage.{node.attr}")
    return found


def _embedded_sources(tree: ast.AST) -> list[tuple[int, ast.AST]]:
    """Parse string constants that are themselves Python programs (subprocess drivers)."""
    out: list[tuple[int, ast.AST]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "import" in node.value
        ):
            try:
                out.append((node.lineno, ast.parse(node.value)))
            except SyntaxError:
                continue
    return out


def test_contract_no_fastmcp_import_or_jsonrpcmessage_model_validate() -> None:
    """[contract] shipped package and tests/ hold no mcp.server.fastmcp import and no
    JSONRPCMessage.model_validate*."""
    pkg_parent = _pkg_root().parent
    violations: list[str] = []
    for path in _python_files():
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path.relative_to(pkg_parent)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        violations += _violations(tree, str(rel))
        for lineno, sub in _embedded_sources(tree):
            violations += _violations(sub, f"{rel}:{lineno}<embedded>")
    assert not violations, "mcp 1.x API still referenced:\n" + "\n".join(violations)


def test_contract_scanner_flags_each_removed_form() -> None:
    """[contract] the scanner itself catches every form it guards (no vacuous pass)."""
    samples = {
        "from mcp.server.fastmcp import FastMCP": 1,
        "from mcp.server.fastmcp.exceptions import ToolError": 1,
        "import mcp.server.fastmcp": 1,
        "JSONRPCMessage.model_validate_json(b'{}')": 1,
        "JSONRPCMessage.model_validate({})": 1,
        "from mcp.server.mcpserver import MCPServer": 0,
        "jsonrpc_message_adapter.validate_json(b'{}')": 0,
    }
    for src, expected in samples.items():
        assert len(_violations(ast.parse(src), "<sample>")) == expected, src
    driver = (
        'DRIVER = """\nimport sys\nfrom mcp.types import JSONRPCMessage\n'
        'JSONRPCMessage.model_validate({})\n"""'
    )
    embedded = _embedded_sources(ast.parse(driver))
    assert sum(len(_violations(t, "<e>")) for _, t in embedded) == 1
