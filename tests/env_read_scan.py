"""AST walk resolving every environment-variable read under ``src/`` to literal names.

Backs the config-contract pin in ``test_config_contract.py`` (REVIEW B4b). Every
``os.environ.get`` / ``os.getenv`` / ``environ.get`` / ``getenv`` call, ``map(<getter>, names)``
and ``environ[...]`` subscript is resolved through literal and module-level string (tuple)
constants. Any other reference to ``os.environ`` / ``os.getenv`` / ``environ`` / ``getenv`` (an
alias, an aliased or star import, a non-call use, a computed name) raises
:class:`UnanalysableEnvReadError`, so a read the walk cannot analyse is red, never invisible.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


class UnanalysableEnvReadError(ValueError):
    """An env access the scan cannot resolve to literal names: the pin must go red."""


_MUTANT = re.compile(r"__mutmut_\d+$|^_mutmut_\w*trampoline$")

# Names that reach the process environment. ``environ``/``getenv`` are analysed; the bytes
# variants are never analysed, so any use of them is red.
_ENV_REFS = frozenset({"environ", "getenv", "environb", "getenvb"})

# Whole-environment copies that forward ``os.environ`` verbatim to a child process. They read
# no named variable; any other non-call use of ``os.environ`` is unanalysable and red.
_PASS_THROUGH = frozenset(
    {
        ("linc_crucible/core/deployer.py", "dict(os.environ)"),
        ("linc_crucible/core/subprocess_utils.py", "{**os.environ, **env}"),
    }
)


def _nodes(tree: ast.AST) -> list[ast.AST]:
    """Every node except mutmut's trampoline and ``*__mutmut_N`` copies (injected reads)."""
    out: list[ast.AST] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _MUTANT.search(node.name):
            continue
        out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _module_strings(tree: ast.Module) -> dict[str, str | tuple[str, ...]]:
    """Module-level ``NAME = "X"`` / ``NAME = ("X", "Y")`` constants (annotated or not)."""
    found: dict[str, str | tuple[str, ...]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found[target.id] = value.value
        elif isinstance(value, ast.Tuple | ast.List):
            items = [e.value for e in value.elts if isinstance(e, ast.Constant)]
            if len(items) == len(value.elts) and all(isinstance(i, str) for i in items):
                found[target.id] = tuple(items)
    return found


class _EnvScan:
    """Resolve every env access in one module to literal names, or raise."""

    def __init__(self, source: str, rel: str) -> None:
        self.rel = rel
        self.tree = ast.parse(source)
        self.consts = _module_strings(self.tree)
        self.nodes = _nodes(self.tree)
        self.parents = {c: n for n in self.nodes for c in ast.iter_child_nodes(n)}

    def fail(self, node: ast.AST, why: str) -> UnanalysableEnvReadError:
        lineno = getattr(node, "lineno", 0)
        return UnanalysableEnvReadError(f"{self.rel}:{lineno}: {why}: {ast.unparse(node)}")

    def names(self) -> set[str]:
        found: set[str] = set()
        for node in self.nodes:
            self.check_import(node)
            kind = _env_ref(node)
            if kind == "environ":
                found |= self.environ_use(node)
            elif kind == "getenv":
                found |= self.getter_use(node)
            elif kind is not None:
                raise self.fail(node, f"unanalysable {kind} reference")
        return found

    def check_import(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os" and alias.asname not in (None, "os"):
                    raise self.fail(node, "os imported under an alias")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name == "*" or (
                    alias.name in _ENV_REFS and alias.asname not in (None, alias.name)
                ):
                    raise self.fail(node, "env accessor imported under an alias or via *")

    def environ_use(self, ref: ast.AST) -> set[str]:
        parent = self.parents.get(ref)
        if isinstance(parent, ast.Subscript) and parent.value is ref:
            return {self.one(parent.slice)}
        if isinstance(parent, ast.Attribute) and parent.attr == "get":
            return self.getter_use(parent)
        if parent is not None and (self.rel, ast.unparse(parent)) in _PASS_THROUGH:
            return set()
        raise self.fail(parent or ref, "unanalysable environ use")

    def getter_use(self, getter: ast.AST) -> set[str]:
        call = self.parents.get(getter)
        if isinstance(call, ast.Call) and call.func is getter and call.args:
            return {self.one(call.args[0])}
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "map"
            and len(call.args) == 2
            and call.args[0] is getter
        ):
            return set(self.many(call.args[1]))
        raise self.fail(call or getter, "unanalysable getter use")

    def one(self, expr: ast.expr) -> str:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        value = self.consts.get(expr.id) if isinstance(expr, ast.Name) else None
        if isinstance(value, str):
            return value
        raise self.fail(expr, "env-var name is not a literal or module-level constant")

    def many(self, expr: ast.expr) -> tuple[str, ...]:
        if isinstance(expr, ast.Tuple | ast.List):
            return tuple(self.one(e) for e in expr.elts)
        value = self.consts.get(expr.id) if isinstance(expr, ast.Name) else None
        if isinstance(value, tuple):
            return value
        raise self.fail(expr, "env-var names are not a literal or module-level tuple")


def _env_ref(node: ast.AST) -> str | None:
    """``environ``/``getenv``/... when ``node`` is ``os.<that>`` or the bare name, else None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr in _ENV_REFS
    ):
        return node.attr
    if isinstance(node, ast.Name) and node.id in _ENV_REFS:
        return node.id
    return None


def names_read_under(root: Path) -> set[str]:
    """Env-var names read by every ``*.py`` under ``root / "src"`` (no git needed).

    Raises:
        UnanalysableEnvReadError: an env access the walk cannot resolve to literal names.
    """
    names: set[str] = set()
    src = root / "src"
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src).as_posix()
        names |= _EnvScan(path.read_text(encoding="utf-8"), rel).names()
    return names
