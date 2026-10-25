"""Store-name allowlist and containment proofs for crucible's on-disk stores.

Every store (screenshot baselines, a11y baselines, ``.compare`` artifacts,
quality scores) builds paths from caller text — an MCP argument, a CLI
argument, a BVT screenshot stem. This module is the single gate between that
text and the filesystem:

* :func:`validate_name` — a ``[A-Za-z0-9._-]`` allowlist (first char
  alphanumeric, at most 128 chars, no ``..``), so a name can never be an
  absolute path, a traversal, a hidden dir or a CLI-flag look-alike.
* :func:`contained_dir` — validates each name and proves, on the RESOLVED
  path, that the joined directory stays inside the resolved store root (a
  symlinked ``<store>/<project>`` pointing elsewhere is refused).
* :func:`prepare_dir` — :func:`contained_dir` + ``mkdir`` + a second proof
  after the mkdir, so a symlink raced in between is refused.
* :func:`remove_store_dir` — the only ``rmtree`` a prune performs, refusing
  symlinked dirs and anything that resolves outside its parent.

Failure mode (input-boundary zone): raise :class:`InvalidStoreName` (a
``ValueError``); the MCP and CLI layers turn it into a validation error.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from .state_paths import ensure_private_dir

logger = structlog.get_logger().bind(domain="build_pipeline")

MAX_NAME_LEN = 128
NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ECHO_LIMIT = 40
_FIELDS = ("project", "screen_name")

# Ordered (predicate, rule) table — the first matching predicate names the rule.
_RULES: tuple[tuple[Callable[[str], bool], str], ...] = (
    (lambda v: not v, "must not be empty"),
    (lambda v: len(v) > MAX_NAME_LEN, f"must be at most {MAX_NAME_LEN} characters"),
    (lambda v: "\x00" in v, "must not contain a NUL byte"),
    (lambda v: "/" in v or "\\" in v, "must not contain a path separator"),
    (lambda v: ".." in v, "must not contain '..'"),
    (lambda v: any(c.isspace() for c in v), "must not contain whitespace"),
    (lambda v: v[0] in ".-", "must not start with '.' or '-'"),
    (
        lambda v: NAME_RE.fullmatch(v) is None,
        "may only contain letters, digits, '.', '_' and '-'",
    ),
)


def _preview(value: object) -> str:
    """Bounded, escaped echo of a rejected value — never more than 40 of its chars."""
    if not isinstance(value, str):
        return type(value).__name__
    suffix = "..." if len(value) > _ECHO_LIMIT else ""
    return repr(value[:_ECHO_LIMIT]) + suffix


class InvalidStoreName(ValueError):
    """A store name failed the allowlist or resolved outside its store.

    Attributes:
        field: The argument that was rejected (``project``, ``screen_name``, ``name``).
        rule: The human-readable rule it broke.
    """

    def __init__(self, field: str, rule: str, value: object) -> None:
        self.field = field
        self.rule = rule
        super().__init__(f"invalid {field}: {rule} (got {_preview(value)})")


def validate_name(value: object, field: str) -> str:
    """Return ``value`` if it is a safe store name, else raise.

    Args:
        value: Caller-supplied name (project, screen, timestamp).
        field: Argument name reported in the error.

    Returns:
        The unchanged name.

    Raises:
        InvalidStoreName: ``value`` is not a ``str`` or breaks an allowlist rule.
    """
    if not isinstance(value, str):
        raise InvalidStoreName(field, "must be a string", value)
    for broken, rule in _RULES:
        if broken(value):
            raise InvalidStoreName(field, rule, value)
    return value


def _field_for(index: int) -> str:
    """Field label for the ``index``-th name of a store path (project, screen, then name)."""
    return _FIELDS[index] if index < len(_FIELDS) else "name"


def contained_dir(root: Path, *names: str) -> Path:
    """Join validated ``names`` under ``root``, proven to stay inside it.

    Each prefix is resolved (``strict=False``) and must be relative to the
    resolved root, so a symlinked component pointing outside is refused and
    blamed on the right field. Creates nothing.

    Args:
        root: Store root.
        *names: Path components in store order (project, screen, timestamp).

    Returns:
        ``root / names...`` (unresolved, so stored path strings keep their shape).

    Raises:
        InvalidStoreName: A name fails :func:`validate_name` or a prefix resolves
            outside the store.
    """
    resolved_root = Path(root).resolve()
    for index, name in enumerate(names):
        field = _field_for(index)
        validate_name(name, field)
        candidate = resolved_root.joinpath(*names[: index + 1]).resolve(strict=False)
        if not candidate.is_relative_to(resolved_root):
            raise InvalidStoreName(field, "resolves outside the store (symlink?)", name)
    return Path(root).joinpath(*names)


def prepare_dir(root: Path, *names: str) -> Path:
    """Create a contained store directory and re-prove containment after the mkdir.

    Raises:
        InvalidStoreName: As :func:`contained_dir`, before or after the mkdir.
    """
    path = contained_dir(root, *names)
    ensure_private_dir(path)
    contained_dir(root, *names)
    return path


def scope_dirs(root: Path, project: str, screen_name: str = "") -> tuple[Path, Path | None]:
    """Contained ``(project_dir, screen_dir)`` for a list/prune scope; validates both up front.

    ``screen_name=""`` means "every screen" and yields ``screen_dir=None``. A
    bad screen name raises even when the project dir does not exist yet, so a
    read never answers a hostile name with a silent empty result.

    Raises:
        InvalidStoreName: As :func:`contained_dir`.
    """
    project_dir = contained_dir(root, project)
    if screen_name == "":
        return project_dir, None
    return project_dir, contained_dir(root, project, screen_name)


def name_validation_error(project: object, screen_name: object) -> dict[str, Any] | None:
    """MCP-layer precheck: the validation envelope for a bad name, else ``None``.

    An empty ``screen_name`` means "all screens" (list/prune) and is not
    validated; a non-empty one is.
    """
    try:
        validate_name(project, "project")
        if screen_name != "":
            validate_name(screen_name, "screen_name")
    except InvalidStoreName as exc:
        return validation_error(exc)
    return None


def validation_error(exc: ValueError) -> dict[str, Any]:
    """The cli-kit validation envelope (maps to exit 2) for a refused store name or tool arg."""
    return {"error": str(exc), "error_category": "validation", "validation_error": True}


def required_project_error(project: object) -> dict[str, Any] | None:
    """MCP precheck for a project-only store (scores): required, then allowlisted."""
    if project == "":
        return {"error": "project is required."}
    try:
        validate_name(project, "project")
    except InvalidStoreName as exc:
        return validation_error(exc)
    return None


def required_names_error(project: object, screen_name: object) -> dict[str, Any] | None:
    """MCP precheck: :func:`required_project_error`, then :func:`name_validation_error`."""
    return required_project_error(project) or name_validation_error(project, screen_name)


def name_guarded(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run a store-backed tool body, mapping a store-layer refusal to the validation envelope.

    The MCP precheck catches bad names up front; this catches what only the
    store can see (e.g. a symlinked ``<store>/<project>`` resolving outside).
    """
    try:
        return fn()
    except InvalidStoreName as exc:
        return validation_error(exc)


def store_error(exc: OSError | ValueError) -> dict[str, Any]:
    """Error dict for a capture/promote failure; a refused store name is a validation error."""
    if isinstance(exc, InvalidStoreName):
        return validation_error(exc)
    return {"error": str(exc)}


def _prune_refusal(path: Path, parent: Path) -> str | None:
    """Why ``path`` must not be rmtree'd, or ``None`` when it is safe."""
    if path.is_symlink() or path.parent.is_symlink():
        return "symlink"
    if not path.resolve().is_relative_to(parent.resolve()):
        return "outside"
    return None


def remove_store_dir(path: Path, parent: Path, *, event_prefix: str) -> int:
    """``rmtree`` one pruned store dir iff it is a real dir proven inside ``parent``.

    A symlinked dir (or one reached through a symlinked parent) is skipped with
    a ``<event_prefix>_prune_skipped_symlink`` warning; one resolving outside
    ``parent`` with ``<event_prefix>_prune_skipped_outside``.

    Returns:
        1 when removed, 0 when skipped.
    """
    refusal = _prune_refusal(path, parent)
    if refusal is not None:
        logger.warning(f"{event_prefix}_prune_skipped_{refusal}", path=str(path))
        return 0
    shutil.rmtree(path)
    logger.info(f"{event_prefix}_pruned", path=str(path))
    return 1
