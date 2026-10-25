"""Single source of truth for crucible on-disk state locations.

Canonical root: ``$LINC_HOME/crucible`` when ``LINC_HOME`` is set, otherwise
``~/.linc/crucible``. Everything crucible persists between runs — the fleet
registry, visual/a11y/perf baselines, quality scores, extracted assets —
resolves through :func:`state_path`, so no module can drift onto a separate
legacy root (such as ``~/.crucible`` or ``~/.linc/baselines``).

Migration shim: :func:`state_path` returns the canonical location, falling
back to a legacy location only when the canonical one does not exist yet and
the legacy one does — pre-migration state stays readable in place, while new
state is always created under the canonical root.

All helpers re-evaluate the environment on every call so ``os.environ``
monkeypatching in tests takes effect.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from pathlib import Path

import structlog

logger = structlog.get_logger().bind(domain="build_pipeline")

_PRIVATE_DIR_MODE = stat.S_IRWXU


def ensure_private_dir(path: Path) -> Path:
    """Create *path* and any missing parents with mode ``0700``.

    Pre-existing directories (such as a user-managed ``LINC_HOME``) keep their
    current permissions; only directories created by this call are tightened
    with ``os.chmod(..., 0o700)`` so the process umask cannot leave them wider.
    """
    missing = [d for d in (path, *path.parents) if not d.exists()]
    for directory in reversed(missing):
        directory.mkdir()
        os.chmod(directory, _PRIVATE_DIR_MODE)
    return path


def linc_home() -> Path:
    """Return ``$LINC_HOME`` when set, otherwise ``~/.linc``."""
    return Path(os.getenv("LINC_HOME", str(Path.home() / ".linc")))


def crucible_root() -> Path:
    """Return the canonical crucible state root: ``<linc_home>/crucible``."""
    return linc_home() / "crucible"


def legacy_crucible_root() -> Path:
    """Return the legacy root: ``$LINC_HOME/.crucible`` when set, else ``~/.crucible``."""
    home = os.getenv("LINC_HOME")
    if home:
        return Path(home) / ".crucible"
    return Path.home() / ".crucible"


def state_path(*segments: str, extra_legacy_roots: Sequence[Path] = ()) -> Path:
    """Resolve a state location under the canonical root, with legacy read-fallback.

    Args:
        segments: Path segments below the state root (e.g. ``"baselines"``,
            ``"fleet.yaml"``).
        extra_legacy_roots: Additional pre-migration roots to consult after
            ``legacy_crucible_root()`` (e.g. perf baselines' old
            ``<linc_home>/baselines`` home, passed as ``linc_home()`` with
            segment ``"baselines"``).

    Returns:
        The canonical path when it exists or no legacy candidate does;
        otherwise the first existing legacy candidate. The exists-check is on
        the resolved path itself, so a canonical store created later takes
        precedence from that point on.
    """
    canonical = crucible_root().joinpath(*segments)
    if canonical.exists():
        return canonical
    for root in (legacy_crucible_root(), *extra_legacy_roots):
        candidate = root.joinpath(*segments)
        if candidate.exists():
            logger.info(
                "crucible_legacy_state_fallback",
                canonical=str(canonical),
                legacy=str(candidate),
            )
            return candidate
    return canonical
