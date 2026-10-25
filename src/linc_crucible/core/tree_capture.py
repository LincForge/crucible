"""A11y tree baseline storage — versioned accessibility tree capture and retrieval.

Baselines stored at ``~/.linc/crucible/baselines/<project>/<screen_name>/<timestamp>/``
(shared store with visual baselines; legacy ``~/.crucible/baselines`` read if
only it exists — see :mod:`linc_crucible.core.state_paths`).
Each a11y baseline directory contains ``a11y_tree.json`` and ``metadata.json``.
Directories containing only ``screenshot.png`` are visual baselines and are ignored.
``project`` and ``screen_name`` must pass the
:mod:`~linc_crucible.core.store_names` allowlist; every read, write and prune
is proven to stay inside the store (``InvalidStoreName`` otherwise).

Capture, retrieve, list, and prune a11y baselines per project and screen.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog
from linc_device_kit.models import ElementNode

from .state_paths import state_path
from .store_names import (
    contained_dir,
    prepare_dir,
    remove_store_dir,
    scope_dirs,
)

logger = structlog.get_logger().bind(domain="build_pipeline")


def _default_baselines_dir() -> Path:
    """Return the default baselines directory, honouring ``LINC_HOME``.

    Returns:
        ``<state root>/baselines`` via :func:`state_paths.state_path` —
        canonical ``$LINC_HOME/crucible/baselines`` (default
        ``~/.linc/crucible/baselines``, the same store visual baselines use),
        legacy ``~/.crucible/baselines`` when only it exists. Re-evaluated on
        every call so that ``os.environ`` monkeypatching in tests takes effect.
    """
    return state_path("baselines")


@dataclass
class A11yBaselineEntry:
    """A single a11y tree baseline capture."""

    project: str
    screen_name: str
    capture_time: str  # Compact ISO 8601 e.g. "20260413T120000Z"
    tree_path: str  # Absolute path to stored a11y_tree.json
    device_info: str
    app_version: str
    element_count: int

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "mode": "a11y",
            "project": self.project,
            "screen_name": self.screen_name,
            "capture_time": self.capture_time,
            "tree_path": self.tree_path,
            "device_info": self.device_info,
            "app_version": self.app_version,
            "element_count": self.element_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> A11yBaselineEntry:
        """Deserialize from a dictionary."""
        return cls(
            project=data.get("project", ""),
            screen_name=data.get("screen_name", ""),
            capture_time=data.get("capture_time", ""),
            tree_path=data.get("tree_path", ""),
            device_info=data.get("device_info", ""),
            app_version=data.get("app_version", ""),
            element_count=data.get("element_count", 0),
        )


class TreeCapture:
    """Manages a11y tree baselines with versioned storage and pruning."""

    def __init__(self, baselines_dir: Path | None = None) -> None:
        env_path = os.environ.get("CRUCIBLE_BASELINES_DIR")
        if baselines_dir is not None:
            self._baselines_dir = baselines_dir
        elif env_path:
            self._baselines_dir = Path(env_path)
        else:
            self._baselines_dir = _default_baselines_dir()

    @property
    def baselines_dir(self) -> Path:
        """Resolved store root (for honest-inventory reporting)."""
        return self._baselines_dir

    def store(
        self,
        project: str,
        screen_name: str,
        elements: list[ElementNode],
        device_info: str = "",
        app_version: str = "",
    ) -> A11yBaselineEntry:
        """Store an a11y tree baseline for a project screen.

        Args:
            project: Project name.
            screen_name: Screen identifier (e.g. "home", "settings").
            elements: List of ElementNode objects to serialize.
            device_info: Device description string.
            app_version: App version string.

        Returns:
            A11yBaselineEntry with metadata and stored tree path.

        Raises:
            InvalidStoreName: If project/screen_name is not a safe store name
                (checked before anything is written).
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        baseline_dir = prepare_dir(self._baselines_dir, project, screen_name, timestamp)

        tree_data = [el.to_dict() for el in elements]
        tree_path = baseline_dir / "a11y_tree.json"
        with open(tree_path, "w") as f:
            json.dump(tree_data, f, indent=2)

        entry = A11yBaselineEntry(
            project=project,
            screen_name=screen_name,
            capture_time=timestamp,
            tree_path=str(tree_path),
            device_info=device_info,
            app_version=app_version,
            element_count=len(elements),
        )

        metadata_path = baseline_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        logger.info(
            "a11y_tree_captured",
            project=project,
            screen=screen_name,
            element_count=len(elements),
            path=str(baseline_dir),
        )
        return entry

    def get_latest(self, project: str, screen_name: str) -> A11yBaselineEntry | None:
        """Get the most recent a11y baseline for a project screen.

        Args:
            project: Project name.
            screen_name: Screen identifier.

        Returns:
            The newest A11yBaselineEntry, or None if no baselines exist.

        Raises:
            InvalidStoreName: If project/screen_name is not a safe store name.
        """
        screen_dir = contained_dir(self._baselines_dir, project, screen_name)
        if not screen_dir.is_dir():
            return None

        timestamp_dirs = sorted(
            (d for d in screen_dir.iterdir() if d.is_dir() and (d / "a11y_tree.json").exists()),
            key=lambda d: d.name,
            reverse=True,
        )
        if not timestamp_dirs:
            return None

        return self._load_entry(timestamp_dirs[0])

    def load_elements(self, entry: A11yBaselineEntry) -> list[ElementNode]:
        """Load and deserialize ElementNode list from an a11y baseline entry.

        Args:
            entry: A11yBaselineEntry pointing to a stored a11y_tree.json.

        Returns:
            List of ElementNode objects.
        """
        with open(entry.tree_path) as f:
            data = json.load(f)
        return [ElementNode.from_dict(d) for d in data]

    def list_entries(self, project: str, screen_name: str = "") -> list[A11yBaselineEntry]:
        """List a11y baselines for a project, optionally filtered by screen.

        Args:
            project: Project name.
            screen_name: If provided, list only this screen's baselines.

        Returns:
            List of A11yBaselineEntry sorted newest-first.

        Raises:
            InvalidStoreName: If project/screen_name is not a safe store name.
        """
        project_dir, screen_dir = scope_dirs(self._baselines_dir, project, screen_name)
        if not project_dir.is_dir():
            return []

        if screen_dir is not None:
            screens = [screen_dir]
        else:
            screens = [d for d in project_dir.iterdir() if d.is_dir()]

        entries: list[A11yBaselineEntry] = []
        for screen_dir in screens:
            if not screen_dir.is_dir():
                continue
            for ts_dir in screen_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                # Only process dirs that contain a11y_tree.json (not screenshot baselines)
                if not (ts_dir / "a11y_tree.json").exists():
                    continue
                entry = self._load_entry(ts_dir)
                if entry is not None:
                    entries.append(entry)

        entries.sort(key=lambda e: e.capture_time, reverse=True)
        return entries

    def prune(self, project: str, screen_name: str = "", keep: int = 5) -> int:
        """Remove old a11y baselines beyond the retention limit.

        Args:
            project: Project name.
            screen_name: If provided, prune only this screen. Otherwise prune all screens.
            keep: Number of newest baselines to keep per screen.

        Returns:
            Number of baselines pruned. A symlinked timestamp dir (or one under
            a symlinked screen dir) is skipped with a warning, never followed.

        Raises:
            InvalidStoreName: If project/screen_name is not a safe store name
                (checked before anything is deleted).
        """
        project_dir, screen_dir = scope_dirs(self._baselines_dir, project, screen_name)
        if not project_dir.is_dir():
            return 0

        if screen_dir is not None:
            screens = [screen_dir]
        else:
            screens = [d for d in project_dir.iterdir() if d.is_dir()]

        pruned = 0
        for screen_dir in screens:
            if not screen_dir.is_dir():
                continue
            timestamp_dirs = sorted(
                (
                    d
                    for d in screen_dir.iterdir()
                    if d.is_dir() and (d / "a11y_tree.json").exists()
                ),
                key=lambda d: d.name,
                reverse=True,
            )
            for old_dir in timestamp_dirs[keep:]:
                pruned += remove_store_dir(old_dir, screen_dir, event_prefix="a11y_baseline")

        return pruned

    def _load_entry(self, baseline_dir: Path) -> A11yBaselineEntry | None:
        """Load an A11yBaselineEntry from a baseline directory's metadata.json."""
        metadata_path = baseline_dir / "metadata.json"
        try:
            with open(metadata_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "a11y_baseline_metadata_read_failed",
                path=str(metadata_path),
                error=str(exc),
            )
            return None

        return A11yBaselineEntry.from_dict(data)
