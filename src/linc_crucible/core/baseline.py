"""Baseline manager — versioned visual baseline storage with lifecycle management.

Baselines stored at ``~/.linc/crucible/baselines/<project>/<screen>/<timestamp>/``.
Each baseline directory contains ``screenshot.png`` and ``metadata.json``.
``project`` and ``screen_name`` must pass the
:mod:`~linc_crucible.core.store_names` allowlist; every read, write and prune
is proven to stay inside the store (``InvalidStoreName`` otherwise).

Capture, retrieve, list, and prune baselines per project and screen.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from linc_crucible.core.bvt.visual_compare import compute_ssim
from linc_crucible.core.state_paths import state_path
from linc_crucible.core.store_names import (
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
        ``~/.linc/crucible/baselines``). All crucible
        state-path helpers (fleet, a11y baselines, scores, assets, perf
        baselines) share this one root through ``state_paths``.
        Re-evaluated on every call so that ``os.environ`` monkeypatching in
        tests takes effect.
    """
    return state_path("baselines")


def _read_png_dimensions(image_path: Path) -> tuple[int, int]:
    """Read width and height from a PNG file's IHDR chunk.

    Args:
        image_path: Path to the PNG file.

    Returns:
        Tuple of (width, height).

    Raises:
        ValueError: If the file is not a valid PNG.
    """
    png_signature = b"\x89PNG\r\n\x1a\n"
    try:
        with open(image_path, "rb") as f:
            sig = f.read(8)
            if sig != png_signature:
                raise ValueError(f"Not a valid PNG file: {image_path}")
            # Skip IHDR chunk length (4 bytes) and type (4 bytes)
            f.read(8)
            width, height = struct.unpack(">II", f.read(8))
            return width, height
    except (OSError, struct.error) as exc:
        raise ValueError(f"Failed to read PNG dimensions: {exc}") from exc


@dataclass
class BaselineEntry:
    """A single visual baseline capture."""

    project: str
    screen_name: str
    capture_time: str  # ISO 8601 e.g. "20260318T180552Z"
    screenshot_path: str  # Absolute path to stored PNG
    device_info: str
    app_version: str
    width: int
    height: int
    source: str = ""  # "promoted" | "" (normal capture)
    source_score: float | None = None  # SSIM at promotion time
    promoted_at: str = ""  # ISO 8601 UTC, set on promote()

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "project": self.project,
            "screen_name": self.screen_name,
            "capture_time": self.capture_time,
            "screenshot_path": self.screenshot_path,
            "device_info": self.device_info,
            "app_version": self.app_version,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "source_score": self.source_score,
            "promoted_at": self.promoted_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BaselineEntry:
        """Deserialize from a dictionary."""
        return cls(
            project=data.get("project", ""),
            screen_name=data.get("screen_name", ""),
            capture_time=data.get("capture_time", ""),
            screenshot_path=data.get("screenshot_path", ""),
            device_info=data.get("device_info", ""),
            app_version=data.get("app_version", ""),
            width=data.get("width", 0),
            height=data.get("height", 0),
            source=data.get("source", ""),
            source_score=data.get("source_score"),
            promoted_at=data.get("promoted_at", ""),
        )


class BaselineManager:
    """Manages visual baseline screenshots with versioned storage and pruning."""

    def __init__(self, baselines_dir: Path | None = None) -> None:
        env_path = os.environ.get("CRUCIBLE_BASELINES_DIR")
        if baselines_dir:
            self._baselines_dir = baselines_dir
        elif env_path:
            self._baselines_dir = Path(env_path)
        else:
            self._baselines_dir = _default_baselines_dir()

    @property
    def baselines_dir(self) -> Path:
        """Resolved store root (for honest-inventory reporting)."""
        return self._baselines_dir

    def capture(
        self,
        project: str,
        screen_name: str,
        image_path: Path,
        device_info: str = "",
        app_version: str = "",
    ) -> BaselineEntry:
        """Capture a baseline screenshot for a project screen.

        Args:
            project: Project name.
            screen_name: Screen identifier (e.g. "home", "settings").
            image_path: Path to the source PNG file.
            device_info: Device description string.
            app_version: App version string.

        Returns:
            BaselineEntry with metadata and stored screenshot path.

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If image_path is not a valid PNG.
            InvalidStoreName: If project/screen_name is not a safe store name
                (checked before anything is written).
        """
        contained_dir(self._baselines_dir, project, screen_name)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        width, height = _read_png_dimensions(image_path)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        baseline_dir = prepare_dir(self._baselines_dir, project, screen_name, timestamp)

        screenshot_dest = baseline_dir / "screenshot.png"
        shutil.copy2(image_path, screenshot_dest)

        entry = BaselineEntry(
            project=project,
            screen_name=screen_name,
            capture_time=timestamp,
            screenshot_path=str(screenshot_dest),
            device_info=device_info,
            app_version=app_version,
            width=width,
            height=height,
        )

        metadata_path = baseline_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        logger.info(
            "baseline_captured",
            project=project,
            screen=screen_name,
            path=str(baseline_dir),
        )
        return entry

    def get_latest(self, project: str, screen_name: str) -> BaselineEntry | None:
        """Get the most recent baseline for a project screen.

        Args:
            project: Project name.
            screen_name: Screen identifier.

        Returns:
            The newest BaselineEntry, or None if no baselines exist.

        Raises:
            InvalidStoreName: If project/screen_name is not a safe store name.
        """
        screen_dir = contained_dir(self._baselines_dir, project, screen_name)
        if not screen_dir.is_dir():
            return None

        timestamp_dirs = sorted(
            (d for d in screen_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
        if not timestamp_dirs:
            return None

        return self._load_entry(timestamp_dirs[0])

    def compare(
        self,
        project: str,
        screen_name: str,
        candidate_path: Path,
        *,
        pass_threshold: float = 0.95,
        review_threshold: float = 0.85,
    ) -> dict:
        """Compare a candidate screenshot against the latest stored baseline.

        Resolves the golden from the same store this manager captures to, so a
        freshly captured baseline is always the file compared against. Returns
        the ``compute_ssim`` result dict, or a benign surfaced skip when no
        baseline exists yet (``ok=True, skipped=True, no_baseline=True,
        verdict="UNBASELINED"``) — distinct from the loud deps-missing result
        (``ok=False, available=False, deps_missing=True``).
        """
        latest = self.get_latest(project, screen_name)
        if latest is None:
            logger.info("baseline_compare_no_baseline", project=project, screen=screen_name)
            return {
                "ok": True,
                "available": True,
                "skipped": True,
                "no_baseline": True,
                "verdict": "UNBASELINED",
                "reason": f"no baseline for {project}/{screen_name}",
            }

        return compute_ssim(
            Path(latest.screenshot_path),
            Path(candidate_path),
            pass_threshold=pass_threshold,
            review_threshold=review_threshold,
            diff_dir=self._compare_dir(project, screen_name),
        )

    def _compare_dir(self, project: str, screen_name: str) -> Path:
        """Working dir for compare artifacts, OUTSIDE the per-screen timestamp dirs.

        Lives at ``<baselines_dir>/.compare/<project>/<screen>/`` so it is never
        iterated by get_latest/list_baselines/prune (which walk
        ``baselines_dir/<project>/...``) and cannot out-sort a timestamp dir.
        Created through ``prepare_dir`` (names validated, containment proven).
        """
        return prepare_dir(self._baselines_dir / ".compare", project, screen_name)

    def promote(
        self,
        project: str,
        screen_name: str,
        image_path: Path,
        *,
        source_score: float | None = None,
        device_info: str = "",
        app_version: str = "",
    ) -> BaselineEntry:
        """Install a candidate screenshot as the new golden, stamped with
        promotion provenance.

        Explicit-only: nothing auto-promotes on a passing compare. Reuses the
        ``capture`` storage path, then rewrites the stored ``metadata.json``
        with ``source="promoted"``, ``source_score`` and ``promoted_at`` so the
        golden's origin is auditable. Raises the same errors as ``capture``
        (``FileNotFoundError`` / ``ValueError``) on a bad image.
        """
        entry = self.capture(
            project=project,
            screen_name=screen_name,
            image_path=image_path,
            device_info=device_info,
            app_version=app_version,
        )
        entry.source = "promoted"
        entry.source_score = source_score
        entry.promoted_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        metadata_path = Path(entry.screenshot_path).parent / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        logger.info(
            "baseline_promoted",
            project=project,
            screen=screen_name,
            score=source_score,
        )
        return entry

    def list_baselines(self, project: str, screen_name: str = "") -> list[BaselineEntry]:
        """List baselines for a project, optionally filtered by screen.

        Args:
            project: Project name.
            screen_name: If provided, list only this screen's baselines.

        Returns:
            List of BaselineEntry sorted newest-first.

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

        entries: list[BaselineEntry] = []
        for screen_dir in screens:
            if not screen_dir.is_dir():
                continue
            for ts_dir in screen_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                entry = self._load_entry(ts_dir)
                if entry:
                    entries.append(entry)

        entries.sort(key=lambda e: e.capture_time, reverse=True)
        return entries

    def prune(self, project: str, screen_name: str = "", keep: int = 5) -> int:
        """Remove old baselines beyond the retention limit.

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
                (d for d in screen_dir.iterdir() if d.is_dir()),
                key=lambda d: d.name,
                reverse=True,
            )
            for old_dir in timestamp_dirs[keep:]:
                pruned += remove_store_dir(old_dir, screen_dir, event_prefix="baseline")

        return pruned

    def _load_entry(self, baseline_dir: Path) -> BaselineEntry | None:
        """Load a BaselineEntry from a baseline directory's metadata.json."""
        metadata_path = baseline_dir / "metadata.json"
        try:
            with open(metadata_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "baseline_metadata_read_failed",
                path=str(metadata_path),
                error=str(exc),
            )
            return None

        return BaselineEntry.from_dict(data)
