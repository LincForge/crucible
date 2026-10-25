"""[contract] Import hygiene pins: importing crucible modules touches no private LINC paths."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_TARGET_MODULES = (
    "linc_crucible.mcp_server",
    "linc_crucible.cli",
    "linc_crucible.core.preflight",
    "linc_crucible.core.forge_integration",
    "linc_crucible.core.device_policy",
    "linc_crucible.core.ios_signing",
)

_UNSET_ENV_VARS = (
    "LINC_HOME",
    "LINC_WORKSPACE",
    "LINC_PROJECTS_DIR",
    "LINC_DEVICE_FLEET_FILE",
    "FORGE_PROJECTS_FILE",
)

_FORBIDDEN_FRAGMENTS = (
    "projects/LINC",
    "device-fleet.yaml",
    ".forge",
    ".claude.json",
)


def test_contract_import_hygiene_subprocess(tmp_path: Path) -> None:
    """[contract] Importing crucible entrypoints and core modules with empty HOME exits 0."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    for var in _UNSET_ENV_VARS:
        env.pop(var, None)

    stmt = f"import {', '.join(_TARGET_MODULES)}"
    proc = subprocess.run(
        [sys.executable, "-c", stmt],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
    stderr_lines = [line for line in proc.stderr.splitlines() if not line.startswith("objc[")]
    assert stderr_lines == []


def test_contract_import_hygiene_path_recorder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] Reloading crucible modules never probes private LINC/forge/claude paths."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in _UNSET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    modules = [importlib.import_module(name) for name in _TARGET_MODULES]
    snapshots = {mod: dict(mod.__dict__) for mod in modules}

    recorded: list[str] = []
    real_is_file = Path.is_file
    real_is_dir = Path.is_dir
    real_read_text = Path.read_text
    real_open = Path.open

    def rec_is_file(self: Path, *args, **kwargs):
        recorded.append(str(self))
        return real_is_file(self, *args, **kwargs)

    def rec_is_dir(self: Path, *args, **kwargs):
        recorded.append(str(self))
        return real_is_dir(self, *args, **kwargs)

    def rec_read_text(self: Path, *args, **kwargs):
        recorded.append(str(self))
        return real_read_text(self, *args, **kwargs)

    def rec_open(self: Path, *args, **kwargs):
        recorded.append(str(self))
        return real_open(self, *args, **kwargs)

    try:
        with (
            patch.object(Path, "is_file", rec_is_file),
            patch.object(Path, "is_dir", rec_is_dir),
            patch.object(Path, "read_text", rec_read_text),
            patch.object(Path, "open", rec_open),
        ):
            for mod in modules:
                importlib.reload(mod)
    finally:
        for mod, snap in snapshots.items():
            mod.__dict__.clear()
            mod.__dict__.update(snap)

    offenders = [
        path for path in recorded if any(fragment in path for fragment in _FORBIDDEN_FRAGMENTS)
    ]
    assert offenders == [], f"Forbidden paths touched during import: {offenders}"
