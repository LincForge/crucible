"""Honest-inventory guard on crucible_baseline list (device-verification audit 2026-08-06).

A bare ``{"count": 0, "entries": []}`` reads identically for "no entries match
this filter" and "the baseline store has never existed on this host" — the
2026-08-06 audit found the store absent on the mini while the tool succeeded
vacuously. A zero-entry list must now say explicitly whether the store root
exists and where it was looked for.
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from linc_crucible.core.baseline import BaselineManager
from linc_crucible.core.tree_capture import TreeCapture


def _write_png(path: Path, width: int = 4, height: int = 4) -> Path:
    """Minimal PNG with a valid signature + IHDR width/height."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    path.write_bytes(sig + ihdr + b"\x00" * 16)
    return path


class TestScreenshotListInventory:
    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_list_absent_store_says_so(self, mock_init, tmp_path):
        from linc_crucible.baseline_tools import crucible_baseline

        manager = BaselineManager(baselines_dir=tmp_path / "never-created")
        with patch("linc_crucible.server_context._baseline_manager", manager):
            result = await crucible_baseline(action="list", project="SniperPulse")

        assert result["count"] == 0
        assert result["empty"] is True
        assert result["store_exists"] is False
        assert result["store_root"] == str(tmp_path / "never-created")
        assert "does not exist" in result["message"]

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_list_existing_but_empty_store_says_so(self, mock_init, tmp_path):
        from linc_crucible.baseline_tools import crucible_baseline

        store = tmp_path / "baselines"
        store.mkdir()
        manager = BaselineManager(baselines_dir=store)
        with patch("linc_crucible.server_context._baseline_manager", manager):
            result = await crucible_baseline(action="list", project="SniperPulse")

        assert result["count"] == 0
        assert result["empty"] is True
        assert result["store_exists"] is True
        assert "SniperPulse" in result["message"]

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_list_with_entries_has_no_empty_flag(self, mock_init, tmp_path):
        from linc_crucible.baseline_tools import crucible_baseline

        store = tmp_path / "baselines"
        manager = BaselineManager(baselines_dir=store)
        manager.capture(
            project="SniperPulse",
            screen_name="home",
            image_path=_write_png(tmp_path / "shot.png"),
        )
        with patch("linc_crucible.server_context._baseline_manager", manager):
            result = await crucible_baseline(action="list", project="SniperPulse")

        assert result["count"] == 1
        assert "empty" not in result
        assert "store_exists" not in result


class TestA11yListInventory:
    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_a11y_list_absent_store_says_so(self, mock_init, tmp_path):
        from linc_crucible.baseline_tools import crucible_baseline

        capture = TreeCapture(baselines_dir=tmp_path / "never-created")
        with patch("linc_crucible.server_context._tree_capture", capture):
            result = await crucible_baseline(
                action="list", project="SniperPulse", mode="a11y"
            )

        assert result["count"] == 0
        assert result["empty"] is True
        assert result["store_exists"] is False
        assert result["store_root"] == str(tmp_path / "never-created")
        assert "does not exist" in result["message"]

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    async def test_a11y_list_with_entries_has_no_empty_flag(self, mock_init, tmp_path):
        from linc_device_kit.models import ElementNode

        from linc_crucible.baseline_tools import crucible_baseline

        capture = TreeCapture(baselines_dir=tmp_path / "baselines")
        capture.store(
            project="SniperPulse",
            screen_name="home",
            elements=[ElementNode(id="root", text="Home")],
        )
        with patch("linc_crucible.server_context._tree_capture", capture):
            result = await crucible_baseline(
                action="list", project="SniperPulse", mode="a11y"
            )

        assert result["count"] == 1
        assert "empty" not in result
