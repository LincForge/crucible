"""Tests for core.state_paths — single source of truth for crucible state roots.

Issue #117 (full fix): the original patch moved ``baseline.py``/``asset_config.py``
to ``~/.linc/crucible/`` but left ``tree_capture.py``, ``fleet_registry.py`` and
``scorer.py`` on the legacy ``~/.crucible/`` root (and perf baselines on
``~/.linc/baselines``). Every default-path helper now routes through
``state_paths.state_path()``: canonical ``$LINC_HOME/crucible`` (default
``~/.linc/crucible``), read-fallback to a legacy root only when the canonical
location does not exist and the legacy one does.
"""

from __future__ import annotations

from pathlib import Path

from linc_crucible.core.state_paths import (
    crucible_root,
    legacy_crucible_root,
    linc_home,
    state_path,
)


class TestRoots:
    def test_crucible_root_honours_linc_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        assert crucible_root() == tmp_path / "crucible"

    def test_crucible_root_defaults_to_home_dot_linc(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LINC_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert crucible_root() == tmp_path / ".linc" / "crucible"

    def test_legacy_root_honours_linc_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        assert legacy_crucible_root() == tmp_path / ".crucible"

    def test_legacy_root_defaults_to_home_dot_crucible(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LINC_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert legacy_crucible_root() == tmp_path / ".crucible"

    def test_linc_home_honours_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        assert linc_home() == tmp_path


class TestStatePath:
    def test_canonical_when_neither_exists(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        assert state_path("baselines") == tmp_path / "crucible" / "baselines"

    def test_canonical_wins_when_both_exist(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        (tmp_path / "crucible" / "baselines").mkdir(parents=True)
        (tmp_path / ".crucible" / "baselines").mkdir(parents=True)
        assert state_path("baselines") == tmp_path / "crucible" / "baselines"

    def test_falls_back_to_legacy_when_only_legacy_exists(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        (tmp_path / ".crucible" / "baselines").mkdir(parents=True)
        assert state_path("baselines") == tmp_path / ".crucible" / "baselines"

    def test_legacy_file_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        legacy = tmp_path / ".crucible" / "fleet.yaml"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("devices: []\n")
        assert state_path("fleet.yaml") == legacy

    def test_extra_legacy_roots_consulted_after_dot_crucible(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        old_perf = tmp_path / "baselines"
        old_perf.mkdir()
        assert state_path("baselines", extra_legacy_roots=(tmp_path,)) == old_perf

    def test_extra_legacy_roots_lose_to_canonical(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        (tmp_path / "baselines").mkdir()
        (tmp_path / "crucible" / "baselines").mkdir(parents=True)
        assert (
            state_path("baselines", extra_legacy_roots=(tmp_path,))
            == tmp_path / "crucible" / "baselines"
        )


class TestModuleDefaultsUnified:
    """Every state-path helper resolves under the ONE canonical root."""

    def test_all_defaults_share_the_canonical_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        from linc_crucible.core.asset_config import _default_assets_dir
        from linc_crucible.core.baseline import (
            _default_baselines_dir as visual_baselines_dir,
        )
        from linc_crucible.core.fleet_registry import _default_fleet_path
        from linc_crucible.core.scorer import _default_scores_dir
        from linc_crucible.core.tree_capture import (
            _default_baselines_dir as a11y_baselines_dir,
        )

        root = tmp_path / "crucible"
        assert _default_fleet_path() == root / "fleet.yaml"
        assert a11y_baselines_dir() == root / "baselines"
        assert visual_baselines_dir() == root / "baselines"
        assert _default_scores_dir() == root / "scores"
        assert _default_assets_dir() == root / "assets"

    def test_fleet_path_reads_legacy_root_if_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        from linc_crucible.core.fleet_registry import _default_fleet_path

        legacy = tmp_path / ".crucible" / "fleet.yaml"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("devices: []\n")
        assert _default_fleet_path() == legacy

    def test_a11y_baselines_read_legacy_root_if_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        from linc_crucible.core.tree_capture import _default_baselines_dir

        legacy = tmp_path / ".crucible" / "baselines"
        legacy.mkdir(parents=True)
        assert _default_baselines_dir() == legacy

    def test_scores_read_legacy_root_if_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        from linc_crucible.core.scorer import _default_scores_dir

        legacy = tmp_path / ".crucible" / "scores"
        legacy.mkdir(parents=True)
        assert _default_scores_dir() == legacy


class TestEnsurePrivateDir:
    """[contract] fresh LINC_HOME and crucible state subdirectories are created 0700;
    a pre-existing 0755 LINC_HOME is not tightened by a subdirectory write."""

    def test_creates_missing_parents_and_target_0700(self, tmp_path):
        import os
        import stat

        from linc_crucible.core.state_paths import ensure_private_dir

        fresh_home = tmp_path / "fresh" / "h"
        target = fresh_home / "crucible" / "scores"
        old = os.umask(0o022)
        try:
            returned = ensure_private_dir(target)
        finally:
            os.umask(old)

        assert returned == target
        assert stat.S_IMODE(fresh_home.stat().st_mode) == 0o700
        assert stat.S_IMODE((fresh_home / "crucible").stat().st_mode) == 0o700
        assert stat.S_IMODE(target.stat().st_mode) == 0o700

    def test_preserves_preexisting_0755_linc_home_across_store_writers(
        self, monkeypatch, tmp_path
    ):
        import os
        import stat
        from unittest.mock import patch

        import pytest

        import linc_crucible.core.asset_extractor as ae
        from linc_crucible.core.asset_config import AssetExtractorConfig
        from linc_crucible.core.device import Platform
        from linc_crucible.core.fleet_registry import FleetDevice, FleetRegistry
        from linc_crucible.core.state_paths import ensure_private_dir
        from linc_crucible.core.store_names import prepare_dir

        existing_home = tmp_path / "linc_home"
        existing_home.mkdir(mode=0o755)
        os.chmod(existing_home, 0o755)
        monkeypatch.setenv("LINC_HOME", str(existing_home))

        video_file = tmp_path / "clip.mp4"
        video_file.write_text("fake")
        assets_base = existing_home / "crucible" / "assets"
        cfg = AssetExtractorConfig(base_output_dir=assets_base)

        old = os.umask(0o022)
        try:
            assert ensure_private_dir(existing_home) == existing_home
            scores_dir = prepare_dir(existing_home / "crucible" / "scores", "demo")
            reg = FleetRegistry(fleet_path=existing_home / "crucible" / "fleet" / "fleet.yaml")
            reg.add(FleetDevice(serial="emulator-5554", platform=Platform.ANDROID))
            with (
                patch.object(ae, "_decode_frames", return_value=[tmp_path / "f.png"]),
                patch.object(ae, "filter_frames", return_value=[]),
                pytest.raises(ValueError, match="quality too low"),
            ):
                ae.extract_assets(str(video_file), cfg)
        finally:
            os.umask(old)

        assert stat.S_IMODE(existing_home.stat().st_mode) == 0o755
        assert stat.S_IMODE((existing_home / "crucible").stat().st_mode) == 0o700
        assert stat.S_IMODE(scores_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(reg._path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE((assets_base / "clip").stat().st_mode) == 0o700
