"""Regression: crucible default paths must respect LINC_HOME monkeypatch.

Reference: linc-reactor PR #5 (commit ac8e14f).
"""

from __future__ import annotations


def test_asset_config_respects_LINC_HOME(monkeypatch, tmp_path):
    """AssetExtractorConfig.base_output_dir uses LINC_HOME when set."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    # Force re-import so default_factory sees the new env
    import importlib

    import linc_crucible.core.asset_config as mod

    importlib.reload(mod)
    cfg = mod.AssetExtractorConfig()
    assert cfg.base_output_dir == tmp_path / "crucible" / "assets"


def test_log_mcp_call_respects_LINC_HOME(monkeypatch, tmp_path):
    """log_mcp_call writes to LINC_HOME-based path when LINC_HOME is set."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    from linc_crucible.core.observability import log_mcp_call

    log_mcp_call(
        server="linc-crucible",
        tool="crucible_test",
        params={},
        duration_ms=1.0,
        success=True,
    )
    assert (tmp_path / "mcp-calls.jsonl").exists()


def test_baseline_default_respects_LINC_HOME(monkeypatch, tmp_path):
    """_default_baselines_dir() returns path under LINC_HOME when set.

    Issue #117: unified with asset_config's convention (segment "crucible",
    not ".crucible") — this assertion updated to match the corrected helper.
    """
    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    from linc_crucible.core.baseline import _default_baselines_dir

    assert _default_baselines_dir() == tmp_path / "crucible" / "baselines"


def test_scorer_default_respects_LINC_HOME(monkeypatch, tmp_path):
    """_default_scores_dir() resolves under the unified crucible root (issue #117)."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    from linc_crucible.core.scorer import _default_scores_dir

    assert _default_scores_dir() == tmp_path / "crucible" / "scores"


def test_fleet_default_respects_LINC_HOME(monkeypatch, tmp_path):
    """_default_fleet_path() resolves under the unified crucible root (issue #117)."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    from linc_crucible.core.fleet_registry import _default_fleet_path

    assert _default_fleet_path() == tmp_path / "crucible" / "fleet.yaml"


def test_forge_registry_default_respects_LINC_HOME(monkeypatch, tmp_path):
    """_default_forge_registry() returns path under LINC_HOME when set."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    from linc_crucible.core.forge_integration import _default_forge_registry

    assert _default_forge_registry() == tmp_path / ".forge" / "projects.yaml"


def test_preflight_default_respects_LINC_HOME(monkeypatch, tmp_path):
    """_default_linc_workspace() returns path under LINC_HOME when set."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    from linc_crucible.core.preflight import _default_linc_workspace

    assert _default_linc_workspace() == tmp_path / "projects" / "LINC"


def test_tree_capture_default_respects_LINC_HOME(monkeypatch, tmp_path):
    """tree_capture's _default_baselines_dir() resolves under the unified crucible root (#117)."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    from linc_crucible.core.tree_capture import _default_baselines_dir

    assert _default_baselines_dir() == tmp_path / "crucible" / "baselines"
