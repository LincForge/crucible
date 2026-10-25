# tests/test_asset_config.py

from linc_crucible.core.asset_config import AssetExtractorConfig


def test_asset_config_defaults():
    config = AssetExtractorConfig()
    assert config.frames_per_second == 1.0
    assert config.blur_threshold == 100.0
    assert config.phash_threshold == 5
    assert "crucible" in str(config.base_output_dir)


def test_config_variant_and_ai_defaults():
    cfg = AssetExtractorConfig()
    assert cfg.variant_formats == {"x": (1200, 630), "ig": (1080, 1080)}
    assert cfg.max_ai_candidates == 24
    assert cfg.ai_timeout_seconds == 180
    # the default dict must be a per-instance copy, not shared
    cfg.variant_formats["tiktok"] = (1080, 1920)
    assert "tiktok" not in AssetExtractorConfig().variant_formats
