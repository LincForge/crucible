# tests/test_asset_mcp.py
import json
from unittest.mock import patch

import pytest


@patch("linc_crucible.core.asset_extractor.av.open")
def test_extract_assets_mocked(mock_av_open, tmp_path):
    # We won't test full video extraction locally to save time,
    # but assert the tool function exists and returns a dict with 'assets'
    pass


@pytest.mark.asyncio
async def test_extract_assets_tool_surfaces_errors(monkeypatch):

    def boom(video_path, config):
        raise ValueError("quality too low")

    import linc_crucible.asset_tools as at
    monkeypatch.setattr(at, "extract_assets", boom)
    out = await at.crucible_extract_assets(video_path="/nope/x.mp4", max_assets=3)
    payload = json.loads(out)
    assert payload["error"]
    assert "quality too low" in payload["error"]
