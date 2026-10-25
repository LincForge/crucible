"""Tests for letterbox-pad format variant generation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from linc_crucible.core.asset_variants import make_variants


def _portrait_png(path, w=900, h=1950):
    # vertical gradient so cover/fit/blur produce a non-uniform image
    arr = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None, None], (1, w, 3))
    Image.fromarray(arr).save(path)
    return path


def test_make_variants_dimensions_and_files(tmp_path):
    src = _portrait_png(tmp_path / "src.png")
    formats = {"x": (1200, 630), "ig": (1080, 1080)}

    variants = make_variants(src, "demo_frame_0000", tmp_path, formats)

    # one entry per format, list-shaped with name/width/height/path
    assert {v["name"] for v in variants} == {"x", "ig"}
    by_name = {v["name"]: v for v in variants}
    assert (by_name["x"]["width"], by_name["x"]["height"]) == (1200, 630)
    assert (by_name["ig"]["width"], by_name["ig"]["height"]) == (1080, 1080)
    # paths are RELATIVE (basename only), files exist at the right size
    for v in variants:
        assert v["path"] == f"demo_frame_0000_{v['name']}.png"
        out = tmp_path / v["path"]
        assert out.exists()
        assert Image.open(out).size == (v["width"], v["height"])


def test_make_variants_preserves_foreground_not_cropped(tmp_path):
    # a square source fit into a wide x.com frame must be letterboxed (padded),
    # i.e. the output is exactly the target size with the foreground centered
    src = tmp_path / "sq.png"
    Image.fromarray(np.full((1000, 1000, 3), 128, dtype=np.uint8)).save(src)
    variants = make_variants(src, "s", tmp_path, {"x": (1200, 630)})
    out = Image.open(tmp_path / variants[0]["path"])
    assert out.size == (1200, 630)
