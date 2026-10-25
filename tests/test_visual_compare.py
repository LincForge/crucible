"""Tests for the root-agnostic SSIM kernel + dep declaration."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from linc_crucible.core.bvt.visual_compare import compute_ssim

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _gradient_png(path: Path, *, invert: bool = False, size: int = 64) -> Path:
    """A real PNG with deterministic horizontal-gradient structure."""
    img = Image.new("L", (size, size))
    px = img.load()
    for x in range(size):
        v = int(255 * x / (size - 1))
        if invert:
            v = 255 - v
        for y in range(size):
            px[x, y] = v
    img.save(path, "PNG")
    return path


def test_ssim_deps_declared() -> None:
    """scikit-image and numpy must be first-class deps, not transitive (AC2)."""
    data = tomllib.loads(PYPROJECT.read_text())
    deps = " ".join(data["project"]["dependencies"]).lower()
    assert "scikit-image" in deps
    assert "numpy" in deps


def test_compute_ssim_identical_is_pass(tmp_path: Path) -> None:
    g = _gradient_png(tmp_path / "g.png")
    c = _gradient_png(tmp_path / "c.png")
    res = compute_ssim(g, c)
    assert res["ok"] is True
    assert res["available"] is True
    assert res["verdict"] == "PASS"
    assert res["score"] == pytest.approx(1.0, abs=1e-6)


def test_compute_ssim_different_is_fail(tmp_path: Path) -> None:
    g = _gradient_png(tmp_path / "g.png")
    c = _gradient_png(tmp_path / "c.png", invert=True)
    res = compute_ssim(g, c)
    assert res["ok"] is True
    assert res["verdict"] == "FAIL"
    assert res["score"] < 0.85


def test_compute_ssim_candidate_missing(tmp_path: Path) -> None:
    g = _gradient_png(tmp_path / "g.png")
    res = compute_ssim(g, tmp_path / "nope.png")
    assert res["ok"] is False
    assert res["available"] is True
    assert res["verdict"] == "FAIL"
    assert "not found" in res["reason"].lower()


def test_compute_ssim_deps_missing_is_loud(tmp_path: Path, monkeypatch, caplog) -> None:
    """Missing skimage must NOT return a silent pass (AC3)."""
    g = _gradient_png(tmp_path / "g.png")
    c = _gradient_png(tmp_path / "c.png")
    monkeypatch.setitem(sys.modules, "skimage", None)  # force ImportError
    monkeypatch.setitem(sys.modules, "skimage.metrics", None)
    res = compute_ssim(g, c)
    assert res["ok"] is False
    assert res["available"] is False
    assert res["deps_missing"] is True
    assert res.get("verdict") != "PASS"


def test_compute_ssim_writes_diff_when_diff_dir_given(tmp_path: Path) -> None:
    g = _gradient_png(tmp_path / "g.png")
    c = _gradient_png(tmp_path / "c.png", invert=True)
    out = tmp_path / "out"
    res = compute_ssim(g, c, diff_dir=out)
    assert Path(res["diff_path"]).exists()
    assert Path(res["current_path"]).exists()


def test_compute_ssim_unreadable_candidate_returns_error(tmp_path: Path) -> None:
    """A 0-byte candidate returns ok False, verdict ERROR, no exception."""
    g = _gradient_png(tmp_path / "g.png")
    c = tmp_path / "c_empty.png"
    c.write_bytes(b"")
    res = compute_ssim(g, c)
    assert res["ok"] is False
    assert res["verdict"] == "ERROR"
    assert "error" in res


def test_compute_ssim_geometry_mismatch_returns_error(tmp_path: Path) -> None:
    """100x100 candidate vs 200x200 golden returns verdict ERROR and never resizes."""
    g = _gradient_png(tmp_path / "g.png", size=200)
    c = _gradient_png(tmp_path / "c.png", size=100)
    res = compute_ssim(g, c)
    assert res["ok"] is False
    assert res["verdict"] == "ERROR"
    assert res["error"] == "geometry mismatch 100x100 vs 200x200"

