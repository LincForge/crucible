"""Tests for asset extractor — includes characterization tests for extract_assets."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import structlog
import structlog.testing

from linc_crucible.core.asset_config import AssetExtractorConfig
from linc_crucible.core.asset_extractor import filter_frames, parse_ai_response


@pytest.fixture(autouse=True)
def _capture_structlog():
    """Route structlog through an in-memory capture for every test in this module.

    The module-level PrintLogger in asset_extractor binds whatever stdout/stderr was
    live at first use; under mutmut's repeated in-process pytest sessions that stream
    can already be closed, so a plain warning raises "I/O operation on closed file".
    """
    with structlog.testing.capture_logs():
        yield


def test_filter_frames(tmp_path):
    # Just a mock unit test to ensure the filtering function can be imported and runs
    import numpy as np
    from PIL import Image

    # Create a blank (blurry/empty) image
    img_path = tmp_path / "test.png"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)

    # Expect it to be filtered out because variance is 0 (below 100)
    survivors = filter_frames([img_path], blur_threshold=100.0, phash_threshold=5)
    assert len(survivors) == 0


def _synthetic_laplacian_images():
    import numpy as np
    from PIL import Image, ImageFilter

    checkerboard = (((np.indices((32, 32)).sum(axis=0)) % 2) * 255).astype(np.uint8)
    uniform = np.full((32, 32), 128, dtype=np.uint8)
    blurred = np.asarray(
        Image.fromarray(checkerboard).filter(ImageFilter.GaussianBlur(radius=2)),
        dtype=np.uint8,
    )
    img_1x1 = np.array([[42]], dtype=np.uint8)
    img_3x3 = np.array([[10, 200, 30], [150, 50, 220], [80, 190, 110]], dtype=np.uint8)
    return {
        "checkerboard": (checkerboard, 1040400.0),
        "uniform": (uniform, 0.0),
        "gaussian_blurred_checkerboard": (blurred, 10.59765625),
        "1x1": (img_1x1, 0.0),
        "3x3": (img_3x3, 305335.80246913584),
    }


def test_contract_no_cv2_or_opencv_dependency():
    """[contract] No `import cv2` under src/ and no opencv in pyproject dependencies."""
    import tomllib

    import linc_crucible

    repo_root = Path(__file__).resolve().parents[1]
    git_root = repo_root.parent if repo_root.name == "mutants" else repo_root
    src_dir = git_root / "src"
    pkg_root = src_dir if src_dir.is_dir() else Path(linc_crucible.__file__).resolve().parent
    listed = sorted(
        p.relative_to(pkg_root).as_posix() for p in pkg_root.rglob("*.py")
    )
    assert listed
    cv2_imports = []
    for rel_path in listed:
        text = (pkg_root / rel_path).read_text(encoding="utf-8")
        if "import cv2" in text or "from cv2" in text:
            cv2_imports.append(rel_path)
    assert cv2_imports == []

    pyproject = tomllib.loads((git_root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    opt_groups = pyproject["project"].get("optional-dependencies", {}).values()
    opt_deps = [dep for group in opt_groups for dep in group]
    assert not any("opencv" in d.lower() or "cv2" in d.lower() for d in [*deps, *opt_deps])


def test_laplacian_variance_cv2_parity():
    """[test] `_laplacian_variance` matches `cv2.Laplacian(..., cv2.CV_64F).var()` within 1e-6."""
    cv2 = pytest.importorskip("cv2")
    from linc_crucible.core.asset_extractor import _laplacian_variance

    for name, (gray, _golden) in _synthetic_laplacian_images().items():
        expected = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        actual = _laplacian_variance(gray)
        assert isinstance(actual, float)
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), f"mismatch on {name}"


def test_laplacian_variance_golden_values():
    """[test] `_laplacian_variance` matches pinned cv2 golden variances on synthetic images."""
    from linc_crucible.core.asset_extractor import _laplacian_variance

    for name, (gray, golden) in _synthetic_laplacian_images().items():
        actual = _laplacian_variance(gray)
        assert isinstance(actual, float)
        assert actual == pytest.approx(golden, rel=1e-6, abs=1e-9), f"mismatch on {name}"


def test_filter_frames_survivors_and_unreadable(tmp_path: Path):
    """[test] filter_frames keeps sharp unique frames, drops blurry/dups, skips bad files."""
    import numpy as np
    from PIL import Image

    import linc_crucible.core.asset_extractor as ae

    unreadable = tmp_path / "unreadable.png"
    unreadable.write_bytes(b"not-a-valid-image")
    missing = tmp_path / "does_not_exist.png"

    blurry = tmp_path / "blurry.png"
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(blurry)

    sharp1 = tmp_path / "sharp1.png"
    grid = (((np.indices((64, 64)).sum(axis=0)) % 2) * 255).astype(np.uint8)
    rgb1 = np.stack([grid, np.roll(grid, 1, axis=0), grid], axis=-1)
    Image.fromarray(rgb1).save(sharp1)

    sharp1_dup = tmp_path / "sharp1_dup.png"
    Image.fromarray(rgb1).save(sharp1_dup)

    sharp2 = tmp_path / "sharp2.png"
    yy, xx = np.indices((64, 64))
    pattern2 = np.where((xx < 32) ^ (yy < 16), 255, 0).astype(np.uint8)
    edge_boost = (((xx + yy) % 2) * 60).astype(np.uint8)
    rgb2 = np.stack([np.bitwise_xor(pattern2, edge_boost)] * 3, axis=-1)
    Image.fromarray(rgb2).save(sharp2)

    seen_dtypes = []
    real_lap = ae._laplacian_variance

    def spy_lap(gray):
        seen_dtypes.append(gray.dtype)
        return real_lap(gray)

    with patch.object(ae, "_laplacian_variance", side_effect=spy_lap):
        survivors = filter_frames(
            [unreadable, missing, blurry, sharp1, sharp1_dup, sharp2],
            blur_threshold=100.0,
            phash_threshold=5,
        )

    assert survivors == [sharp1, sharp2]
    assert all(dt == np.float64 for dt in seen_dtypes)


def test_evenly_spaced_selection():
    from linc_crucible.core.asset_extractor import evenly_spaced

    items = list(range(10))  # 0..9
    # n >= len → all items
    assert evenly_spaced(items, 20) == items
    assert evenly_spaced(items, 10) == items
    # n == 1 → middle frame
    assert evenly_spaced(items, 1) == [5]
    # empty / non-positive
    assert evenly_spaced([], 5) == []
    assert evenly_spaced(items, 0) == []
    # even distribution: first and last always present, monotonic, no dups
    picks = evenly_spaced(items, 4)
    assert len(picks) == 4
    assert picks[0] == 0 and picks[-1] == 9
    assert picks == sorted(picks)
    assert len(set(picks)) == 4


def test_parse_ai_response():
    resp = (
        '{"assets": [{"filename": "frame_001.png", "rationale": "good"}], "fallback_used": false}'
    )
    parsed = parse_ai_response(resp)
    assert len(parsed["assets"]) == 1
    assert parsed["assets"][0]["filename"] == "frame_001.png"


class TestExtractAssets:
    """Characterization tests for extract_assets — covers CRAP-critical branches."""

    def test_extract_assets_missing_video(self, tmp_path: Path) -> None:
        """FileNotFoundError when video path does not exist."""
        from linc_crucible.core.asset_extractor import extract_assets

        with pytest.raises(FileNotFoundError, match="not found"):
            extract_assets(str(tmp_path / "nonexistent.mp4"))

    def test_extract_assets_low_quality_raises(self, tmp_path: Path) -> None:
        """survivors == [] → descriptive 'quality too low' error, temp still cleaned."""
        import linc_crucible.core.asset_extractor as ae

        video_file = tmp_path / "v.mp4"
        video_file.write_text("fake")
        cfg = AssetExtractorConfig(base_output_dir=tmp_path / "out")

        with (
            patch.object(ae, "_decode_frames", return_value=[tmp_path / "f.png"]),
            patch.object(ae, "filter_frames", return_value=[]),
            pytest.raises(ValueError, match="quality too low"),
        ):
            ae.extract_assets(str(video_file), cfg)

    def test_extract_assets_temp_under_base_output(self, tmp_path: Path) -> None:
        """Temp scratch dir lives under base_output_dir (same FS), not /tmp."""
        import linc_crucible.core.asset_extractor as ae

        video_file = tmp_path / "v.mp4"
        video_file.write_text("fake")
        base = tmp_path / "out"
        cfg = AssetExtractorConfig(base_output_dir=base)
        seen = {}

        def capture_decode(video, temp_dir, config):
            seen["temp"] = temp_dir
            raise RuntimeError("stop after temp creation")

        with (
            patch.object(ae, "_decode_frames", side_effect=capture_decode),
            pytest.raises(RuntimeError, match="stop after"),
        ):
            ae.extract_assets(str(video_file), cfg)
        assert str(seen["temp"]).startswith(str(base))
        assert not seen["temp"].exists()  # finally cleaned it up


# --- Task 5: rank_frames ---


def _ai_result(output: str, success: bool = True):
    from linc_ai import AIResult

    return AIResult(success=success, output=output, model="claude-test")


def _mk_candidates(tmp_path, names):
    import numpy as np
    from PIL import Image

    paths = []
    for nm in names:
        p = tmp_path / nm
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def test_rank_frames_builds_cli_request(tmp_path):
    """Contract: the ClaudeCLIRequest carries Read tool, add-dir, schema, derived turns."""
    pytest.importorskip("linc_ai")
    from linc_ai import ModelTier

    from linc_crucible.core.asset_extractor import rank_frames

    cands = _mk_candidates(tmp_path, ["frame_0000.png", "frame_0001.png"])
    captured = {}

    def fake_complete(request):
        captured["req"] = request
        return _ai_result('{"assets": [{"filename": "frame_0000.png", "rationale": "hero"}]}')

    with patch("linc_ai.complete", side_effect=fake_complete):
        cfg = AssetExtractorConfig()
        out = rank_frames(cands, tmp_path, cfg, max_assets=5)

    req = captured["req"]
    assert req.tools == ["Read"]
    assert req.allowed_tools == ["Read"]
    assert str(tmp_path) in req.context_dirs
    assert req.json_schema is not None
    assert req.model_tier == ModelTier.HIGH
    assert req.max_turns == len(cands) + 2
    assert req.timeout_seconds == cfg.ai_timeout_seconds
    assert req.dangerously_skip_permissions is True
    assert out["fallback_used"] is False
    assert out["picks"][0]["path"] == cands[0]
    assert out["picks"][0]["rationale"] == "hero"


def test_rank_frames_drops_hallucinated_filenames(tmp_path):
    pytest.importorskip("linc_ai")
    from linc_crucible.core.asset_extractor import rank_frames

    cands = _mk_candidates(tmp_path, ["frame_0000.png", "frame_0001.png"])
    resp = (
        '{"assets": [{"filename": "frame_0000.png", "rationale": "ok"}, '
        '{"filename": "frame_9999.png", "rationale": "ghost"}]}'
    )
    with patch("linc_ai.complete", return_value=_ai_result(resp)):
        out = rank_frames(cands, tmp_path, AssetExtractorConfig(), max_assets=5)
    assert [p["path"] for p in out["picks"]] == [cands[0]]
    assert out["fallback_used"] is False


def test_rank_frames_all_hallucinated_falls_back(tmp_path):
    pytest.importorskip("linc_ai")
    from linc_crucible.core.asset_extractor import rank_frames

    cands = _mk_candidates(tmp_path, ["frame_0000.png", "frame_0001.png", "frame_0002.png"])
    resp = '{"assets": [{"filename": "ghost.png", "rationale": "nope"}]}'
    with patch("linc_ai.complete", return_value=_ai_result(resp)):
        out = rank_frames(cands, tmp_path, AssetExtractorConfig(), max_assets=2)
    assert out["fallback_used"] is True
    assert len(out["picks"]) == 2


def test_rank_frames_provider_error_falls_back(tmp_path):
    pytest.importorskip("linc_ai")
    from linc_ai import AIProviderError

    from linc_crucible.core.asset_extractor import rank_frames

    cands = _mk_candidates(
        tmp_path, ["frame_0000.png", "frame_0001.png", "frame_0002.png", "frame_0003.png"]
    )
    with patch("linc_ai.complete", side_effect=AIProviderError("no claude binary")):
        out = rank_frames(cands, tmp_path, AssetExtractorConfig(), max_assets=2)
    assert out["fallback_used"] is True
    assert len(out["picks"]) == 2
    assert all(p["rationale"] == "Fallback selection (evenly spaced)" for p in out["picks"])


def test_rank_frames_unparseable_falls_back(tmp_path):
    pytest.importorskip("linc_ai")
    from linc_crucible.core.asset_extractor import rank_frames

    cands = _mk_candidates(tmp_path, ["frame_0000.png", "frame_0001.png"])
    with patch("linc_ai.complete", return_value=_ai_result("not json at all")):
        out = rank_frames(cands, tmp_path, AssetExtractorConfig(), max_assets=2)
    assert out["fallback_used"] is True


def test_rank_frames_caps_candidates(tmp_path):
    pytest.importorskip("linc_ai")
    from linc_crucible.core.asset_extractor import rank_frames

    names = [f"frame_{i:04d}.png" for i in range(30)]
    cands = _mk_candidates(tmp_path, names)
    captured = {}

    def fake_complete(request):
        captured["req"] = request
        return _ai_result('{"assets": [{"filename": "frame_0000.png", "rationale": "x"}]}')

    cfg = AssetExtractorConfig(max_ai_candidates=24)
    with patch("linc_ai.complete", side_effect=fake_complete):
        rank_frames(cands, tmp_path, cfg, max_assets=5)
    # derived turns reflect the capped candidate count, not the full survivor list
    assert captured["req"].max_turns == 24 + 2


def test_rank_frames_import_error_falls_back(tmp_path):
    """When linc_ai is unavailable (headless/CI), ImportError routes to fallback."""
    import builtins

    from linc_crucible.core.asset_extractor import rank_frames

    cands = _mk_candidates(tmp_path, ["frame_0000.png", "frame_0001.png"])
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "linc_ai" or name.startswith("linc_ai."):
            raise ImportError("linc_ai not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocked):
        out = rank_frames(cands, tmp_path, AssetExtractorConfig(), max_assets=1)
    assert out["fallback_used"] is True
    assert len(out["picks"]) == 1


def test_rank_frames_broken_linc_ai_import_falls_back(tmp_path):
    """[contract] A non-ImportError raised while importing linc_ai never escapes rank_frames."""
    import builtins

    from linc_crucible.core.asset_extractor import rank_frames

    cands = _mk_candidates(tmp_path, ["frame_0000.png", "frame_0001.png"])
    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "linc_ai" or name.startswith("linc_ai."):
            raise RuntimeError("linc_ai import-time failure")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=broken),
        structlog.testing.capture_logs() as captured,
    ):
        out = rank_frames(cands, tmp_path, AssetExtractorConfig(), max_assets=1)
    assert out["fallback_used"] is True
    assert len(out["picks"]) == 1
    events = [c["event"] for c in captured]
    # routed through the explicit guard, not the runtime-failure degrade
    assert events == ["asset_ai_import_failed", "asset_ai_unavailable"]
    assert captured[0]["log_level"] == "warning"
    assert captured[0]["error"] == "RuntimeError: linc_ai import-time failure"


def test_rank_frames_ai_unavailable_warns_and_falls_back(tmp_path):
    """Simulates missing linc-ai package by patching _ai_available to False."""
    import sys
    from unittest.mock import MagicMock

    import structlog

    import linc_crucible.core.asset_extractor as ae

    with patch.dict(sys.modules, {"linc_ai": MagicMock()}):
        assert ae._ai_available() is True
    with patch.dict(sys.modules, {"linc_ai": None}):
        assert ae._ai_available() is False

    cands = _mk_candidates(tmp_path, ["frame_0000.png", "frame_0001.png"])
    complete_mock = MagicMock()
    try:
        import linc_ai

        complete_ctx = patch.object(linc_ai, "complete", complete_mock)
    except ImportError:
        complete_ctx = patch.dict(sys.modules, {"linc_ai": MagicMock(complete=complete_mock)})

    with (
        complete_ctx,
        patch.object(ae, "_ai_available", return_value=False),
        structlog.testing.capture_logs() as captured,
    ):
        out = ae.rank_frames(cands, tmp_path, AssetExtractorConfig(), max_assets=1)

    assert out["fallback_used"] is True
    assert len(out["picks"]) == 1
    complete_mock.assert_not_called()
    warnings = [c for c in captured if c.get("event") == "asset_ai_unavailable"]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert warnings[0]["hint"] == (
        "AI frame ranking needs the optional linc-ai package, which is not published; "
        "using evenly-spaced frames"
    )
    assert "[ai]" not in warnings[0]["hint"]


# --- Task 6: extract_assets end-to-end ---


def test_extract_assets_end_to_end(tmp_path, monkeypatch):
    """av + AI mocked; assert originals + variants + manifest (relative paths) land in output."""
    pytest.importorskip("linc_ai")
    import json as _json

    import numpy as np
    from PIL import Image

    import linc_crucible.core.asset_extractor as ae

    video_file = tmp_path / "demo.mp4"
    video_file.write_text("fake")
    base = tmp_path / "out"
    cfg = AssetExtractorConfig(base_output_dir=base)

    # _decode_frames writes two real PNGs into the temp dir and returns them
    def fake_decode(video, temp_dir, config):
        paths = []
        for i in range(2):
            p = temp_dir / f"frame_{i:04d}.png"
            Image.fromarray(np.full((1950, 900, 3), 100 + i, dtype=np.uint8)).save(p)
            paths.append(p)
        return paths

    monkeypatch.setattr(ae, "_decode_frames", fake_decode)
    monkeypatch.setattr(ae, "filter_frames", lambda paths, *a: paths)

    def fake_complete(request):
        from linc_ai import AIResult

        return AIResult(
            success=True,
            output='{"assets": [{"filename": "frame_0000.png", "rationale": "hero shot"}]}',
            model="claude-test",
        )

    monkeypatch.setattr("linc_ai.complete", fake_complete)

    result = ae.extract_assets(str(video_file), cfg)

    out_dir = base / "demo"
    assert result["fallback_used"] is False
    assert (out_dir / "demo_frame_0000.png").exists()
    assert (out_dir / "demo_frame_0000_x.png").exists()
    assert (out_dir / "demo_frame_0000_ig.png").exists()

    manifest = _json.loads((out_dir / "manifest.json").read_text())
    assert manifest["video"] == "demo.mp4"
    assert manifest["fallback_used"] is False
    assert "model" not in manifest
    asset0 = manifest["assets"][0]
    assert asset0["original"] == "demo_frame_0000.png"  # RELATIVE
    assert asset0["rationale"] == "hero shot"
    names = {v["name"] for v in asset0["variants"]}
    assert names == {"x", "ig"}
    assert all("/" not in v["path"] for v in asset0["variants"])  # relative basenames

    # temp scratch dir cleaned up — only the stem output dir remains under base
    leftovers = [
        p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("crucible_assets_")
    ]
    assert leftovers == []
