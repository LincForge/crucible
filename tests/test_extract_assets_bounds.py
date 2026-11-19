"""Contract and boundary tests for crucible_extract_assets frame caps and bomb handling."""

from __future__ import annotations

import asyncio
import json
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
import structlog.testing
from PIL import Image

import linc_crucible.core.asset_extractor as ae
from linc_crucible.asset_tools import crucible_extract_assets
from linc_crucible.core.asset_config import AssetExtractorConfig
from linc_crucible.core.asset_extractor import (
    _decode_frames,
    extract_assets,
    filter_frames,
)


@pytest.fixture(autouse=True)
def _capture_structlog():
    with structlog.testing.capture_logs():
        yield


@pytest.fixture(autouse=True)
def _disable_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic fallback ranking so tests never shell out to the claude CLI."""
    monkeypatch.setattr(ae, "_ai_available", lambda: False)


def _write_synthetic_mp4(path: Path, *, num_frames: int = 40, fps: int = 10) -> Path:
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        yy, xx = np.indices((64, 64))
        for idx in range(num_frames):
            checker = (((xx // 4 + yy // 4 + idx) % 2) * 255).astype(np.uint8)
            band = np.where((xx < 32) ^ (yy < (8 + (idx * 3) % 48)), 255, 0).astype(np.uint8)
            plane = np.bitwise_xor(checker, band // 2).astype(np.uint8)
            rgb = np.stack([plane, np.roll(plane, idx + 1, axis=0), plane], axis=-1)
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return path


def _make_sharp_png(path: Path, *, seed: int = 0) -> Path:
    yy, xx = np.indices((64, 64))
    checker = (((xx + yy + seed) % 2) * 255).astype(np.uint8)
    block = np.where((xx < 32) ^ (yy < (16 + 8 * seed)), 200, 20).astype(np.uint8)
    plane = np.bitwise_xor(checker, block).astype(np.uint8)
    rgb = np.stack([plane, np.roll(plane, seed + 1, axis=0), plane], axis=-1)
    Image.fromarray(rgb).save(path)
    return path


def test_contract_max_sampled_frames_caps_40_frame_video(tmp_path: Path) -> None:
    """[contract] With max_sampled_frames=5 and frames_per_second high enough to sample every
    frame, a 40-frame synthetic video yields <= 5 PNGs written to the temp dir."""
    video_path = _write_synthetic_mp4(tmp_path / "forty.mp4", num_frames=40, fps=10)
    out_base = tmp_path / "out"
    cfg = AssetExtractorConfig(frames_per_second=100.0, max_assets=3, base_output_dir=out_base)
    cfg.max_sampled_frames = 5

    saved_temp_pngs: list[Path] = []
    real_save = Image.Image.save

    def spy_save(self: Image.Image, fp, *args, **kwargs):
        p = Path(fp)
        if p.name.startswith("frame_") and p.suffix == ".png":
            saved_temp_pngs.append(p)
        return real_save(self, fp, *args, **kwargs)

    with (
        patch.object(Image.Image, "save", autospec=True, side_effect=spy_save),
        structlog.testing.capture_logs() as captured,
    ):
        result = extract_assets(str(video_path), cfg)

    assert len(saved_temp_pngs) <= 5
    assert len(saved_temp_pngs) == 5
    assert len(result["assets"]) == 3
    default_cfg = AssetExtractorConfig()
    assert default_cfg.max_sampled_frames == 600
    assert default_cfg.max_frame_pixels == 40_000_000
    capped_events = [e for e in captured if e.get("event") == "asset_frames_capped"]
    assert len(capped_events) == 1
    assert capped_events[0].get("cap") == 5
    assert capped_events[0].get("max_sampled_frames") == 5


def test_contract_decode_frames_stops_at_cap_when_stream_length_unknown(tmp_path: Path) -> None:
    """[contract] Even when stream.frames and duration are unknown, _decode_frames stops writing
    after max_sampled_frames and logs asset_frames_capped."""
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    fake_frames = [av.VideoFrame.from_ndarray(rgb, format="rgb24") for _ in range(12)]
    fake_stream = MagicMock()
    fake_stream.average_rate = 10
    fake_stream.frames = 0
    fake_stream.duration = None
    fake_stream.time_base = None

    fake_container = MagicMock()
    fake_container.streams.video = [fake_stream]
    fake_container.decode.return_value = iter(fake_frames)

    cfg = AssetExtractorConfig(frames_per_second=100.0)
    cfg.max_sampled_frames = 5

    with (
        patch("linc_crucible.core.asset_extractor.av.open", return_value=fake_container),
        structlog.testing.capture_logs() as captured,
    ):
        extracted = _decode_frames(tmp_path / "stream.mp4", tmp_path, cfg)

    assert len(extracted) == 5
    assert len(list(tmp_path.glob("frame_*.png"))) == 5
    capped_events = [e for e in captured if e.get("event") == "asset_frames_capped"]
    assert len(capped_events) == 1
    assert capped_events[0]["cap"] == 5
    fake_container.close.assert_called_once()


def test_contract_sampling_interval_from_duration_and_exact_frame_boundaries() -> None:
    """[test] _estimated_stream_frames and _sampling_interval use duration*time_base*fps when
    stream.frames is 0 and respect exact boundaries."""
    from linc_crucible.core.asset_extractor import _estimated_stream_frames, _sampling_interval

    stream_from_dur = MagicMock()
    stream_from_dur.frames = 0
    stream_from_dur.duration = 400
    stream_from_dur.time_base = Fraction(1, 10)
    stream_from_dur.average_rate = 10
    assert _estimated_stream_frames(stream_from_dur, 10) == 400

    cfg = AssetExtractorConfig(frames_per_second=100.0, max_sampled_frames=5)
    interval, capped = _sampling_interval(stream_from_dur, cfg)
    assert interval == 80
    assert capped is True

    # When total_frames <= max_sampled_frames, fps_interval wins and capped is False
    stream_short = MagicMock()
    stream_short.frames = 5
    stream_short.duration = None
    stream_short.time_base = None
    stream_short.average_rate = 10
    interval_short, capped_short = _sampling_interval(stream_short, cfg)
    assert interval_short == 1
    assert capped_short is False

    # Unknown fps falls back to 30
    stream_no_fps = MagicMock()
    stream_no_fps.frames = 0
    stream_no_fps.duration = None
    stream_no_fps.time_base = None
    stream_no_fps.average_rate = None
    assert _estimated_stream_frames(stream_no_fps, None) == 0
    assert _sampling_interval(stream_no_fps, cfg) == (30, False)

    # stream.frames == 1 is respected directly without falling back to duration
    stream_one = MagicMock()
    stream_one.frames = 1
    stream_one.duration = 999
    stream_one.time_base = Fraction(1, 1)
    assert _estimated_stream_frames(stream_one, 10) == 1

    # Fractional duration * time_base * fps is rounded up via math.ceil
    stream_frac = MagicMock()
    stream_frac.frames = 0
    stream_frac.duration = 41
    stream_frac.time_base = Fraction(1, 10)
    stream_frac.average_rate = 1
    assert _estimated_stream_frames(stream_frac, 1) == 5


@pytest.mark.parametrize(
    ("width", "height", "max_pixels", "expected_msg"),
    [
        (20000, 20000, 40_000_000, "frame size 20000x20000 exceeds the cap"),
        (8001, 5000, 40_000_000, "frame size 8001x5000 exceeds the cap"),
        (0, 100, 40_000_000, "frame size 0x100 exceeds the cap"),
        (100, 0, 40_000_000, "frame size 100x0 exceeds the cap"),
        (-10, 100, 40_000_000, "frame size -10x100 exceeds the cap"),
        (100, -5, 40_000_000, "frame size 100x-5 exceeds the cap"),
    ],
    ids=[
        "oversized_20000x20000",
        "one_over_cap_8001x5000",
        "zero_width",
        "zero_height",
        "negative_width",
        "negative_height",
    ],
)
def test_contract_oversized_or_nonpositive_frame_refused_before_to_image(
    tmp_path: Path,
    width: int,
    height: int,
    max_pixels: int,
    expected_msg: str,
) -> None:
    """[contract] A fake frame reporting 20000x20000 (or <=0 dimensions) is refused before
    to_image() is called, while a frame right at the cap (8000x5000) is accepted."""
    fake_frame = MagicMock()
    fake_frame.width = width
    fake_frame.height = height
    fake_img = MagicMock()
    fake_frame.to_image.return_value = fake_img

    fake_stream = MagicMock()
    fake_stream.average_rate = 1
    fake_stream.frames = 1
    fake_stream.duration = None
    fake_stream.time_base = None

    fake_container = MagicMock()
    fake_container.streams.video = [fake_stream]
    fake_container.decode.return_value = iter([fake_frame])

    cfg = AssetExtractorConfig(frames_per_second=1.0)
    cfg.max_frame_pixels = max_pixels

    with patch("linc_crucible.core.asset_extractor.av.open", return_value=fake_container):
        with pytest.raises(ValueError, match=rf"^{expected_msg}$"):
            _decode_frames(tmp_path / "huge.mp4", tmp_path, cfg)
        fake_frame.to_image.assert_not_called()
    fake_container.close.assert_called_once()

    # Exact-cap frame (8000x5000 == 40_000_000) and 1x1 frame are allowed
    for ok_w, ok_h in ((8000, 5000), (1, 1)):
        ok_frame = MagicMock()
        ok_frame.width = ok_w
        ok_frame.height = ok_h
        ok_container = MagicMock()
        ok_container.streams.video = [fake_stream]
        ok_container.decode.return_value = iter([ok_frame])
        with patch("linc_crucible.core.asset_extractor.av.open", return_value=ok_container):
            out = _decode_frames(tmp_path / "ok.mp4", tmp_path, cfg)
            assert len(out) == 1
            ok_frame.to_image.assert_called_once()


def test_contract_oversized_frame_returns_documented_error_json_via_mcp_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] crucible_extract_assets returns {error, assets: [], fallback_used: false}
    when a frame exceeds max_frame_pixels."""
    linc_home = tmp_path / "linc_home"
    monkeypatch.setenv("LINC_HOME", str(linc_home))
    vid = _write_synthetic_mp4(tmp_path / "clip.mp4", num_frames=2, fps=1)

    fake_frame = MagicMock()
    fake_frame.width = 20000
    fake_frame.height = 20000
    fake_stream = MagicMock()
    fake_stream.average_rate = 1
    fake_stream.frames = 1
    fake_container = MagicMock()
    fake_container.streams.video = [fake_stream]
    fake_container.decode.return_value = iter([fake_frame])

    with patch("linc_crucible.core.asset_extractor.av.open", return_value=fake_container):
        raw = asyncio.run(crucible_extract_assets(video_path=str(vid), max_assets=2))

    payload = json.loads(raw)
    assert payload == {
        "error": "frame size 20000x20000 exceeds the cap",
        "assets": [],
        "fallback_used": False,
    }
    fake_frame.to_image.assert_not_called()


def test_contract_filter_frames_skips_decompression_bomb_and_counts_reasons(
    tmp_path: Path,
) -> None:
    """[contract] filter_frames given a path whose Image.open raises DecompressionBombError
    skips it and reports {'decompression_bomb': 1}."""
    bomb_path = tmp_path / "bomb.png"
    bomb_path.write_bytes(b"fake")

    with patch(
        "linc_crucible.core.asset_extractor.Image.open",
        side_effect=Image.DecompressionBombError("Image size exceeds limit"),
    ):
        survivors = filter_frames([bomb_path], blur_threshold=100.0, phash_threshold=5)

    assert survivors == []
    assert survivors.skipped_frames == {"decompression_bomb": 1}

    skipped: dict[str, int] = {}
    with patch(
        "linc_crucible.core.asset_extractor.Image.open",
        side_effect=Image.DecompressionBombError("Image size exceeds limit"),
    ):
        survivors_with_out = filter_frames(
            [bomb_path], blur_threshold=100.0, phash_threshold=5, skipped_frames=skipped
        )
    assert survivors_with_out == []
    assert skipped == {"decompression_bomb": 1}

    # Also test all skip reasons together (decompression_bomb, unreadable, blurry, duplicate)
    unreadable_path = tmp_path / "bad.png"
    unreadable_path.write_bytes(b"not-a-png")
    blurry_path = tmp_path / "blurry.png"
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(blurry_path)
    sharp1 = _make_sharp_png(tmp_path / "sharp1.png", seed=0)
    sharp1_dup = _make_sharp_png(tmp_path / "sharp1_dup.png", seed=0)
    sharp2 = _make_sharp_png(tmp_path / "sharp2.png", seed=3)

    real_open = Image.open

    def selective_open(fp, *args, **kwargs):
        if Path(fp) == bomb_path:
            raise Image.DecompressionBombError("bomb")
        return real_open(fp, *args, **kwargs)

    all_skipped: dict[str, int] = {}
    with patch("linc_crucible.core.asset_extractor.Image.open", side_effect=selective_open):
        kept = filter_frames(
            [bomb_path, bomb_path, unreadable_path, blurry_path, sharp1, sharp1_dup, sharp2],
            blur_threshold=100.0,
            phash_threshold=5,
            skipped_frames=all_skipped,
        )

    assert kept == [sharp1, sharp2]
    assert all_skipped == {
        "decompression_bomb": 2,
        "unreadable": 1,
        "blurry": 1,
        "duplicate": 1,
    }
    assert kept.skipped_frames == all_skipped

    # Exact blur_threshold survives; phash distance == phash_threshold is duplicate
    class _FakeHash:
        def __init__(self, val: int) -> None:
            self.val = val

        def __sub__(self, other: _FakeHash) -> int:
            return abs(self.val - other.val)

    boundary_skipped: dict[str, int] = {}
    with (
        patch("linc_crucible.core.asset_extractor._laplacian_variance", return_value=100.0),
        patch(
            "linc_crucible.core.asset_extractor.imagehash.phash",
            side_effect=[_FakeHash(0), _FakeHash(5), _FakeHash(6)],
        ),
    ):
        boundary_kept = filter_frames(
            [sharp1, sharp1_dup, sharp2],
            blur_threshold=100.0,
            phash_threshold=5,
            skipped_frames=boundary_skipped,
        )
    assert boundary_kept == [sharp1, sharp2]
    assert boundary_skipped == {"duplicate": 1}


def test_contract_extract_assets_includes_skipped_frames_in_result_and_manifest(
    tmp_path: Path,
) -> None:
    """[contract] extract_assets returns skipped_frames: {reason: count} in its result and
    manifest when frames are skipped."""
    vid = _write_synthetic_mp4(tmp_path / "clip.mp4", num_frames=3, fps=1)
    out_base = tmp_path / "out"
    cfg = AssetExtractorConfig(frames_per_second=1.0, max_assets=2, base_output_dir=out_base)

    real_open = Image.open

    def bomb_on_first_frame(fp, *args, **kwargs):
        if Path(fp).name == "frame_0000.png":
            raise Image.DecompressionBombError("bomb on frame 0")
        return real_open(fp, *args, **kwargs)

    with patch("linc_crucible.core.asset_extractor.Image.open", side_effect=bomb_on_first_frame):
        result = extract_assets(str(vid), cfg)

    assert result["skipped_frames"] == {"decompression_bomb": 1}
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["skipped_frames"] == {"decompression_bomb": 1}


def test_default_config_on_normal_short_clip_behaves_as_before_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[test] The default config on a normal short clip behaves as before (pin)."""
    linc_home = tmp_path / "linc_home"
    monkeypatch.setenv("LINC_HOME", str(linc_home))

    vid = _write_synthetic_mp4(tmp_path / "short.mp4", num_frames=2, fps=1)
    result = extract_assets(str(vid))

    assert result["fallback_used"] is True
    assert len(result["assets"]) == 2
    assert result["assets"][0]["filename"] == "short_frame_0000.png"
    assert {v["name"] for v in result["assets"][0]["variants"]} == {"x", "ig"}
    assert result.get("skipped_frames", {}) == {}
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["video"] == "short.mp4"
    assert manifest["fallback_used"] is True
    assert len(manifest["assets"]) == 2
