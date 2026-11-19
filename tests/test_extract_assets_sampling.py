"""Boundary tests for asset-extraction frame sampling, size refusal, and blur input."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import linc_crucible.core.asset_extractor as ae
from linc_crucible.core.asset_config import AssetExtractorConfig
from linc_crucible.core.asset_extractor import (
    _classify_frame,
    _estimated_stream_frames,
    _sampling_interval,
    _validate_frame_dimensions,
)


def _stream(*, frames: int = 0, duration=None, time_base=None, average_rate=None):
    return SimpleNamespace(
        frames=frames, duration=duration, time_base=time_base, average_rate=average_rate
    )


@pytest.mark.parametrize(
    ("stream", "fps", "expected"),
    [
        (_stream(frames=0, duration=400, time_base=None), 10, 0),
        (_stream(frames=0, duration=None, time_base=Fraction(1, 10)), 10, 0),
        (_stream(frames=0, duration=None, time_base=None), 10, 0),
        (_stream(frames=0, duration=400, time_base=Fraction(1, 10)), None, 0),
        (_stream(frames=0, duration=0, time_base=Fraction(1, 10)), 10, 0),
        (_stream(frames=0, duration=400, time_base=Fraction(1, 10)), 10, 400),
        (_stream(frames=1, duration=400, time_base=Fraction(1, 10)), 10, 1),
        (_stream(frames=7, duration=None, time_base=None), None, 7),
    ],
    ids=[
        "duration_without_time_base",
        "time_base_without_duration",
        "nothing_known",
        "duration_and_time_base_without_fps",
        "zero_duration",
        "duration_estimate",
        "one_frame_wins_over_duration",
        "frames_known_without_fps",
    ],
)
def test_estimated_stream_frames_requires_every_estimate_input(
    stream: SimpleNamespace, fps, expected: int
) -> None:
    """[test] The duration estimate needs duration, time_base, and fps all present."""
    assert _estimated_stream_frames(stream, fps) == expected


def test_contract_negative_duration_never_caps_sampling() -> None:
    """[contract] A hostile negative duration yields no positive estimate and no cap."""
    stream = _stream(frames=0, duration=-400, time_base=Fraction(1, 10), average_rate=10)
    assert _estimated_stream_frames(stream, 10) <= 0
    cfg = AssetExtractorConfig(frames_per_second=100.0, max_sampled_frames=5)
    assert _sampling_interval(stream, cfg) == (1, False)


@pytest.mark.parametrize(
    ("frames", "max_sampled", "expected"),
    [
        (40, 1, (40, True)),
        (40, 0, (1, False)),
        (40, -1, (1, False)),
        (1, 1, (1, False)),
        (2, 1, (2, True)),
        (0, 1, (1, False)),
        (41, 5, (9, True)),
    ],
    ids=[
        "cap_one_samples_once",
        "cap_zero_no_division",
        "cap_negative_no_division",
        "one_frame_cap_one",
        "two_frames_cap_one",
        "unknown_length_cap_one",
        "interval_rounds_up",
    ],
)
def test_sampling_interval_cap_boundaries(
    frames: int, max_sampled: int, expected: tuple[int, bool]
) -> None:
    """[test] Cap interval boundaries at max_sampled_frames 1/0/-1 and total_frames 0/1/2."""
    stream = _stream(frames=frames, average_rate=10)
    cfg = AssetExtractorConfig(frames_per_second=100.0, max_sampled_frames=max_sampled)
    assert _sampling_interval(stream, cfg) == expected


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (1, 0, "frame size 1x0 exceeds the cap"),
        (0, 1, "frame size 0x1 exceeds the cap"),
        (101, 100, "frame size 101x100 exceeds the cap"),
    ],
    ids=["one_by_zero", "zero_by_one", "one_pixel_row_over"],
)
def test_validate_frame_dimensions_refuses_edge_sizes(width: int, height: int, message: str):
    """[contract] Zero-sized and one-over-cap frames are refused with the documented text."""
    frame = SimpleNamespace(width=width, height=height)
    with pytest.raises(ValueError, match=rf"^{message}$"):
        _validate_frame_dimensions(frame, 10_000)


def test_validate_frame_dimensions_accepts_exact_cap() -> None:
    """[test] A frame exactly at max_frame_pixels is accepted."""
    _validate_frame_dimensions(SimpleNamespace(width=100, height=100), 10_000)


@pytest.mark.parametrize(
    "mode", ["RGB", "RGBA", "P", "L", "LA", "I;16"], ids=["rgb", "rgba", "p", "l", "la", "i16"]
)
def test_classify_frame_feeds_two_dimensional_luma_to_blur_check(tmp_path: Path, mode: str):
    """[test] The blur check always receives a 2-D float64 luma array, whatever the PNG mode."""
    yy, xx = np.indices((32, 32))
    rgb = np.stack([(xx * 8) % 256, (yy * 8) % 256, ((xx + yy) * 4) % 256], axis=-1)
    img = Image.fromarray(rgb.astype(np.uint8))
    path = tmp_path / f"frame_{mode.replace(';', '_')}.png"
    if mode == "I;16":
        Image.fromarray((rgb[..., 0] * 3).astype(np.uint16)).save(path)
    else:
        img.convert(mode).save(path)
    expected = np.asarray(Image.open(path).convert("L"), dtype=np.float64)

    seen: list[np.ndarray] = []

    def spy(gray: np.ndarray) -> float:
        seen.append(gray)
        return 1e9

    with patch.object(ae, "_laplacian_variance", side_effect=spy):
        assert _classify_frame(path, 1.0, 0, []) is None

    assert len(seen) == 1
    assert seen[0].ndim == 2
    assert seen[0].dtype == np.float64
    np.testing.assert_array_equal(seen[0], expected)
