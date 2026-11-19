"""Contract and regression tests for crucible_extract_assets input and output confinement."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
import structlog.testing
from click.testing import CliRunner

import linc_crucible.core.asset_extractor as ae
from linc_crucible.asset_tools import _extract_assets_json, crucible_extract_assets
from linc_crucible.cli import cli
from linc_crucible.core.asset_config import AssetExtractorConfig
from linc_crucible.core.asset_extractor import (
    _decode_frames,
    _ensure_inside_base,
    _has_manifest_collision,
    _redact_path_text,
    _resolve_output_dir,
    _resolve_regular_video,
    _safe_stem,
    extract_assets,
)


@pytest.fixture(autouse=True)
def _capture_structlog():
    with structlog.testing.capture_logs():
        yield


@pytest.fixture(autouse=True)
def _disable_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic fallback ranking so tests never shell out to the claude CLI."""
    monkeypatch.setattr(ae, "_ai_available", lambda: False)


def _write_audio_only_mp4(path: Path) -> Path:
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("aac", rate=44100)
        samples = np.zeros((1, 1024), dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
        frame.sample_rate = 44100
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return path


def _write_valid_2frame_mp4(path: Path, *, seed: int = 0) -> Path:
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        yy, xx = np.indices((64, 64))
        for idx in range(2):
            checker = (((xx // 4 + yy // 4 + idx + seed) % 2) * 255).astype(np.uint8)
            band = np.where((xx < 32) ^ (yy < (16 + 8 * idx + 4 * seed)), 255, 0).astype(
                np.uint8
            )
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


def _files_outside_assets(linc_home: Path) -> list[Path]:
    crucible_dir = linc_home / "crucible"
    if not crucible_dir.exists():
        return []
    assets_dir = (crucible_dir / "assets").resolve()
    outside: list[Path] = []
    for root, _dirs, files in os.walk(crucible_dir):
        root_path = Path(root)
        for fname in files:
            fpath = (root_path / fname).resolve()
            if not fpath.is_relative_to(assets_dir):
                outside.append(fpath)
    return outside


def _all_files_under(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    found: list[Path] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            found.append(Path(root) / fname)
    return found


@pytest.mark.parametrize(
    ("case_kind", "expected_error_substring"),
    [
        ("parent_traversal", "video_path must be a regular file"),
        ("root_dir", "video_path must be a regular file"),
        ("directory", "video_path must be a regular file"),
        ("fifo", "video_path must be a regular file"),
        ("audio_only", "no video stream in audio.mp4"),
        ("text_mp4", "unreadable video: InvalidDataError"),
    ],
    ids=["parent_traversal", "root_dir", "directory", "fifo", "audio_only", "text_mp4"],
)
def test_contract_hostile_or_unreadable_video_path_returns_error_json_and_confines_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_kind: str,
    expected_error_substring: str,
) -> None:
    """[contract] crucible_extract_assets(video_path=X) for hostile/unreadable inputs returns
    JSON with non-empty 'error', assets == [], fallback_used is False, does not raise, returns
    within 2 s (including FIFO), and leaves zero files outside $LINC_HOME/crucible/assets."""
    linc_home = tmp_path / "linc_home"
    linc_home.mkdir()
    monkeypatch.setenv("LINC_HOME", str(linc_home))

    work = tmp_path / "inputs"
    work.mkdir()

    if case_kind == "parent_traversal":
        target = str(linc_home) + "/.."
    elif case_kind == "root_dir":
        target = "/"
    elif case_kind == "directory":
        sub = work / "adir"
        sub.mkdir()
        target = str(sub)
    elif case_kind == "fifo":
        fifo = work / "pipe.mp4"
        os.mkfifo(fifo)
        target = str(fifo)
    elif case_kind == "audio_only":
        audio = _write_audio_only_mp4(work / "audio.mp4")
        target = str(audio)
    elif case_kind == "text_mp4":
        txt = work / "clip.mp4"
        txt.write_text("not a video container at all\n", encoding="utf-8")
        target = str(txt)
    else:  # pragma: no cover
        raise AssertionError(case_kind)

    raw = asyncio.run(
        asyncio.wait_for(crucible_extract_assets(video_path=target, max_assets=3), timeout=2.0)
    )
    payload = json.loads(raw)
    assert set(payload.keys()) == {"error", "assets", "fallback_used"}
    assert isinstance(payload.get("error"), str) and payload["error"]
    assert expected_error_substring in payload["error"]
    assert payload["assets"] == []
    assert payload["fallback_used"] is False
    assert _files_outside_assets(linc_home) == []
    assert _all_files_under(linc_home / "crucible") == []
    if case_kind in {"parent_traversal", "root_dir", "directory", "fifo"}:
        assert not (linc_home / "crucible").exists()


def test_two_videos_same_basename_produce_distinct_output_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[test] Two different valid 2-frame videos both named clip.mp4 in different folders
    produce two distinct output folders and both manifests survive."""
    linc_home = tmp_path / "linc_home"
    monkeypatch.setenv("LINC_HOME", str(linc_home))

    dir_a = tmp_path / "folder_a"
    dir_b = tmp_path / "folder_b"
    dir_a.mkdir()
    dir_b.mkdir()
    vid_a = _write_valid_2frame_mp4(dir_a / "clip.mp4", seed=0)
    vid_b = _write_valid_2frame_mp4(dir_b / "clip.mp4", seed=1)

    res_a = json.loads(asyncio.run(crucible_extract_assets(video_path=str(vid_a), max_assets=2)))
    res_b = json.loads(asyncio.run(crucible_extract_assets(video_path=str(vid_b), max_assets=2)))

    assert "error" not in res_a
    assert "error" not in res_b
    manifest_a = Path(res_a["manifest_path"])
    manifest_b = Path(res_b["manifest_path"])
    assert manifest_a != manifest_b
    assert manifest_a.parent != manifest_b.parent
    assert manifest_a.exists()
    assert manifest_b.exists()

    expected_suffix = hashlib.sha256(str(vid_b.resolve()).encode("utf-8")).hexdigest()[:8]
    assert manifest_a.parent.name == "clip"
    assert manifest_b.parent.name == f"clip_{expected_suffix}"

    data_a = json.loads(manifest_a.read_text(encoding="utf-8"))
    data_b = json.loads(manifest_b.read_text(encoding="utf-8"))
    assert data_a["video"] == "clip.mp4"
    assert data_b["video"] == "clip.mp4"
    assert data_a["video_path"] == str(vid_a.resolve())
    assert data_b["video_path"] == str(vid_b.resolve())

    # Re-running the first video reuses its own folder rather than colliding with itself
    res_a_again = json.loads(
        asyncio.run(crucible_extract_assets(video_path=str(vid_a), max_assets=2))
    )
    assert Path(res_a_again["manifest_path"]) == manifest_a


def test_valid_video_result_shape_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[test] A valid video produces the documented result shape."""
    linc_home = tmp_path / "linc_home"
    monkeypatch.setenv("LINC_HOME", str(linc_home))

    vid = _write_valid_2frame_mp4(tmp_path / "demo.mp4", seed=0)
    raw = asyncio.run(crucible_extract_assets(video_path=str(vid), max_assets=2))
    payload = json.loads(raw)

    assert set(payload.keys()) == {"assets", "fallback_used", "manifest_path", "skipped_frames"}
    assert payload["skipped_frames"] == {}
    assert payload["fallback_used"] is True
    assert len(payload["assets"]) >= 1
    first = payload["assets"][0]
    assert set(first.keys()) == {"filename", "rationale", "variants"}
    assert first["filename"] == "demo_frame_0000.png"
    assert {v["name"] for v in first["variants"]} == {"x", "ig"}
    manifest_path = Path(payload["manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["video"] == "demo.mp4"
    assert manifest["video_path"] == str(vid.resolve())
    assert manifest["fallback_used"] is True
    assert len(manifest["assets"]) == len(payload["assets"])


def test_safe_stem_filtering_length_and_refusals(tmp_path: Path) -> None:
    """[test] _safe_stem filters to [A-Za-z0-9._-], caps at 100 chars, refuses empty/dot stems."""
    assert _safe_stem("Az09._-ok") == "Az09._-ok"
    assert _safe_stem("my clip@#1!.mp4") == "my_clip__1_.mp4"
    assert _safe_stem("x" * 100) == "x" * 100
    assert _safe_stem("x" * 101) == "x" * 100
    assert len(_safe_stem("a" * 150)) == 100

    for bad in ("", ".", ".."):
        with pytest.raises(ValueError, match="stem must be a non-empty safe name"):
            _safe_stem(bad)

    # Regular files named '..mp4' (stem '.') and '...mp4' (stem '..') are refused before mkdir
    base = tmp_path / "assets_out"
    for fname in ("..mp4", "...mp4"):
        f = tmp_path / fname
        f.write_bytes(b"data")
        with pytest.raises(ValueError, match="stem must be a non-empty safe name"):
            extract_assets(str(f), AssetExtractorConfig(base_output_dir=base))
        assert not base.exists()


def test_ensure_inside_base_and_symlink_escape_refused(tmp_path: Path) -> None:
    """[test] output_dir resolving outside base_output_dir (or equal to it) is refused."""
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="resolves outside base_output_dir"):
        _ensure_inside_base(base, base)
    with pytest.raises(ValueError, match="resolves outside base_output_dir"):
        _ensure_inside_base(outside, base)

    # Symlinked base_output_dir resolves cleanly for a valid child
    sym_base = tmp_path / "sym_base"
    sym_base.symlink_to(base)
    _ensure_inside_base(sym_base / "valid_child", sym_base)
    sym_base.unlink()

    symlink_dir = base / "clip"
    symlink_dir.symlink_to(outside)
    vid = _write_valid_2frame_mp4(tmp_path / "clip.mp4", seed=0)
    with pytest.raises(ValueError, match="resolves outside base_output_dir"):
        extract_assets(str(vid), AssetExtractorConfig(base_output_dir=base))
    assert list(outside.iterdir()) == []

    # Suffixed collision path resolving outside base is also refused
    symlink_dir.unlink()
    real_clip = base / "clip"
    real_clip.mkdir()
    (real_clip / "manifest.json").write_text(json.dumps({"video": "/other/clip.mp4"}))
    suffix = hashlib.sha256(str(vid.resolve()).encode("utf-8")).hexdigest()[:8]
    (base / f"clip_{suffix}").symlink_to(outside)
    with pytest.raises(ValueError, match="resolves outside base_output_dir"):
        _resolve_output_dir("clip", vid.resolve(), base)


def test_has_manifest_collision_variants(tmp_path: Path) -> None:
    """[test] _has_manifest_collision handles missing, corrupt, non-dict, legacy, and matching."""
    vid_a = (tmp_path / "a" / "clip.mp4").resolve()
    vid_b = (tmp_path / "b" / "clip.mp4").resolve()
    mpath = tmp_path / "manifest.json"

    assert _has_manifest_collision(mpath, vid_a) is False

    mpath.write_text("{not valid json", encoding="utf-8")
    assert _has_manifest_collision(mpath, vid_a) is True

    mpath.write_text("[1, 2, 3]", encoding="utf-8")
    assert _has_manifest_collision(mpath, vid_a) is True

    # Legacy manifest with only 'video' field
    mpath.write_text(json.dumps({"video": "clip.mp4"}), encoding="utf-8")
    assert _has_manifest_collision(mpath, vid_a) is True

    mpath.write_text(json.dumps({"video": str(vid_a)}), encoding="utf-8")
    assert _has_manifest_collision(mpath, vid_a) is False
    assert _has_manifest_collision(mpath, vid_b) is True

    # Current manifest with 'video_path'
    mpath.write_text(
        json.dumps({"video": "clip.mp4", "video_path": str(vid_a)}), encoding="utf-8"
    )
    assert _has_manifest_collision(mpath, vid_a) is False
    assert _has_manifest_collision(mpath, vid_b) is True

    # Unreadable manifest (OSError on read_text)
    with patch.object(Path, "read_text", side_effect=OSError("perm denied")):
        assert _has_manifest_collision(mpath, vid_a) is True


def test_resolve_regular_video_oserror_and_decode_frames_oserror(tmp_path: Path) -> None:
    """[test] Non-FileNotFoundError OSError in os.stat or av.open becomes ValueError."""
    sub = tmp_path / "sub"
    sub.mkdir()
    f = sub / "clip.mp4"
    f.write_bytes(b"x")
    unnormalized = str(sub / ".." / "sub" / "clip.mp4")
    assert _resolve_regular_video(unnormalized) == f.resolve()

    with (
        patch("linc_crucible.core.asset_extractor.os.stat", side_effect=PermissionError("denied")),
        pytest.raises(ValueError, match=r"^unreadable video: PermissionError$"),
    ):
        _resolve_regular_video(str(f))

    with (
        patch("linc_crucible.core.asset_extractor.av.open", side_effect=OSError("io fail")),
        pytest.raises(ValueError, match=r"^unreadable video: OSError$"),
    ):
        _decode_frames(f, tmp_path, AssetExtractorConfig())


def test_decode_frames_closes_container_on_missing_video_stream(tmp_path: Path) -> None:
    """[test] _decode_frames closes the PyAV container even when streams.video is empty."""
    fake_container = MagicMock()
    fake_container.streams.video = ()
    with (
        patch("linc_crucible.core.asset_extractor.av.open", return_value=fake_container),
        pytest.raises(ValueError, match=r"^no video stream in sample\.mp4$"),
    ):
        _decode_frames(tmp_path / "sample.mp4", tmp_path, AssetExtractorConfig())
    fake_container.close.assert_called_once()


def test_error_text_redacts_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] Error text passes paths through redact_home (#211)."""
    fake_home = tmp_path / "secret_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("LINC_HOME", str(fake_home / ".linc"))

    missing = fake_home / "videos" / "missing.mp4"
    with pytest.raises(FileNotFoundError) as exc_missing:
        extract_assets(str(missing))
    assert str(fake_home) not in str(exc_missing.value)
    assert "~/videos/missing.mp4" in str(exc_missing.value)

    raw = asyncio.run(_extract_assets_json(str(missing), 3))
    payload = json.loads(raw)
    assert str(fake_home) not in payload["error"]
    assert "~/videos/missing.mp4" in payload["error"]
    assert payload["assets"] == []
    assert payload["fallback_used"] is False

    # Low-quality error also redacts HOME directly and via _extract_assets_json
    low_q = fake_home / "videos"
    low_q.mkdir()
    vid = low_q / "blurry.mp4"
    vid.write_bytes(b"fake")
    with (
        patch.object(ae, "_decode_frames", return_value=[tmp_path / "f.png"]),
        patch.object(ae, "filter_frames", return_value=[]),
    ):
        with pytest.raises(ValueError) as exc_low:
            extract_assets(str(vid))
        assert str(fake_home) not in str(exc_low.value)
        assert "~/videos/blurry.mp4" in str(exc_low.value)

        raw_low = asyncio.run(_extract_assets_json(str(vid), 3))
    payload_low = json.loads(raw_low)
    assert str(fake_home) not in payload_low["error"]
    assert "~/videos/blurry.mp4" in payload_low["error"]

    # _extract_assets_json redacts even if extract_assets raises an unredacted path
    with patch(
        "linc_crucible.asset_tools.extract_assets",
        side_effect=ValueError(f"boom at {fake_home}/raw/clip.mp4"),
    ):
        raw_patched = asyncio.run(_extract_assets_json(str(vid), 3))
    payload_patched = json.loads(raw_patched)
    assert str(fake_home) not in payload_patched["error"]
    assert "~/raw/clip.mp4" in payload_patched["error"]

    # When Path.home() raises RuntimeError, _redact_path_text degrades cleanly
    monkeypatch.setattr(
        Path, "home", classmethod(lambda _cls: (_ for _ in ()).throw(RuntimeError("no HOME")))
    )
    assert _redact_path_text("/some/path/video.mp4") == "/some/path/video.mp4"


def test_cli_extract_assets_reflects_error_json_on_hostile_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[test] CLI command reflected from crucible_extract_assets exits 1 with error envelope."""
    linc_home = tmp_path / "linc_home"
    monkeypatch.setenv("LINC_HOME", str(linc_home))
    txt = tmp_path / "clip.mp4"
    txt.write_text("not a video", encoding="utf-8")

    res = CliRunner().invoke(cli, ["extract-assets", "--video-path", str(txt)])
    assert res.exit_code == 1
    assert res.stdout == ""
    err = json.loads(res.stderr)
    assert "unreadable video: InvalidDataError" in err["error"]["message"]
