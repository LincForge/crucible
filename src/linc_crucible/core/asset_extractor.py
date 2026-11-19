import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import av
import imagehash
import numpy as np
import structlog
from PIL import Image, UnidentifiedImageError

from linc_crucible._redaction import redact_home
from linc_crucible.core.asset_config import AssetExtractorConfig
from linc_crucible.core.state_paths import ensure_private_dir

logger = structlog.get_logger().bind(domain="asset_extractor")

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_STEM_LEN = 100

_ASO_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["filename"],
                },
            },
            "fallback_used": {"type": "boolean"},
        },
        "required": ["assets"],
    }
)


def evenly_spaced(items: list, n: int) -> list:
    """Select ``n`` evenly-spaced items across ``items`` (deterministic).

    ``n >= len(items)`` returns all items; ``n == 1`` returns the middle item;
    ``n <= 0`` or empty input returns ``[]``. First and last are always included
    when ``n >= 2``.
    """
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def _laplacian_variance(gray: np.ndarray) -> float:
    """Return the variance of the 3x3 discrete Laplacian with reflect-101 border padding."""
    padded = np.pad(np.asarray(gray, dtype=np.float64), 1, mode="reflect")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(laplacian.var())


class _FilteredFrames(list):
    """List of surviving frame paths carrying ``skipped_frames`` reason counts."""

    def __init__(self, survivors: list[Path], skipped_frames: dict[str, int]) -> None:
        super().__init__(survivors)
        self.skipped_frames = skipped_frames


def _classify_frame(
    path: Path, blur_threshold: float, phash_threshold: int, seen_hashes: list
) -> str | None:
    """Return None when ``path`` survives filtering, or the skip reason key."""
    try:
        luma = Image.open(path).convert("L")
    except Image.DecompressionBombError:
        return "decompression_bomb"
    except (OSError, UnidentifiedImageError):
        return "unreadable"

    if _laplacian_variance(np.asarray(luma, dtype=np.float64)) < blur_threshold:
        return "blurry"

    # phash reduces to luma internally, so hashing the L image matches hashing RGB.
    phash = imagehash.phash(luma)
    if any(phash - seen <= phash_threshold for seen in seen_hashes):
        return "duplicate"

    seen_hashes.append(phash)
    return None


def filter_frames(
    frame_paths: list[Path],
    blur_threshold: float,
    phash_threshold: int,
    skipped_frames: dict[str, int] | None = None,
) -> list[Path]:
    survivors: list[Path] = []
    seen_hashes: list = []
    counts = skipped_frames if skipped_frames is not None else {}

    for path in frame_paths:
        reason = _classify_frame(path, blur_threshold, phash_threshold, seen_hashes)
        if reason is None:
            survivors.append(path)
        else:
            counts[reason] = counts.get(reason, 0) + 1

    return _FilteredFrames(survivors, counts)


def parse_ai_response(response_text: str) -> dict:
    try:
        data = json.loads(response_text)
        if "assets" in data:
            return data
    except json.JSONDecodeError:
        pass
    # Fallback structure
    return {"assets": [], "fallback_used": True}


def _build_aso_prompt(names: list[str], max_assets: int) -> str:
    listing = ", ".join(names)
    return (
        "You are an App Store Optimization (ASO) expert. The directory contains "
        f"sequential frames from an app screen recording, named: {listing}. "
        f"Read each image, then select the top {max_assets} frames that represent "
        "distinct, core features of the app. Exclude mid-transition animations, "
        "loading screens, and blank states. Respond ONLY with the JSON schema provided."
    )


def _fallback_picks(survivors: list[Path], max_assets: int) -> dict:
    picks = evenly_spaced(survivors, max_assets)
    return {
        "picks": [{"path": p, "rationale": "Fallback selection (evenly spaced)"} for p in picks],
        "fallback_used": True,
    }


def _ai_available() -> bool:
    """Return True when the optional ``linc_ai`` extra imports cleanly.

    Any import-time failure (missing extra, or a broken install raising a non-ImportError)
    counts as unavailable so ``rank_frames`` keeps its never-raises contract.
    """
    try:
        import linc_ai  # noqa: F401
    except ImportError:
        return False
    except Exception as exc:  # broken linc_ai install must degrade, never propagate
        logger.warning("asset_ai_import_failed", error=f"{type(exc).__name__}: {exc}")
        return False
    return True


def _rank_with_retries(request, retries: int = 2) -> dict:
    """Call the AI adapter, parsing + retrying. Raises ValueError if no assets parse."""
    from linc_ai import complete

    for attempt in range(retries + 1):
        result = complete(request)
        text = result.output if result.success else ""
        parsed = parse_ai_response(text)
        if parsed.get("assets"):
            return parsed
        if attempt < retries:
            time.sleep(2**attempt)  # 1s, 2s
    raise ValueError("AI response had no parseable assets after retries")


def _validate_picks(data: dict, candidates: list[Path]) -> list[dict]:
    """Keep only assets whose filename matches a real candidate; drop hallucinations."""
    by_name = {p.name: p for p in candidates}
    picks: list[dict] = []
    for asset in data.get("assets", []):
        fname = asset.get("filename")
        if fname in by_name:
            picks.append({"path": by_name[fname], "rationale": asset.get("rationale", "")})
        else:
            logger.warning("asset_hallucinated_filename_dropped", filename=fname)
    return picks


def rank_frames(
    survivors: list[Path], temp_dir: Path, config: AssetExtractorConfig, max_assets: int
) -> dict:
    """Rank survivor frames via Claude-Vision (CLI adapter); degrade to fallback on any failure.

    Returns ``{"picks": [{"path": Path, "rationale": str}], "fallback_used": bool}``.
    Never raises -- every failure mode routes to the deterministic evenly-spaced fallback.
    """
    if not survivors:
        return {"picks": [], "fallback_used": True}
    if not _ai_available():
        logger.warning(
            "asset_ai_unavailable",
            hint=(
                "AI frame ranking needs the optional linc-ai package, which is not published; "
                "using evenly-spaced frames"
            ),
        )
        return _fallback_picks(survivors, max_assets)

    candidates = survivors[: config.max_ai_candidates]
    if len(survivors) > config.max_ai_candidates:
        logger.warning(
            "asset_candidates_truncated",
            survivors=len(survivors),
            cap=config.max_ai_candidates,
        )

    try:
        from linc_ai import ClaudeCLIRequest, ModelTier

        names = [p.name for p in candidates]
        request = ClaudeCLIRequest(
            prompt=_build_aso_prompt(names, max_assets),
            tools=["Read"],
            allowed_tools=["Read"],
            context_dirs=[str(temp_dir)],
            json_schema=_ASO_SCHEMA,
            model_tier=ModelTier.HIGH,
            max_turns=len(candidates) + 2,  # one Read per frame + headroom (derived, not fixed)
            timeout_seconds=config.ai_timeout_seconds,
            # skips all permission prompts; with tools=["Read"] the only built-in
            # tool is Read, which can read any file this user can read; the frame
            # dir is added with --add-dir
            dangerously_skip_permissions=True,
        )
        data = _rank_with_retries(request)
        picks = _validate_picks(data, candidates)
        if not picks:
            logger.warning("asset_ai_no_valid_picks")
            return _fallback_picks(survivors, max_assets)
        return {"picks": picks[:max_assets], "fallback_used": False}
    except Exception as exc:  # broad by design: degrade to the deterministic fallback
        logger.warning(
            "asset_ai_ranking_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return _fallback_picks(survivors, max_assets)


def _redact_path_text(text: str) -> str:
    """Redact the user's home directory from error text via :func:`redact_home`."""
    try:
        home = Path.home()
    except RuntimeError:
        home = None
    return redact_home(text, home)


def _resolve_regular_video(video_path: str) -> Path:
    """Resolve ``video_path`` and verify it is a regular file via :func:`os.stat`."""
    resolved = Path(video_path).resolve()
    try:
        st = os.stat(resolved)
    except FileNotFoundError as exc:
        raise FileNotFoundError(_redact_path_text(f"Video {video_path} not found")) from exc
    except OSError as exc:
        raise ValueError(f"unreadable video: {type(exc).__name__}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("video_path must be a regular file")
    return resolved


def _safe_stem(raw_stem: str) -> str:
    """Filter ``raw_stem`` to ``[A-Za-z0-9._-]`` (max 100 chars), refusing empty or dot stems."""
    stem = _SAFE_STEM_RE.sub("_", raw_stem)[:_MAX_STEM_LEN]
    if not stem or stem in {".", ".."}:
        raise ValueError("video_path stem must be a non-empty safe name")
    return stem


def _ensure_inside_base(output_dir: Path, base_output_dir: Path) -> None:
    """Prove ``output_dir.resolve()`` is strictly inside ``base_output_dir.resolve()``."""
    base_resolved = base_output_dir.resolve()
    out_resolved = output_dir.resolve()
    if out_resolved == base_resolved or not out_resolved.is_relative_to(base_resolved):
        raise ValueError("output_dir resolves outside base_output_dir")


def _has_manifest_collision(manifest_path: Path, resolved_video: Path) -> bool:
    """Return True when ``manifest_path`` exists and records a different source video."""
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(data, dict):
        return True
    recorded = data.get("video_path") or data.get("video")
    return recorded != str(resolved_video)


def _resolve_output_dir(stem: str, resolved_video: Path, base_output_dir: Path) -> Path:
    """Return an ``output_dir`` proven inside ``base_output_dir``, disambiguating collisions."""
    output_dir = base_output_dir / stem
    _ensure_inside_base(output_dir, base_output_dir)
    if _has_manifest_collision(output_dir / "manifest.json", resolved_video):
        suffix = hashlib.sha256(str(resolved_video).encode("utf-8")).hexdigest()[:8]
        output_dir = base_output_dir / f"{stem}_{suffix}"
        _ensure_inside_base(output_dir, base_output_dir)
    return output_dir


def _estimated_stream_frames(stream, fps) -> int:
    """Return the known or duration-estimated frame count for ``stream`` (<= 0 when unknown).

    PyAV always exposes ``frames`` (0 when unknown), ``duration`` and ``time_base`` (None when
    unknown); a non-positive result means "length unknown" and never caps sampling.
    """
    if stream.frames > 0:
        return stream.frames
    if stream.duration and stream.time_base and fps:
        return math.ceil(stream.duration * stream.time_base * fps)
    return 0


def _sampling_interval(stream, config: AssetExtractorConfig) -> tuple[int, bool]:
    """Return ``(frame_interval, capped_by_estimate)`` for ``stream`` under ``config``."""
    fps = stream.average_rate
    fps_interval = max(1, int(fps / config.frames_per_second)) if fps else 30
    if config.max_sampled_frames > 0:
        # a non-positive estimate gives cap_interval <= 0, which never beats fps_interval >= 1
        cap_interval = math.ceil(_estimated_stream_frames(stream, fps) / config.max_sampled_frames)
        if cap_interval > fps_interval:
            return cap_interval, True
    return fps_interval, False


def _validate_frame_dimensions(frame, max_frame_pixels: int) -> None:
    """Refuse non-positive or oversized frame dimensions before pixel conversion."""
    width, height = frame.width, frame.height
    if width <= 0 or height <= 0 or width * height > max_frame_pixels:
        raise ValueError(f"frame size {width}x{height} exceeds the cap")


def _decode_frames(
    video_path_obj: Path, temp_dir: Path, config: AssetExtractorConfig
) -> list[Path]:
    """Decode the video at ``frames_per_second``, writing sampled frames to ``temp_dir``."""
    extracted: list[Path] = []
    try:
        container = av.open(str(video_path_obj))
    except (av.error.FFmpegError, OSError) as exc:
        raise ValueError(f"unreadable video: {type(exc).__name__}") from exc
    try:
        if not container.streams.video:
            raise ValueError(f"no video stream in {video_path_obj.name}")
        stream = container.streams.video[0]
        frame_interval, capped = _sampling_interval(stream, config)
        for count, frame in enumerate(container.decode(stream)):
            if count % frame_interval != 0:
                continue
            if len(extracted) >= config.max_sampled_frames:
                capped = True
                break
            _validate_frame_dimensions(frame, config.max_frame_pixels)
            temp_path = temp_dir / f"frame_{count:04d}.png"
            frame.to_image().save(temp_path)
            extracted.append(temp_path)
        if capped:
            logger.warning(
                "asset_frames_capped",
                cap=config.max_sampled_frames,
                max_sampled_frames=config.max_sampled_frames,
            )
    finally:
        container.close()
    return extracted


def _finalize(
    ranking: dict,
    stem: str,
    output_dir: Path,
    config: AssetExtractorConfig,
    video_name: str,
    video_path: str = "",
    skipped_frames: dict[str, int] | None = None,
) -> dict:
    """Generate variants, move originals out of temp, write manifest. Paths are relative."""
    from linc_crucible.core.asset_variants import make_variants

    skipped = dict(skipped_frames) if skipped_frames else {}
    manifest_assets: list[dict] = []
    result_assets: list[dict] = []
    for idx, pick in enumerate(ranking["picks"]):
        out_stem = f"{stem}_frame_{idx:04d}"
        # variants read the original while it is still in the temp dir (ordering matters)
        variants = make_variants(pick["path"], out_stem, output_dir, config.variant_formats)
        original_name = f"{out_stem}.png"
        shutil.move(str(pick["path"]), str(output_dir / original_name))
        manifest_assets.append(
            {
                "index": idx,
                "original": original_name,
                "rationale": pick["rationale"],
                "variants": variants,
            }
        )
        result_assets.append(
            {"filename": original_name, "rationale": pick["rationale"], "variants": variants}
        )

    manifest = {
        "video": video_name,
        "video_path": video_path,
        "generated_at": datetime.now(UTC).isoformat(),
        "fallback_used": ranking["fallback_used"],
        "skipped_frames": skipped,
        "assets": manifest_assets,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "assets": result_assets,
        "fallback_used": ranking["fallback_used"],
        "manifest_path": str(manifest_path),
        "skipped_frames": skipped,
    }


def extract_assets(video_path: str, config: AssetExtractorConfig = None) -> dict:
    if config is None:
        config = AssetExtractorConfig()

    video_path_obj = _resolve_regular_video(video_path)
    stem = _safe_stem(video_path_obj.stem)
    output_dir = _resolve_output_dir(stem, video_path_obj, config.base_output_dir)
    ensure_private_dir(output_dir)  # also creates base_output_dir
    # temp scratch on the SAME filesystem as the output (cheap rename, no /tmp pressure)
    temp_dir = Path(tempfile.mkdtemp(prefix="crucible_assets_", dir=config.base_output_dir))

    try:
        extracted = _decode_frames(video_path_obj, temp_dir, config)
        skipped_frames: dict[str, int] = {}
        survivors = filter_frames(
            extracted, config.blur_threshold, config.phash_threshold, skipped_frames
        )
        if not survivors:
            raise ValueError(
                _redact_path_text(
                    f"Video {video_path} quality too low: no sharp, distinct frames "
                    "survived pre-filtering"
                )
            )
        ranking = rank_frames(survivors, temp_dir, config, config.max_assets)
        return _finalize(
            ranking,
            stem,
            output_dir,
            config,
            video_path_obj.name,
            video_path=str(video_path_obj),
            skipped_frames=skipped_frames,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
