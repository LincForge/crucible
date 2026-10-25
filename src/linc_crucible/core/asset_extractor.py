import json
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import av
import imagehash
import numpy as np
import structlog
from PIL import Image, UnidentifiedImageError

from linc_crucible.core.asset_config import AssetExtractorConfig
from linc_crucible.core.state_paths import ensure_private_dir

logger = structlog.get_logger().bind(domain="asset_extractor")

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


def filter_frames(
    frame_paths: list[Path], blur_threshold: float, phash_threshold: int
) -> list[Path]:
    survivors = []
    seen_hashes = []

    for path in frame_paths:
        # Load image
        try:
            img_pil = Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError):
            continue

        # Check blur
        gray = np.asarray(img_pil.convert("L"), dtype=np.float64)
        variance = _laplacian_variance(gray)
        if variance < blur_threshold:
            continue

        # Check dedup
        phash = imagehash.phash(img_pil)

        is_duplicate = False
        for seen in seen_hashes:
            if phash - seen <= phash_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            seen_hashes.append(phash)
            survivors.append(path)

    return survivors


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


def _decode_frames(
    video_path_obj: Path, temp_dir: Path, config: AssetExtractorConfig
) -> list[Path]:
    """Decode the video at ``frames_per_second``, writing sampled frames to ``temp_dir``."""
    extracted: list[Path] = []
    container = av.open(str(video_path_obj))
    try:
        stream = container.streams.video[0]
        fps = stream.average_rate
        frame_interval = max(1, int(fps / config.frames_per_second)) if fps else 30
        for count, frame in enumerate(container.decode(stream)):
            if count % frame_interval == 0:
                temp_path = temp_dir / f"frame_{count:04d}.png"
                frame.to_image().save(temp_path)
                extracted.append(temp_path)
    finally:
        container.close()
    return extracted


def _finalize(
    ranking: dict, stem: str, output_dir: Path, config: AssetExtractorConfig, video_name: str
) -> dict:
    """Generate variants, move originals out of temp, write manifest. Paths are relative."""
    from linc_crucible.core.asset_variants import make_variants

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
        "generated_at": datetime.now(UTC).isoformat(),
        "fallback_used": ranking["fallback_used"],
        "assets": manifest_assets,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "assets": result_assets,
        "fallback_used": ranking["fallback_used"],
        "manifest_path": str(manifest_path),
    }


def extract_assets(video_path: str, config: AssetExtractorConfig = None) -> dict:
    if config is None:
        config = AssetExtractorConfig()

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video {video_path} not found")

    stem = video_path_obj.stem
    output_dir = config.base_output_dir / stem
    ensure_private_dir(output_dir)  # also creates base_output_dir
    # temp scratch on the SAME filesystem as the output (cheap rename, no /tmp pressure)
    temp_dir = Path(tempfile.mkdtemp(prefix="crucible_assets_", dir=config.base_output_dir))

    try:
        extracted = _decode_frames(video_path_obj, temp_dir, config)
        survivors = filter_frames(extracted, config.blur_threshold, config.phash_threshold)
        if not survivors:
            raise ValueError(
                f"Video {video_path} quality too low: no sharp, distinct frames "
                "survived pre-filtering"
            )
        ranking = rank_frames(survivors, temp_dir, config, config.max_assets)
        return _finalize(ranking, stem, output_dir, config, video_path_obj.name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
