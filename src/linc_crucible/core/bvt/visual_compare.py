"""SSIM visual comparison kernel — root-agnostic.

Pure pixel comparison: takes two explicit image paths and returns a verdict.
Knows nothing about baseline storage roots — ``BaselineManager.compare()`` owns
golden resolution. Dependencies (Pillow, numpy, scikit-image) are lazy-imported
and a missing dependency yields a LOUD result (``ok=False, available=False``),
never a silent pass.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

PASS_THRESHOLD = 0.95
REVIEW_THRESHOLD = 0.85


def compute_ssim(
    golden_path: Path,
    candidate_path: Path,
    *,
    pass_threshold: float = PASS_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
    diff_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare two screenshots by grayscale SSIM.

    Returns a result dict; never raises on missing deps / missing files.
    Keys: ``ok``, ``available``, ``skipped``, and (when a comparison ran)
    ``score``, ``verdict``; ``deps_missing`` on the missing-dependency path;
    ``diff_path``/``current_path`` when ``diff_dir`` is provided. The
    deps-missing path is the ONLY one that returns ``ok=False`` AND
    ``available=False``, so a caller checking either flag can never mistake a
    missing-deps skip for a pass.
    """
    try:
        import numpy as np
        from PIL import Image, UnidentifiedImageError
        from skimage.metrics import structural_similarity as ssim
    except ImportError as exc:
        log.warning("visual_compare_deps_missing", error=str(exc))
        return {
            "ok": False,
            "available": False,
            "deps_missing": True,
            "skipped": True,
            "reason": f"Visual comparison dependencies not installed: {exc}",
        }

    candidate = Path(candidate_path)
    if not candidate.exists():
        return _missing_image("Candidate", candidate)

    golden = Path(golden_path)
    if not golden.exists():
        return _missing_image("Golden", golden)

    try:
        with Image.open(golden) as img_g:
            golden_size = img_g.size
            golden_gray = img_g.convert("L")
            golden_arr = np.array(golden_gray)
    except (UnidentifiedImageError, OSError) as exc:
        return _ssim_error(str(exc), candidate=candidate, golden=golden)

    try:
        with Image.open(candidate) as img_c:
            candidate_size = img_c.size
            candidate_gray = img_c.convert("L")
            if candidate_size != golden_size:
                cw, ch = candidate_size
                gw, gh = golden_size
                return _ssim_error(
                    f"geometry mismatch {cw}x{ch} vs {gw}x{gh}",
                    candidate=candidate,
                    golden=golden,
                )
            candidate_arr = np.array(candidate_gray)
    except (UnidentifiedImageError, OSError) as exc:
        return _ssim_error(str(exc), candidate=candidate, golden=golden)

    score, diff = ssim(golden_arr, candidate_arr, full=True, data_range=255)

    verdict = _verdict_for(float(score), pass_threshold, review_threshold)

    result: dict[str, Any] = {
        "ok": True,
        "available": True,
        "skipped": False,
        "score": round(float(score), 4),
        "verdict": verdict,
    }

    if diff_dir is not None:
        result.update(_write_diff(diff, candidate, Path(diff_dir), np, Image))

    log.info("visual_compare_done", score=result["score"], verdict=verdict)
    return result


def _missing_image(label: str, path: Path) -> dict[str, Any]:
    """Loud (but non-deps) FAIL result for an absent golden or candidate."""
    return {
        "ok": False,
        "available": True,
        "skipped": False,
        "verdict": "FAIL",
        "score": 0.0,
        "reason": f"{label} image not found: {path}",
    }


def _ssim_error(
    error: str,
    candidate: Path | str,
    golden: Path | str,
) -> dict[str, Any]:
    """Build an error result dict for unreadable images or geometry mismatch."""
    return {
        "ok": False,
        "verdict": "ERROR",
        "error": error,
        "candidate": str(candidate),
        "golden": str(golden),
    }



def _verdict_for(score: float, pass_threshold: float, review_threshold: float) -> str:
    """Map an SSIM score to PASS / REVIEW / FAIL."""
    if score >= pass_threshold:
        return "PASS"
    if score >= review_threshold:
        return "REVIEW"
    return "FAIL"


def _write_diff(
    diff: Any, candidate: Path, diff_dir: Path, np: Any, image_mod: Any
) -> dict[str, str]:
    """Persist the inverted diff image and a copy of the candidate. Returns paths."""
    diff_dir.mkdir(parents=True, exist_ok=True)
    diff_normalized = ((1.0 - diff) * 255).astype(np.uint8)
    diff_path = diff_dir / "diff.png"
    image_mod.fromarray(diff_normalized).save(diff_path, "PNG")
    current_ref = diff_dir / "current.png"
    shutil.copy2(candidate, current_ref)
    return {"diff_path": str(diff_path), "current_path": str(current_ref)}
