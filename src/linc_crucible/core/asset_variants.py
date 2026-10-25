"""Letterbox-pad format variants for marketing assets.

Each selected asset is rendered into social/store target aspect ratios by
fitting the whole screenshot (never cropping it) onto a blurred, cover-resized
copy of itself. Deterministic, PIL-only -- no AI, CI-safe.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

_BLUR_RADIUS = 20


def _cover_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to fill ``w`` by ``h`` (cover) and center-crop the overflow."""
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - w) // 2, (new_h - h) // 2
    return resized.crop((left, top, left + w, top + h))


def _fit_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to fit within ``w`` by ``h`` (contain), preserving aspect ratio."""
    src_w, src_h = img.size
    scale = min(w / src_w, h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def make_variants(
    image_path: Path,
    out_stem: str,
    out_dir: Path,
    formats: dict[str, tuple[int, int]],
) -> list[dict]:
    """Render letterbox-padded variants of ``image_path`` for each format.

    Returns a list of ``{"name", "width", "height", "path"}`` dicts where
    ``path`` is the variant's filename **relative to** ``out_dir`` (so the
    manifest stays machine-portable).
    """
    src = Image.open(image_path).convert("RGB")
    variants: list[dict] = []
    for name, (w, h) in formats.items():
        background = _cover_resize(src, w, h).filter(ImageFilter.GaussianBlur(_BLUR_RADIUS))
        foreground = _fit_resize(src, w, h)
        offset = ((w - foreground.width) // 2, (h - foreground.height) // 2)
        background.paste(foreground, offset)
        filename = f"{out_stem}_{name}.png"
        background.save(out_dir / filename)
        variants.append({"name": name, "width": w, "height": h, "path": filename})
    return variants
