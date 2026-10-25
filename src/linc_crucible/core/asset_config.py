from dataclasses import dataclass, field
from pathlib import Path

from .state_paths import state_path


def _default_assets_dir() -> Path:
    """Return the default asset output directory, honouring ``LINC_HOME``.

    Returns:
        ``<state root>/assets`` via :func:`state_paths.state_path` — canonical
        ``$LINC_HOME/crucible/assets`` (default ``~/.linc/crucible/assets``).
        Re-evaluated on every call so that ``os.environ`` monkeypatching in
        tests takes effect.
    """
    return state_path("assets")


@dataclass
class AssetExtractorConfig:
    frames_per_second: float = 1.0
    max_assets: int = 5
    blur_threshold: float = 100.0  # Laplacian variance threshold
    phash_threshold: int = 5  # Max Hamming distance for duplicates
    base_output_dir: Path = field(default_factory=_default_assets_dir)
    # Social/store format variants (target WxH in pixels)
    variant_formats: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"x": (1200, 630), "ig": (1080, 1080)}
    )
    # AI ranking knobs (budget_usd intentionally omitted — meaningless under subscription billing)
    max_ai_candidates: int = 24  # hard cap on frames sent to the vision model
    ai_timeout_seconds: int = 180  # wall-clock ceiling on the claude subprocess
