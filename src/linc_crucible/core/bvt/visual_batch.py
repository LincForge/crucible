"""Multi-screen visual-compare orchestration at the BVT-runner layer.

Recovers the pre-T2 batch logic but delegates golden resolution + SSIM to
``BaselineManager.compare`` (the T2 single entry point). The pixel kernel
(``visual_compare.compute_ssim``) stays root-agnostic and untouched; this
module only fans a list of screenshots across the manager and aggregates one
verdict. Imports flow downward only (runner layer -> BaselineManager).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from linc_crucible.core.store_names import InvalidStoreName

if TYPE_CHECKING:
    from linc_crucible.core.baseline import BaselineManager


def aggregate_verdicts(comparisons: list[dict[str, Any]]) -> str:
    """Aggregate one verdict from per-screenshot compare results.

    ``DEPS_MISSING`` short-circuits FIRST: if ANY comparison reports
    ``deps_missing``, the whole run is ``DEPS_MISSING`` so a broken environment
    can never produce a partially-blind ``PASS``. Otherwise only non-skipped
    comparisons vote: ``FAIL`` if any FAIL, ``REVIEW`` if any REVIEW, ``PASS``
    if all PASS. If nothing voted: an empty ``comparisons`` list (nothing was
    asked to run) is ``SKIPPED``; a non-empty list where every entry is a
    no-baseline skip is ``UNBASELINED`` — a coverage gap, not "nothing ran" —
    so a whole suite with zero goldens (including goldens silently orphaned by
    a screenshot-filename change) reads as a gap instead of a silent pass-through.
    """
    if any(c.get("deps_missing") for c in comparisons):
        return "DEPS_MISSING"
    verdicts = [c["verdict"] for c in comparisons if not c.get("skipped")]
    if not verdicts:
        if comparisons and any(c.get("no_baseline") for c in comparisons):
            return "UNBASELINED"
        return "SKIPPED"
    if any(v == "ERROR" for v in verdicts):
        return "ERROR"
    if any(v == "FAIL" for v in verdicts):
        return "FAIL"
    if any(v == "REVIEW" for v in verdicts):
        return "REVIEW"
    return "PASS"


def _compare_one(
    manager: BaselineManager, project: str, screen: str, screenshot: Path
) -> dict[str, Any]:
    """One screenshot's comparison; a name the store allowlist refuses is an ERROR entry.

    Stems come from ``step_handlers._safe_file_stem`` (already ``[A-Za-z0-9._-]``,
    leading dots stripped), so only a stem starting with ``-`` or longer than
    128 chars — or a hostile ``project`` — reaches this branch.
    """
    try:
        result = manager.compare(project, screen, screenshot)
    except InvalidStoreName:
        return {"screen": screen, "verdict": "ERROR", "reason": "invalid baseline name"}
    return {"screen": screen, **result}


def compare_run(
    manager: BaselineManager,
    project: str,
    screenshots: list[str],
) -> dict[str, Any]:
    """Compare each screenshot against its latest golden via ``manager``.

    The baseline lookup key for each screenshot is ``Path(p).stem`` — i.e. the
    ``{test_name}_{label}`` BVT screenshot filename. THE STEM IS THE BASELINE
    KEY: if the BVT screenshot filename format ever changes, every existing
    golden is silently orphaned (-> ``no_baseline`` -> ``SKIPPED``), so a
    format change must be paired with a baseline re-capture.

    Returns ``{comparisons, aggregate_verdict, available}``. ``available`` is
    ``False`` iff the aggregate is ``DEPS_MISSING``. A screen whose name the
    store refuses is recorded as ``verdict="ERROR"`` instead of raising.
    """
    comparisons = [_compare_one(manager, project, Path(p).stem, Path(p)) for p in screenshots]

    aggregate = aggregate_verdicts(comparisons)
    return {
        "comparisons": comparisons,
        "aggregate_verdict": aggregate,
        "available": aggregate != "DEPS_MISSING",
    }
