"""Quality scorer — weighted 0-100 composite score with JSON persistence.

Scores saved to ``~/.linc/crucible/scores/<project>/<ISO_timestamp>.json``
(legacy ``~/.crucible/scores`` read if only it exists — see
:mod:`linc_crucible.core.state_paths`).

Score formula — weights are normalised dynamically from ``BASE_WEIGHTS``
raw importances.  Active dimensions are determined at runtime and their raw
values are divided by their sum so weights always total 1.0.

  Dimensions:
    tests:    pass_rate * 100     (raw importance 77)
    lint:     max(0, 100 - N*2)  (raw importance 63)
    security: MobSF score 0-100  (raw importance 60, present in "full" mode)
    perf:     APK size → 0-100   (raw importance 35)
    a11y:     a11y score 0-100   (raw importance 35)

  Visual (when baselines exist):
    visual:   avg SSIM → 0-100   weight 0.15 (redistributed from other dims)
    SSIM >= 0.95 = 100, <= 0.70 = 0, linear between

Ship gate: composite >= 85
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

from .state_paths import state_path
from .store_names import contained_dir, prepare_dir

logger = structlog.get_logger().bind(domain="build_pipeline")


def _default_scores_dir() -> Path:
    """Return the default quality scores directory, honouring ``LINC_HOME``.

    Returns:
        ``<state root>/scores`` via :func:`state_paths.state_path` — canonical
        ``$LINC_HOME/crucible/scores`` (default ``~/.linc/crucible/scores``),
        legacy ``~/.crucible/scores`` when only it exists. Re-evaluated on
        every call so that ``os.environ`` monkeypatching in tests takes effect.
    """
    return state_path("scores")


SHIP_GATE_THRESHOLD = 85

# Bumped whenever the SCORING SEMANTICS (not just the on-disk shape) change in
# a way that makes an older persisted record untrustworthy as-is.
# Pre-v2 records were computed before attempted-unmeasured dimensions closed
# the ship gate (a crashed toolchain could read composite=100.0,
# ship_gate=True) — they are stamped `stale` on load (see `_load_score`) so a
# future consumer (e.g. crucible_preflight) can choose to warn or refuse
# rather than trust a pre-fix score as authoritative. Bump this again the
# next time a scoring-semantics change (not just a field addition) needs the
# same staleness signal.
SCORE_SCHEMA_VERSION = 2

# Raw importance values per dimension.  To add a new scoring dimension, add one
# entry here — compute() normalises active dimensions to sum to 1.0 at runtime.
#
# Values chosen so that normalisation reproduces the exact weights previously
# used by the hard-coded combinatorial table (verified by regression tests).
BASE_WEIGHTS: dict[str, float] = {
    "tests": 77,
    "lint": 63,
    "security": 60,
    "code_quality": 55,
    "perf": 35,
    "a11y": 35,
    "a11y_structural": 40,
    "a11y_rules": 35,
}

# APK/IPA size scoring thresholds (MB)
APK_SIZE_EXCELLENT = 10.0
APK_SIZE_POOR = 30.0

# SSIM scoring thresholds
SSIM_EXCELLENT = 0.95
SSIM_POOR = 0.70


def apk_size_to_score(size_mb: float) -> float:
    """Convert artifact size to a 0-100 score.

    <= 10 MB = 100, >= 30 MB = 0, linear between.
    """
    if size_mb <= APK_SIZE_EXCELLENT:
        return 100.0
    if size_mb >= APK_SIZE_POOR:
        return 0.0
    return 100.0 * (APK_SIZE_POOR - size_mb) / (APK_SIZE_POOR - APK_SIZE_EXCELLENT)


def ssim_to_score(ssim: float) -> float:
    """Convert SSIM similarity to a 0-100 score.

    >= 0.95 = 100, <= 0.70 = 0, linear between.
    """
    if ssim >= SSIM_EXCELLENT:
        return 100.0
    if ssim <= SSIM_POOR:
        return 0.0
    return 100.0 * (ssim - SSIM_POOR) / (SSIM_EXCELLENT - SSIM_POOR)


# Accessibility severity penalties
A11Y_PENALTY_CRITICAL = 10
A11Y_PENALTY_MAJOR = 5
A11Y_PENALTY_MINOR = 2


def a11y_issues_to_score(critical: int = 0, major: int = 0, minor: int = 0) -> float:
    """Convert accessibility issue counts to a 0-100 score.

    Deducts per issue: critical=10, major=5, minor=2.
    """
    deduction = (
        critical * A11Y_PENALTY_CRITICAL + major * A11Y_PENALTY_MAJOR + minor * A11Y_PENALTY_MINOR
    )
    return max(0.0, 100.0 - deduction)


@dataclass
class ScoreDimension:
    """A single dimension of the quality score."""

    name: str
    raw_value: float  # The input (e.g. pass_rate, lint_issues, mobsf_score)
    score: float  # Normalized 0-100
    weight: float
    weighted_score: float = 0.0


@dataclass
class QualityScore:
    """Complete quality score for a project."""

    project: str
    timestamp: str
    mode: str  # "quick" or "full"
    composite: float = 0.0
    ship_gate: bool = False
    dimensions: list[ScoreDimension] = field(default_factory=list)
    commit_sha: str | None = None
    # Names of dimensions that were ATTEMPTED (a command was configured and
    # invoked) but produced no usable score (crashed, refused, or exited
    # without parseable output). Non-empty
    # here means `compute()` forced `ship_gate=False` regardless of the
    # composite of whatever sibling dimensions survived.
    attempted_unmeasured: list[str] = field(default_factory=list)
    # True when this record was LOADED from disk and its `schema_version`
    # predates `SCORE_SCHEMA_VERSION` (or is absent entirely) — i.e. it was
    # computed under pre-fix scoring semantics and should not be trusted as
    # authoritative without a re-assay. Always False for a freshly computed
    # score (see `compute()`); only `_load_score()` sets it True.
    stale: bool = False

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "project": self.project,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "composite": round(self.composite, 1),
            "ship_gate": self.ship_gate,
            "commit_sha": self.commit_sha,
            "schema_version": SCORE_SCHEMA_VERSION,
            "attempted_unmeasured": list(self.attempted_unmeasured),
            "dimensions": [
                {
                    "name": d.name,
                    "raw_value": d.raw_value,
                    "score": round(d.score, 1),
                    "weight": d.weight,
                    "weighted_score": round(d.weighted_score, 1),
                }
                for d in self.dimensions
            ],
        }


class QualityScorer:
    """Computes and persists quality scores for LINC projects."""

    def __init__(self, scores_dir: Path | None = None) -> None:
        env_path = os.environ.get("CRUCIBLE_SCORES_DIR")
        if scores_dir:
            self._scores_dir = scores_dir
        elif env_path:
            self._scores_dir = Path(env_path)
        else:
            self._scores_dir = _default_scores_dir()

    @property
    def scores_dir(self) -> Path:
        """Directory this scorer persists and reads score records from."""
        return self._scores_dir

    @staticmethod
    def _build_mode(
        has_security: bool,
        active_dims: list[str],
        has_visual: bool,
        partial_unmeasured: list[str],
    ) -> str:
        """Assemble the `mode` tag string (e.g. "full+a11y+partial-unmeasured").

        Pure string assembly, no scoring semantics — extracted out of
        `compute()` to keep that function's cyclomatic complexity bounded.
        """
        mode_parts = ["full" if (has_security or "security" in partial_unmeasured) else "quick"]
        for dim in ("code_quality", "perf", "a11y", "a11y_structural", "a11y_rules"):
            if dim in active_dims:
                mode_parts.append(dim)
        mode = "+".join(mode_parts)

        if has_visual:
            mode += "+visual"

        # A dimension that was ATTEMPTED but never measured (crashed,
        # refused, unparseable) must stay visible even though it has already
        # been excluded from `active_dims`/`dimensions` — otherwise a
        # sibling dimension's green score silently renormalises to fill the
        # gap and nothing downstream can tell the broken toolchain happened.
        if partial_unmeasured:
            mode += f"+partial-unmeasured({', '.join(partial_unmeasured)})"

        return mode

    def _build_dimensions(
        self,
        test_pass_rate: float | None,
        lint_issues: int | None,
        security_score: float | None,
        code_quality_score: float | None,
        perf_score: float | None,
        a11y_score: float | None,
        a11y_structural_score: float | None,
        a11y_rules_score: float | None,
        visual_scores: list[float] | None,
        weights: dict[str, float],
        visual_weight: float,
    ) -> list[ScoreDimension]:
        dimensions: list[ScoreDimension] = []

        # Tests dimension — skipped when not measured (no test command configured).
        # A missing dimension must never be silently scored as a perfect pass.
        if test_pass_rate is not None:
            test_score = max(0.0, min(100.0, test_pass_rate * 100))
            dimensions.append(
                ScoreDimension(
                    "tests",
                    test_pass_rate,
                    test_score,
                    weights.get("tests", 0),
                    test_score * weights.get("tests", 0),
                )
            )

        # Lint dimension — skipped when not measured (no lint command configured).
        if lint_issues is not None:
            lint_score = max(0.0, 100.0 - lint_issues * 2)
            dimensions.append(
                ScoreDimension(
                    "lint",
                    float(lint_issues),
                    lint_score,
                    weights.get("lint", 0),
                    lint_score * weights.get("lint", 0),
                )
            )

        # Security dimension (full mode only)
        if security_score is not None:
            sec_score = max(0.0, min(100.0, security_score))
            dimensions.append(
                ScoreDimension(
                    "security",
                    security_score,
                    sec_score,
                    weights.get("security", 0),
                    sec_score * weights.get("security", 0),
                )
            )

        # Code quality dimension (CRAP score, 0-100)
        if code_quality_score is not None:
            cq_normalized = max(0.0, min(100.0, code_quality_score))
            dimensions.append(
                ScoreDimension(
                    "code_quality",
                    code_quality_score,
                    cq_normalized,
                    weights.get("code_quality", 0),
                    cq_normalized * weights.get("code_quality", 0),
                )
            )

        # Performance dimension (artifact size)
        if perf_score is not None:
            perf_normalized = apk_size_to_score(perf_score)
            dimensions.append(
                ScoreDimension(
                    "perf",
                    perf_score,
                    perf_normalized,
                    weights.get("perf", 0),
                    perf_normalized * weights.get("perf", 0),
                )
            )

        # Accessibility dimension
        if a11y_score is not None:
            a11y_normalized = max(0.0, min(100.0, a11y_score))
            dimensions.append(
                ScoreDimension(
                    "a11y",
                    a11y_score,
                    a11y_normalized,
                    weights.get("a11y", 0),
                    a11y_normalized * weights.get("a11y", 0),
                )
            )

        # A11y tree structural diff dimension
        if a11y_structural_score is not None:
            a11y_structural_normalized = max(0.0, min(100.0, a11y_structural_score))
            dimensions.append(
                ScoreDimension(
                    "a11y_structural",
                    a11y_structural_score,
                    a11y_structural_normalized,
                    weights.get("a11y_structural", 0),
                    a11y_structural_normalized * weights.get("a11y_structural", 0),
                )
            )

        # A11y WCAG rules checker dimension
        if a11y_rules_score is not None:
            a11y_rules_normalized = max(0.0, min(100.0, a11y_rules_score))
            dimensions.append(
                ScoreDimension(
                    "a11y_rules",
                    a11y_rules_score,
                    a11y_rules_normalized,
                    weights.get("a11y_rules", 0),
                    a11y_rules_normalized * weights.get("a11y_rules", 0),
                )
            )

        # Visual dimension (SSIM-based)
        if visual_scores:
            avg_ssim = sum(visual_scores) / len(visual_scores)
            visual_normalized = ssim_to_score(avg_ssim)
            dimensions.append(
                ScoreDimension(
                    "visual",
                    round(avg_ssim, 4),
                    visual_normalized,
                    visual_weight,
                    visual_normalized * visual_weight,
                )
            )

        return dimensions

    def compute(
        self,
        project: str,
        test_pass_rate: float | None,
        lint_issues: int | None,
        security_score: float | None = None,
        code_quality_score: float | None = None,
        perf_score: float | None = None,
        visual_scores: list[float] | None = None,
        visual_weight: float = 0.15,
        a11y_score: float | None = None,
        a11y_structural_score: float | None = None,
        a11y_rules_score: float | None = None,
        commit_sha: str | None = None,
        attempted_unmeasured: list[str] | None = None,
    ) -> QualityScore:
        """Compute a quality score.

        Args:
            project: Project name.
            test_pass_rate: Test pass rate as a fraction (0.0 - 1.0), or None
                when tests were not measured (no test command configured). A
                None dimension is excluded — never scored as a perfect pass.
            lint_issues: Number of lint issues found, or None when lint was not
                measured (no lint command configured).
            security_score: Optional MobSF security score (0-100).
            perf_score: Optional artifact size in MB (converted to 0-100 internally).
            visual_scores: Optional list of SSIM values (0.0-1.0) per screen.
                None or empty list skips visual dimension.
            visual_weight: Fraction of composite allocated to visual (default 0.15).
            a11y_score: Optional pre-computed accessibility score (0-100).
            a11y_structural_score: Optional a11y tree structural diff score (0-100).
            a11y_rules_score: Optional WCAG rules checker score (0-100).
            commit_sha: Optional commit SHA.
            attempted_unmeasured: Names of dimensions that were ATTEMPTED (a
                command was configured and actually invoked) but produced no
                usable score — as opposed to a dimension that was simply never
                configured. A `None`/excluded dimension caused by "not
                configured" is fine to renormalise around; one caused by
                "attempted and crashed/unparseable" is NOT — it forces
                `ship_gate=False` regardless of what any sibling dimension
                scored. Callers pass the SAME
                dimension names here that they also passed `None` for above;
                this list is purely an additional signal, not a substitute.

        Returns:
            QualityScore with composite, dimensions, and ship gate status.
        """
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Normalise once up front so both return paths below see the same
        # concrete list (never `None`) — a plain list identity, not something
        # re-derived per branch.
        partial_unmeasured = list(attempted_unmeasured) if attempted_unmeasured else []

        has_security = security_score is not None
        has_visual = bool(visual_scores)

        # Build active dimension list and normalise raw importances to 1.0.
        # tests/lint are only active when actually measured — an unmeasured
        # dimension is excluded so it can't masquerade as a perfect pass.
        active_dims: list[str] = []
        for dim, present in [
            ("tests", test_pass_rate is not None),
            ("lint", lint_issues is not None),
            ("security", security_score is not None),
            ("code_quality", code_quality_score is not None),
            ("perf", perf_score is not None),
            ("a11y", a11y_score is not None),
            ("a11y_structural", a11y_structural_score is not None),
            ("a11y_rules", a11y_rules_score is not None),
        ]:
            if present:
                active_dims.append(dim)

        # Nothing was measured (e.g. a project with no lint/test command and no
        # other dimensions). A blank assay must read as 0/not-shippable, never
        # as a false-green 100 — "nothing checked" is not "everything passed".
        if not active_dims:
            return QualityScore(
                project=project,
                timestamp=now,
                mode="unmeasured",
                composite=0.0,
                ship_gate=False,
                dimensions=[],
                commit_sha=commit_sha,
                attempted_unmeasured=partial_unmeasured,
            )

        raw = {d: BASE_WEIGHTS[d] for d in active_dims}
        total = sum(raw.values())
        weights = {d: v / total for d, v in raw.items()}

        mode = self._build_mode(has_security, active_dims, has_visual, partial_unmeasured)

        if has_visual:
            scale = 1.0 - visual_weight
            weights = {k: v * scale for k, v in weights.items()}

        dimensions = self._build_dimensions(
            test_pass_rate,
            lint_issues,
            security_score,
            code_quality_score,
            perf_score,
            a11y_score,
            a11y_structural_score,
            a11y_rules_score,
            visual_scores,
            weights,
            visual_weight,
        )

        composite = sum(d.weighted_score for d in dimensions)
        # gate on the displayed (rounded) composite so PASS/FAIL never contradicts
        # the shown score — a raw composite in [84.95, 85.0) displays as 85.0 and
        # must not fail the gate. Do NOT revert this to the raw `composite`.
        rounded = round(composite, 1)

        # A broken toolchain must never produce a shippable score just because
        # a sibling dimension happened to be green — force the gate closed
        # whenever ANY attempted dimension failed to measure, regardless of
        # the composite.
        ship_gate = (rounded >= SHIP_GATE_THRESHOLD) and not partial_unmeasured

        result = QualityScore(
            project=project,
            timestamp=now,
            mode=mode,
            composite=rounded,
            ship_gate=ship_gate,
            dimensions=dimensions,
            commit_sha=commit_sha,
            attempted_unmeasured=partial_unmeasured,
        )

        return result

    def save(self, score: QualityScore) -> Path:
        """Persist a quality score to disk.

        Saved to ``~/.linc/crucible/scores/<project>/<timestamp>.json``.

        Raises:
            InvalidStoreName: If ``score.project`` is not a safe store name or
                its dir resolves outside the store (checked before writing).
        """
        project_dir = prepare_dir(self._scores_dir, score.project)

        # Use filesystem-safe timestamp
        safe_ts = score.timestamp.replace(":", "-")
        filepath = project_dir / f"{safe_ts}.json"

        with open(filepath, "w") as f:
            json.dump(score.to_dict(), f, indent=2)

        logger.info("score_saved", path=str(filepath))
        return filepath

    def load_latest(self, project: str) -> QualityScore | None:
        """Load the most recent score for a project.

        Raises:
            InvalidStoreName: If ``project`` is not a safe store name.
        """
        project_dir = contained_dir(self._scores_dir, project)
        if not project_dir.is_dir():
            return None

        files = sorted(project_dir.glob("*.json"), reverse=True)
        if not files:
            return None

        return self._load_score(files[0])

    def load_history(self, project: str, limit: int = 10) -> list[QualityScore]:
        """Load score history for a project, most recent first.

        Raises:
            InvalidStoreName: If ``project`` is not a safe store name.
        """
        project_dir = contained_dir(self._scores_dir, project)
        if not project_dir.is_dir():
            return []

        files = sorted(project_dir.glob("*.json"), reverse=True)[:limit]
        scores = []
        for f in files:
            score = self._load_score(f)
            if score:
                scores.append(score)
        return scores

    def _load_score(self, filepath: Path) -> QualityScore | None:
        """Load a single score from a JSON file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("score_file_read_failed", path=str(filepath), error=str(exc))
            return None

        dimensions = []
        for dim_data in data.get("dimensions", []):
            dimensions.append(
                ScoreDimension(
                    name=dim_data.get("name", ""),
                    raw_value=dim_data.get("raw_value", 0.0),
                    score=dim_data.get("score", 0.0),
                    weight=dim_data.get("weight", 0.0),
                    weighted_score=dim_data.get("weighted_score", 0.0),
                )
            )

        # A record written before schema v2 carries no `schema_version` at all
        # and was computed before attempted-unmeasured dimensions closed the
        # ship gate — mark it `stale` so a consumer (e.g. crucible_preflight)
        # can choose to warn or refuse rather than silently trust it as
        # authoritative. `data.get(..., 1)` treats a missing key as version 1
        # (pre-fix); only a record written by THIS `to_dict()` (which always
        # stamps the current `SCORE_SCHEMA_VERSION`) reads as fresh.
        schema_version = data.get("schema_version", 1)

        return QualityScore(
            project=data.get("project", ""),
            timestamp=data.get("timestamp", ""),
            mode=data.get("mode", "quick"),
            composite=data.get("composite", 0.0),
            ship_gate=data.get("ship_gate", False),
            dimensions=dimensions,
            commit_sha=data.get("commit_sha", None),
            attempted_unmeasured=list(data.get("attempted_unmeasured", [])),
            stale=schema_version < SCORE_SCHEMA_VERSION,
        )
