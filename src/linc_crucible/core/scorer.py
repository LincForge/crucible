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
import math
import os
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from linc_crucible._bounded_read import RegistryTooLargeError

from .state_paths import state_path
from .store_names import contained_dir, prepare_dir

logger = structlog.get_logger().bind(domain="build_pipeline")

# A score dated further ahead than this was saved under a wrong clock (or restored from
# elsewhere). Its filename sorts after every real later run, so it is never "latest".
SCORE_FUTURE_SLACK = timedelta(minutes=5)


def parse_score_timestamp(raw_timestamp: str | None) -> datetime | None:
    """Parse an ISO timestamp from a quality score, treating naive as UTC."""
    if not isinstance(raw_timestamp, str):
        return None
    raw = raw_timestamp.strip()
    if not raw:
        return None
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        return None


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
    # Filenames of corrupt/malformed score files skipped while loading from disk.
    skipped_files: list[str] = field(default_factory=list)

    def _skipped_extra(self) -> dict[str, Any]:
        """Optional skipped-file fields included only when corrupt files were bypassed."""
        if not self.skipped_files:
            return {}
        return {
            "skipped_count": len(self.skipped_files),
            "skipped_files": list(self.skipped_files),
        }

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
            **self._skipped_extra(),
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
        self._skipped_tls = threading.local()

    @property
    def scores_dir(self) -> Path:
        """Directory this scorer persists and reads score records from."""
        return self._scores_dir

    @property
    def last_skipped_files(self) -> list[str]:
        """Skipped corrupt/malformed filenames from the latest load on this thread."""
        return list(getattr(self._skipped_tls, "files", ()))

    def _set_last_skipped(self, skipped: list[str]) -> None:
        self._skipped_tls.files = list(skipped)

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

        _write_json_atomic(filepath, score.to_dict())

        logger.info("score_saved", path=str(filepath))
        return filepath

    def _scan_scores(self, project: str, limit: int) -> tuple[list[QualityScore], list[str]]:
        """Scan score files newest-first, skipping corrupt/malformed entries.

        Scores dated more than ``SCORE_FUTURE_SLACK`` ahead of the host clock are ignored
        (logged, not reported as skipped): they would otherwise shadow every real run.
        """
        project_dir = contained_dir(self._scores_dir, project)
        if not project_dir.is_dir():
            self._set_last_skipped([])
            return [], []

        scores: list[QualityScore] = []
        skipped: list[str] = []
        horizon = datetime.now(UTC) + SCORE_FUTURE_SLACK
        for filepath in sorted(project_dir.glob("*.json"), reverse=True):
            score = self._load_score(filepath)
            if score is None:
                skipped.append(filepath.name)
                continue
            stamped = parse_score_timestamp(score.timestamp)
            if stamped is not None and stamped > horizon:
                logger.warning("score_file_future_dated_ignored", path=filepath.name)
                continue
            scores.append(score)
            if len(scores) >= limit:
                break
        self._set_last_skipped(skipped)
        return scores, skipped

    def load_latest(self, project: str) -> QualityScore | None:
        """Load the most recent readable score for a project.

        Falls back past unreadable or malformed newest files and records their
        filenames on ``skipped_files`` (and ``last_skipped_files``).

        Raises:
            InvalidStoreName: If ``project`` is not a safe store name.
        """
        scores, skipped = self._scan_scores(project, limit=1)
        if not scores:
            return None
        latest = scores[0]
        latest.skipped_files = skipped
        return latest

    def load_history(self, project: str, limit: int = 10) -> list[QualityScore]:
        """Load readable score history for a project, most recent first.

        Skips unreadable or malformed files and records their filenames in
        ``last_skipped_files``.

        Raises:
            InvalidStoreName: If ``project`` is not a safe store name.
        """
        if limit <= 0:
            self._set_last_skipped([])
            return []
        scores, _ = self._scan_scores(project, limit=limit)
        return scores

    def _load_score(self, filepath: Path) -> QualityScore | None:
        """Load a single score from a JSON file."""
        if _reject_non_regular_score(filepath):
            return None
        try:
            raw = _read_regular_nonblock(filepath, _MAX_SCORE_FILE_BYTES)
            data = json.loads(raw.decode("utf-8"))
        except (OSError, RecursionError, ValueError) as exc:
            logger.warning("score_file_read_failed", path=str(filepath), error=str(exc))
            return None

        return _parse_score_data(data, filepath)


_MAX_SCORE_FILE_BYTES = 1_048_576


def _write_json_atomic(filepath: Path, payload: dict[str, Any]) -> None:
    """Write *payload* as JSON to *filepath* via a 0600 temp file, fsync, and ``os.replace``."""
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=filepath.parent,
            prefix=".score-",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            os.fchmod(fh.fileno(), 0o600)
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, filepath)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _reject_non_regular_score(filepath: Path) -> bool:
    """Return True and log if *filepath* is a symlink or not a regular file."""
    if filepath.is_symlink():
        logger.warning("score_file_symlink_skipped", path=str(filepath))
        return True
    if not filepath.is_file():
        logger.warning("score_file_not_regular_skipped", path=str(filepath))
        return True
    return False


def _read_regular_nonblock(filepath: Path, cap: int) -> bytes:
    """Open *filepath* with ``O_NONBLOCK``, verify regular file via ``fstat``, and cap reads."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(filepath, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(f"not a regular file: {filepath}")
        with os.fdopen(fd, "rb") as fh:
            fd = -1
            data = fh.read(cap + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > cap:
        raise RegistryTooLargeError(f"{st.st_size} bytes, over the {cap}-byte cap")
    return data


def _is_str(value: object) -> bool:
    return isinstance(value, str)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_score_dimensions(raw_dims: object) -> list[ScoreDimension] | None:
    """Validate and build ``ScoreDimension`` objects from JSON, or return None if malformed."""
    if not isinstance(raw_dims, list):
        return None
    dimensions: list[ScoreDimension] = []
    for dim_data in raw_dims:
        if not isinstance(dim_data, dict):
            return None
        name = dim_data.get("name", "")
        raw_value = dim_data.get("raw_value", 0.0)
        score = dim_data.get("score", 0.0)
        weight = dim_data.get("weight", 0.0)
        weighted_score = dim_data.get("weighted_score", 0.0)
        if not _is_str(name):
            return None
        if not all(_is_finite_number(v) for v in (raw_value, score, weight, weighted_score)):
            return None
        dimensions.append(
            ScoreDimension(
                name=name,
                raw_value=float(raw_value),
                score=float(score),
                weight=float(weight),
                weighted_score=float(weighted_score),
            )
        )
    return dimensions


def _parse_score_data(data: object, filepath: Path) -> QualityScore | None:
    """Validate a decoded JSON value as a score dict and return ``QualityScore`` or None."""
    if not isinstance(data, dict):
        logger.warning("score_file_not_object", path=str(filepath), type=type(data).__name__)
        return None

    project = data.get("project", "")
    timestamp = data.get("timestamp", "")
    mode = data.get("mode", "quick")
    composite = data.get("composite", 0.0)
    ship_gate = data.get("ship_gate", False)
    commit_sha = data.get("commit_sha", None)
    schema_version = data.get("schema_version", 1)
    attempted = data.get("attempted_unmeasured", [])
    dimensions = _parse_score_dimensions(data.get("dimensions", []))

    if (
        not (_is_str(project) and _is_str(timestamp) and _is_str(mode))
        or not _is_finite_number(composite)
        or not isinstance(ship_gate, bool)
        or (commit_sha is not None and not _is_str(commit_sha))
        or not _is_strict_int(schema_version)
        or not isinstance(attempted, list)
        or not all(_is_str(item) for item in attempted)
        or dimensions is None
    ):
        logger.warning("score_file_invalid_schema", path=str(filepath))
        return None

    return QualityScore(
        project=project,
        timestamp=timestamp,
        mode=mode,
        composite=float(composite),
        ship_gate=ship_gate,
        dimensions=dimensions,
        commit_sha=commit_sha,
        attempted_unmeasured=list(attempted),
        stale=schema_version < SCORE_SCHEMA_VERSION,
    )
