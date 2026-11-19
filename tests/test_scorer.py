"""Tests for quality scorer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog

from linc_crucible.core.scorer import (
    BASE_WEIGHTS,
    SCORE_SCHEMA_VERSION,
    SHIP_GATE_THRESHOLD,
    QualityScorer,
    a11y_issues_to_score,
    apk_size_to_score,
    ssim_to_score,
)


@pytest.fixture
def scorer(tmp_path: Path) -> QualityScorer:
    return QualityScorer(scores_dir=tmp_path / "scores")


class TestQualityScoreCompute:
    def test_perfect_quick_score(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
        )
        assert score.composite == 100.0
        assert score.ship_gate is True
        assert score.mode == "quick"
        assert len(score.dimensions) == 2

    def test_zero_quick_score(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=0.0,
            lint_issues=100,
        )
        assert score.composite == 0.0
        assert score.ship_gate is False

    def test_nothing_measured_is_not_green(self, scorer: QualityScorer) -> None:
        """Both tests and lint unmeasured (no commands) must not score as 100.

        Regression for the gateway Quality 'instant 100' false-green: a project
        with no lint/test command configured ran zero checks, so the composite
        must read 0/not-shippable, never a perfect pass.
        """
        score = scorer.compute(
            project="Blank",
            test_pass_rate=None,
            lint_issues=None,
        )
        assert score.composite == 0.0
        assert score.ship_gate is False
        assert score.mode == "unmeasured"
        assert score.dimensions == []

    def test_partial_measurement_excludes_unmeasured_dim(self, scorer: QualityScorer) -> None:
        """An unmeasured dimension is excluded, not scored as a perfect pass.

        Tests unmeasured + lint clean → composite is the lint score alone,
        and the tests dimension is absent (renders as '—' downstream).
        """
        score = scorer.compute(
            project="LintOnly",
            test_pass_rate=None,
            lint_issues=0,
        )
        names = {d.name for d in score.dimensions}
        assert names == {"lint"}
        assert score.composite == 100.0  # lint clean, weight normalised to 1.0
        # linc-crucible#136 finding 2 (the complementary case, made explicit):
        # `test_pass_rate=None` here means "no test command was configured
        # at all" — nothing was attempted, so it is fine to renormalise
        # around it and still ship. This is UNLIKE `attempted_unmeasured`
        # (see TestAttemptedUnmeasuredForcesShipGate below), which is for a
        # command that WAS configured and then crashed.
        assert score.ship_gate is True
        assert score.attempted_unmeasured == []
        assert "partial-unmeasured" not in score.mode

    def test_quick_score_weights(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=0.8,
            lint_issues=5,
        )
        # tests: 80 * 0.55 = 44.0
        # lint: (100 - 10) * 0.45 = 90 * 0.45 = 40.5
        # total: 84.5
        assert score.composite == 84.5
        assert score.ship_gate is False  # < 85

    def test_full_score_with_security(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            security_score=90.0,
        )
        # tests: 100 * 0.40 = 40.0
        # lint: 100 * 0.30 = 30.0
        # security: 90 * 0.30 = 27.0
        # total: 97.0
        assert score.mode == "full"
        assert score.composite == 97.0
        assert score.ship_gate is True
        assert len(score.dimensions) == 3

    def test_gate_agrees_with_rounded_display(self, scorer: QualityScorer) -> None:
        # Raw composite ~84.9575 rounds to 85.0 for display/persistence.
        # The gate must agree with the SHOWN score: composite==85.0 and ship_gate True.
        # Before the fix, ship_gate used the unrounded 84.9575 < 85 and was False,
        # contradicting the displayed 85.0.
        # tests: 72.65 * 0.55 = 39.9575
        # lint:  100.0 * 0.45 = 45.0
        # raw total: 84.9575 -> round(.,1) = 85.0
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=0.7265,
            lint_issues=0,
        )
        assert score.composite == 85.0
        assert score.ship_gate is True

    def test_ship_gate_boundary(self, scorer: QualityScorer) -> None:
        # Exactly at threshold
        # Need composite = 85
        # tests=100*0.55=55, lint needs 30/0.45 = 66.67 -> 100-66.67=33.33 -> ~17 issues
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=7,  # lint = 100-14 = 86, weighted = 86*0.45 = 38.7 + 55 = 93.7
        )
        assert score.ship_gate is True

    def test_lint_floor_at_zero(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=200,
        )
        lint_dim = next(d for d in score.dimensions if d.name == "lint")
        assert lint_dim.score == 0.0

    def test_pass_rate_clamped(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.5,  # Over 1.0
            lint_issues=0,
        )
        test_dim = next(d for d in score.dimensions if d.name == "tests")
        assert test_dim.score == 100.0  # Clamped


class TestAttemptedUnmeasuredForcesShipGate:
    """linc-crucible#136 finding 2 (MINOS round 0, BLOCKER): `compute()`
    previously emitted `mode='unmeasured', composite=0.0, ship_gate=False`
    ONLY when the active-dimension list was EMPTY. Any surviving dimension
    was renormalised to weight 1.0, so an excluded-because-crashed dimension
    was indistinguishable from one the project never configured. The
    `attempted_unmeasured` parameter closes that hole: it forces
    `ship_gate=False` whenever ANY attempted dimension failed to measure,
    regardless of what a sibling dimension scored.
    """

    def test_attempted_unmeasured_forces_ship_gate_false(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="Mixed",
            test_pass_rate=None,  # tests crashed, excluded from the composite
            lint_issues=0,  # lint is clean
            attempted_unmeasured=["tests"],
        )
        assert {d.name for d in score.dimensions} == {"lint"}
        assert score.composite == 100.0  # the surviving dimension alone still scores 100
        assert score.ship_gate is False  # ...but the gate is forced closed
        assert "partial-unmeasured" in score.mode
        assert score.attempted_unmeasured == ["tests"]

    def test_attempted_unmeasured_empty_list_does_not_force_gate(
        self, scorer: QualityScorer
    ) -> None:
        """An explicit empty list must behave identically to omitting the
        parameter — no forced closure, no mode suffix."""
        score = scorer.compute(
            project="Clean",
            test_pass_rate=1.0,
            lint_issues=0,
            attempted_unmeasured=[],
        )
        assert score.ship_gate is True
        assert score.attempted_unmeasured == []
        assert "partial-unmeasured" not in score.mode

    def test_all_active_dims_attempted_unmeasured_reads_fully_unmeasured(
        self, scorer: QualityScorer
    ) -> None:
        """Both tests and lint crashed (both excluded, both attempted) — the
        empty-active-dims early-return path must still carry the
        attempted_unmeasured list through, not silently drop it.
        """
        score = scorer.compute(
            project="Broken",
            test_pass_rate=None,
            lint_issues=None,
            attempted_unmeasured=["tests", "lint"],
        )
        assert score.mode == "unmeasured"
        assert score.ship_gate is False
        assert score.dimensions == []
        assert score.attempted_unmeasured == ["tests", "lint"]

    def test_attempted_unmeasured_forces_gate_even_above_threshold(
        self, scorer: QualityScorer
    ) -> None:
        """The forced closure is unconditional on the composite value — even
        a would-be-perfect surviving composite must not ship."""
        score = scorer.compute(
            project="PerfectButBroken",
            test_pass_rate=1.0,
            lint_issues=0,
            security_score=100.0,
            attempted_unmeasured=["a11y"],
        )
        assert score.composite == 100.0
        assert score.ship_gate is False


class TestScoreSchemaVersion:
    """linc-crucible#136 finding 6: a record persisted before this fix
    carries no `schema_version` and was computed under the false-green
    defect this PR closes. `_load_score` stamps such a record `stale` so a
    future consumer (crucible_preflight) can warn or refuse rather than
    trust it as authoritative.
    """

    def test_to_dict_stamps_current_schema_version(self, scorer: QualityScorer) -> None:
        score = scorer.compute(project="P", test_pass_rate=1.0, lint_issues=0)
        d = score.to_dict()
        assert d["schema_version"] == SCORE_SCHEMA_VERSION
        assert d["attempted_unmeasured"] == []

    def test_freshly_computed_score_is_not_stale(self, scorer: QualityScorer) -> None:
        score = scorer.compute(project="P", test_pass_rate=1.0, lint_issues=0)
        assert score.stale is False

    def test_load_score_without_schema_version_is_stale(
        self, scorer: QualityScorer, tmp_path: Path
    ) -> None:
        # A hand-built pre-fix record: no `schema_version`, no
        # `attempted_unmeasured` key at all — exactly what `to_dict()` wrote
        # before this PR.
        project_dir = scorer._scores_dir / "OldProject"
        project_dir.mkdir(parents=True)
        legacy = {
            "project": "OldProject",
            "timestamp": "2026-01-01T00:00:00Z",
            "mode": "quick",
            "composite": 100.0,
            "ship_gate": True,
            "commit_sha": None,
            "dimensions": [],
        }
        filepath = project_dir / "2026-01-01T00-00-00Z.json"
        filepath.write_text(json.dumps(legacy))

        loaded = scorer.load_latest("OldProject")
        assert loaded is not None
        assert loaded.stale is True
        assert loaded.attempted_unmeasured == []

    def test_load_score_freshly_saved_is_not_stale(self, scorer: QualityScorer) -> None:
        score = scorer.compute(project="Fresh", test_pass_rate=1.0, lint_issues=0)
        scorer.save(score)

        loaded = scorer.load_latest("Fresh")
        assert loaded is not None
        assert loaded.stale is False

    def test_load_score_preserves_attempted_unmeasured_round_trip(
        self, scorer: QualityScorer
    ) -> None:
        score = scorer.compute(
            project="RoundTrip",
            test_pass_rate=None,
            lint_issues=0,
            attempted_unmeasured=["tests"],
        )
        scorer.save(score)

        loaded = scorer.load_latest("RoundTrip")
        assert loaded is not None
        assert loaded.attempted_unmeasured == ["tests"]
        assert loaded.ship_gate is False


class TestQualityScorePersistence:
    def test_save_and_load(self, scorer: QualityScorer, tmp_path: Path) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=0.95,
            lint_issues=3,
        )
        filepath = scorer.save(score)

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["project"] == "TestProject"
        assert data["composite"] == score.composite

    def test_load_latest(self, scorer: QualityScorer) -> None:
        # Save two scores
        s1 = scorer.compute("Proj", test_pass_rate=0.5, lint_issues=0)
        scorer.save(s1)
        s2 = scorer.compute("Proj", test_pass_rate=1.0, lint_issues=0)
        scorer.save(s2)

        latest = scorer.load_latest("Proj")
        assert latest is not None
        assert latest.composite == 100.0

    def test_load_latest_no_scores(self, scorer: QualityScorer) -> None:
        assert scorer.load_latest("NonExistent") is None

    def test_load_history(self, scorer: QualityScorer) -> None:
        # Create scores with distinct timestamps to avoid file collisions
        for i in range(5):
            s = scorer.compute("Proj", test_pass_rate=0.5 + i * 0.1, lint_issues=0)
            # Override timestamp to ensure uniqueness
            s.timestamp = f"2026-02-20T00:00:0{i}Z"
            scorer.save(s)

        history = scorer.load_history("Proj", limit=3)
        assert len(history) == 3

    def test_load_history_empty(self, scorer: QualityScorer) -> None:
        assert scorer.load_history("NonExistent") == []


class TestScoreToDict:
    def test_to_dict(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=0.9,
            lint_issues=5,
            security_score=85.0,
        )
        d = score.to_dict()
        assert d["project"] == "TestProject"
        assert d["mode"] == "full"
        assert "composite" in d
        assert "ship_gate" in d
        assert len(d["dimensions"]) == 3
        assert d["dimensions"][0]["name"] == "tests"
        assert d["schema_version"] == SCORE_SCHEMA_VERSION
        assert d["attempted_unmeasured"] == []


class TestScoreConstants:
    def test_ship_gate_threshold(self) -> None:
        assert SHIP_GATE_THRESHOLD == 85

    def test_base_weights_is_flat_dict(self) -> None:
        """BASE_WEIGHTS must be a flat dict[str, float] of raw importances."""
        assert isinstance(BASE_WEIGHTS, dict)
        for key, value in BASE_WEIGHTS.items():
            assert isinstance(key, str), f"Key {key!r} should be str, not {type(key)}"
            assert isinstance(value, (int, float)), f"Value for {key!r} should be numeric"

    def test_base_weights_has_all_dimensions(self) -> None:
        """BASE_WEIGHTS must contain entries for all scoring dimensions."""
        expected_dims = {"tests", "lint", "security", "perf", "a11y"}
        assert set(BASE_WEIGHTS.keys()) >= expected_dims

    def test_base_weights_all_positive(self) -> None:
        """All raw importance values must be positive."""
        for dim, value in BASE_WEIGHTS.items():
            assert value > 0, f"BASE_WEIGHTS[{dim!r}] must be positive, got {value}"

    @pytest.mark.parametrize(
        "active_dims",
        [
            {"tests", "lint"},
            {"tests", "lint", "security"},
            {"tests", "lint", "perf"},
            {"tests", "lint", "security", "perf"},
            {"tests", "lint", "a11y"},
            {"tests", "lint", "security", "a11y"},
            {"tests", "lint", "perf", "a11y"},
            {"tests", "lint", "security", "perf", "a11y"},
        ],
        ids=lambda s: "+".join(sorted(s)),
    )
    def test_normalized_weights_sum_to_one(self, active_dims: set[str]) -> None:
        """Dynamic normalization must produce weights that sum to 1.0."""
        raw = {d: BASE_WEIGHTS[d] for d in active_dims}
        total = sum(raw.values())
        normalized = {d: v / total for d, v in raw.items()}
        assert abs(sum(normalized.values()) - 1.0) < 0.001

    def test_no_frozenset_keys_in_base_weights(self) -> None:
        """BASE_WEIGHTS must NOT use frozenset keys (old combinatorial structure)."""
        for key in BASE_WEIGHTS:
            assert not isinstance(key, frozenset), (
                "BASE_WEIGHTS still uses frozenset keys — not refactored"
            )


class TestApkSizeToScore:
    def test_excellent(self) -> None:
        assert apk_size_to_score(5.0) == 100.0

    def test_at_excellent_boundary(self) -> None:
        assert apk_size_to_score(10.0) == 100.0

    def test_at_poor_boundary(self) -> None:
        assert apk_size_to_score(30.0) == 0.0

    def test_above_poor(self) -> None:
        assert apk_size_to_score(50.0) == 0.0

    def test_midpoint(self) -> None:
        assert apk_size_to_score(20.0) == pytest.approx(50.0)


class TestPerfScoring:
    def test_quick_perf_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            perf_score=10.0,  # 10 MB = score 100
        )
        assert score.mode == "quick+perf"
        assert len(score.dimensions) == 3
        assert score.composite == 100.0

    def test_full_perf_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            security_score=100.0,
            perf_score=10.0,
        )
        assert score.mode == "full+perf"
        assert len(score.dimensions) == 4
        assert score.composite == 100.0

    def test_no_perf_backward_compat(self, scorer: QualityScorer) -> None:
        """Without perf_score, behavior is identical to before."""
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
        )
        assert score.mode == "quick"
        assert len(score.dimensions) == 2
        assert score.composite == 100.0

    def test_large_apk_penalizes_score(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            perf_score=30.0,  # 30 MB = score 0
        )
        # tests: 100*0.45=45, lint: 100*0.35=35, perf: 0*0.20=0
        assert score.composite == 80.0
        assert score.ship_gate is False


class TestSsimToScore:
    def test_excellent(self) -> None:
        assert ssim_to_score(0.98) == 100.0

    def test_at_excellent_boundary(self) -> None:
        assert ssim_to_score(0.95) == 100.0

    def test_at_poor_boundary(self) -> None:
        assert ssim_to_score(0.70) == 0.0

    def test_below_poor(self) -> None:
        assert ssim_to_score(0.50) == 0.0

    def test_midpoint(self) -> None:
        assert ssim_to_score(0.825) == pytest.approx(50.0)


class TestVisualScoring:
    def test_quick_visual_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            visual_scores=[0.98],
        )
        assert score.mode == "quick+visual"
        assert len(score.dimensions) == 3
        assert score.composite == pytest.approx(100.0)

    def test_full_visual_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            security_score=100.0,
            visual_scores=[0.98],
        )
        assert score.mode == "full+visual"
        assert len(score.dimensions) == 4

    def test_full_perf_visual_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            security_score=100.0,
            perf_score=10.0,
            visual_scores=[0.98],
        )
        assert score.mode == "full+perf+visual"
        assert len(score.dimensions) == 5
        assert score.composite == pytest.approx(100.0)

    def test_no_visual_backward_compat(self, scorer: QualityScorer) -> None:
        """Without visual_scores, behavior is identical to before."""
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
        )
        assert score.mode == "quick"
        assert len(score.dimensions) == 2
        assert score.composite == 100.0

    def test_empty_list_skips_visual(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            visual_scores=[],
        )
        assert score.mode == "quick"
        assert len(score.dimensions) == 2

    def test_degraded_ssim_penalizes(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            visual_scores=[0.825],
        )
        visual_dim = next(d for d in score.dimensions if d.name == "visual")
        assert visual_dim.score == pytest.approx(50.0)
        assert score.composite < 100.0

    def test_multiple_screens_averaged(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            visual_scores=[0.95, 0.70],
        )
        visual_dim = next(d for d in score.dimensions if d.name == "visual")
        # avg SSIM = 0.825 → score = 50.0
        assert visual_dim.score == pytest.approx(50.0)

    def test_weight_redistribution_sums_to_one(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            visual_scores=[0.98],
        )
        total_weight = sum(d.weight for d in score.dimensions)
        assert total_weight == pytest.approx(1.0)

    def test_custom_visual_weight(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            visual_scores=[0.98],
            visual_weight=0.25,
        )
        visual_dim = next(d for d in score.dimensions if d.name == "visual")
        assert visual_dim.weight == 0.25
        other_weights = sum(d.weight for d in score.dimensions if d.name != "visual")
        assert other_weights == pytest.approx(0.75)


class TestA11yIssuesToScore:
    def test_no_issues(self) -> None:
        assert a11y_issues_to_score() == 100.0

    def test_critical_issues(self) -> None:
        assert a11y_issues_to_score(critical=5) == 50.0

    def test_major_issues(self) -> None:
        assert a11y_issues_to_score(major=10) == 50.0

    def test_minor_issues(self) -> None:
        assert a11y_issues_to_score(minor=25) == 50.0

    def test_mixed_issues(self) -> None:
        # 2*10 + 3*5 + 5*2 = 20+15+10 = 45
        assert a11y_issues_to_score(critical=2, major=3, minor=5) == 55.0

    def test_floor_at_zero(self) -> None:
        assert a11y_issues_to_score(critical=20) == 0.0


class TestA11yStructuralAndRulesScoring:
    """Tests for the new a11y_structural and a11y_rules scoring dimensions."""

    def test_a11y_structural_dimension_present_when_score_provided(
        self, scorer: QualityScorer
    ) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_structural_score=100.0,
        )
        names = [d.name for d in score.dimensions]
        assert "a11y_structural" in names

    def test_a11y_structural_dimension_absent_when_not_provided(
        self, scorer: QualityScorer
    ) -> None:
        score = scorer.compute(project="T", test_pass_rate=1.0, lint_issues=0)
        names = [d.name for d in score.dimensions]
        assert "a11y_structural" not in names

    def test_a11y_rules_dimension_present_when_score_provided(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_rules_score=80.0,
        )
        names = [d.name for d in score.dimensions]
        assert "a11y_rules" in names

    def test_a11y_rules_dimension_absent_when_not_provided(self, scorer: QualityScorer) -> None:
        score = scorer.compute(project="T", test_pass_rate=1.0, lint_issues=0)
        names = [d.name for d in score.dimensions]
        assert "a11y_rules" not in names

    def test_both_dimensions_together(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_structural_score=100.0,
            a11y_rules_score=100.0,
        )
        names = [d.name for d in score.dimensions]
        assert "a11y_structural" in names
        assert "a11y_rules" in names
        assert score.composite == 100.0

    def test_both_dimensions_coexist_with_old_a11y(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_score=100.0,
            a11y_structural_score=100.0,
            a11y_rules_score=100.0,
        )
        names = [d.name for d in score.dimensions]
        assert "a11y" in names
        assert "a11y_structural" in names
        assert "a11y_rules" in names
        assert score.composite == 100.0

    def test_a11y_structural_penalizes_low_score(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_structural_score=0.0,
        )
        assert score.composite < 100.0

    def test_a11y_rules_penalizes_low_score(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_rules_score=0.0,
        )
        assert score.composite < 100.0

    def test_a11y_structural_score_clamped_to_100(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_structural_score=150.0,
        )
        dim = next(d for d in score.dimensions if d.name == "a11y_structural")
        assert dim.score == 100.0

    def test_a11y_rules_score_clamped_to_0(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_rules_score=-10.0,
        )
        dim = next(d for d in score.dimensions if d.name == "a11y_rules")
        assert dim.score == 0.0

    def test_weights_sum_to_one_with_both_new_dims(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_structural_score=100.0,
            a11y_rules_score=100.0,
        )
        total_weight = sum(d.weight for d in score.dimensions)
        assert total_weight == pytest.approx(1.0)

    def test_mode_includes_a11y_structural(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_structural_score=100.0,
        )
        assert "a11y_structural" in score.mode

    def test_mode_includes_a11y_rules(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_rules_score=100.0,
        )
        assert "a11y_rules" in score.mode

    def test_base_weights_has_new_dimensions(self) -> None:
        assert "a11y_structural" in BASE_WEIGHTS
        assert "a11y_rules" in BASE_WEIGHTS
        assert BASE_WEIGHTS["a11y_structural"] == 40
        assert BASE_WEIGHTS["a11y_rules"] == 35


class TestA11yScoring:
    def test_quick_a11y_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(project="T", test_pass_rate=1.0, lint_issues=0, a11y_score=100.0)
        assert score.mode == "quick+a11y"
        assert len(score.dimensions) == 3
        assert score.composite == 100.0

    def test_full_a11y_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            security_score=100.0,
            a11y_score=100.0,
        )
        assert score.mode == "full+a11y"
        assert len(score.dimensions) == 4

    def test_full_perf_a11y_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            security_score=100.0,
            perf_score=10.0,
            a11y_score=100.0,
        )
        assert score.mode == "full+perf+a11y"
        assert len(score.dimensions) == 5

    def test_quick_perf_a11y_mode(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            perf_score=10.0,
            a11y_score=100.0,
        )
        assert score.mode == "quick+perf+a11y"
        assert len(score.dimensions) == 4

    def test_a11y_issues_penalize(self, scorer: QualityScorer) -> None:
        score = scorer.compute(project="T", test_pass_rate=1.0, lint_issues=0, a11y_score=50.0)
        assert score.composite < 100.0

    def test_no_a11y_backward_compat(self, scorer: QualityScorer) -> None:
        score = scorer.compute(project="T", test_pass_rate=1.0, lint_issues=0)
        assert score.mode == "quick"
        assert len(score.dimensions) == 2

    def test_a11y_with_visual_overlay(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T",
            test_pass_rate=1.0,
            lint_issues=0,
            a11y_score=100.0,
            visual_scores=[0.98],
        )
        assert "+a11y+visual" in score.mode
        total_weight = sum(d.weight for d in score.dimensions)
        assert total_weight == pytest.approx(1.0)


class TestCodeQualityScoring:
    """Tests for the code_quality (CRAP) scoring dimension."""

    def test_code_quality_dimension_present_when_provided(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T", test_pass_rate=1.0, lint_issues=0, code_quality_score=95.0
        )
        names = [d.name for d in score.dimensions]
        assert "code_quality" in names

    def test_code_quality_dimension_absent_when_not_provided(self, scorer: QualityScorer) -> None:
        score = scorer.compute(project="T", test_pass_rate=1.0, lint_issues=0)
        names = [d.name for d in score.dimensions]
        assert "code_quality" not in names

    def test_code_quality_perfect_score(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T", test_pass_rate=1.0, lint_issues=0, code_quality_score=100.0
        )
        assert score.composite == 100.0

    def test_code_quality_low_score_penalizes(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T", test_pass_rate=1.0, lint_issues=0, code_quality_score=50.0
        )
        assert score.composite < 100.0

    def test_code_quality_clamped_to_100(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T", test_pass_rate=1.0, lint_issues=0, code_quality_score=120.0
        )
        dim = next(d for d in score.dimensions if d.name == "code_quality")
        assert dim.score == 100.0

    def test_code_quality_clamped_to_0(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T", test_pass_rate=1.0, lint_issues=0, code_quality_score=-10.0
        )
        dim = next(d for d in score.dimensions if d.name == "code_quality")
        assert dim.score == 0.0

    def test_mode_includes_code_quality(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T", test_pass_rate=1.0, lint_issues=0, code_quality_score=90.0
        )
        assert "code_quality" in score.mode

    def test_weights_sum_to_one_with_code_quality(self, scorer: QualityScorer) -> None:
        score = scorer.compute(
            project="T", test_pass_rate=1.0, lint_issues=0, code_quality_score=90.0
        )
        total_weight = sum(d.weight for d in score.dimensions)
        assert total_weight == pytest.approx(1.0)

    def test_base_weights_has_code_quality(self) -> None:
        assert "code_quality" in BASE_WEIGHTS
        assert BASE_WEIGHTS["code_quality"] > 0

    def test_backward_compat_no_code_quality(self, scorer: QualityScorer) -> None:
        """Existing scores without code_quality remain unchanged."""
        score = scorer.compute(project="T", test_pass_rate=0.8, lint_issues=5)
        assert score.composite == 84.5  # Same as before


class TestQualityScoreCommitSha:
    def test_compute_defaults_commit_sha_to_none(self, scorer: QualityScorer) -> None:
        """compute() defaults commit_sha to None when omitted."""
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
        )
        assert score.commit_sha is None

    def test_compute_with_commit_sha(self, scorer: QualityScorer) -> None:
        """compute() stores passed commit_sha on the QualityScore."""
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            commit_sha="abcdef123456",
        )
        assert score.commit_sha == "abcdef123456"

    def test_commit_sha_round_trip(self, scorer: QualityScorer) -> None:
        """A populated commit_sha survives to_dict/save/load unchanged."""
        score = scorer.compute(
            project="TestProject",
            test_pass_rate=1.0,
            lint_issues=0,
            commit_sha="abcdef123456",
        )
        # to_dict roundtrip
        d = score.to_dict()
        assert d["commit_sha"] == "abcdef123456"

        # save and load
        filepath = scorer.save(score)
        loaded = scorer._load_score(filepath)
        assert loaded is not None
        assert loaded.commit_sha == "abcdef123456"

    def test_legacy_score_file_loads_none(self, scorer: QualityScorer, tmp_path: Path) -> None:
        """A legacy score file with NO commit_sha key loads with commit_sha=None."""
        legacy_data = {
            "project": "LegacyProj",
            "timestamp": "2026-06-25T12:00:00Z",
            "mode": "quick",
            "composite": 90.0,
            "ship_gate": True,
            "dimensions": [
                {
                    "name": "tests",
                    "raw_value": 1.0,
                    "score": 100.0,
                    "weight": 0.55,
                    "weighted_score": 55.0,
                }
            ],
        }
        legacy_file = tmp_path / "legacy.json"
        with open(legacy_file, "w") as f:
            json.dump(legacy_data, f)

        loaded = scorer._load_score(legacy_file)
        assert loaded is not None
        assert loaded.project == "LegacyProj"
        assert loaded.commit_sha is None  # Loaded cleanly without KeyError


class TestResolveCommitSha:
    """Tests for _resolve_commit_sha helper in assayer.py."""

    def test_none_path(self) -> None:
        from linc_crucible.core.assayer import _resolve_commit_sha

        assert _resolve_commit_sha(None) is None

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        from linc_crucible.core.assayer import _resolve_commit_sha

        assert _resolve_commit_sha(tmp_path / "nonexistent") is None

    @patch("linc_crucible.core.assayer.safe_subprocess_run")
    def test_git_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from linc_crucible.core.assayer import _resolve_commit_sha

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "  abcdef123456 \n"
        mock_run.return_value = mock_res

        # Path must exist
        assert _resolve_commit_sha(tmp_path) == "abcdef123456"
        mock_run.assert_called_once_with("git rev-parse HEAD", cwd=str(tmp_path), timeout=10)

    @patch("linc_crucible.core.assayer.safe_subprocess_run")
    def test_git_failure(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from linc_crucible.core.assayer import _resolve_commit_sha

        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_run.return_value = mock_res

        assert _resolve_commit_sha(tmp_path) is None

    @patch("linc_crucible.core.assayer.safe_subprocess_run")
    def test_git_exception(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from linc_crucible.core.assayer import _resolve_commit_sha

        mock_run.side_effect = Exception("git failed")

        assert _resolve_commit_sha(tmp_path) is None


class TestBuildDimensions:
    def test_build_dimensions_empty(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            None, None, None, None, None, None, None, None, None, {}, 0.15
        )
        assert dims == []

    def test_build_dimensions_tests(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            test_pass_rate=0.85,
            lint_issues=None,
            security_score=None,
            code_quality_score=None,
            perf_score=None,
            a11y_score=None,
            a11y_structural_score=None,
            a11y_rules_score=None,
            visual_scores=None,
            weights={"tests": 0.3},
            visual_weight=0.15,
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "tests"
        assert d.raw_value == 0.85
        assert d.score == 85.0
        assert d.weight == 0.3
        assert d.weighted_score == 85.0 * 0.3

    def test_build_dimensions_tests_clamp_low(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            test_pass_rate=-0.1,
            lint_issues=None,
            security_score=None,
            code_quality_score=None,
            perf_score=None,
            a11y_score=None,
            a11y_structural_score=None,
            a11y_rules_score=None,
            visual_scores=None,
            weights={},
            visual_weight=0.15,
        )
        assert dims[0].score == 0.0
        assert dims[0].weight == 0.0

    def test_build_dimensions_tests_clamp_high(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            test_pass_rate=1.5,
            lint_issues=None,
            security_score=None,
            code_quality_score=None,
            perf_score=None,
            a11y_score=None,
            a11y_structural_score=None,
            a11y_rules_score=None,
            visual_scores=None,
            weights={"tests": 0.5},
            visual_weight=0.15,
        )
        assert dims[0].score == 100.0

    def test_build_dimensions_lint(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            test_pass_rate=None,
            lint_issues=5,
            security_score=None,
            code_quality_score=None,
            perf_score=None,
            a11y_score=None,
            a11y_structural_score=None,
            a11y_rules_score=None,
            visual_scores=None,
            weights={"lint": 0.2},
            visual_weight=0.15,
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "lint"
        assert d.raw_value == 5.0
        assert d.score == 90.0
        assert d.weight == 0.2
        assert d.weighted_score == 90.0 * 0.2

    def test_build_dimensions_lint_clamp(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            test_pass_rate=None,
            lint_issues=60,
            security_score=None,
            code_quality_score=None,
            perf_score=None,
            a11y_score=None,
            a11y_structural_score=None,
            a11y_rules_score=None,
            visual_scores=None,
            weights={"lint": 0.2},
            visual_weight=0.15,
        )
        assert dims[0].score == 0.0

    def test_build_dimensions_security(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            test_pass_rate=None,
            lint_issues=None,
            security_score=85.5,
            code_quality_score=None,
            perf_score=None,
            a11y_score=None,
            a11y_structural_score=None,
            a11y_rules_score=None,
            visual_scores=None,
            weights={"security": 0.1},
            visual_weight=0.15,
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "security"
        assert d.raw_value == 85.5
        assert d.score == 85.5
        assert d.weight == 0.1
        assert d.weighted_score == 85.5 * 0.1

    def test_build_dimensions_security_clamp(self, scorer: QualityScorer) -> None:
        dims_low = scorer._build_dimensions(
            None, None, -10.0, None, None, None, None, None, None, {}, 0.15
        )
        assert dims_low[0].score == 0.0
        dims_high = scorer._build_dimensions(
            None, None, 110.0, None, None, None, None, None, None, {}, 0.15
        )
        assert dims_high[0].score == 100.0

    def test_build_dimensions_code_quality(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            None, None, None, 75.2, None, None, None, None, None, {"code_quality": 0.4}, 0.15
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "code_quality"
        assert d.raw_value == 75.2
        assert d.score == 75.2
        assert d.weight == 0.4
        assert d.weighted_score == 75.2 * 0.4

    def test_build_dimensions_code_quality_clamp(self, scorer: QualityScorer) -> None:
        dims_low = scorer._build_dimensions(
            None, None, None, -5.0, None, None, None, None, None, {}, 0.15
        )
        assert dims_low[0].score == 0.0
        dims_high = scorer._build_dimensions(
            None, None, None, 105.0, None, None, None, None, None, {}, 0.15
        )
        assert dims_high[0].score == 100.0

    @patch("linc_crucible.core.scorer.apk_size_to_score")
    def test_build_dimensions_perf(
        self, mock_apk_size_to_score: MagicMock, scorer: QualityScorer
    ) -> None:
        mock_apk_size_to_score.return_value = 60.0
        dims = scorer._build_dimensions(
            None, None, None, None, 15.0, None, None, None, None, {"perf": 0.25}, 0.15
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "perf"
        assert d.raw_value == 15.0
        assert d.score == 60.0
        assert d.weight == 0.25
        assert d.weighted_score == 60.0 * 0.25
        mock_apk_size_to_score.assert_called_once_with(15.0)

    def test_build_dimensions_a11y(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            None, None, None, None, None, 92.5, None, None, None, {"a11y": 0.2}, 0.15
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "a11y"
        assert d.raw_value == 92.5
        assert d.score == 92.5
        assert d.weight == 0.2
        assert d.weighted_score == 92.5 * 0.2

    def test_build_dimensions_a11y_clamp(self, scorer: QualityScorer) -> None:
        dims_low = scorer._build_dimensions(
            None, None, None, None, None, -1.0, None, None, None, {}, 0.15
        )
        assert dims_low[0].score == 0.0
        dims_high = scorer._build_dimensions(
            None, None, None, None, None, 101.0, None, None, None, {}, 0.15
        )
        assert dims_high[0].score == 100.0

    def test_build_dimensions_a11y_structural(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            None, None, None, None, None, None, 88.0, None, None, {"a11y_structural": 0.1}, 0.15
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "a11y_structural"
        assert d.raw_value == 88.0
        assert d.score == 88.0
        assert d.weight == 0.1
        assert d.weighted_score == 88.0 * 0.1

    def test_build_dimensions_a11y_structural_clamp(self, scorer: QualityScorer) -> None:
        dims_low = scorer._build_dimensions(
            None, None, None, None, None, None, -1.0, None, None, {}, 0.15
        )
        assert dims_low[0].score == 0.0
        dims_high = scorer._build_dimensions(
            None, None, None, None, None, None, 101.0, None, None, {}, 0.15
        )
        assert dims_high[0].score == 100.0

    def test_build_dimensions_a11y_rules(self, scorer: QualityScorer) -> None:
        dims = scorer._build_dimensions(
            None, None, None, None, None, None, None, 77.0, None, {"a11y_rules": 0.15}, 0.15
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "a11y_rules"
        assert d.raw_value == 77.0
        assert d.score == 77.0
        assert d.weight == 0.15
        assert d.weighted_score == 77.0 * 0.15

    def test_build_dimensions_a11y_rules_clamp(self, scorer: QualityScorer) -> None:
        dims_low = scorer._build_dimensions(
            None, None, None, None, None, None, None, -5.0, None, {}, 0.15
        )
        assert dims_low[0].score == 0.0
        dims_high = scorer._build_dimensions(
            None, None, None, None, None, None, None, 105.0, None, {}, 0.15
        )
        assert dims_high[0].score == 100.0

    @patch("linc_crucible.core.scorer.ssim_to_score")
    def test_build_dimensions_visual(
        self, mock_ssim_to_score: MagicMock, scorer: QualityScorer
    ) -> None:
        mock_ssim_to_score.return_value = 95.0
        dims = scorer._build_dimensions(
            None, None, None, None, None, None, None, None, [0.9, 0.95, 0.91], {}, 0.15
        )
        assert len(dims) == 1
        d = dims[0]
        assert d.name == "visual"
        assert d.raw_value == 0.92
        assert d.score == 95.0
        assert d.weight == 0.15
        assert d.weighted_score == 95.0 * 0.15
        mock_ssim_to_score.assert_called_once_with(0.9199999999999999)


class TestBuildMode:
    """Direct coverage for `_build_mode` — extracted out of `compute()` in
    the linc-crucible#136 repair round specifically to bring `compute()`'s
    CRAP score back under the ratchet baseline. Every branch here needs its
    own test since the extraction only helps if this new function itself
    stays at (near-)full branch coverage.
    """

    def test_quick_mode_no_extras(self) -> None:
        mode = QualityScorer._build_mode(
            has_security=False,
            active_dims=["tests", "lint"],
            has_visual=False,
            partial_unmeasured=[],
        )
        assert mode == "quick"

    def test_full_mode_when_security_present(self) -> None:
        mode = QualityScorer._build_mode(
            has_security=True,
            active_dims=["tests", "lint", "security"],
            has_visual=False,
            partial_unmeasured=[],
        )
        assert mode == "full"

    def test_extra_dims_appended_in_fixed_order(self) -> None:
        mode = QualityScorer._build_mode(
            has_security=False,
            active_dims=["tests", "perf", "a11y", "code_quality"],
            has_visual=False,
            partial_unmeasured=[],
        )
        # Fixed emission order regardless of active_dims order: code_quality,
        # perf, a11y, a11y_structural, a11y_rules.
        assert mode == "quick+code_quality+perf+a11y"

    def test_a11y_structural_and_rules_appended(self) -> None:
        mode = QualityScorer._build_mode(
            has_security=False,
            active_dims=["tests", "a11y_structural", "a11y_rules"],
            has_visual=False,
            partial_unmeasured=[],
        )
        assert mode == "quick+a11y_structural+a11y_rules"

    def test_visual_suffix_appended(self) -> None:
        mode = QualityScorer._build_mode(
            has_security=False,
            active_dims=["tests"],
            has_visual=True,
            partial_unmeasured=[],
        )
        assert mode == "quick+visual"

    def test_partial_unmeasured_suffix_appended(self) -> None:
        mode = QualityScorer._build_mode(
            has_security=False,
            active_dims=["tests"],
            has_visual=False,
            partial_unmeasured=["lint"],
        )
        assert mode == "quick+partial-unmeasured(lint)"

    def test_all_suffixes_combine(self) -> None:
        mode = QualityScorer._build_mode(
            has_security=True,
            active_dims=["tests", "perf"],
            has_visual=True,
            partial_unmeasured=["lint"],
        )
        assert mode == "full+perf+visual+partial-unmeasured(lint)"

    def test_security_in_partial_unmeasured_yields_full_mode(self) -> None:
        mode = QualityScorer._build_mode(
            has_security=False,
            active_dims=["tests"],
            has_visual=False,
            partial_unmeasured=["security"],
        )
        assert mode == "full+partial-unmeasured(security)"


# ── Store containment (R4 High 1 + Medium 6) ─────────────────────────────────


def _attempt(fn, *args, **kwargs) -> BaseException | None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # the contract asserts the exact type below
        return exc
    return None


def _assert_refused(exc: BaseException | None) -> None:
    from linc_crucible.core.store_names import InvalidStoreName

    assert isinstance(exc, InvalidStoreName), f"expected InvalidStoreName, got {exc!r}"
    assert exc.field == "project"


class TestScoreStoreContainment:
    @pytest.mark.parametrize("project", ["../../escaped", "ABS", "a b", "", "a\x00b"])
    def test_contract_save_hostile_project_writes_nothing(
        self, tmp_path: Path, project: str
    ) -> None:
        """[contract] save() with a hostile project writes nothing anywhere."""
        if project == "ABS":  # an absolute path, kept under tmp_path so a red run stays local
            project = str(tmp_path / "abs")
        scores = tmp_path / "x" / "scores"
        scorer = QualityScorer(scores_dir=scores)
        score = scorer.compute(project=project, test_pass_rate=1.0, lint_issues=0)

        _assert_refused(_attempt(scorer.save, score))

        assert not (tmp_path / "escaped").exists()
        assert not (tmp_path / "abs").exists()
        assert not scores.exists()

    @pytest.mark.parametrize("op", ["load_latest", "load_history"])
    def test_read_paths_with_invalid_project_raise(self, scorer: QualityScorer, op: str) -> None:
        _assert_refused(_attempt(getattr(scorer, op), "../x"))

    def test_contract_symlinked_project_refused(self, tmp_path: Path) -> None:
        """[contract] ``<scores>/proj`` → outside: save/load refused, target untouched."""
        scores = tmp_path / "scores"
        scores.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "2026-01-01T00-00-00Z.json").write_text("{}")
        (scores / "proj").symlink_to(outside, target_is_directory=True)
        scorer = QualityScorer(scores_dir=scores)
        score = scorer.compute(project="proj", test_pass_rate=1.0, lint_issues=0)

        _assert_refused(_attempt(scorer.save, score))
        _assert_refused(_attempt(scorer.load_latest, "proj"))
        _assert_refused(_attempt(scorer.load_history, "proj"))
        assert sorted(p.name for p in outside.iterdir()) == ["2026-01-01T00-00-00Z.json"]

    def test_valid_layout_unchanged(self, scorer: QualityScorer, tmp_path: Path) -> None:
        score = scorer.compute(project="My.App_2-x", test_pass_rate=1.0, lint_issues=0)
        path = scorer.save(score)
        assert path.parent == tmp_path / "scores" / "My.App_2-x"
        assert path.name == score.timestamp.replace(":", "-") + ".json"


class TestScoreStoreFallbackAndTypeChecksContract:
    """[contract] Corrupt/list-shaped score files fall back and are counted (R4 finding 10)."""

    def test_contract_corrupt_newest_falls_back_and_list_history_safe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] A store with a corrupt newest file and three valid older files ->
        crucible score reports the newest valid one; a list-shaped file does not crash
        history=true."""
        from click.testing import CliRunner

        import linc_crucible.server_context as server_ctx
        from linc_crucible.assay_tools import crucible_score
        from linc_crucible.cli import cli

        store = tmp_path / "scores"
        scorer = QualityScorer(scores_dir=store)
        for ts, pass_rate in (
            ("2026-03-01T00:00:00Z", 0.6),
            ("2026-03-02T00:00:00Z", 0.8),
            ("2026-03-03T00:00:00Z", 1.0),
        ):
            s = scorer.compute("DemoProj", test_pass_rate=pass_rate, lint_issues=0)
            s.timestamp = ts
            scorer.save(s)

        proj_dir = store / "DemoProj"
        newest_corrupt = proj_dir / "2026-03-04T00-00-00Z.json"
        newest_corrupt.write_text("{corrupt json", encoding="utf-8")

        monkeypatch.setattr(server_ctx, "_initialized", True)
        monkeypatch.setattr(server_ctx, "_scorer", scorer)

        with structlog.testing.capture_logs():
            runner = CliRunner()
            res = runner.invoke(cli, ["score", "DemoProj"])
            assert res.exit_code == 0, res.output
            payload = json.loads(res.stdout)
            assert payload["timestamp"] == "2026-03-03T00:00:00Z"
            assert payload["composite"] == 100.0
            assert payload["score"] == 100.0
            assert payload["skipped_files"] == ["2026-03-04T00-00-00Z.json"]
            assert payload["skipped_count"] == 1

            # Now replace the newest corrupt file with a list-shaped JSON file and verify
            # both latest and history=true skip & report/count it without AttributeError.
            newest_corrupt.write_text("[1, 2, 3]", encoding="utf-8")

            latest_res = runner.invoke(cli, ["score", "DemoProj"])
            assert latest_res.exit_code == 0, latest_res.output
            latest_payload = json.loads(latest_res.stdout)
            assert latest_payload["timestamp"] == "2026-03-03T00:00:00Z"
            assert latest_payload["skipped_files"] == ["2026-03-04T00-00-00Z.json"]

            history_dict = crucible_score(project="DemoProj", history=True, limit=10)
            assert history_dict["count"] == 3
            assert len(history_dict["scores"]) == 3
            assert [s["timestamp"] for s in history_dict["scores"]] == [
                "2026-03-03T00:00:00Z",
                "2026-03-02T00:00:00Z",
                "2026-03-01T00:00:00Z",
            ]
            assert history_dict["skipped_count"] == 1
            assert history_dict["skipped_files"] == ["2026-03-04T00-00-00Z.json"]

            history_cli = runner.invoke(cli, ["score", "DemoProj", "--history"])
            assert history_cli.exit_code == 0, history_cli.output
            history_payload = json.loads(history_cli.stdout)
            assert history_payload["count"] == 3
            assert history_payload["skipped_count"] == 1
            assert history_payload["skipped_files"] == ["2026-03-04T00-00-00Z.json"]

            # Human mode also reports skipped files literally without Rich markup crash
            (proj_dir / "2026-03-03T12[00]Z.json").write_text("[9]", encoding="utf-8")
            human_latest = runner.invoke(cli, ["--human", "score", "DemoProj"])
            assert human_latest.exit_code == 0
            expected_skip_line = (
                "Skipped 2 corrupt score file(s): "
                "2026-03-04T00-00-00Z.json, 2026-03-03T12[00]Z.json"
            )
            assert expected_skip_line in human_latest.stdout

            human_history = runner.invoke(cli, ["--human", "score", "DemoProj", "--history"])
            assert human_history.exit_code == 0
            assert expected_skip_line in human_history.stdout

    def test_subprocess_cli_score_keeps_corrupt_warning_off_stdout(self, tmp_path: Path) -> None:
        """[contract] Real CLI subprocess keeps score_file_read_failed off stdout."""
        import os
        import subprocess
        import sys

        store = tmp_path / "scores"
        scorer = QualityScorer(scores_dir=store)
        s = scorer.compute("SubProj", test_pass_rate=1.0, lint_issues=0)
        s.timestamp = "2026-03-01T00:00:00Z"
        scorer.save(s)
        (store / "SubProj" / "2026-03-02T00-00-00Z.json").write_text("{corrupt", encoding="utf-8")

        env = {
            **os.environ,
            "CRUCIBLE_SCORES_DIR": str(store),
            "LINC_HOME": str(tmp_path / "linc_home"),
        }
        proc = subprocess.run(
            [sys.executable, "-m", "linc_crucible.cli", "score", "SubProj"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        payload = json.loads(proc.stdout)
        assert payload["timestamp"] == "2026-03-01T00:00:00Z"
        assert payload["skipped_files"] == ["2026-03-02T00-00-00Z.json"]

    def test_load_history_respects_limit_across_corrupt_files_and_zero_limit(
        self, scorer: QualityScorer
    ) -> None:
        for ts, pr in (
            ("2026-03-01T00:00:00Z", 0.5),
            ("2026-03-02T00:00:00Z", 0.8),
            ("2026-03-04T00:00:00Z", 1.0),
        ):
            s = scorer.compute("Proj", test_pass_rate=pr, lint_issues=0)
            s.timestamp = ts
            scorer.save(s)
        proj_dir = scorer.scores_dir / "Proj"
        (proj_dir / "2026-03-03T00-00-00Z.json").write_text('["bad"]', encoding="utf-8")

        assert scorer.load_history("Proj", limit=0) == []
        assert scorer.last_skipped_files == []

        # With limit=2, skips the corrupt 03-03 file and still returns 2 valid scores.
        history = scorer.load_history("Proj", limit=2)
        assert [s.timestamp for s in history] == ["2026-03-04T00:00:00Z", "2026-03-02T00:00:00Z"]
        assert scorer.last_skipped_files == ["2026-03-03T00-00-00Z.json"]

    def test_all_corrupt_files_reports_not_found_with_skipped_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from click.testing import CliRunner

        import linc_crucible.server_context as server_ctx
        from linc_crucible.assay_tools import crucible_score
        from linc_crucible.cli import cli

        store = tmp_path / "scores"
        proj_dir = store / "AllBad"
        proj_dir.mkdir(parents=True)
        (proj_dir / "2026-03-01T00-00-00Z.json").write_text("[1]", encoding="utf-8")
        (proj_dir / "2026-03-02T00-00-00Z.json").write_text("{bad", encoding="utf-8")

        scorer = QualityScorer(scores_dir=store)
        monkeypatch.setattr(server_ctx, "_initialized", True)
        monkeypatch.setattr(server_ctx, "_scorer", scorer)

        with structlog.testing.capture_logs():
            latest_err = crucible_score(project="AllBad", history=False)
            assert latest_err["error_category"] == "not_found"
            assert latest_err["skipped_count"] == 2
            assert latest_err["skipped_files"] == [
                "2026-03-02T00-00-00Z.json",
                "2026-03-01T00-00-00Z.json",
            ]

            hist_err = crucible_score(project="AllBad", history=True)
            assert hist_err["count"] == 0
            assert hist_err["empty"] is True
            assert hist_err["skipped_count"] == 2

            human_empty_hist = CliRunner().invoke(cli, ["--human", "score", "AllBad", "--history"])
            assert human_empty_hist.exit_code == 0
            assert "Skipped 2 corrupt score file(s):" in human_empty_hist.stdout

    @pytest.mark.parametrize(
        "payload",
        [
            "[1, 2, 3]",
            '"a string"',
            "42",
            "null",
            '{"project": 123}',
            '{"timestamp": []}',
            '{"mode": false}',
            '{"composite": "95"}',
            '{"composite": true}',
            '{"composite": NaN}',
            '{"ship_gate": "yes"}',
            '{"commit_sha": 99}',
            '{"schema_version": "2"}',
            '{"schema_version": true}',
            '{"attempted_unmeasured": "tests"}',
            '{"attempted_unmeasured": [1]}',
            '{"dimensions": "bad"}',
            '{"dimensions": ["not-a-dict"]}',
            '{"dimensions": [{"name": 5}]}',
            '{"dimensions": [{"name": "t", "score": true}]}',
            '{"dimensions": [{"name": "t", "weight": "0.5"}]}',
            '{"dimensions": [{"name": "t", "raw_value": Infinity}]}',
        ],
    )
    def test_load_score_rejects_malformed_json_shapes_and_types(
        self, scorer: QualityScorer, tmp_path: Path, payload: str
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(payload, encoding="utf-8")
        assert scorer._load_score(bad) is None

    def test_load_score_rejects_symlink_oversized_and_bad_utf8(
        self, scorer: QualityScorer, tmp_path: Path
    ) -> None:
        from linc_crucible.core.scorer import _MAX_SCORE_FILE_BYTES

        target = tmp_path / "real.json"
        target.write_text('{"project": "P"}', encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(target)
        with structlog.testing.capture_logs() as cap_logs:
            assert scorer._load_score(link) is None
        assert any(
            e.get("event") == "score_file_symlink_skipped" and str(link) in e.get("path", "")
            for e in cap_logs
        )

        bad_utf8 = tmp_path / "bad_utf8.json"
        bad_utf8.write_bytes(b"\xff\xfe\x00\x00")
        with structlog.testing.capture_logs() as cap_logs:
            assert scorer._load_score(bad_utf8) is None
        assert any(e.get("event") == "score_file_read_failed" for e in cap_logs)

        at_cap = tmp_path / "at_cap.json"
        at_cap_prefix = b'{"project":"P"}'
        at_cap.write_bytes(
            at_cap_prefix[:-1] + b" " * (_MAX_SCORE_FILE_BYTES - len(at_cap_prefix)) + b"}"
        )
        assert scorer._load_score(at_cap) is not None

        oversized = tmp_path / "huge.json"
        oversized.write_bytes(
            at_cap_prefix[:-1] + b" " * (_MAX_SCORE_FILE_BYTES - len(at_cap_prefix) + 1) + b"}"
        )
        assert scorer._load_score(oversized) is None

        with structlog.testing.capture_logs() as cap_logs:
            list_file = tmp_path / "list.json"
            list_file.write_text("[1]", encoding="utf-8")
            assert scorer._load_score(list_file) is None
            bad_schema = tmp_path / "bad_schema.json"
            bad_schema.write_text('{"composite": "bad"}', encoding="utf-8")
            assert scorer._load_score(bad_schema) is None
        assert any(
            e.get("event") == "score_file_not_object" and e.get("type") == "list" for e in cap_logs
        )
        assert any(e.get("event") == "score_file_invalid_schema" for e in cap_logs)

        minimal = tmp_path / "minimal.json"
        minimal.write_text('{"dimensions": [{}]}', encoding="utf-8")
        loaded_min = scorer._load_score(minimal)
        assert loaded_min is not None
        assert loaded_min.project == ""
        assert loaded_min.timestamp == ""
        assert loaded_min.mode == "quick"
        assert loaded_min.composite == 0.0
        assert loaded_min.ship_gate is False
        assert loaded_min.commit_sha is None
        assert loaded_min.attempted_unmeasured == []
        assert loaded_min.stale is True
        assert len(loaded_min.dimensions) == 1
        d0 = loaded_min.dimensions[0]
        assert (d0.name, d0.raw_value, d0.score, d0.weight, d0.weighted_score) == (
            "",
            0.0,
            0.0,
            0.0,
            0.0,
        )

        valid_ints = tmp_path / "valid_ints.json"
        valid_ints.write_text(
            json.dumps(
                {
                    "project": "P",
                    "timestamp": "T",
                    "mode": "quick",
                    "composite": 90,
                    "ship_gate": True,
                    "commit_sha": "deadbeef",
                    "schema_version": 2,
                    "attempted_unmeasured": [],
                    "dimensions": [
                        {
                            "name": "tests",
                            "raw_value": 1,
                            "score": 100,
                            "weight": 1,
                            "weighted_score": 100,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        loaded = scorer._load_score(valid_ints)
        assert loaded is not None
        assert loaded.composite == 90.0
        assert loaded.commit_sha == "deadbeef"
        assert loaded.stale is False
        assert loaded.dimensions[0].score == 100.0


def test_contract_load_latest_stops_at_the_newest_readable_score(scorer: QualityScorer) -> None:
    """[contract] load_latest returns the newest readable score and reports only the
    corrupt files newer than it: it does not scan past it into older files."""
    project_dir = scorer._scores_dir / "Seq"
    project_dir.mkdir(parents=True)
    good = {
        "project": "Seq",
        "timestamp": "2026-01-03T00:00:00Z",
        "mode": "quick",
        "composite": 91.0,
        "ship_gate": True,
        "commit_sha": "abc123",
        "schema_version": 2,
        "dimensions": [
            {
                "name": "tests",
                "raw_value": 0.9,
                "score": 90.0,
                "weight": 1.0,
                "weighted_score": 90.5,
            }
        ],
    }
    (project_dir / "2026-01-03T00-00-00Z.json").write_text(json.dumps(good))
    (project_dir / "2026-01-02T00-00-00Z.json").write_text("{corrupt")
    (project_dir / "2026-01-01T00-00-00Z.json").write_text(json.dumps({**good, "composite": 10.0}))

    latest = scorer.load_latest("Seq")
    assert latest is not None
    assert latest.composite == 91.0
    assert latest.skipped_files == []
    assert latest.commit_sha == "abc123"
    assert latest.dimensions[0].weighted_score == 90.5


def test_contract_invalid_score_file_warning_names_the_file(scorer: QualityScorer) -> None:
    """[contract] a score file that is not a JSON object is skipped with a warning that
    names that file, so the operator can find it."""
    import structlog.testing

    project_dir = scorer._scores_dir / "Bad"
    project_dir.mkdir(parents=True)
    bad = project_dir / "2026-01-01T00-00-00Z.json"
    bad.write_text("[1, 2]")
    with structlog.testing.capture_logs() as logs:
        assert scorer.load_latest("Bad") is None
    hits = [e for e in logs if e.get("event") == "score_file_not_object"]
    assert hits and hits[0]["path"] == str(bad)
    assert hits[0]["type"] == "list"


def test_contract_save_interrupted_after_temp_write_preserves_previous_score(
    scorer: QualityScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] a save interrupted after the temp write (monkeypatch os.replace to raise)
    leaves the previous newest score intact and no partial final file."""
    import os
    import stat

    s1 = scorer.compute("AtomicProj", test_pass_rate=0.95, lint_issues=0)
    s1.timestamp = "2026-03-01T00:00:00Z"
    p1 = scorer.save(s1)
    assert stat.S_IMODE(p1.stat().st_mode) == 0o600
    original_bytes = p1.read_bytes()

    s2 = scorer.compute("AtomicProj", test_pass_rate=0.20, lint_issues=10)
    s2.timestamp = "2026-03-02T00:00:00Z"
    p2_expected = p1.parent / "2026-03-02T00-00-00Z.json"

    fsync_calls: list[int] = []
    real_fsync = os.fsync

    def _spy_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    replace_calls: list[tuple[Path, Path]] = []
    expected_ts = s2.timestamp

    def _failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        src_p = Path(src)
        dst_p = Path(dst)
        replace_calls.append((src_p, dst_p))
        assert src_p.parent == dst_p.parent
        assert stat.S_IMODE(src_p.stat().st_mode) == 0o600
        assert json.loads(src_p.read_text(encoding="utf-8"))["timestamp"] == expected_ts
        assert len(fsync_calls) >= 1
        raise OSError("simulated crash during os.replace")

    monkeypatch.setattr(os, "fsync", _spy_fsync)
    monkeypatch.setattr(os, "replace", _failing_replace)

    with pytest.raises(OSError, match="simulated crash"):
        scorer.save(s2)

    assert len(replace_calls) == 1
    assert not p2_expected.exists()
    assert [p.name for p in p1.parent.iterdir()] == [p1.name]
    assert p1.read_bytes() == original_bytes
    latest = scorer.load_latest("AtomicProj")
    assert latest is not None
    assert latest.timestamp == "2026-03-01T00:00:00Z"
    assert latest.composite == s1.composite
    assert latest.skipped_files == []

    # Also interrupting an overwrite of the existing timestamp leaves p1 intact.
    s1_overwrite = scorer.compute("AtomicProj", test_pass_rate=0.10, lint_issues=20)
    s1_overwrite.timestamp = "2026-03-01T00:00:00Z"
    expected_ts = s1_overwrite.timestamp
    with pytest.raises(OSError, match="simulated crash"):
        scorer.save(s1_overwrite)
    assert [p.name for p in p1.parent.iterdir()] == [p1.name]
    assert p1.read_bytes() == original_bytes


def test_atomic_save_propagates_temp_creation_failure(
    scorer: QualityScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When NamedTemporaryFile fails before creating a file, save propagates the OSError."""
    import tempfile

    s1 = scorer.compute("FailOpen", test_pass_rate=1.0, lint_issues=0)

    def _fail_tmp(*_args: object, **_kwargs: object) -> object:
        raise OSError("disk full")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _fail_tmp)
    with pytest.raises(OSError, match="disk full"):
        scorer.save(s1)


def test_load_score_refuses_directory_and_nonblock_toctou_fifo(
    scorer: QualityScorer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-regular files (directories, raced FIFOs) are refused without blocking."""
    import os

    dir_entry = tmp_path / "dir.json"
    dir_entry.mkdir()
    assert scorer._load_score(dir_entry) is None

    fifo_entry = tmp_path / "raced_fifo.json"
    os.mkfifo(fifo_entry)
    monkeypatch.setattr(Path, "is_file", lambda self: self == fifo_entry)
    assert scorer._load_score(fifo_entry) is None


def test_contract_fifo_score_file_skipped_within_2s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] a FIFO named like the newest score file: crucible score returns within 2 s
    with that file skipped (fails/hangs on 278647a; run with a thread timeout)."""
    import os
    import threading
    import time

    from click.testing import CliRunner

    import linc_crucible.server_context as server_ctx
    from linc_crucible.assay_tools import crucible_score
    from linc_crucible.cli import cli

    store = tmp_path / "scores"
    scorer = QualityScorer(scores_dir=store)
    s1 = scorer.compute("FifoProj", test_pass_rate=0.9, lint_issues=0)
    s1.timestamp = "2026-03-01T00:00:00Z"
    scorer.save(s1)

    fifo_name = "2026-03-02T00-00-00Z.json"
    fifo_path = store / "FifoProj" / fifo_name
    os.mkfifo(fifo_path)

    monkeypatch.setattr(server_ctx, "_initialized", True)
    monkeypatch.setattr(server_ctx, "_scorer", scorer)

    outcome: dict[str, object] = {}

    def _run_score() -> None:
        try:
            outcome["mcp"] = crucible_score(project="FifoProj")
        except BaseException as exc:  # pragma: no cover - failure capture
            outcome["exc"] = exc

    t0 = time.monotonic()
    thread = threading.Thread(target=_run_score, daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    elapsed = time.monotonic() - t0

    assert not thread.is_alive(), "crucible_score hung on FIFO score file (>2s)"
    assert elapsed < 2.0
    assert "exc" not in outcome
    mcp_res = outcome["mcp"]
    assert isinstance(mcp_res, dict)
    assert mcp_res["timestamp"] == "2026-03-01T00:00:00Z"
    assert mcp_res["skipped_files"] == [fifo_name]

    # Exercise the CLI `crucible score` command as well (now known not to hang in open()).
    with structlog.testing.capture_logs():
        cli_res = CliRunner().invoke(cli, ["score", "FifoProj"])
    assert cli_res.exit_code == 0, cli_res.output
    payload = json.loads(cli_res.stdout)
    assert payload["timestamp"] == "2026-03-01T00:00:00Z"
    assert payload["skipped_files"] == [fifo_name]


def test_contract_skipped_filename_with_esc_printed_escaped_in_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] a skipped filename containing ESC is printed escaped in --human."""
    from click.testing import CliRunner

    import linc_crucible.server_context as server_ctx
    from linc_crucible.cli import cli

    store = tmp_path / "scores"
    scorer = QualityScorer(scores_dir=store)
    s1 = scorer.compute("EscProj", test_pass_rate=1.0, lint_issues=0)
    s1.timestamp = "2026-03-01T00:00:00Z"
    scorer.save(s1)

    esc_name = "2026-03-04T00-00-00Z\x1b[31mEVIL\x1b[0m\u202e.json"
    (store / "EscProj" / esc_name).write_text("{corrupt", encoding="utf-8")

    monkeypatch.setattr(server_ctx, "_initialized", True)
    monkeypatch.setattr(server_ctx, "_scorer", scorer)

    with structlog.testing.capture_logs():
        runner = CliRunner()
        res_latest = runner.invoke(cli, ["--human", "score", "EscProj"])
        assert res_latest.exit_code == 0, res_latest.output
        assert "\x1b" not in res_latest.stdout
        assert "\u202e" not in res_latest.stdout
        assert r"2026-03-04T00-00-00Z\x1b[31mEVIL\x1b[0m\u202e.json" in res_latest.stdout

        res_hist = runner.invoke(cli, ["--human", "score", "EscProj", "--history"])
        assert res_hist.exit_code == 0, res_hist.output
        assert "\x1b" not in res_hist.stdout
        assert "\u202e" not in res_hist.stdout
        assert r"2026-03-04T00-00-00Z\x1b[31mEVIL\x1b[0m\u202e.json" in res_hist.stdout


def test_contract_atomic_save_uses_hidden_tmp_name_and_indented_json(
    scorer: QualityScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[contract] a save writes a hidden `.score-*.tmp` file in the same directory (never a
    `*.json` a reader could pick up), 0600, then replaces it onto the final name; the final
    file is two-space indented UTF-8 JSON."""
    import os as _os
    import stat as _stat

    from linc_crucible.core import scorer as scorer_mod

    seen: list[tuple[Path, Path, int]] = []
    real_replace = _os.replace

    def _spy(src: str | Path, dst: str | Path) -> None:
        src_p = Path(src)
        seen.append((src_p, Path(dst), _stat.S_IMODE(src_p.stat().st_mode)))
        real_replace(src, dst)

    monkeypatch.setattr(scorer_mod.os, "replace", _spy)
    score = scorer.compute(project="AtomicNames", test_pass_rate=1.0, lint_issues=0)
    final = scorer.save(score)

    assert len(seen) == 1
    tmp, dst, mode = seen[0]
    assert dst == final
    assert tmp.parent == final.parent
    assert tmp.name.startswith(".score-") and tmp.name.endswith(".tmp")
    assert mode == 0o600
    assert not tmp.exists()
    text = final.read_text(encoding="utf-8")
    assert text.startswith('{\n  "')
    assert sorted(p.name for p in final.parent.iterdir()) == [final.name]


def test_contract_reject_non_regular_score_decides_each_kind(tmp_path: Path) -> None:
    """[contract] the store's first guard rejects a symlink and a non-regular file (FIFO,
    directory) by itself, and accepts a regular file: it does not rely on the open()
    flags behind it."""
    import os as _os

    from linc_crucible.core.scorer import _reject_non_regular_score

    regular = tmp_path / "a.json"
    regular.write_text("{}")
    link = tmp_path / "b.json"
    link.symlink_to(regular)
    fifo = tmp_path / "c.json"
    _os.mkfifo(fifo)
    directory = tmp_path / "d.json"
    directory.mkdir()

    assert _reject_non_regular_score(regular) is False
    assert _reject_non_regular_score(link) is True
    assert _reject_non_regular_score(fifo) is True
    assert _reject_non_regular_score(directory) is True
