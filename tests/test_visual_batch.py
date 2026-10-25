"""Tests for visual_batch — multi-screen compare orchestration + aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from linc_crucible.core.bvt.visual_batch import aggregate_verdicts, compare_run


class _FakeManager:
    """Stub BaselineManager.compare returning canned results keyed by screen."""

    def __init__(self, by_screen: dict[str, dict[str, Any]]) -> None:
        self._by_screen = by_screen
        self.calls: list[tuple[str, str, str]] = []

    def compare(self, project: str, screen_name: str, candidate_path: Path) -> dict[str, Any]:
        self.calls.append((project, screen_name, str(candidate_path)))
        return dict(self._by_screen[screen_name])


def _ran(verdict: str, score: float = 0.9) -> dict[str, Any]:
    return {"ok": True, "available": True, "skipped": False, "score": score, "verdict": verdict}


def _no_baseline() -> dict[str, Any]:
    return {
        "ok": True,
        "available": True,
        "skipped": True,
        "no_baseline": True,
        "verdict": "UNBASELINED",
    }


def _deps_missing() -> dict[str, Any]:
    return {"ok": False, "available": False, "deps_missing": True, "skipped": True}


class TestAggregateVerdicts:
    def test_all_pass(self) -> None:
        assert aggregate_verdicts([_ran("PASS"), _ran("PASS")]) == "PASS"

    def test_any_review(self) -> None:
        assert aggregate_verdicts([_ran("PASS"), _ran("REVIEW")]) == "REVIEW"

    def test_any_fail_beats_review(self) -> None:
        assert aggregate_verdicts([_ran("REVIEW"), _ran("FAIL")]) == "FAIL"

    def test_all_no_baseline_is_unbaselined(self) -> None:
        # A batch where every screen lacks a golden is a coverage gap, not
        # "nothing was asked to run" — it must read distinctly from SKIPPED.
        assert aggregate_verdicts([_no_baseline(), _no_baseline()]) == "UNBASELINED"

    def test_empty_is_skipped(self) -> None:
        assert aggregate_verdicts([]) == "SKIPPED"

    def test_deps_missing_short_circuits_over_pass(self) -> None:
        # The masquerade case: a deps_missing mixed with a real PASS must NOT be PASS.
        assert aggregate_verdicts([_deps_missing(), _ran("PASS")]) == "DEPS_MISSING"

    def test_deps_missing_short_circuits_over_fail(self) -> None:
        assert aggregate_verdicts([_deps_missing(), _ran("FAIL")]) == "DEPS_MISSING"

    def test_healthy_cases_unchanged_around_unbaselined(self) -> None:
        # AC4: adding UNBASELINED must not disturb any pre-existing verdict.
        assert aggregate_verdicts([]) == "SKIPPED"
        assert aggregate_verdicts([_ran("PASS"), _no_baseline()]) == "PASS"
        assert aggregate_verdicts([_ran("FAIL"), _no_baseline()]) == "FAIL"
        assert aggregate_verdicts([_deps_missing(), _no_baseline()]) == "DEPS_MISSING"
        assert aggregate_verdicts([_no_baseline()]) == "UNBASELINED"

    def test_aggregate_over_pass_error_is_not_pass(self) -> None:
        """Aggregate over [PASS, ERROR] is not PASS."""
        res = aggregate_verdicts([_ran("PASS"), {"ok": False, "verdict": "ERROR"}])
        assert res != "PASS"
        assert res == "ERROR"



class TestCompareRun:
    def test_derives_screen_from_stem_and_aggregates(self) -> None:
        mgr = _FakeManager({"login_home": _ran("PASS"), "login_settings": _ran("REVIEW")})
        out = compare_run(mgr, "MyApp", ["/out/login_home.png", "/out/login_settings.png"])
        assert out["aggregate_verdict"] == "REVIEW"
        assert out["available"] is True
        assert [c["screen"] for c in out["comparisons"]] == ["login_home", "login_settings"]
        # screen name passed to manager.compare is the filename stem
        assert mgr.calls[0][1] == "login_home"

    def test_empty_screenshots(self) -> None:
        out = compare_run(_FakeManager({}), "MyApp", [])
        assert out == {"comparisons": [], "aggregate_verdict": "SKIPPED", "available": True}

    def test_deps_missing_sets_available_false(self) -> None:
        mgr = _FakeManager({"a": _deps_missing(), "b": _ran("PASS")})
        out = compare_run(mgr, "MyApp", ["/o/a.png", "/o/b.png"])
        assert out["aggregate_verdict"] == "DEPS_MISSING"
        assert out["available"] is False

    def test_all_no_baseline_surfaces_unbaselined_available_true(self) -> None:
        mgr = _FakeManager({"a": _no_baseline(), "b": _no_baseline()})
        out = compare_run(mgr, "MyApp", ["/o/a.png", "/o/b.png"])
        assert out["aggregate_verdict"] == "UNBASELINED"
        assert out["available"] is True
