"""T5: BVT conditional branching (if/then/else). reactor b46ad539."""

from __future__ import annotations

from pathlib import Path

import pytest

from linc_crucible.core.bvt import runner as runner_mod
from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.runner import BVTRunner, BVTStepResult
from linc_crucible.core.bvt.schema import BVTStep, BVTSuite, BVTTest
from linc_crucible.core.nerve_client import UiTreeUnavailableError
from tests.conftest import make_mock_nerve_client


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(runner_mod.time, "sleep", lambda *_: None)


def _runner(mock, tmp_path: Path) -> BVTRunner:
    return BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)


DIALOG_ELEMENTS = [
    {"id": "consent_dialog", "text": "", "bounds": "[0,0][100,100]"},
    {"id": "accept_btn", "text": "Accept", "bounds": "[10,10][90,40]"},
]
NO_DIALOG_ELEMENTS = [
    {"id": "home_screen", "text": "", "bounds": "[0,0][100,100]"},
]


def _if_step(condition, then_steps, else_steps=None):
    return BVTStep(
        action="if",
        params={"condition": condition, "then": then_steps, "else": else_steps or []},
    )


# --- Task 5: BVTStepResult plumbing ---


def test_stepresult_has_extra_lists() -> None:
    r = BVTStepResult(action="if", passed=True)
    assert r.extra_screenshots == []
    assert r.extra_performance == []


# --- Task 6: _do_if handler ---


def test_if_condition_true_runs_then(tmp_path) -> None:
    mock = make_mock_nerve_client(elements_response=DIALOG_ELEMENTS)
    step = _if_step({"element": "consent_dialog"}, [BVTStep(action="tap", target="accept_btn")])
    r = _runner(mock, tmp_path)._execute_step(step, "dev", "pkg", "t")
    assert r.passed
    # then-branch tap fired: tap_by_id synthesizes a down→up via long_press at the
    # accept_btn center ([10,10][90,40] -> (50, 25)).
    mock.long_press.assert_called_once_with("dev", 50, 25, 50)
    assert "then" in r.message


def test_if_condition_false_runs_else(tmp_path) -> None:
    mock = make_mock_nerve_client(elements_response=NO_DIALOG_ELEMENTS)
    step = _if_step(
        {"element": "consent_dialog"},
        [BVTStep(action="tap", target="accept_btn")],
        else_steps=[BVTStep(action="assert_element", target="home_screen")],
    )
    r = _runner(mock, tmp_path)._execute_step(step, "dev", "pkg", "t")
    assert r.passed
    assert not mock.tap_calls  # then-branch did NOT run
    assert "else" in r.message


def test_if_text_condition_true(tmp_path) -> None:
    mock = make_mock_nerve_client(elements_response=DIALOG_ELEMENTS)
    step = _if_step({"text": "Accept"}, [BVTStep(action="tap", target="accept_btn")])
    assert _runner(mock, tmp_path)._execute_step(step, "dev", "pkg", "t").passed


def test_if_branch_substep_failure_fails_if_with_attribution(tmp_path) -> None:
    mock = make_mock_nerve_client(elements_response=DIALOG_ELEMENTS)
    # then-branch asserts a missing element -> fails -> if step fails
    step = _if_step(
        {"element": "consent_dialog"},
        [BVTStep(action="assert_element", target="missing_widget")],
    )
    r = _runner(mock, tmp_path)._execute_step(step, "dev", "pkg", "t")
    assert not r.passed
    assert "then-step 1" in r.message
    assert "assert_element" in r.message


def test_if_branch_screenshot_surfaces_in_extra(tmp_path) -> None:
    mock = make_mock_nerve_client(elements_response=DIALOG_ELEMENTS)
    runner = _runner(mock, tmp_path)
    # run_suite normally creates this; we drive _execute_step directly here.
    runner._screenshots_dir.mkdir(parents=True, exist_ok=True)
    step = _if_step(
        {"element": "consent_dialog"},
        [BVTStep(action="screenshot", target="after_accept")],
    )
    r = runner._execute_step(step, "dev", "pkg", "t")
    assert r.passed
    assert len(r.extra_screenshots) == 1
    assert r.extra_screenshots[0].endswith("after_accept.png")


def test_if_branch_screenshot_flows_to_test_result(tmp_path) -> None:
    """End-to-end: a screenshot taken inside a then-branch lands in the test's
    screenshots list (extra_screenshots surfaced by _run_test, not lost)."""
    mock = make_mock_nerve_client(
        elements_response=DIALOG_ELEMENTS,
        screen_state_responses=[{"foreground": {"package": "pkg", "activity": ".M"}}],
    )
    runner = _runner(mock, tmp_path)
    suite = BVTSuite(
        package="pkg",
        tests=[
            BVTTest(
                name="consent_flow",
                steps=[
                    _if_step(
                        {"element": "consent_dialog"},
                        [BVTStep(action="screenshot", target="dismissed")],
                    )
                ],
            )
        ],
    )
    result = runner.run_suite(suite, device="dev")
    assert result.passed
    assert any(s.endswith("dismissed.png") for s in result.tests[0].screenshots)


# --- audit F7: an unreadable tree is a HARNESS fault, never a branch decision ---


def test_if_condition_unavailable_tree_is_harness_and_skips_else(tmp_path) -> None:
    # The condition read fails (lenient read would say [] -> "no dialog"). The
    # else branch must NOT run: an unread tree is not evidence the element is absent.
    mock = make_mock_nerve_client(elements_response=[])
    mock.elements_strict.side_effect = UiTreeUnavailableError("could not get idle state")
    step = _if_step(
        {"element": "consent_dialog"},
        [BVTStep(action="tap_xy", params={"x": 1, "y": 1})],
        else_steps=[BVTStep(action="tap_xy", params={"x": 9, "y": 9})],
    )
    r = _runner(mock, tmp_path)._execute_step(step, "dev", "pkg", "t")
    assert not r.passed
    assert r.harness_error is True
    assert r.message.startswith("HARNESS:")
    assert mock.tap_calls == []  # neither branch ran


def test_if_text_condition_unavailable_tree_is_harness(tmp_path) -> None:
    mock = make_mock_nerve_client(elements_response=[])
    mock.elements_strict.side_effect = UiTreeUnavailableError("dump failed")
    step = _if_step({"text": "Accept"}, [], else_steps=[BVTStep(action="wait", target="0s")])
    r = _runner(mock, tmp_path)._execute_step(step, "dev", "pkg", "t")
    assert not r.passed
    assert r.harness_error is True


def test_run_branch_propagates_substep_harness_error(tmp_path) -> None:
    # A then-step that is itself a harness fault (a non-PNG screenshot) must
    # make the `if` result a harness fault too, with the sub-step attribution.
    mock = make_mock_nerve_client(elements_response=DIALOG_ELEMENTS, screenshot_bytes=b"junk")
    runner = _runner(mock, tmp_path)
    runner._screenshots_dir.mkdir(parents=True, exist_ok=True)
    step = _if_step({"element": "consent_dialog"}, [BVTStep(action="screenshot", target="s")])
    r = runner._execute_step(step, "dev", "pkg", "t")
    assert not r.passed
    assert r.harness_error is True
    assert "then-step 1" in r.message
    assert "HARNESS:" in r.message


def test_run_branch_product_failure_is_not_harness(tmp_path) -> None:
    # Pin: a then-step that fails on a READ tree stays a product fault.
    mock = make_mock_nerve_client(elements_response=DIALOG_ELEMENTS)
    step = _if_step(
        {"element": "consent_dialog"},
        [BVTStep(action="assert_element", target="missing_widget")],
    )
    r = _runner(mock, tmp_path)._execute_step(step, "dev", "pkg", "t")
    assert not r.passed
    assert r.harness_error is False
