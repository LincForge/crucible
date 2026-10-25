"""T4: wait_for_element step + per-step retries (reactor 5992552d)."""

from __future__ import annotations

from pathlib import Path

import pytest

from linc_crucible.core.bvt import runner as runner_mod
from linc_crucible.core.bvt import step_handlers
from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.runner import (
    BVTRunner,
    BVTStepResult,
    _duration_seconds,
    _parse_retries,
)
from linc_crucible.core.bvt.schema import BVTStep
from linc_crucible.core.nerve_client import UiTreeUnavailableError
from tests.conftest import make_mock_nerve_client


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(runner_mod.time, "sleep", lambda *_: None)


def _runner(mock, tmp_path: Path) -> BVTRunner:
    return BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)


# --- _duration_seconds ---


def test_duration_none_returns_default() -> None:
    assert _duration_seconds(None, 5.0, what="timeout") == 5.0


def test_duration_int_coerces() -> None:
    # YAML writes `timeout: 5` as a Python int — must not crash.
    assert _duration_seconds(5, 5.0, what="timeout") == 5.0


def test_duration_float_coerces() -> None:
    assert _duration_seconds(1.5, 5.0, what="timeout") == 1.5


def test_duration_seconds_string() -> None:
    assert _duration_seconds("5s", 5.0, what="timeout") == 5.0


def test_duration_ms_string() -> None:
    assert _duration_seconds("500ms", 5.0, what="timeout") == 0.5


def test_duration_bare_numeric_string() -> None:
    assert _duration_seconds("2", 5.0, what="timeout") == 2.0


def test_duration_garbage_returns_default_and_warns(monkeypatch) -> None:
    warned = []
    monkeypatch.setattr(step_handlers.log, "warning", lambda *a, **k: warned.append((a, k)))
    assert _duration_seconds("not-a-duration", 5.0, what="timeout") == 5.0
    assert warned  # warned, did not crash


def test_duration_seconds_maximum_below() -> None:
    assert _duration_seconds(10.0, 5.0, what="timeout", maximum=20.0) == 10.0


def test_duration_seconds_maximum_equal() -> None:
    assert _duration_seconds(20.0, 5.0, what="timeout", maximum=20.0) == 20.0


def test_duration_seconds_maximum_above(monkeypatch) -> None:
    warned = []
    monkeypatch.setattr(step_handlers.log, "warning", lambda *a, **k: warned.append((a, k)))
    assert _duration_seconds(30.0, 5.0, what="timeout", maximum=20.0) == 20.0
    assert any(a[0] == "bvt_duration_clamped" for a, _ in warned)


def test_duration_seconds_maximum_none() -> None:
    assert _duration_seconds(30.0, 5.0, what="timeout", maximum=None) == 30.0


def test_wait_sleep_clamped_to_max(tmp_path: Path, monkeypatch) -> None:
    slept = []
    monkeypatch.setattr(step_handlers.time, "sleep", lambda s: slept.append(s))
    warned = []
    monkeypatch.setattr(step_handlers.log, "warning", lambda *a, **k: warned.append((a, k)))
    runner = _runner(make_mock_nerve_client(), tmp_path)
    res = runner._do_wait(BVTStep(action="wait", target="99999s"), "DEV", "pkg", "t")
    assert res.passed
    assert slept == [120.0]
    assert any(a[0] == "bvt_duration_clamped" for a, _ in warned)


# --- _parse_retries ---


def test_retries_default_zero() -> None:
    assert _parse_retries({}) == 0


def test_retries_reads_int() -> None:
    assert _parse_retries({"retries": 3}) == 3


def test_retries_negative_clamped_to_zero() -> None:
    assert _parse_retries({"retries": -2}) == 0


def test_retries_malformed_returns_zero_and_warns(monkeypatch) -> None:
    warned = []
    monkeypatch.setattr(step_handlers.log, "warning", lambda *a, **k: warned.append((a, k)))
    assert _parse_retries({"retries": "lots"}) == 0
    assert warned


def test_retries_clamped_to_max(monkeypatch) -> None:
    warned = []
    monkeypatch.setattr(step_handlers.log, "warning", lambda *a, **k: warned.append((a, k)))
    assert _parse_retries({"retries": 9999}) == step_handlers._MAX_RETRIES
    assert warned


# --- wait_for_element ---

_PRESENT = [{"id": "com.app:id/home_btn", "bounds": "[0,0][100,100]"}]
_ABSENT: list[dict] = [{"id": "com.app:id/other", "bounds": "[0,0][1,1]"}]


def test_wait_for_element_present_immediately(tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: _PRESENT
    step = BVTStep(action="wait_for_element", target="home_btn")
    res = _runner(mock, tmp_path)._do_wait_for_element(step, "DEV", "pkg", "t")
    assert res.passed
    assert mock.elements_strict.call_count == 1  # no polling needed


def test_wait_for_element_appears_after_polls(tmp_path: Path) -> None:
    seq = [_ABSENT, _ABSENT, _PRESENT]
    state = {"i": 0}

    def _elements(device):
        out = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return out

    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = _elements
    step = BVTStep(action="wait_for_element", target="home_btn", params={"timeout": "5s"})
    res = _runner(mock, tmp_path)._do_wait_for_element(step, "DEV", "pkg", "t")
    assert res.passed
    assert state["i"] == 3  # polled until present


def test_wait_for_element_times_out(tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: _ABSENT
    # timeout 0 -> one poll, then deadline passed -> fail (deterministic, instant).
    step = BVTStep(action="wait_for_element", target="home_btn", params={"timeout": 0})
    res = _runner(mock, tmp_path)._do_wait_for_element(step, "DEV", "pkg", "t")
    assert not res.passed
    assert "home_btn" in res.message
    assert "did not appear" in res.message


def test_wait_for_element_unavailable_tree_every_poll_is_harness_error(tmp_path: Path) -> None:
    # audit F7: the STRICT read raises on every poll (the lenient read would
    # have swallowed it to []). Must be a HARNESS step, not "did not appear".
    mock = make_mock_nerve_client(elements_response=[])
    mock.elements_strict.side_effect = UiTreeUnavailableError("could not get idle state")
    step = BVTStep(action="wait_for_element", target="home_btn", params={"timeout": 0})
    res = _runner(mock, tmp_path)._do_wait_for_element(step, "DEV", "pkg", "t")
    assert not res.passed
    assert res.harness_error is True
    assert res.message.startswith("HARNESS:")
    assert "could not get idle state" in res.message
    assert "did not appear" not in res.message


def test_wait_for_element_polls_through_transient_unavailable_tree(tmp_path: Path) -> None:
    # An unreadable tree mid-window is retried like an absent element; the step
    # passes once a later poll reads the element (harness result only at deadline).
    reads: list = [UiTreeUnavailableError("busy"), UiTreeUnavailableError("busy"), _PRESENT]
    mock = make_mock_nerve_client()
    mock.elements_strict.side_effect = reads
    step = BVTStep(action="wait_for_element", target="home_btn", params={"timeout": "5s"})
    res = _runner(mock, tmp_path)._do_wait_for_element(step, "DEV", "pkg", "t")
    assert res.passed
    assert not res.harness_error
    assert mock.elements_strict.call_count == 3


def test_wait_for_element_empty_target_fails_fast(tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    step = BVTStep(action="wait_for_element", target="", params={})
    res = _runner(mock, tmp_path)._do_wait_for_element(step, "DEV", "pkg", "t")
    assert not res.passed
    assert mock.elements.call_count == 0  # never polled
    assert mock.elements_strict.call_count == 0


def test_wait_for_element_dict_form_target(tmp_path: Path) -> None:
    # dict-valued shorthand: `wait_for_element: {target: home_btn, timeout: 5s}`
    # -> step.target="" and params={"target": ...}; handler falls back to params.
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: _PRESENT
    step = BVTStep(action="wait_for_element", target="", params={"target": "home_btn"})
    res = _runner(mock, tmp_path)._do_wait_for_element(step, "DEV", "pkg", "t")
    assert res.passed


def test_wait_for_element_registered_in_dispatch(tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: _PRESENT
    step = BVTStep(action="wait_for_element", target="home_btn")
    res = _runner(mock, tmp_path)._execute_step(step, "DEV", "pkg", "t")
    assert res.passed  # routed through dispatch, not "Unknown action"


# --- retries wrapper ---


def _stub_step_executor(runner, monkeypatch, outcomes):
    """Patch runner._execute_step to return `outcomes` (bools) per call,
    recording call count."""
    calls = {"n": 0}

    def fake(step, device, package, test_name):
        i = calls["n"]
        calls["n"] += 1
        return BVTStepResult(action=step.action, passed=outcomes[i])

    monkeypatch.setattr(runner, "_execute_step", fake)
    return calls


def test_retries_zero_runs_once(tmp_path, monkeypatch) -> None:
    runner = _runner(make_mock_nerve_client(), tmp_path)
    calls = _stub_step_executor(runner, monkeypatch, [False])
    step = BVTStep(action="tap", target="btn")  # no retries
    res = runner._execute_step_with_retries(step, "DEV", "pkg", "t")
    assert not res.passed
    assert calls["n"] == 1


def test_retries_passes_on_second_attempt(tmp_path, monkeypatch) -> None:
    runner = _runner(make_mock_nerve_client(), tmp_path)
    calls = _stub_step_executor(runner, monkeypatch, [False, True])
    step = BVTStep(action="tap", target="btn", params={"retries": 2})
    res = runner._execute_step_with_retries(step, "DEV", "pkg", "t")
    assert res.passed
    assert calls["n"] == 2  # first + one retry


def test_retries_exhausted_returns_last_failure(tmp_path, monkeypatch) -> None:
    runner = _runner(make_mock_nerve_client(), tmp_path)
    calls = _stub_step_executor(runner, monkeypatch, [False, False, False])
    step = BVTStep(action="tap", target="btn", params={"retries": 2})
    res = runner._execute_step_with_retries(step, "DEV", "pkg", "t")
    assert not res.passed
    assert calls["n"] == 3  # first + two retries, all failed


def test_retries_does_not_swallow_exceptions(tmp_path, monkeypatch) -> None:
    # A raised nerve error (daemon down) is a hard failure, NOT a retryable flake.
    runner = _runner(make_mock_nerve_client(), tmp_path)
    calls = {"n": 0}

    def boom(step, device, package, test_name):
        calls["n"] += 1
        raise RuntimeError("daemon down")

    monkeypatch.setattr(runner, "_execute_step", boom)
    step = BVTStep(action="tap", target="btn", params={"retries": 3})
    with pytest.raises(RuntimeError, match="daemon down"):
        runner._execute_step_with_retries(step, "DEV", "pkg", "t")
    assert calls["n"] == 1  # not retried


def test_retry_backoff_override_honored(tmp_path, monkeypatch) -> None:
    runner = _runner(make_mock_nerve_client(), tmp_path)
    _stub_step_executor(runner, monkeypatch, [False, True])
    slept = []
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: slept.append(s))
    step = BVTStep(action="tap", target="btn", params={"retries": 1, "retry_backoff": "500ms"})
    runner._execute_step_with_retries(step, "DEV", "pkg", "t")
    assert slept == [0.5]  # "500ms" parsed via _duration_seconds


def test_retries_on_type_warns(tmp_path, monkeypatch) -> None:
    runner = _runner(make_mock_nerve_client(), tmp_path)
    _stub_step_executor(runner, monkeypatch, [True])
    warned = []
    monkeypatch.setattr(step_handlers.log, "warning", lambda *a, **k: warned.append((a, k)))
    step = BVTStep(action="type", target="hello", params={"retries": 2})
    runner._execute_step_with_retries(step, "DEV", "pkg", "t")
    assert any("non_idempotent" in (a[0] if a else "") for a, _ in warned)


def test_run_test_routes_through_retries(tmp_path, monkeypatch) -> None:
    # _run_test must call the retry wrapper, not _execute_step directly.
    from linc_crucible.core.bvt.schema import BVTTest

    runner = _runner(make_mock_nerve_client(), tmp_path)
    monkeypatch.setattr(runner, "_reset_app", lambda *a, **k: None)
    seen = {"via_wrapper": False}
    orig = runner._execute_step_with_retries

    def spy(step, device, package, test_name):
        seen["via_wrapper"] = True
        return orig(step, device, package, test_name)

    monkeypatch.setattr(runner, "_execute_step_with_retries", spy)
    monkeypatch.setattr(
        runner,
        "_execute_step",
        lambda step, d, p, t: BVTStepResult(action=step.action, passed=True),
    )
    test = BVTTest(name="x", steps=[BVTStep(action="tap", target="btn")], package="pkg")
    runner._run_test(test, "DEV", "pkg")
    assert seen["via_wrapper"]


# --- backward compatibility ---


def test_existing_suite_runs_unchanged(tmp_path, monkeypatch) -> None:
    # A suite with neither wait_for_element nor retries behaves exactly as before:
    # each step dispatched exactly once.
    from linc_crucible.core.bvt.schema import BVTTest

    runner = _runner(make_mock_nerve_client(), tmp_path)
    monkeypatch.setattr(runner, "_reset_app", lambda *a, **k: None)
    calls = {"n": 0}

    def fake(step, device, package, test_name):
        calls["n"] += 1
        return BVTStepResult(action=step.action, passed=True)

    monkeypatch.setattr(runner, "_execute_step", fake)
    test = BVTTest(
        name="legacy",
        steps=[
            BVTStep(action="tap", target="a"),
            BVTStep(action="assert_element", target="b"),
        ],
        package="pkg",
    )
    runner._run_test(test, "DEV", "pkg")
    assert calls["n"] == 2  # one dispatch per step, no extra attempts


def test_schema_docstring_documents_new_forms() -> None:
    from linc_crucible.core.bvt import schema

    doc = schema.__doc__ or ""
    assert "wait_for_element" in doc
    assert "retries" in doc
