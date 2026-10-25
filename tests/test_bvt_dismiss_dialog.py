"""dismiss_dialog must verify the dialog actually closed, not just that the tap
was injected (improvement 0f0d6576).

A tap against an animated/stateful modal can be accepted (input injected) without
dismissing it; the verb must poll until the target text is GONE before passing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from linc_crucible.core.bvt import runner as runner_mod
from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.runner import BVTRunner
from linc_crucible.core.bvt.schema import BVTStep
from linc_crucible.core.nerve_client import UiTreeUnavailableError
from tests.conftest import make_mock_nerve_client

GATE = [{"text": "No thanks", "bounds": "[572,1823][790,1892]"}]
GONE = [{"id": "level_screen", "bounds": "[0,0][100,100]"}]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(runner_mod.time, "sleep", lambda *_: None)


def _runner(mock, tmp_path: Path) -> BVTRunner:
    return BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)


def _step(optional: bool = False, timeout: float = 0.3) -> BVTStep:
    return BVTStep(
        action="dismiss_dialog",
        target="No thanks",
        params={"optional": optional, "timeout": timeout},
    )


def test_passes_only_after_dialog_gone(tmp_path: Path) -> None:
    # Dialog present until the tap fires, then gone. Must confirm-gone before pass.
    state = {"tapped": False}
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: (
        GONE if state["tapped"] else GATE
    )

    def _tap(device, x, y):
        state["tapped"] = True
        return {"success": True, "action": "tap", "x": x, "y": y}

    mock.tap.side_effect = _tap

    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(), "DEV", "pkg", "t")
    assert res.passed
    assert state["tapped"]  # it actually tapped


def test_fails_when_dialog_stays_after_tap(tmp_path: Path) -> None:
    # Tap is "successful" (injected) but the dialog never closes -> must NOT pass.
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: (
        GATE
    )  # always present
    mock.tap.side_effect = lambda device, x, y: {"success": True, "action": "tap", "x": x, "y": y}

    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(), "DEV", "pkg", "t")
    assert not res.passed
    assert "still present" in res.message


def test_optional_passes_when_dialog_stays(tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: GATE
    mock.tap.side_effect = lambda device, x, y: {"success": True, "action": "tap", "x": x, "y": y}

    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(optional=True), "DEV", "pkg", "t")
    assert res.passed


def test_optional_noop_when_never_present(tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: (
        GONE
    )  # button never present
    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(optional=True), "DEV", "pkg", "t")
    assert res.passed
    mock.tap.assert_not_called()


def test_nonoptional_fails_when_never_present(tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: GONE
    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(optional=False), "DEV", "pkg", "t")
    assert not res.passed
    assert "not found" in res.message


def test_dismiss_dialog_accepts_duration_string(tmp_path: Path) -> None:
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: GATE
    mock.tap.side_effect = lambda device, x, y: {"success": True, "action": "tap", "x": x, "y": y}
    step = BVTStep(
        action="dismiss_dialog",
        target="No thanks",
        params={"optional": True, "timeout": "3s"},
    )
    res = _runner(mock, tmp_path)._do_dismiss_dialog(step, "DEV", "pkg", "t")
    assert res.passed


def _unavailable_after(first_reads: list[list[dict]]):
    """Tree reads that return ``first_reads`` in order, then the tree goes dark.

    Returns (strict, lenient) side-effect callables: the STRICT read raises
    UiTreeUnavailableError once ``first_reads`` is exhausted, while the lenient
    read swallows the same fault to ``[]`` (NerveClient's lenient contract).
    """
    strict_q = list(first_reads)
    lenient_q = list(first_reads)

    def strict(device):
        if strict_q:
            return strict_q.pop(0)
        raise UiTreeUnavailableError("could not get idle state")

    def lenient(device):
        return lenient_q.pop(0) if lenient_q else []

    return strict, lenient


def test_unavailable_tree_every_poll_is_harness_error(tmp_path: Path) -> None:
    # audit F7: never read as "dialog not found" when the tree was never read.
    mock = make_mock_nerve_client()
    mock.elements_strict.side_effect, mock.elements.side_effect = _unavailable_after([])
    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(), "DEV", "pkg", "t")
    assert not res.passed
    assert res.harness_error is True
    assert res.message.startswith("HARNESS:")
    assert "not found" not in res.message
    mock.tap.assert_not_called()


def test_tree_dark_after_tap_is_not_a_dismissal(tmp_path: Path) -> None:
    # The button is read and tapped, then the tree goes unreadable: an unread
    # tree must not count as "the dialog is gone" (pre-fix: a false PASS).
    mock = make_mock_nerve_client()
    # Two good reads: the poll that sees the button, then tap_by_text's own read.
    mock.elements_strict.side_effect, mock.elements.side_effect = _unavailable_after([GATE, GATE])
    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(), "DEV", "pkg", "t")
    assert not res.passed
    assert res.harness_error is True
    assert res.message.startswith("HARNESS:")
    assert len(mock.tap_calls) == 1


def test_optional_does_not_mask_unavailable_tree(tmp_path: Path) -> None:
    # `optional` forgives an absent dialog, not an unreadable tree.
    mock = make_mock_nerve_client()
    mock.elements_strict.side_effect, mock.elements.side_effect = _unavailable_after([])
    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(optional=True), "DEV", "pkg", "t")
    assert not res.passed
    assert res.harness_error is True


def test_read_tree_without_dialog_is_not_harness(tmp_path: Path) -> None:
    # Pin: a readable tree lacking the button is the ordinary product verdict.
    mock = make_mock_nerve_client()
    mock.elements.side_effect = mock.elements_strict.side_effect = lambda device: GONE
    res = _runner(mock, tmp_path)._do_dismiss_dialog(_step(), "DEV", "pkg", "t")
    assert not res.passed
    assert not res.harness_error
    assert "not found" in res.message
