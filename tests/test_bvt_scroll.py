"""scroll_to step + NerveAdapter.scroll_to_element (issue #112).

Scroll-until-visible for content below the fold — SensorKitExplorer#20 was
blocked on exactly this after the S10e->S24 bench swap shrank the effective
viewport. Fake-adapter style follows tests/test_bvt_runner.py /
tests/test_bvt_wait_retries.py: drive the runner with a scripted mock nerve
client rather than a real device.
"""

from __future__ import annotations

from pathlib import Path

from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.runner import BVTRunner
from linc_crucible.core.bvt.schema import BVTStep, BVTSuite, BVTTest
from linc_crucible.core.nerve_client import UiTreeUnavailableError
from tests.conftest import make_mock_nerve_client

_SCREEN = {"screen_size": {"width": 1080, "height": 2280}}


def _make_suite(steps: list[BVTStep], test_name: str = "test1", package: str = "com.test"):
    return BVTSuite(package=package, tests=[BVTTest(name=test_name, steps=steps)])


class TestScrollToStep:
    def test_visible_after_two_swipes(self, tmp_path: Path) -> None:
        """Element appears only once >= 2 swipes have fired; exactly 2 recorded."""
        state = {"swipes": 0}
        target_el = [{"id": "reading_card", "text": "Card", "bounds": "[10,500][300,600]"}]
        swipe_calls: list[tuple] = []

        def _elements(device: str) -> list[dict]:
            return target_el if state["swipes"] >= 2 else []

        def _swipe(device, x1, y1, x2, y2, duration_ms=300):
            state["swipes"] += 1
            swipe_calls.append((device, x1, y1, x2, y2, duration_ms))
            return {"success": True, "action": "swipe"}

        mock = make_mock_nerve_client()
        mock.elements.side_effect = _elements
        mock.elements_strict.side_effect = _elements
        mock.swipe.side_effect = _swipe
        mock.screen_state.side_effect = lambda device: _SCREEN

        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite([BVTStep(action="scroll_to", target="reading_card")])
        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        assert result.tests[0].steps_passed == 1
        assert len(swipe_calls) == 2

    def test_never_visible_fails_after_max_swipes(self, tmp_path: Path) -> None:
        """max_swipes: 3, element never appears -> fails with the swipe-count message."""
        swipe_calls: list[tuple] = []

        def _swipe(device, x1, y1, x2, y2, duration_ms=300):
            swipe_calls.append((device, x1, y1, x2, y2, duration_ms))
            return {"success": True, "action": "swipe"}

        mock = make_mock_nerve_client(elements_response=[])
        mock.swipe.side_effect = _swipe
        mock.screen_state.side_effect = lambda device: _SCREEN

        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite(
            [
                BVTStep(
                    action="scroll_to",
                    target="reading_card",
                    params={"max_swipes": 3},
                ),
            ]
        )
        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        assert "'reading_card' not visible after 3 swipes" in result.tests[0].failure_message
        assert len(swipe_calls) == 3

    def test_already_visible_costs_zero_swipes(self, tmp_path: Path) -> None:
        """An element already on screen passes with no swipe calls."""
        target_el = [{"id": "reading_card", "text": "Card", "bounds": "[10,500][300,600]"}]
        swipe_calls: list[tuple] = []

        mock = make_mock_nerve_client(elements_response=target_el)
        mock.swipe.side_effect = lambda *a, **k: swipe_calls.append(a) or {"success": True}
        mock.screen_state.side_effect = lambda device: _SCREEN

        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite([BVTStep(action="scroll_to", target="reading_card")])
        result = runner.run_suite(suite, device="DEV1")

        assert result.passed
        assert swipe_calls == []

    def test_unavailable_tree_is_harness_error_step(self, tmp_path: Path) -> None:
        """audit F7: an unreadable tree is a HARNESS step, not "not visible after N swipes"."""
        mock = make_mock_nerve_client(elements_response=[])
        mock.elements_strict.side_effect = UiTreeUnavailableError("could not get idle state")
        mock.screen_state.side_effect = lambda device: _SCREEN

        runner = BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path)
        step = BVTStep(action="scroll_to", target="reading_card", params={"max_swipes": 2})
        res = runner._execute_step(step, "DEV1", "com.test", "t")

        assert not res.passed
        assert res.harness_error is True
        assert res.message.startswith("HARNESS:")
        mock.swipe.assert_not_called()

    def test_missing_target_fails_fast(self, tmp_path: Path) -> None:
        mock = make_mock_nerve_client()
        adapter = NerveAdapter(nerve=mock)
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        suite = _make_suite([BVTStep(action="scroll_to", target="")])
        result = runner.run_suite(suite, device="DEV1")

        assert not result.passed
        assert "requires a target" in result.tests[0].failure_message


class TestScrollToElementAdapter:
    """Direct unit coverage of NerveAdapter.scroll_to_element."""

    def test_screen_size_unavailable_gives_up_without_swiping(self) -> None:
        swipe_calls: list[tuple] = []
        mock = make_mock_nerve_client(elements_response=[])
        mock.swipe.side_effect = lambda *a, **k: swipe_calls.append(a) or {"success": True}
        mock.screen_state.side_effect = lambda device: {}

        adapter = NerveAdapter(nerve=mock)
        result = adapter.scroll_to_element("DEV1", "reading_card")

        assert result == {
            "found": False,
            "swipes": 0,
            "error": "screen size unavailable",
        }
        assert swipe_calls == []

    def test_max_swipes_respected(self) -> None:
        swipe_calls: list[tuple] = []
        mock = make_mock_nerve_client(elements_response=[])
        mock.swipe.side_effect = lambda *a, **k: swipe_calls.append(a) or {"success": True}
        mock.screen_state.side_effect = lambda device: _SCREEN

        adapter = NerveAdapter(nerve=mock)
        result = adapter.scroll_to_element("DEV1", "reading_card", max_swipes=2)

        assert result == {"found": False, "swipes": 2}
        assert len(swipe_calls) == 2
