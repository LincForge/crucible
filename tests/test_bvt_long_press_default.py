"""Long-press default duration must EXCEED the Samsung bench devices'
500ms `long_press_timeout`, not tie it — a tie resolves as a click and
Compose `combinedClickable` fires onClick instead of onLongClick
(reactor f01fed85, linc-ops#649).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from linc_crucible.core.bvt.nerve_adapter import DEFAULT_LONG_PRESS_MS, NerveAdapter
from linc_crucible.core.bvt.runner import BVTRunner
from linc_crucible.core.bvt.schema import BVTStep
from tests.conftest import make_mock_nerve_client


class TestDefaultLongPressMsConstant:
    def test_default_exceeds_samsung_tie_threshold(self) -> None:
        # Samsung bench devices' secure long_press_timeout is 500ms; the
        # default must strictly EXCEED it, never tie it.
        assert DEFAULT_LONG_PRESS_MS > 500


class TestRunnerLongPressDefault:
    def test_do_long_press_uses_1200ms_default(self, tmp_path: Path) -> None:
        adapter = MagicMock(spec=NerveAdapter)
        adapter.metrics.return_value = None
        adapter.elements_strict.side_effect = lambda device: adapter.elements(device)
        adapter.long_press_by_id.return_value = {"success": True}
        runner = BVTRunner(nerve=adapter, output_dir=tmp_path)

        step = BVTStep(action="long_press", target="info_button")
        result = runner._execute_step(step, "DEV1", "pkg", "t")

        assert result.passed
        adapter.long_press_by_id.assert_called_once_with(
            "DEV1", "info_button", duration_ms=1200
        )


class TestAdapterLongPressDefault:
    def test_long_press_by_id_uses_1200ms_default(self) -> None:
        mock = make_mock_nerve_client(
            elements_response=[{"id": "info_button", "bounds": "[0,0][100,100]"}]
        )
        mock.long_press.return_value = {"success": True}
        adapter = NerveAdapter(nerve=mock)

        adapter.long_press_by_id("DEV1", "info_button")

        mock.long_press.assert_called_once_with("DEV1", 50, 50, 1200)
