"""[contract] The lazy singleton set is built exactly once, whichever thread calls first.

mcp 2.x runs crucible's plain-``def`` tools (crucible_score, crucible_preflight,
crucible_usage_stats) on anyio worker threads, so concurrent first calls reach
``server_context._ensure_initialized`` from several threads at once. Every heavy
constructor is replaced by a counting fake; the first fake blocks until either all
callers have piled up behind it or a short deadline passes, which turns an unlocked
init into several constructions.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import anyio
import pytest

import linc_crucible.mcp_server  # noqa: F401  (registers every tool)
from linc_crucible import server_context as ctx

_SINGLETONS = (
    "_forge_reader",
    "_deployer",
    "_assayer",
    "_scorer",
    "_preflight_checker",
    "_baseline_manager",
    "_nerve_client",
    "_tree_capture",
    "_bvt_nerve_adapter",
)

_CONSTRUCTORS = {
    "linc_crucible.core.forge_integration": "ForgeProjectReader",
    "linc_crucible.core.scorer": "QualityScorer",
    "linc_crucible.core.nerve_client": "NerveClient",
    "linc_crucible.core.tree_capture": "TreeCapture",
    "linc_crucible.core.deployer": "Deployer",
    "linc_crucible.core.assayer": "Assayer",
    "linc_crucible.core.preflight": "PreflightChecker",
    "linc_crucible.core.baseline": "BaselineManager",
    "linc_crucible.core.bvt.nerve_adapter": "NerveAdapter",
}

_CALLERS = 8
_PILE_UP_DEADLINE_S = 1.0
_GATE_TIMEOUT_S = 5.0


class _Gate:
    """Counts constructions; the first gated construction waits for ``release``."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.entered = 0
        self.release = threading.Event()
        self._lock = threading.Lock()

    def enter(self, name: str) -> None:
        with self._lock:
            self.counts[name] += 1
            if name == "ForgeProjectReader":
                self.entered += 1
        if name == "ForgeProjectReader":
            self.release.wait(_GATE_TIMEOUT_S)

    def wait_for_pile_up(self) -> None:
        deadline = time.monotonic() + _PILE_UP_DEADLINE_S
        while self.entered < _CALLERS and time.monotonic() < deadline:
            time.sleep(0.01)
        self.release.set()


def _install_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Gate:
    import importlib

    gate = _Gate()
    scores_dir = tmp_path / "scores"

    for module_name, cls_name in _CONSTRUCTORS.items():
        module = importlib.import_module(module_name)

        def _make(name: str = cls_name) -> type:
            class _Fake:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    gate.enter(name)
                    self.scores_dir = scores_dir

                def load_latest(self, project: str) -> None:
                    return None

            _Fake.__name__ = f"Fake{name}"
            return _Fake

        monkeypatch.setattr(module, cls_name, _make())

    monkeypatch.setattr(ctx, "_initialized", False)
    for attr in _SINGLETONS:
        monkeypatch.setattr(ctx, attr, None)
    return gate


def test_contract_concurrent_ensure_initialized_builds_each_singleton_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] 8 threads calling _ensure_initialized() construct each singleton exactly once."""
    gate = _install_fakes(monkeypatch, tmp_path)
    start = threading.Barrier(_CALLERS)
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            start.wait(_GATE_TIMEOUT_S)
            ctx._ensure_initialized()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(_CALLERS)]
    for t in threads:
        t.start()
    gate.wait_for_pile_up()
    for t in threads:
        t.join(_GATE_TIMEOUT_S * 2)
        assert not t.is_alive()

    assert not errors, errors
    assert ctx._initialized is True
    assert gate.counts == Counter(dict.fromkeys(_CONSTRUCTORS.values(), 1))
    for attr in _SINGLETONS:
        assert getattr(ctx, attr) is not None, attr


def test_contract_concurrent_crucible_score_calls_build_one_scorer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] 8 concurrent crucible_score calls through MCPServer build one Scorer.

    mcp 2.x runs the sync tool on worker threads, so this is the real-server form of
    the race the direct test pins.
    """
    gate = _install_fakes(monkeypatch, tmp_path)
    results: list[Any] = []

    async def _call() -> None:
        results.append(await ctx.server.call_tool("crucible_score", {"project": "p"}))

    async def _main() -> None:
        async with anyio.create_task_group() as tg:
            for _ in range(_CALLERS):
                tg.start_soon(_call)
            await anyio.to_thread.run_sync(gate.wait_for_pile_up)

    anyio.run(_main)

    assert len(results) == _CALLERS
    assert gate.counts["QualityScorer"] == 1
    assert gate.counts["ForgeProjectReader"] == 1
    for result in results:
        assert result.is_error is False
        assert result.structured_content["error_category"] == "not_found"


def test_contract_failed_init_publishes_nothing_and_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] A constructor that raises publishes no singleton; the next call retries."""
    _install_fakes(monkeypatch, tmp_path).release.set()
    import linc_crucible.core.baseline as baseline_mod

    real_fake = baseline_mod.BaselineManager
    calls = {"n": 0}

    def _flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("baseline dir unreadable")
        return real_fake(*args, **kwargs)

    monkeypatch.setattr(baseline_mod, "BaselineManager", _flaky)

    with pytest.raises(OSError, match="baseline dir unreadable"):
        ctx._ensure_initialized()
    assert ctx._initialized is False
    for attr in _SINGLETONS:
        assert getattr(ctx, attr) is None, attr

    ctx._ensure_initialized()
    assert ctx._initialized is True
    assert all(getattr(ctx, attr) is not None for attr in _SINGLETONS)
