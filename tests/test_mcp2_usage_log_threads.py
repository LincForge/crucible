"""[contract] Usage-log writes stay complete when tools run on worker threads (mcp 2.x).

mcp 2.x runs crucible's sync tools on anyio worker threads, so the first calls of a
process can race to create ``$LINC_HOME`` and append to ``mcp-calls.jsonl``. Before the
port a lost mkdir race raised ``FileExistsError`` (errno 17), which ``_log_call`` counted
as a usage-log failure and dropped the record.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from linc_crucible.core import state_paths
from linc_crucible.core.observability import log_mcp_call

_THREADS = 16


def test_contract_concurrent_first_writes_lose_no_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] 16 threads logging into a fresh LINC_HOME write 16 lines, none raise."""
    home = tmp_path / "fresh" / "linc_home"
    monkeypatch.setenv("LINC_HOME", str(home))
    start = threading.Barrier(_THREADS)
    errors: list[BaseException] = []

    def _worker(i: int) -> None:
        try:
            start.wait(5)
            log_mcp_call("linc-crucible", f"tool_{i}", {}, 1.0, True)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
        assert not t.is_alive()

    assert not errors, errors
    lines = (home / "mcp-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == _THREADS
    assert sorted(json.loads(line)["tool"] for line in lines) == sorted(
        f"tool_{i}" for i in range(_THREADS)
    )


def test_contract_ensure_private_dir_tolerates_a_concurrent_creator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] A directory created between the exists() probe and mkdir() is not an error."""
    target = tmp_path / "a" / "b"
    real_mkdir = Path.mkdir

    def _racing_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        real_mkdir(self, *args, **kwargs)  # the "other thread" wins the race
        raise FileExistsError(17, "File exists", str(self))

    monkeypatch.setattr(Path, "mkdir", _racing_mkdir)
    assert state_paths.ensure_private_dir(target) == target
    assert target.is_dir()
