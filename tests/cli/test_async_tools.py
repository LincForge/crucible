"""End-to-end proof that crucible's ASYNC tools work through the real (unstubbed)
LocalTransport — `crucible extract-assets` and `crucible bvt`.

This proves the async bridge (`LocalTransport.call` -> `anyio.run(fn(**payload))` for a
coroutine function, kit `test_local_sync_and_async`), not just the parity count from
`tests/test_cli_parity.py`. Both tools' heavy backends are monkeypatched at the CORE
boundary (`asset_extractor.extract_assets`, `BVTRunner`, `load_bvt_dir`) — no real video
decode, no real build, no real device — so a coroutine that is scheduled but never awaited
(the historical class of async-CLI-bridge bug this test guards against) would surface as a
`RuntimeWarning: coroutine ... was never awaited` on stderr.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import linc_crucible.server_context as server_ctx
from linc_crucible.cli import cli
from linc_crucible.core.device_policy import DevicePolicy


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _no_real_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze server_context's lazy singleton init so per-test mocks below aren't clobbered.

    `_ensure_initialized()` runs unconditionally as the shared LocalTransport's `before`
    hook on every mounted command; forcing `_initialized = True` up front makes it a no-op
    for the duration of the test, so a mock assigned to `server_ctx._forge_reader` etc.
    survives the whole CLI round trip untouched.
    """
    monkeypatch.setattr(server_ctx, "_initialized", True)


class TestExtractAssetsAsync:
    def test_extract_assets_end_to_end(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        stub_result = {
            "assets": [{"filename": "shot1.png", "rationale": "hero shot", "variants": {}}],
            "fallback_used": False,
            "manifest_path": str(tmp_path / "manifest.json"),
        }
        # Core boundary: the heavy frame-sampling/AI-ranking function, imported by name
        # into asset_tools at module load — patch it there, not at its definition site.
        monkeypatch.setattr(
            "linc_crucible.asset_tools.extract_assets",
            lambda video_path, config: stub_result,
        )

        video_path = tmp_path / "walkthrough.mp4"
        video_path.write_bytes(b"\x00")  # never read — extract_assets is stubbed

        result = runner.invoke(
            cli, ["extract-assets", "--video-path", str(video_path), "--max-assets", "1"]
        )

        assert result.exit_code == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        assert json.loads(result.stdout) == stub_result
        assert "RuntimeWarning" not in result.stderr
        assert "was never awaited" not in result.stderr

    def test_extract_assets_failing_call(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A missing/unreadable video is the evidence AC's 'one failing call' example."""
        video_path = tmp_path / "missing.mp4"  # never created

        result = runner.invoke(
            cli, ["extract-assets", "--video-path", str(video_path), "--max-assets", "1"]
        )

        # asset_tools maps FileNotFoundError/ValueError to a JSON *payload* with an "error"
        # key (not a raised exception) — the kit's classify() still turns that into exit 1
        # with empty stdout and the envelope on stderr, same as any other tool failure.
        assert result.exit_code == 1
        assert result.stdout == ""
        envelope = json.loads(result.stderr)
        assert envelope["error"]
        assert "RuntimeWarning" not in result.stderr


class TestBvtAsync:
    def _fake_suite(self) -> MagicMock:
        suite = MagicMock()
        suite.package = None
        suite.tests = []
        return suite

    def test_bvt_end_to_end(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from linc_crucible.core.bvt.step_handlers import BVTRunResult

        project_dir = tmp_path / "TestProject"
        (project_dir / ".bvt").mkdir(parents=True)  # crucible_bvt only checks .is_dir()

        forge_project = MagicMock()
        forge_project.path = str(project_dir)
        monkeypatch.setattr(server_ctx, "_forge_reader", MagicMock(get=lambda name: forge_project))

        bvt_adapter = MagicMock()
        bvt_adapter.available = True
        monkeypatch.setattr(server_ctx, "_bvt_nerve_adapter", bvt_adapter)

        nerve_client = MagicMock()
        nerve_client.pool.return_value = {}  # falsy reservations -> skip the reservation guard
        monkeypatch.setattr(server_ctx, "_nerve_client", nerve_client)
        # A loaded policy with no personal devices — without it the host's (or CI's
        # absent) device-fleet.yaml decides, and an absent registry fails closed.
        monkeypatch.setattr(
            server_ctx,
            "_cached_device_policy",
            DevicePolicy(personal_serials=frozenset(), loaded=True, source="test-fleet.yaml"),
        )

        # Core boundary #1: yaml loading — no real .bvt yaml needed.
        monkeypatch.setattr(
            "linc_crucible.core.bvt.schema.load_bvt_dir", lambda _dir: self._fake_suite()
        )

        # Core boundary #2: the runner that would otherwise drive a real device via nerve.
        fake_run_result = BVTRunResult(device="EMULATOR-X", package="com.example.app", tests=[])
        fake_runner = MagicMock()
        fake_runner.run_suite.return_value = fake_run_result
        monkeypatch.setattr(
            "linc_crucible.core.bvt.runner.BVTRunner", lambda **kwargs: fake_runner
        )

        result = runner.invoke(
            cli,
            [
                "bvt",
                "--project",
                "TestProject",
                "--device",
                "EMULATOR-X",
            ],
        )

        assert result.exit_code == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        payload = json.loads(result.stdout)
        assert payload["ran"] is True
        assert payload["device"] == "EMULATOR-X"
        assert payload["total_run"] == 0
        assert "RuntimeWarning" not in result.stderr
        assert "was never awaited" not in result.stderr

    def test_bvt_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["bvt", "--help"])
        assert result.exit_code == 0
        assert "--project" in result.stdout
        assert "--device" in result.stdout
