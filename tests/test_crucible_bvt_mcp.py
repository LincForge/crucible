"""Sanity tests for crucible_bvt MCP tool registration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import structlog

from linc_crucible.bvt_tools import _BVT_PRECONDITION_REASONS
from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.runner import BVTRunResult, BVTTestResult
from linc_crucible.core.device_policy import DevicePolicy
from linc_crucible.core.nerve_client import NerveClient
from linc_crucible.mcp_server import server

_PERSONAL_WATCH_SERIAL = "PERSONALWATCH01"


@pytest.fixture(autouse=True)
def _bench_device_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded policy with no personal devices, so these tests never depend on
    the host's real device-fleet.yaml (absent in CI -> the policy fails closed)."""
    import linc_crucible.server_context as ctx

    monkeypatch.setattr(
        ctx,
        "_cached_device_policy",
        DevicePolicy(personal_serials=frozenset(), loaded=True, source="test-fleet.yaml"),
    )


def _set_personal_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    import linc_crucible.server_context as ctx

    monkeypatch.setattr(
        ctx,
        "_cached_device_policy",
        DevicePolicy(
            personal_serials=frozenset({_PERSONAL_WATCH_SERIAL}),
            loaded=True,
            source="test-fleet.yaml",
        ),
    )


@pytest.fixture(autouse=True)
def captured_logs() -> Iterator[list[dict[str, Any]]]:
    """Capture structlog events for every test here instead of printing them.

    crucible's loggers bind their output stream at import time, and importing
    ``linc_crucible.cli`` points structlog at whatever ``sys.stderr`` is current — a
    pytest capture stream. Under in-process re-runs (mutmut's stats + clean-test
    sessions) that stream is closed by the next session, so any test here that logged
    died with "I/O operation on closed file". ``capture_logs`` swaps the shared
    processor chain so events are recorded and never reach the stream; it also lets
    tests assert the exact events logged.
    """
    with structlog.testing.capture_logs() as logs:
        yield logs


def test_crucible_bvt_is_registered() -> None:
    """The tool must appear in the FastMCP server's tool registry."""
    import asyncio

    tools = asyncio.run(server.list_tools())
    names = [t.name for t in tools]
    assert "crucible_bvt" in names, f"crucible_bvt not registered. Found: {names}"


def test_crucible_bvt_has_complete_docstring() -> None:
    """The docstring must include the fail_fast behavior note (Lattner+Hotz finding)."""
    import asyncio

    tools = asyncio.run(server.list_tools())
    bvt = next(t for t in tools if t.name == "crucible_bvt")
    desc = bvt.description or ""
    # Behavior must be explicit per spec 0351fdd7
    assert "fail-fast" in desc.lower() or "fail fast" in desc.lower()
    assert "between tests" in desc.lower()


def _fake_run_result() -> BVTRunResult:
    return BVTRunResult(
        device="DEV1",
        package="com.x",
        tests=[
            BVTTestResult(
                name="login",
                passed=True,
                steps_run=1,
                steps_passed=1,
                screenshots=["/out/login_home.png"],
            )
        ],
    )


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.server_context._baseline_manager")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_compare_against_baseline_true_attaches_block(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_bm: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()
    mock_bm.compare.return_value = {
        "ok": True,
        "available": True,
        "skipped": False,
        "score": 0.97,
        "verdict": "PASS",
    }

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV1",
            output_dir=str(tmp_path / "out"),
            compare_against_baseline=True,
        )
    )
    assert out["passed"] is True  # visual compare does NOT flip functional pass
    assert out["visual_advisory"] == "PASS"
    assert out["visual_compare"]["aggregate_verdict"] == "PASS"
    assert out["visual_compare"]["comparisons"][0]["screen"] == "login_home"
    mock_bm.compare.assert_called_once()


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.server_context._baseline_manager")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
@patch("linc_crucible.core.bvt.visual_batch.compare_run")
async def test_crucible_bvt_compare_run_exception_returns_suite_result(
    mock_compare_run: MagicMock,
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_bm: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()
    mock_compare_run.side_effect = RuntimeError("visual compare crashed")

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV1",
            output_dir=str(tmp_path / "out"),
            compare_against_baseline=True,
        )
    )
    assert out["ran"] is True
    assert out["passed"] is True
    assert len(out["tests"]) == 1
    assert out["visual_compare"]["verdict"] == "ERROR"
    assert "visual compare crashed" in out["visual_compare"]["error"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.server_context._baseline_manager")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_compare_against_baseline_false_is_inert(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_bm: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV1",
            output_dir=str(tmp_path / "out"),
        )
    )
    assert out["visual_advisory"] is None
    assert "visual_compare" not in out
    mock_bm.compare.assert_not_called()


def _fake_run_result_with_harness_error() -> BVTRunResult:
    """One clean pass + one test whose fail-fast step was a HARNESS fault."""
    return BVTRunResult(
        device="DEV1",
        package="com.x",
        tests=[
            BVTTestResult(
                name="login",
                passed=True,
                steps_run=1,
                steps_passed=1,
                screenshots=["/out/login_home.png"],
            ),
            BVTTestResult(
                name="settings_visible",
                passed=False,
                steps_run=1,
                steps_passed=0,
                failure_message="HARNESS: UI tree unavailable — could not get idle state",
                harness_error=True,
            ),
        ],
    )


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.server_context._baseline_manager")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_result_carries_total_harness_errors_and_names(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_bm: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """A HARNESS-faulted test is surfaced via total_harness_errors/harness_errors
    without displacing any pre-existing key (reactor 6a4164a5)."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result_with_harness_error()

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV1",
            output_dir=str(tmp_path / "out"),
        )
    )

    assert out["total_harness_errors"] == 1
    assert out["harness_errors"] == ["settings_visible"]
    # Every pre-existing key must still be present (additive-only contract).
    for key in (
        "ran",
        "device",
        "package",
        "tests",
        "total_run",
        "total_passed",
        "total_failed",
        "total_skipped",
        "passed",
        "duration_ms",
        "output_dir",
        "visual_advisory",
    ):
        assert key in out, f"missing pre-existing key: {key}"
    assert out["total_failed"] == 1
    assert out["passed"] is False


# ---------------------------------------------------------------------------
# Precondition failures → structured {ran: false, reason} (issue #81)
# ---------------------------------------------------------------------------


def _failing_run_result() -> BVTRunResult:
    return BVTRunResult(
        device="DEV1",
        package="com.x",
        tests=[
            BVTTestResult(
                name="login",
                passed=False,
                steps_run=2,
                steps_passed=1,
                failure_message="Step 2 'tap:btn_login' failed: Element 'btn_login' not found",
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("read timed out"),
    ],
    ids=["connect-error", "read-timeout"],
)
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._forge_reader")
async def test_daemon_unreachable_returns_structured_payload(
    mock_forge: MagicMock,
    mock_init: MagicMock,
    transport_exc: Exception,
    tmp_path,
) -> None:
    """Daemon down (connect error / read timeout) → {ran: false, reason}, no raise."""
    from linc_crucible import server_context as ctx
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))

    # Real adapter over a real NerveClient whose transport always fails —
    # exercises the genuine health-probe path, not a stubbed `available`.
    client = NerveClient(max_retries=1, retry_backoff=0.0)
    client._http = MagicMock()
    client._http.get.side_effect = transport_exc
    client._http.post.side_effect = transport_exc

    with patch.object(ctx, "_bvt_nerve_adapter", NerveAdapter(nerve=client)):
        out = json.loads(await crucible_bvt(project="MyApp", device="DEV1"))

    assert out["ran"] is False
    assert out["reason"] == "nerve-daemon-unreachable"
    assert out["detail"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._forge_reader")
async def test_unknown_project_returns_structured_payload(
    mock_forge: MagicMock,
    mock_init: MagicMock,
) -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    mock_forge.get.return_value = None

    out = json.loads(await crucible_bvt(project="NoSuchApp", device="DEV1"))
    assert out["ran"] is False
    assert out["reason"] == "unknown-project"
    assert "NoSuchApp" in out["detail"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._forge_reader")
async def test_suite_dir_missing_returns_structured_payload(
    mock_forge: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    # Project exists but has no .bvt directory.
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))

    out = json.loads(await crucible_bvt(project="MyApp", device="DEV1"))
    assert out["ran"] is False
    assert out["reason"] == "suite-dir-missing"
    assert ".bvt" in out["detail"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_no_tests_match_tags_returns_structured_payload(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(
        tests=[SimpleNamespace(name="t1", tags=["phone"])], package="com.x"
    )

    out = json.loads(await crucible_bvt(project="MyApp", device="DEV1", tags=["watch"]))
    assert out["ran"] is False
    assert out["reason"] == "no-tests-match-tags"
    assert "watch" in out["detail"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_device_not_ready_returns_structured_payload(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """Device-readiness gate (real BVTRunner, no runner patch) → device-not-ready."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_adapter.ensure_ready.return_value = {"ready": False, "reason": "device asleep"}
    mock_load.return_value = SimpleNamespace(
        tests=[SimpleNamespace(name="t1", tags=[])], package="com.x"
    )

    out = json.loads(
        await crucible_bvt(project="MyApp", device="DEV1", output_dir=str(tmp_path / "out"))
    )
    assert out["ran"] is False
    assert out["reason"] == "device-not-ready"
    assert "device asleep" in out["detail"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_device_reserved_returns_structured_payload(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """A device reserved by another nerve session (issue #110) is a precondition, not a run."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_nerve_client.pool.return_value = {
        "reservations": [
            {
                "device_serial": "R58N000TEST",
                "session_id": "other-sess",
                "purpose": "bench-grid",
                "age_seconds": 42,
            }
        ],
        "count": 1,
    }

    out = json.loads(await crucible_bvt(project="MyApp", device="R58N000TEST"))
    assert out["ran"] is False
    assert out["reason"] == "device-reserved"
    assert "other-sess" in out["detail"]
    assert "R58N000TEST" in out["detail"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_reservation_check_swallows_connection_error(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_nerve_client: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """pool() raising ConnectionError degrades gracefully — bvt runs as if unreserved."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_nerve_client.pool.side_effect = ConnectionError("nerve unreachable")
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()

    out = json.loads(
        await crucible_bvt(project="MyApp", device="DEV1", output_dir=str(tmp_path / "out"))
    )
    assert out["ran"] is True


def test_precondition_reason_codes_are_distinct() -> None:
    """One DISTINCT machine-readable code per precondition (no generic bucket)."""
    from linc_crucible.bvt_tools import _BVT_PRECONDITION_REASONS

    assert len(set(_BVT_PRECONDITION_REASONS)) == len(_BVT_PRECONDITION_REASONS)
    assert set(_BVT_PRECONDITION_REASONS) == {
        "unknown-project",
        "suite-dir-missing",
        "nerve-daemon-unreachable",
        "no-tests-match-tags",
        "device-not-ready",
        "package-mismatch",
        "device-reserved",
        "platform-mismatch",
    }


def test_bvt_precondition_unknown_reason_raises_value_error(
    captured_logs: list[dict[str, Any]],
) -> None:
    """[contract] _bvt_precondition raises ValueError naming the reason for a reason
    not in _BVT_PRECONDITION_REASONS, and logs nothing.
    """
    from linc_crucible.bvt_tools import _bvt_precondition

    with pytest.raises(ValueError, match=r"^unknown precondition reason: unknown-reason$"):
        _bvt_precondition("unknown-reason", "detail message")
    assert captured_logs == []


@pytest.mark.parametrize("reason", _BVT_PRECONDITION_REASONS)
def test_bvt_precondition_every_known_reason_accepted(
    reason: str, captured_logs: list[dict[str, Any]]
) -> None:
    """[pin] Every reason in _BVT_PRECONDITION_REASONS is accepted, logged, and returned."""
    from linc_crucible.bvt_tools import _bvt_precondition

    raw = _bvt_precondition(reason, "detail message")
    assert json.loads(raw) == {"ran": False, "reason": reason, "detail": "detail message"}
    assert captured_logs == [
        {
            "event": "crucible_bvt_precondition_failed",
            "log_level": "warning",
            "domain": "build_pipeline",
            "reason": reason,
            "detail": "detail message",
        }
    ]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.package_resolver.resolve_effective_package")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_yaml_package_mismatch_returns_structured_payload(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_resolve: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """YAML package: contradicting the real applicationId must not run (b02fcbc6 class)."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(
        tests=[SimpleNamespace(name="t1", tags=[])], package="com.stale.yaml"
    )
    mock_resolve.return_value = "com.lincmobile.real"

    out = json.loads(await crucible_bvt(project="MyApp", device="DEV1"))
    assert out["ran"] is False
    assert out["reason"] == "package-mismatch"
    assert "com.stale.yaml" in out["detail"]
    assert "com.lincmobile.real" in out["detail"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.package_resolver.resolve_effective_package")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_unresolvable_package_does_not_block_run(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_resolve: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """The guard fires only on a POSITIVE mismatch — resolver returning None trusts YAML."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(
        tests=[SimpleNamespace(name="t1", tags=[])], package="com.x"
    )
    mock_resolve.return_value = None
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()

    out = json.loads(
        await crucible_bvt(project="MyApp", device="DEV1", output_dir=str(tmp_path / "out"))
    )
    assert out["ran"] is True


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_per_test_failure_still_returns_ran_true(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """Regression: real test failures stay ran=true / passed=false with detail."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.return_value = _failing_run_result()

    out = json.loads(
        await crucible_bvt(project="MyApp", device="DEV1", output_dir=str(tmp_path / "out"))
    )
    assert out["ran"] is True
    assert out["passed"] is False
    assert out["total_failed"] == 1
    assert "btn_login" in out["tests"][0]["failure_message"]
    assert "reason" not in out


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_unexpected_exception_still_raises(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """NEGATIVE: an unexpected exception must NOT be converted to ran=false."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.side_effect = ValueError("crucible bug")

    with pytest.raises(ValueError, match="crucible bug"):
        await crucible_bvt(project="MyApp", device="DEV1", output_dir=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# exclude_tags (issue #111) — tags is include-only today; exclude_tags lets
# callers drop tests AFTER the include filter (mirrors nightly runner
# semantics: linc_ops/daily_device_tests/runner.py:380).
# ---------------------------------------------------------------------------


def _make_result_from_suite(suite, device, app_path=None, budget_s=None):
    """Fake BVTRunner.run_suite that reports exactly the tests it was handed.

    Lets the assertion on the JSON result (total_run, test names) genuinely
    reflect the selection logic in crucible_bvt rather than a hardcoded
    fixture — the thing under test IS the filtering, not the runner.
    """
    tests = [
        BVTTestResult(name=t.name, passed=True, steps_run=1, steps_passed=1) for t in suite.tests
    ]
    return BVTRunResult(device=device, package="com.x", tests=tests)


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_exclude_tags_applied_after_include_filter(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """tags=["watch"], exclude_tags=["ios"] drops the watch+ios test, keeps watch-only."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(
        tests=[
            SimpleNamespace(name="watch_ios_sim", tags=["smoke", "watch", "ios"]),
            SimpleNamespace(name="watch_only", tags=["watch"]),
        ],
        package="com.x",
    )
    mock_runner_cls.return_value.run_suite.side_effect = _make_result_from_suite

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV1",
            tags=["watch"],
            exclude_tags=["ios"],
            output_dir=str(tmp_path / "out"),
        )
    )
    assert out["ran"] is True
    assert out["total_run"] == 1
    assert out["tests"][0]["name"] == "watch_only"


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_exclude_tags_removing_all_returns_no_tests_match_tags(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """exclude_tags emptying the (already tag-included) suite is a precondition, not a crash."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(
        tests=[SimpleNamespace(name="watch_ios_sim", tags=["smoke", "watch", "ios"])],
        package="com.x",
    )

    out = json.loads(
        await crucible_bvt(project="MyApp", device="DEV1", tags=["watch"], exclude_tags=["ios"])
    )
    assert out["ran"] is False
    assert out["reason"] == "no-tests-match-tags"
    assert "exclude_tags" in out["detail"]


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_exclude_tags_default_none_is_byte_identical_to_today(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """Default exclude_tags=None must not change today's include-only selection."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(
        tests=[
            SimpleNamespace(name="watch_ios_sim", tags=["smoke", "watch", "ios"]),
            SimpleNamespace(name="watch_only", tags=["watch"]),
        ],
        package="com.x",
    )
    mock_runner_cls.return_value.run_suite.side_effect = _make_result_from_suite

    out = json.loads(
        await crucible_bvt(
            project="MyApp", device="DEV1", tags=["watch"], output_dir=str(tmp_path / "out")
        )
    )
    assert out["ran"] is True
    assert out["total_run"] == 2


def test_crucible_bvt_signature_has_exclude_tags_default_none() -> None:
    """Pins the signature shape: exclude_tags exists with default None."""
    import inspect

    from linc_crucible.bvt_tools import crucible_bvt

    params = inspect.signature(crucible_bvt).parameters
    assert "exclude_tags" in params
    assert params["exclude_tags"].default is None


# ---------------------------------------------------------------------------
# app_path (issue #114) — thread the optional built-artifact path through to
# BVTRunner.run_suite so interactive/MCP-driven iOS-sim cold_start runs can
# reinstall after the uninstall-based clear (runner.py already accepts it —
# only the crucible_bvt call site was missing the kwarg).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_app_path_forwarded_to_run_suite(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """Passing app_path forwards it as an explicit keyword to run_suite."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()

    await crucible_bvt(
        project="MyApp",
        device="DEV1",
        output_dir=str(tmp_path / "out"),
        app_path="/tmp/Foo.app",
    )

    mock_runner_cls.return_value.run_suite.assert_called_once()
    _, call_kwargs = mock_runner_cls.return_value.run_suite.call_args
    assert call_kwargs["app_path"] == "/tmp/Foo.app"


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_app_path_default_is_forwarded_as_explicit_none(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """Omitting app_path still passes it through explicitly as None."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()

    await crucible_bvt(
        project="MyApp",
        device="DEV1",
        output_dir=str(tmp_path / "out"),
    )

    mock_runner_cls.return_value.run_suite.assert_called_once()
    _, call_kwargs = mock_runner_cls.return_value.run_suite.call_args
    assert "app_path" in call_kwargs
    assert call_kwargs["app_path"] is None


def test_crucible_bvt_docstring_documents_app_path() -> None:
    """The docstring Args section must mention app_path (per issue #114 AC)."""
    from linc_crucible.bvt_tools import crucible_bvt

    assert "app_path:" in crucible_bvt.__doc__


# ---------------------------------------------------------------------------
# platform mismatch (reactor cf505981) — a test tagged for the platform the
# resolved device is NOT never reaches the runner; it is reported as a
# skipped BVTTestResult instead of a bogus launch/assert failure.
# ---------------------------------------------------------------------------


def test_device_platform_exact_match() -> None:
    from linc_crucible.bvt_tools import _device_platform

    devices = [{"serial": "DEV", "platform": "android"}]
    assert _device_platform(devices, "DEV") == "android"


def test_device_platform_absent_serial() -> None:
    from linc_crucible.bvt_tools import _device_platform

    devices = [{"serial": "OTHER", "platform": "android"}]
    assert _device_platform(devices, "DEV") is None


def test_device_platform_missing_platform_key() -> None:
    from linc_crucible.bvt_tools import _device_platform

    devices = [{"serial": "DEV"}]
    assert _device_platform(devices, "DEV") is None


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_platform_mismatch_skips_wrong_platform_test(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """An android device runs the android test and skips the ios one, reason attached."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_adapter.discover.return_value = [{"serial": "DEV", "platform": "android"}]
    mock_load.return_value = SimpleNamespace(
        tests=[
            SimpleNamespace(name="a", tags=["smoke", "android"]),
            SimpleNamespace(name="b", tags=["smoke", "ios"]),
        ],
        package="com.x",
    )
    mock_runner_cls.return_value.run_suite.side_effect = _make_result_from_suite

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV",
            tags=["smoke"],
            output_dir=str(tmp_path / "out"),
        )
    )

    call_args, _call_kwargs = mock_runner_cls.return_value.run_suite.call_args
    suite_arg = call_args[0]
    assert [t.name for t in suite_arg.tests] == ["a"]

    by_name = {t["name"]: t for t in out["tests"]}
    assert by_name["a"]["skipped"] is False
    assert by_name["b"]["skipped"] is True
    assert "platform mismatch" in by_name["b"]["skip_reason"]
    assert "ios" in by_name["b"]["skip_reason"]
    assert out["total_skipped"] == 1
    assert out["total_run"] == 2
    assert out["ran"] is True


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_platform_mismatch_all_remaining_returns_precondition(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """Every remaining test mismatched → precondition, run_suite never called."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_adapter.discover.return_value = [{"serial": "DEV", "platform": "android"}]
    mock_load.return_value = SimpleNamespace(
        tests=[SimpleNamespace(name="b", tags=["smoke", "ios"])],
        package="com.x",
    )

    out = json.loads(await crucible_bvt(project="MyApp", device="DEV", tags=["smoke"]))

    assert out == {
        "ran": False,
        "reason": "platform-mismatch",
        "detail": out["detail"],
    }
    assert "ios" in out["detail"]
    assert "DEV" in out["detail"]
    assert "android" in out["detail"]
    mock_runner_cls.return_value.run_suite.assert_not_called()


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_platform_mismatch_unknown_device_leaves_suite_untouched(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """discover() not listing the device → both tests reach run_suite unchanged."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_adapter.discover.return_value = []
    mock_load.return_value = SimpleNamespace(
        tests=[
            SimpleNamespace(name="a", tags=["smoke", "android"]),
            SimpleNamespace(name="b", tags=["smoke", "ios"]),
        ],
        package="com.x",
    )
    mock_runner_cls.return_value.run_suite.side_effect = _make_result_from_suite

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV",
            tags=["smoke"],
            output_dir=str(tmp_path / "out"),
        )
    )

    call_args, _ = mock_runner_cls.return_value.run_suite.call_args
    assert [t.name for t in call_args[0].tests] == ["a", "b"]
    assert out["total_run"] == 2
    assert out["total_skipped"] == 0


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_platform_mismatch_discover_raises_leaves_suite_untouched(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """discover() raising → no crash, both tests reach run_suite unchanged."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_adapter.discover.side_effect = ConnectionError("nerve unreachable")
    mock_load.return_value = SimpleNamespace(
        tests=[
            SimpleNamespace(name="a", tags=["smoke", "android"]),
            SimpleNamespace(name="b", tags=["smoke", "ios"]),
        ],
        package="com.x",
    )
    mock_runner_cls.return_value.run_suite.side_effect = _make_result_from_suite

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV",
            tags=["smoke"],
            output_dir=str(tmp_path / "out"),
        )
    )

    call_args, _ = mock_runner_cls.return_value.run_suite.call_args
    assert [t.name for t in call_args[0].tests] == ["a", "b"]
    assert out["total_run"] == 2
    assert out["total_skipped"] == 0


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_platform_mismatch_untagged_test_always_runs(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """A test with no platform tag always runs, even on a resolved ios device."""
    from linc_crucible.bvt_tools import crucible_bvt

    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_adapter.discover.return_value = [{"serial": "DEV", "platform": "ios"}]
    mock_load.return_value = SimpleNamespace(
        tests=[SimpleNamespace(name="c", tags=["smoke"])],
        package="com.x",
    )
    mock_runner_cls.return_value.run_suite.side_effect = _make_result_from_suite

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device="DEV",
            tags=["smoke"],
            output_dir=str(tmp_path / "out"),
        )
    )

    assert out["ran"] is True
    assert out["total_run"] == 1
    assert out["total_skipped"] == 0
    assert out["tests"][0]["name"] == "c"


# ---------------------------------------------------------------------------
# budget_s (audit F11) — crucible_bvt exposes the suite wall-clock budget and
# forwards it to BVTRunner.run_suite; a budget-cut run serializes passed=false.
# ---------------------------------------------------------------------------


def _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path) -> None:
    (tmp_path / ".bvt").mkdir()
    mock_forge.get.return_value = SimpleNamespace(path=str(tmp_path))
    mock_adapter.available = True
    mock_load.return_value = SimpleNamespace(tests=[], package="com.x")


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_budget_s_forwarded_to_run_suite(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """crucible_bvt(budget_s=...) forwards it as an explicit keyword to run_suite."""
    from linc_crucible.bvt_tools import crucible_bvt

    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()

    await crucible_bvt(
        project="MyApp", device="DEV1", output_dir=str(tmp_path / "out"), budget_s=42.0
    )

    mock_runner_cls.return_value.run_suite.assert_called_once()
    _, call_kwargs = mock_runner_cls.return_value.run_suite.call_args
    assert call_kwargs["budget_s"] == 42.0


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_budget_s_default_is_runner_default(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """Omitting budget_s forwards the runner's own default (_SUITE_BUDGET_DEFAULT_S)."""
    from linc_crucible.bvt_tools import crucible_bvt
    from linc_crucible.core.bvt.runner import _SUITE_BUDGET_DEFAULT_S

    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()

    await crucible_bvt(project="MyApp", device="DEV1", output_dir=str(tmp_path / "out"))

    _, call_kwargs = mock_runner_cls.return_value.run_suite.call_args
    assert call_kwargs["budget_s"] == _SUITE_BUDGET_DEFAULT_S


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_budget_exhausted_run_serializes_red(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
) -> None:
    """A budget-cut run (green executed tests + budget skips) reports passed=false."""
    from linc_crucible.bvt_tools import crucible_bvt
    from linc_crucible.core.bvt.runner import BVTBudgetedRunResult

    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)
    reason = "HARNESS: suite budget 1000s exhausted after 1400s"
    mock_runner_cls.return_value.run_suite.return_value = BVTBudgetedRunResult(
        device="DEV1",
        package="com.x",
        tests=[
            BVTTestResult(name="a", passed=True, steps_run=1, steps_passed=1),
            BVTTestResult(name="b", passed=False, skipped=True, skip_reason=reason),
        ],
        budget_exhausted=True,
    )

    out = json.loads(
        await crucible_bvt(
            project="MyApp", device="DEV1", output_dir=str(tmp_path / "out"), budget_s=1000
        )
    )

    assert out["ran"] is True
    assert out["passed"] is False
    assert out["budget_exhausted"] is True
    assert out["total_passed"] == 1
    assert out["total_skipped"] == 1
    assert out["tests"][1]["skip_reason"] == reason


def test_crucible_bvt_docstring_documents_budget_s() -> None:
    """The docstring Args section documents budget_s."""
    from linc_crucible.bvt_tools import crucible_bvt

    assert "budget_s:" in crucible_bvt.__doc__


# ── Personal-device policy (epic linc-crucible-6jf, bead 6jf.2) ───────


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_personal_device_is_refused_before_reservation_and_runner(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    _set_personal_policy(monkeypatch)
    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)

    out = json.loads(await crucible_bvt(project="MyApp", device=_PERSONAL_WATCH_SERIAL))

    assert out["ran"] is False
    assert out["success"] is False
    assert out["error_category"] == "personal_device"
    assert _PERSONAL_WATCH_SERIAL in out["error"]
    assert "allow_personal=True" in out["error"]
    mock_runner_cls.assert_not_called()
    mock_nerve_client.pool.assert_not_called()  # refused before the #110 reservation guard


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_personal_device_runs_with_allow_personal(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    _set_personal_policy(monkeypatch)
    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)
    mock_nerve_client.pool.return_value = {}
    mock_runner_cls.return_value.run_suite.return_value = _fake_run_result()

    out = json.loads(
        await crucible_bvt(
            project="MyApp",
            device=_PERSONAL_WATCH_SERIAL,
            output_dir=str(tmp_path / "out"),
            allow_personal=True,
        )
    )

    assert out["ran"] is True
    assert "error_category" not in out
    mock_runner_cls.return_value.run_suite.assert_called_once()


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_unreadable_registry_refuses_bvt(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linc_crucible.server_context as ctx
    from linc_crucible.bvt_tools import crucible_bvt
    from linc_crucible.core.device_policy import load_device_policy

    bad_fleet = tmp_path / "absent-fleet.yaml"
    bad_fleet.write_text("::: unparseable :::\n")
    monkeypatch.setattr(ctx, "_cached_device_policy", load_device_policy(bad_fleet))
    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)

    out = json.loads(await crucible_bvt(project="MyApp", device="BENCH1"))

    assert out["ran"] is False
    assert out["error_category"] == "personal_device"
    assert "absent-fleet.yaml" in out["error"]
    mock_runner_cls.assert_not_called()


def test_crucible_bvt_docstring_documents_allow_personal() -> None:
    from linc_crucible.bvt_tools import crucible_bvt

    doc = crucible_bvt.__doc__ or ""
    assert "allow_personal:" in doc
    assert "Personal devices" in doc


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_personal_device_alias_with_personal_hardware_id_is_refused(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[contract] bvt with device 192.0.2.10:5555 whose discover record carries a personal
    hardware_id is refused before any reservation or runner call."""
    from linc_crucible.bvt_tools import crucible_bvt

    _set_personal_policy(monkeypatch)
    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)
    mock_nerve_client.discover.return_value = [
        {
            "serial": "192.0.2.10:5555",
            "id": "192.0.2.10:5555",
            "hardware_id": _PERSONAL_WATCH_SERIAL,
            "platform": "android",
            "state": "online",
        }
    ]

    out = json.loads(await crucible_bvt(project="MyApp", device="192.0.2.10:5555"))

    assert out["ran"] is False
    assert out["success"] is False
    assert out["error_category"] == "personal_device"
    assert "192.0.2.10:5555" in out["error"]
    assert "allow_personal=True" in out["error"]
    mock_runner_cls.assert_not_called()
    mock_nerve_client.pool.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_serial",
    [
        _PERSONAL_WATCH_SERIAL.lower(),
        f" {_PERSONAL_WATCH_SERIAL} ",
        f"{_PERSONAL_WATCH_SERIAL}\n",
    ],
)
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_personal_device_case_and_whitespace_variants_refused(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    raw_serial: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[contract] Casefold and whitespace variants of a personal serial are refused in bvt."""
    from linc_crucible.bvt_tools import crucible_bvt

    _set_personal_policy(monkeypatch)
    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)

    out = json.loads(await crucible_bvt(project="MyApp", device=raw_serial))

    assert out["ran"] is False
    assert out["error_category"] == "personal_device"
    mock_runner_cls.assert_not_called()


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._nerve_client")
@patch("linc_crucible.core.bvt.runner.BVTRunner")
@patch("linc_crucible.core.bvt.schema.load_bvt_dir")
@patch("linc_crucible.server_context._bvt_nerve_adapter")
@patch("linc_crucible.server_context._forge_reader")
async def test_nerve_unreachable_during_precheck_still_refuses_personal_serial(
    mock_forge: MagicMock,
    mock_adapter: MagicMock,
    mock_load: MagicMock,
    mock_runner_cls: MagicMock,
    mock_nerve_client: MagicMock,
    mock_init: MagicMock,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[test] Nerve unreachable (ConnectionError) during pre-check falls back to string-only
    check and still refuses an exact personal serial."""
    from linc_crucible.bvt_tools import crucible_bvt

    _set_personal_policy(monkeypatch)
    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)
    mock_nerve_client.discover.side_effect = ConnectionError("nerve unreachable")

    out = json.loads(await crucible_bvt(project="MyApp", device=_PERSONAL_WATCH_SERIAL))

    assert out["ran"] is False
    assert out["error_category"] == "personal_device"
    mock_runner_cls.assert_not_called()


# ---------------------------------------------------------------------------
# CLI exit codes + error categories for BVT preconditions (linc-crucible-5ro)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected_category", "expected_code"),
    [
        ("unknown-project", "not_found", "UNKNOWN_PROJECT"),
        ("suite-dir-missing", "not_found", "SUITE_DIR_MISSING"),
        ("nerve-daemon-unreachable", "unavailable", "NERVE_DAEMON_UNREACHABLE"),
        ("device-not-ready", "unavailable", "DEVICE_NOT_READY"),
        ("device-reserved", "unavailable", "DEVICE_RESERVED"),
        ("no-tests-match-tags", "internal", "NO_TESTS_MATCH_TAGS"),
        ("package-mismatch", "internal", "PACKAGE_MISMATCH"),
        ("platform-mismatch", "internal", "PLATFORM_MISMATCH"),
    ],
)
def test_cli_bvt_precondition_reasons_exit_1_with_mapped_category(
    reason: str,
    expected_category: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[contract] crucible bvt for each precondition reason exits 1 with the mapped category."""
    from click.testing import CliRunner

    import linc_crucible.bvt_tools as bvt_tools_mod
    import linc_crucible.server_context as ctx
    from linc_crucible.cli import cli

    monkeypatch.setattr(ctx, "_initialized", True)
    mock_forge = MagicMock()
    mock_forge.get.return_value = None
    monkeypatch.setattr(ctx, "_forge_reader", mock_forge)
    real_precondition = bvt_tools_mod._bvt_precondition
    with patch.object(
        bvt_tools_mod,
        "_bvt_precondition",
        side_effect=lambda _r, _d: real_precondition(reason, f"precondition failed: {reason}"),
    ):
        res = CliRunner().invoke(cli, ["bvt", "--project", "MyApp", "--device", "DEV1"])

    assert res.exit_code == 1, res.output
    assert res.stdout == ""
    envelope = json.loads(res.stderr)
    assert envelope["error"]["category"] == expected_category
    assert envelope["error"]["code"] == expected_code
    assert f"precondition failed: {reason}" in envelope["error"]["message"]
    assert envelope["error"]["details"]["ran"] is False
    assert envelope["error"]["details"]["reason"] == reason


def test_cli_bvt_personal_device_refusal_exits_1_with_permission_category(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[contract] crucible bvt personal_device refusal exits 1 with category 'permission'."""
    from click.testing import CliRunner

    import linc_crucible.server_context as ctx
    from linc_crucible.cli import cli

    monkeypatch.setattr(ctx, "_initialized", True)
    mock_forge = MagicMock()
    mock_adapter = MagicMock()
    mock_load = MagicMock()
    _patch_bvt_env(mock_forge, mock_adapter, mock_load, tmp_path)
    monkeypatch.setattr(ctx, "_forge_reader", mock_forge)
    monkeypatch.setattr(ctx, "_bvt_nerve_adapter", mock_adapter)
    monkeypatch.setattr(ctx, "_nerve_client", MagicMock())
    _set_personal_policy(monkeypatch)

    res = CliRunner().invoke(
        cli, ["bvt", "--project", "MyApp", "--device", _PERSONAL_WATCH_SERIAL]
    )
    assert res.exit_code == 1, res.output
    assert res.stdout == ""
    envelope = json.loads(res.stderr)
    assert envelope["error"]["category"] == "permission"
    assert envelope["error"]["code"] == "PERSONAL_DEVICE"
    assert envelope["error"]["details"]["ran"] is False
    assert envelope["error"]["details"]["error_category"] == "personal_device"
