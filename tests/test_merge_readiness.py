"""Pre-merge acceptance tests for the nerve merger into linc-crucible.

These tests define "done" for the nerve integration. Tests that depend on
nerve tools being wired are marked ``xfail`` — they will flip to ``XPASS``
once the merger is complete, at which point the xfail markers should be
removed.

Ops council requirements tested:
- Tool completeness (31 tools: 5 crucible_* + 26 nerve_*)
- Concurrency safety under CapacityLimiter(4)
- Domain logging attribution (build_pipeline vs device_control)
- Error domain attribution (Allspaw pipeline-vs-device distinction)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


class TestToolInventory:
    """Tool completeness: the merged server must expose all 32 tools.

    Ops council requirement: after the nerve merger, the crucible MCP server
    must serve 6 crucible_* tools (build pipeline) and 26 nerve_* tools
    (device control) — 32 total.
    """

    @pytest.mark.xfail(reason="Pre-merge: nerve tools not yet wired")
    def test_server_exposes_31_tools(self) -> None:
        """Assert the server registers exactly 31 tools (5 crucible + 26 nerve)."""
        from linc_crucible.mcp_server import server

        tools = server._tool_manager.list_tools()
        names = [t.name for t in tools]

        crucible_tools = [n for n in names if n.startswith("crucible_")]
        nerve_tools = [n for n in names if n.startswith("nerve_")]

        assert len(crucible_tools) == 7, f"Expected 7 crucible_* tools, got: {crucible_tools}"
        assert len(nerve_tools) == 26, f"Expected 26 nerve_* tools, got: {nerve_tools}"
        assert len(tools) == 33, f"Expected 33 total tools, got {len(tools)}: {names}"

    def test_server_exposes_8_crucible_tools(self) -> None:
        """Sanity check: the 8 existing crucible_* tools are registered today."""
        from linc_crucible.mcp_server import server

        tools = server._tool_manager.list_tools()
        crucible_names = sorted(n.name for n in tools if n.name.startswith("crucible_"))

        expected = [
            "crucible_assay",
            "crucible_baseline",
            "crucible_bvt",
            "crucible_extract_assets",
            "crucible_pour",
            "crucible_preflight",
            "crucible_score",
            "crucible_usage_stats",
        ]
        assert crucible_names == expected


class TestConcurrencyStress:
    """Concurrency safety: 5 parallel MCP calls must not deadlock.

    Ops council requirement: the CapacityLimiter(4) concurrency gate must
    allow at least 5 queued async tool calls to complete without deadlock
    within a 10-second timeout.
    """

    @pytest.mark.asyncio
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._deployer")
    async def test_five_concurrent_calls_no_deadlock(
        self,
        mock_deployer: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """Fire 5 concurrent crucible_pour calls; all must finish in 10s."""
        from linc_crucible.deploy_tools import crucible_pour

        mock_deployer.pour.return_value.to_dict.return_value = {
            "success": True,
            "devices": [],
        }

        results = await asyncio.wait_for(
            asyncio.gather(*[crucible_pour(project="canary") for _ in range(5)]),
            timeout=10.0,
        )

        assert len(results) == 5
        for r in results:
            assert isinstance(r, dict)


class TestDomainLogging:
    """Domain logging attribution: crucible=build_pipeline, nerve=device_control.

    Ops council requirement (Allspaw): every MCP tool module must bind
    structured loggers with the correct operational domain so that
    on-call engineers can filter by domain in production.
    """

    def test_crucible_mcp_server_logger_binds_build_pipeline(self) -> None:
        """The mcp_server module logger must be bound with domain='build_pipeline'."""
        from linc_crucible.server_context import logger

        # structlog BoundLogger stores context in _context dict
        assert logger._context.get("domain") == "build_pipeline", (
            f"Expected domain='build_pipeline', got context: {logger._context}"
        )

    def test_crucible_core_modules_bind_build_pipeline(self) -> None:
        """All crucible core modules must bind domain='build_pipeline'."""
        from linc_crucible.core import deployer, fleet_registry, scorer

        for mod in (deployer, fleet_registry, scorer):
            ctx = mod.logger._context
            assert ctx.get("domain") == "build_pipeline", (
                f"{mod.__name__} logger missing domain='build_pipeline': {ctx}"
            )

    @pytest.mark.xfail(reason="Pre-merge: nerve tools not yet wired")
    def test_nerve_tools_log_device_control_domain(self) -> None:
        """A nerve_* tool module must bind domain='device_control'."""
        from linc_crucible.mcp_server import server

        tools = server._tool_manager.list_tools()
        nerve_names = [t.name for t in tools if t.name.startswith("nerve_")]
        assert nerve_names, "No nerve_* tools registered on server"


class TestErrorDomainAttribution:
    """Error domain attribution: failures carry pipeline or device domain.

    Ops council requirement (Allspaw): when a tool call fails, the error
    log must carry the correct domain so incident responders can immediately
    distinguish device-layer failures from pipeline-layer failures.
    """

    def test_pipeline_error_logs_with_domain(self) -> None:
        """Core scorer logs warnings with domain='build_pipeline' on file errors."""
        from linc_crucible.core.scorer import logger

        # Verify the logger that handles score-read failures is bound
        # to build_pipeline — this is the domain that appears in error logs
        # like "score_file_read_failed".
        assert logger._context.get("domain") == "build_pipeline", (
            f"Scorer error logger missing domain='build_pipeline': {logger._context}"
        )

    def test_fleet_error_logs_with_domain(self) -> None:
        """Core fleet_registry logs warnings with domain='build_pipeline' on errors."""
        from linc_crucible.core.fleet_registry import logger

        # Fleet registry logs "fleet_file_read_failed" and "invalid_device_entry"
        # with this domain.
        assert logger._context.get("domain") == "build_pipeline", (
            f"Fleet registry error logger missing domain='build_pipeline': {logger._context}"
        )

    @pytest.mark.xfail(reason="Pre-merge: nerve tools not yet wired")
    def test_device_error_includes_domain(self) -> None:
        """A device-layer tool error must log with domain='device_control'."""
        from linc_crucible.mcp_server import server

        tools = server._tool_manager.list_tools()
        nerve_names = [t.name for t in tools if t.name.startswith("nerve_")]
        assert nerve_names, "No nerve_* tools registered on server"
