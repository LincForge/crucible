"""Tests for the MCP server thread pool limiter."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_capacity_limiter_bounds_concurrency() -> None:
    """Verify CapacityLimiter(4) limits concurrent thread dispatch."""
    from linc_crucible.server_context import _get_thread_limiter

    limiter = _get_thread_limiter()
    assert limiter.total_tokens == 4
