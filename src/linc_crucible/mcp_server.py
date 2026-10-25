"""MCP server for Linc Crucible — Claude Code integration.

Exposes crucible tools via FastMCP for device fleet management,
deployment, quality assurance, and scoring from within Claude Code.

Start with: ``crucible bellows``
"""

import linc_crucible.assay_tools
import linc_crucible.asset_tools
import linc_crucible.baseline_tools
import linc_crucible.bvt_tools
import linc_crucible.deploy_tools
import linc_crucible.stats_tools  # noqa: F401
from linc_crucible.server_context import _observed_tool, server

__all__ = ["_observed_tool", "server"]
